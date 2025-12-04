import logging
import threading
import time
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional

from src.core.interfaces import Processor
from src.core.message_bus import MessageBus
from src.core.types import DataPoint, DataType, SignalSource, DiagnosisResult
from src.core.config_schema import DiagnosisConfig
from src.modelTrain.diagnosis_engine import DiagnosisEngine


class SignalProcessor(Processor):
    """
    Consumes WAVEFORM DataPoints, runs diagnosis, and produces HEALTH/FAULT DataPoints.
    """

    def __init__(self, config: DiagnosisConfig, bus: MessageBus):
        self.config = config
        self.bus = bus
        self.logger = logging.getLogger(self.__class__.__name__)

        # Initialize the legacy DiagnosisEngine
        # In a full refactor, we would rewrite DiagnosisEngine too,
        # but for now we wrap it to respect the "One Step at a Time" rule.
        # We need to adapt the Pydantic config back to a dict for the legacy engine
        legacy_config_dict = {
            "diagnosis_config": config.dict()
        }
        self.engine = DiagnosisEngine(legacy_config_dict)

        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._latest_scalars: Dict[str, float] = {}  # Cache for scalar values (speed, temp) needed for diagnosis

    def start(self) -> None:
        if self._running: return
        self.logger.info("Starting Signal Processor...")
        self._running = True
        self._thread = threading.Thread(target=self._process_loop, name="SignalProcessor", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._running = False
        if self._thread:
            self._thread.join(timeout=2.0)
        self.logger.info("Signal Processor stopped.")

    def _process_loop(self):
        while self._running:
            # Block for a short time to allow checking _running flag
            data = self.bus.get(timeout=1.0)

            if not data:
                continue

            try:
                if data.type == DataType.SCALAR:
                    self._handle_scalar(data)
                elif data.type == DataType.WAVEFORM:
                    self._handle_waveform(data)

                # Signal task done to the queue
                self.bus.task_done()

            except Exception as e:
                self.logger.error(f"Error processing data point {data.point_name}: {e}", exc_info=True)

    def _handle_scalar(self, data: DataPoint):
        """Cache scalar values (like speed) that might be needed for diagnosis."""
        key = f"{data.device_name}.{data.point_name}"
        # We assume scalar value is float based on DataPoint validation
        self._latest_scalars[key] = float(data.value)

    def _handle_waveform(self, data: DataPoint):
        """Run diagnosis on waveform data."""
        self.logger.debug(f"Processing waveform from {data.device_name}.{data.point_name}")

        # Prepare data for DiagnosisEngine
        # Engine expects: Dict[str, np.ndarray] for vibration data
        # We map point_name (e.g. 'ch1') to the array
        vibration_data = {data.point_name: data.value}

        # Get sampling rate from metadata or config
        sampling_rate = data.metadata.get('sampling_rate', 25600)

        # Get associated speed data if available
        # This relies on conventions or metadata.
        # For now, let's try to find a matching speed scalar in our cache.
        # In a real system, we might need time-aligned speed arrays.
        speed_key = f"{data.device_name}.speed"  # Convention
        current_speed = self._latest_scalars.get(speed_key, 0.0)

        # Construct speed array (constant speed assumption for single data point duration)
        # diagnosis_engine expects an array
        n_samples = len(data.value)
        speed_data = np.full(n_samples, current_speed)

        # Run Diagnosis
        # Modbus data is also passed as a dict of scalars
        result_dict = self.engine.diagnose(
            vibration_data=vibration_data,
            sampling_rate=sampling_rate,
            speed_data=speed_data,
            modbus_data=self._latest_scalars
        )

        if result_dict:
            self._publish_results(data, result_dict)

    def _publish_results(self, source_data: DataPoint, result: Dict[str, Any]):
        """Convert diagnosis results back to DataPoints and publish."""
        timestamp = source_data.timestamp

        # 1. Overall Health Score
        health_score = result['health_scores']['overall']['score']

        hp = DataPoint(
            timestamp=timestamp,
            source=SignalSource.INTERNAL,
            type=DataType.HEALTH_SCORE,
            device_name=source_data.device_name,
            point_name="health_score",
            value=health_score,
            metadata={"components": result['health_scores']['overall'].get('subsystems', {})}
        )
        self.bus.publish(hp)

        # 2. Faults
        for fault in result.get('faults', []):
            fp = DataPoint(
                timestamp=timestamp,
                source=SignalSource.INTERNAL,
                type=DataType.FAULT,
                device_name=fault.get('device_name', source_data.device_name),
                point_name="fault_detected",
                value=1.0,  # Indicator
                metadata={
                    "fault_type": fault.get('fault_type'),
                    "description": fault.get('fault_description'),
                    "severity": fault.get('severity')
                }
            )
            self.bus.publish(fp)
            self.logger.warning(f"Fault detected: {fault.get('fault_description')}")