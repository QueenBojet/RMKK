import logging
import threading
import numpy as np
from typing import Dict, Any, Optional

from src.core.interfaces import Processor
from src.core.message_bus import MessageBus
from src.core.types import DataPoint, DataType, SignalSource
from src.core.config_schema import DiagnosisConfig
from src.modelTrain.diagnosis_engine import DiagnosisEngine


class SignalProcessor(Processor):
    """
    Diagnoses AGGREGATED_WAVEFORM data.
    """

    def __init__(self, config: DiagnosisConfig, bus: MessageBus, speed_config=None):
        self.config = config
        self.bus = bus
        self.logger = logging.getLogger(self.__class__.__name__)

        legacy_config_dict = {"diagnosis_config": config.dict()}
        self.engine = DiagnosisEngine(legacy_config_dict)

        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._latest_modbus = {}

    def start(self) -> None:
        if self._running: return
        self._running = True
        self._thread = threading.Thread(target=self._process_loop, name="SignalProcessor", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._running = False
        if self._thread:
            self._thread.join(timeout=2.0)

    def _process_loop(self):
        while self._running:
            data = self.bus.get(timeout=1.0)
            if not data: continue

            try:
                if data.type == DataType.SCALAR:
                    # Update Modbus Cache (for other params like temp)
                    key = f"{data.device_name}.{data.point_name}"
                    self._latest_modbus[key] = float(data.value)

                elif data.type == DataType.AGGREGATED_WAVEFORM:
                    self._handle_diagnosis(data)

                self.bus.task_done()
            except Exception as e:
                self.logger.error(f"Processor Error: {e}", exc_info=True)

    def _handle_diagnosis(self, data: DataPoint):
        # data.value is {'ch1':..., 'speed':...}
        payload = data.value
        sampling_rate = data.metadata.get('sampling_rate', 25600)

        # 1. Extract Speed (upsampled)
        speed_arr = payload.get('speed')

        # 2. Extract Vibration Channels
        vib_data = {k: v for k, v in payload.items() if k.startswith('ch')}

        if not vib_data or speed_arr is None:
            return

        # 3. Run Diagnosis
        # Aggregator already guaranteed speed validity, so we just run it.
        result = self.engine.diagnose(
            vibration_data=vib_data,
            sampling_rate=sampling_rate,
            speed_data=speed_arr,
            modbus_data=self._latest_modbus
        )

        if result:
            self._publish_results(data, result)

    def _publish_results(self, source_data, result):
        timestamp = source_data.timestamp
        health_score = result['health_scores']['overall']['score']

        hp = DataPoint(
            timestamp=timestamp,
            source=SignalSource.INTERNAL,
            type=DataType.HEALTH_SCORE,
            device_name="system",
            point_name="health_score",
            value=health_score,
            metadata={"components": result['health_scores']['overall'].get('subsystems', {})}
        )
        self.bus.publish(hp)

        for fault in result.get('faults', []):
            fp = DataPoint(
                timestamp=timestamp,
                source=SignalSource.INTERNAL,
                type=DataType.FAULT,
                device_name=fault.get('device_name', source_data.device_name),
                point_name="fault_detected",
                value=1.0,
                metadata={
                    "fault_type": fault.get('fault_type'),
                    "description": fault.get('fault_description'),
                    "severity": fault.get('severity')
                }
            )
            self.bus.publish(fp)
            self.logger.warning(f"Fault detected: {fault.get('fault_description')}")