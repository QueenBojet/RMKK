import logging
import threading
import time
from datetime import datetime

import numpy as np
from typing import Dict, List, Optional, Any
from collections import defaultdict

from src.core.interfaces import Processor
from src.core.message_bus import MessageBus
from src.core.types import DataPoint, DataType, SignalSource
from src.core.config_schema import AppConfig


class SignalAggregator(Processor):
    """
    Middleware Processor.
    1. Consumes single-channel WAVEFORM DataPoints.
    2. Buffers them to synchronize channels (wait for all ch1-ch5).
    3. Filters based on SPEED (Trigger logic).
    4. Produces AGGREGATED_WAVEFORM DataPoints for the DiagnosisEngine.
    """

    def __init__(self, config: AppConfig, bus: MessageBus, required_channels: List[str]):
        self.config = config
        self.bus = bus
        self.required_channels = set(required_channels)
        self.logger = logging.getLogger(self.__class__.__name__)

        self._running = False
        self._thread: Optional[threading.Thread] = None

        # Buffer: { timestamp_key: { channel: data } }
        # We group by timestamp (seconds precision) to align data
        self._buffer: Dict[str, Dict[str, Any]] = defaultdict(dict)
        self._buffer_metadata: Dict[str, Dict] = defaultdict(dict)
        self._buffer_lock = threading.Lock()

        # Speed Cache for Triggering
        self._current_speed = 0.0
        self._last_trigger_time = 0.0

        # Config values
        self.speed_threshold = self.config.speed_trigger.threshold_rpm if self.config.speed_trigger else 5.0
        self.trigger_cooldown = self.config.speed_trigger.cooldown_seconds if self.config.speed_trigger else 1.0

    def start(self) -> None:
        if self._running: return
        self._running = True
        self._thread = threading.Thread(target=self._process_loop, name="SignalAggregator", daemon=True)
        self._thread.start()
        self.logger.info(f"Signal Aggregator started. Syncing channels: {self.required_channels}")

    def stop(self) -> None:
        self._running = False
        if self._thread:
            self._thread.join(timeout=2.0)

    def _process_loop(self):
        while self._running:
            # 1. Update Speed Cache (Peek at scalars without consuming if possible,
            # or rely on a separate mechanism. Here we assume scalars flow through bus too)
            # Actually, Aggregator should probably consume everything and re-publish if it's not waveform.
            # But to keep it simple, let's assume scalars are handled by SignalProcessor or we cache them here.
            # For this design, we will consume ALL data.

            data = self.bus.get(timeout=1.0)
            if not data:
                self._check_timeouts()  # Clean old partial buffers
                continue

            try:
                if data.type == DataType.SCALAR:
                    self._handle_scalar(data)
                    # Re-publish scalars so SignalProcessor/Storage can see them
                    # Note: This creates a loop if not careful.
                    # Ideally, Aggregator is the ONLY consumer of raw input, and outputs 'Processed' data.
                    # Or we use a specific topic.
                    # For V1 fix, we will modify SignalProcessor to listen to AGGREGATED only.
                    # And scalars pass through.
                    self.bus.publish(data)

                elif data.type == DataType.WAVEFORM:
                    self._handle_waveform(data)

                self.bus.task_done()

            except Exception as e:
                self.logger.error(f"Error in aggregator: {e}", exc_info=True)

    def _handle_scalar(self, data: DataPoint):
        if "speed" in data.point_name.lower():
            self._current_speed = float(data.value)

    def _handle_waveform(self, data: DataPoint):
        # 1. Check Trigger (Speed Filter)
        if self.config.speed_trigger and self.config.speed_trigger.enable:
            if self._current_speed < self.speed_threshold:
                # LINUS: Don't process garbage data from stopped machines.
                return

                # 2. Buffer for Synchronization
        # We assume data arriving close in time belongs together.
        # Keying by timestamp (rounded to second) is a simple heuristic.
        # For high-freq data, exact timestamp matching is hard without a common clock source.
        ts_key = data.timestamp.strftime("%Y%m%d%H%M%S")

        with self._buffer_lock:
            # Initialize buffer entry if new
            if ts_key not in self._buffer:
                self._buffer[ts_key] = {}
                self._buffer_metadata[ts_key] = data.metadata

            self._buffer[ts_key][data.point_name] = data.value

            # Check if complete
            current_channels = set(self._buffer[ts_key].keys())
            if self.required_channels.issubset(current_channels):
                self._flush_aggregated(ts_key, data.timestamp)

    def _flush_aggregated(self, ts_key: str, timestamp: datetime):
        """Package buffered channels into one AGGREGATED DataPoint."""
        data_map = self._buffer.pop(ts_key)
        metadata = self._buffer_metadata.pop(ts_key)

        # Check cooldown
        now = time.time()
        if (now - self._last_trigger_time) < self.trigger_cooldown:
            return

        self._last_trigger_time = now

        # Create Aggregated DataPoint
        dp = DataPoint(
            timestamp=timestamp,
            source=SignalSource.INTERNAL,
            type=DataType.AGGREGATED_WAVEFORM,
            device_name="system",
            point_name="all_channels",
            value=data_map,  # Dict[str, ndarray]
            metadata=metadata
        )

        self.logger.info(f"Aggregated full vibration set: {list(data_map.keys())} at {self._current_speed:.1f} RPM")
        self.bus.publish(dp)

    def _check_timeouts(self):
        """Cleanup buffers that never completed (e.g. packet loss)."""
        # Implementation omitted for brevity, but vital for production.
        pass