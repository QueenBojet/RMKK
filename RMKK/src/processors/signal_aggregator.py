import logging
import threading
import time
import numpy as np
from datetime import datetime
from collections import deque, defaultdict
from typing import Dict, List, Optional, Any
from scipy.interpolate import interp1d

from src.core.interfaces import Processor
from src.core.message_bus import MessageBus
from src.core.types import DataPoint, DataType, SignalSource
from src.core.config_schema import AppConfig


class SignalAggregator(Processor):
    def __init__(self, config: AppConfig, bus: MessageBus, required_channels: List[str]):
        self.config = config
        self.bus = bus
        self.required_channels = set(required_channels)
        self.logger = logging.getLogger(self.__class__.__name__)

        self._running = False
        self._thread: Optional[threading.Thread] = None

        self._speed_history = deque(maxlen=1000)
        self._vib_buffer: Dict[str, Dict[str, Any]] = defaultdict(dict)
        self._vib_metadata: Dict[str, Dict] = defaultdict(dict)
        self._buffer_lock = threading.Lock()

        self.trigger_cfg = self.config.speed_trigger
        self.threshold = self.trigger_cfg.threshold_rpm
        self.cooldown = self.trigger_cfg.cooldown_seconds

        self.min_valid_ratio = 0.8
        if hasattr(self.trigger_cfg, 'validation'):
            val_cfg = self.trigger_cfg.validation
            if isinstance(val_cfg, dict):
                self.min_valid_ratio = val_cfg.get('min_valid_ratio', 0.8)
            elif hasattr(val_cfg, 'min_valid_ratio'):
                self.min_valid_ratio = val_cfg.min_valid_ratio

        self.speed_key = f"{self.trigger_cfg.device_name}.{self.trigger_cfg.point_name}"

        self._current_speed = 0.0
        self._last_trigger_time = 0.0

        self.logger.info(f"Aggregator initialized. Watching: {self.speed_key} > {self.threshold} RPM.")

    def start(self) -> None:
        if self._running: return
        self._running = True
        self._thread = threading.Thread(target=self._process_loop, name="SignalAggregator", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._running = False
        if self._thread:
            self._thread.join(timeout=2.0)

    def _process_loop(self):
        while self._running:
            data = self.bus.get(timeout=0.1)
            if not data: continue

            try:
                if data.type == DataType.SCALAR:
                    self._handle_scalar(data)
                    self.bus.publish(data)

                elif data.type == DataType.WAVEFORM:
                    self._handle_waveform(data)

                self.bus.task_done()
            except Exception as e:
                self.logger.error(f"Aggregator Error: {e}", exc_info=True)

    def _handle_scalar(self, data: DataPoint):
        key = f"{data.device_name}.{data.point_name}"

        # --- DEBUG LOG FOR SPEED UPDATES ---
        # This will confirm if Modbus is actually sending the right key
        if key == self.speed_key:
            val = float(data.value)
            self._current_speed = val
            with self._buffer_lock:
                self._speed_history.append((data.timestamp.timestamp(), val))
            # Optional: Log occasionally to verify stream
            # if val > 10 and int(time.time()) % 5 == 0:
            #    self.logger.debug(f"Speed Update: {val:.1f} RPM")

        elif "speed" in data.point_name.lower() and self._current_speed == 0.0:
            self._current_speed = float(data.value)

    def _handle_waveform(self, data: DataPoint):
        # 1. Trigger Check (Instantaneous)
        if self._current_speed < self.threshold:
            # --- EXPLICIT LOGGING FOR DROPS ---
            # Now we log WARNING if we are getting vibration but speed says STOP.
            # Only log for channel 1 to avoid spamming 5 times per event.
            if "ch1" in data.point_name:
                self.logger.warning(
                    f"Ignored Vibration. Current Speed ({self._current_speed:.1f}) < Threshold ({self.threshold}). "
                    f"Waiting for machine to speed up..."
                )
            return

        # 2. Buffer Logic
        ts_key = data.timestamp.strftime("%Y%m%d%H%M%S")

        with self._buffer_lock:
            if ts_key not in self._vib_buffer:
                self._vib_buffer[ts_key] = {}
                self._vib_metadata[ts_key] = data.metadata

            self._vib_buffer[ts_key][data.point_name] = data.value

            if self.required_channels.issubset(self._vib_buffer[ts_key].keys()):
                self._process_complete_set(ts_key, data.timestamp)

    def _process_complete_set(self, ts_key: str, timestamp: datetime):
        data_map = self._vib_buffer.pop(ts_key)
        metadata = self._vib_metadata.pop(ts_key)

        now = time.time()
        if (now - self._last_trigger_time) < self.cooldown:
            return

        # --- VALIDATION ---
        first_ch = list(data_map.values())[0]
        n_points = len(first_ch)
        fs = metadata.get('sampling_rate', 25600)
        duration = n_points / fs

        start_ts = timestamp.timestamp()
        end_ts = start_ts + duration

        # Look for history. Now that we backdated vibration timestamp,
        # we should find plenty of Modbus data here.
        history_slice = [
            (t, v) for t, v in self._speed_history
            if start_ts - 0.5 <= t <= end_ts + 0.5
        ]

        # Extract just the speeds for validation
        speeds_val = [v for t, v in history_slice if start_ts <= t <= end_ts]

        if not speeds_val:
            # Fallback (Should be rare now)
            speeds_val = [self._current_speed]

        speeds_arr = np.array(speeds_val)
        valid_count = np.sum(speeds_arr > self.threshold)
        valid_ratio = valid_count / len(speeds_arr) if len(speeds_arr) > 0 else 0.0
        avg_speed = np.mean(speeds_arr)

        # Log string for debug
        speed_log_str = [round(x, 1) for x in speeds_arr.tolist()]
        if len(speed_log_str) > 10:
            speed_log_str = f"{speed_log_str[:5]} ... {speed_log_str[-5:]}"

        if valid_ratio < self.min_valid_ratio:
            self.logger.warning(
                f"Speed Validation Failed ({valid_ratio:.0%}). "
                f"Window: {speed_log_str}. Dropping data."
            )
            return

        self._last_trigger_time = now

        # Interpolate
        speed_upsampled = np.full(n_points, avg_speed)
        if len(history_slice) > 1:
            try:
                t_x = np.array([x[0] for x in history_slice])
                v_y = np.array([x[1] for x in history_slice])

                # De-duplicate timestamps to avoid RuntimeWarning
                _, unique_idx = np.unique(t_x, return_index=True)
                t_x = t_x[unique_idx]
                v_y = v_y[unique_idx]

                if len(t_x) > 1:
                    t_new = np.linspace(start_ts, end_ts, n_points)
                    f = interp1d(t_x, v_y, kind='linear', fill_value="extrapolate")
                    speed_upsampled = f(t_new)
            except Exception as e:
                self.logger.error(f"Interpolation error: {e}")

        data_map['speed'] = speed_upsampled

        dp = DataPoint(
            timestamp=timestamp,
            source=SignalSource.INTERNAL,
            type=DataType.AGGREGATED_WAVEFORM,
            device_name="system",
            point_name="merged_data",
            value=data_map,
            metadata=metadata
        )

        self.logger.info(
            f"Triggered & Validated ({valid_ratio:.0%}). "
            f"Avg: {avg_speed:.1f} RPM. Window: {speed_log_str}"
        )
        self.bus.publish(dp)