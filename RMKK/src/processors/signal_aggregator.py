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

        # Speed history: (timestamp, value)
        # Using deque is fine for simple history, but be aware of O(N) lookup. 
        # For 1000 items it's acceptable.
        self._speed_history = deque(maxlen=2000) 
        
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

        # REMOVED: self._current_speed. It is evil. Don't use global state for async events.
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
            # Short timeout to keep loop responsive
            data = self.bus.get(timeout=0.1)
            if not data: continue

            try:
                if data.type == DataType.SCALAR:
                    self._handle_scalar(data)
                    # Forward scalar data to storage/others
                    self.bus.publish(data)

                elif data.type == DataType.WAVEFORM:
                    self._handle_waveform(data)

                self.bus.task_done()
            except Exception as e:
                self.logger.error(f"Aggregator Error: {e}", exc_info=True)

    def _handle_scalar(self, data: DataPoint):
        key = f"{data.device_name}.{data.point_name}"
        
        # Only record the configured speed point
        if key == self.speed_key:
            val = float(data.value)
            with self._buffer_lock:
                self._speed_history.append((data.timestamp.timestamp(), val))

    def _handle_waveform(self, data: DataPoint):
        # LINUS: I deleted the idiotic check "if current_speed < threshold" here.
        # We process ALL waveform data into the buffer. 
        # We decide whether to keep it later, based on its TIMESTAMP, not based on "now".
        
        # Group by timestamp (using seconds resolution string for key)
        # Note: Ideally use exact timestamp if synchronized, but string key is safer for slight jitters
        ts_key = data.timestamp.strftime("%Y%m%d%H%M%S")

        with self._buffer_lock:
            if ts_key not in self._vib_buffer:
                self._vib_buffer[ts_key] = {}
                self._vib_metadata[ts_key] = data.metadata

            self._vib_buffer[ts_key][data.point_name] = data.value

            # Check if we have all required channels for this timestamp
            if self.required_channels.issubset(self._vib_buffer[ts_key].keys()):
                self._process_complete_set(ts_key, data.timestamp)

    def _process_complete_set(self, ts_key: str, timestamp: datetime):
        # Pop data from buffer
        data_map = self._vib_buffer.pop(ts_key)
        metadata = self._vib_metadata.pop(ts_key)

        # 1. Check Signal Quality (RMS & Mean Abs > 1e-6)
        # LINUS: This was missing in your code.
        if not self._check_signal_quality(data_map):
            self.logger.warning(f"Signal Quality Failed (RMS/Mean < 1e-6). Dropping set {ts_key}.")
            return

        # 2. Extract Timing Info
        first_ch = list(data_map.values())[0]
        n_points = len(first_ch)
        fs = metadata.get('sampling_rate', 25600)
        duration = n_points / fs
        
        start_ts = timestamp.timestamp()
        end_ts = start_ts + duration

        # 3. Speed Validation (The Truth Source)
        # Retrieve speed data covering this time window [start_ts, end_ts]
        # We add a small buffer (0.5s) to ensure interpolation works at edges
        with self._buffer_lock:
            history_slice = [
                (t, v) for t, v in self._speed_history
                if start_ts - 1.0 <= t <= end_ts + 1.0
            ]

        if not history_slice:
            self.logger.warning(f"No speed data found for window {ts_key}. Dropping.")
            return

        # Extract values strictly within the acquisition window for validation
        speeds_in_window = [v for t, v in history_slice if start_ts <= t <= end_ts]
        
        if not speeds_in_window:
            # Fallback: if data points are sparse (e.g. 1Hz) and window is small
            # use the nearest points from history_slice
            speeds_in_window = [v for t, v in history_slice]

        speeds_arr = np.array(speeds_in_window)
        valid_count = np.sum(speeds_arr > self.threshold)
        valid_ratio = valid_count / len(speeds_arr) if len(speeds_arr) > 0 else 0.0
        avg_speed = np.mean(speeds_arr)

        # 4. Threshold Check
        if valid_ratio < self.min_valid_ratio:
            self.logger.debug(
                f"Speed Validation Failed. Ratio: {valid_ratio:.2f} < {self.min_valid_ratio}. "
                f"Avg Speed: {avg_speed:.1f}. Dropping."
            )
            return

        # 5. Cooldown Check
        # Only apply cooldown if we actually have a valid event
        now = time.time()
        if (now - self._last_trigger_time) < self.cooldown:
            return
        self._last_trigger_time = now

        # 6. Interpolate Speed for Resampling
        speed_upsampled = self._interpolate_speed(history_slice, start_ts, end_ts, n_points, avg_speed)
        data_map['speed'] = speed_upsampled

        # 7. Publish
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
            f"Avg: {avg_speed:.1f} RPM. Window: {ts_key}"
        )
        self.bus.publish(dp)

    def _check_signal_quality(self, data_map: Dict[str, Any]) -> bool:
        """
        Verify that ALL vibration channels have:
        1. RMS > 1e-6
        2. Mean(Abs) > 1e-6
        """
        limit = 1e-6
        for ch_name, signal in data_map.items():
            if ch_name == 'speed': continue # Skip speed if present
            
            arr = np.array(signal)
            rms = np.sqrt(np.mean(arr**2))
            mean_abs = np.mean(np.abs(arr))
            
            if rms <= limit or mean_abs <= limit:
                # self.logger.debug(f"Channel {ch_name} too weak. RMS:{rms:.2e}, MeanAbs:{mean_abs:.2e}")
                return False
        return True

    def _interpolate_speed(self, history, start_ts, end_ts, n_points, default_val):
        try:
            t_x = np.array([x[0] for x in history])
            v_y = np.array([x[1] for x in history])

            # Sort and de-duplicate (time can flow backwards in bad mocks, fix it)
            sorted_indices = np.argsort(t_x)
            t_x = t_x[sorted_indices]
            v_y = v_y[sorted_indices]

            _, unique_idx = np.unique(t_x, return_index=True)
            t_x = t_x[unique_idx]
            v_y = v_y[unique_idx]

            if len(t_x) > 1:
                t_new = np.linspace(start_ts, end_ts, n_points)
                f = interp1d(t_x, v_y, kind='linear', fill_value="extrapolate")
                return f(t_new)
        except Exception as e:
            self.logger.error(f"Interpolation error: {e}")
        
        return np.full(n_points, default_val)
