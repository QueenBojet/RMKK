import time
import logging
import threading
import numpy as np
import pandas as pd
from typing import Dict, Any

from src.core.interfaces import Collector
from src.core.message_bus import MessageBus
from src.core.types import DataPoint, DataType, SignalSource
from src.core.config_schema import VibrationSimConfig
from src.collectors.mock_modbus_collector import MachineState


class MockVibrationCollector(Collector):
    """
    Generates simulated vibration waveform data.
    FIX: Timestamps now reflect the START of acquisition.
    """

    def __init__(self, config: VibrationSimConfig, bus: MessageBus):
        self.config = config
        self.bus = bus
        self.logger = logging.getLogger("MockVibrationCollector")
        self._running = False
        self._thread = None
        self.trigger_threshold = 5.0

    def start(self) -> None:
        if self._running: return
        self.logger.info("[TEST MODE] Starting Mock Vibration Collector...")
        self._running = True
        self._thread = threading.Thread(target=self._sim_loop, name="MockVibLoop", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._running = False
        if self._thread:
            self._thread.join(timeout=1.0)

    def get_status(self) -> Dict[str, Any]:
        return {"running": self._running}

    def _sim_loop(self):
        fs = 25600
        duration = self.config.duration_seconds
        check_interval = 0.5

        while self._running:
            # 1. Check speed BEFORE sleep (Simulate trigger condition)
            current_rpm = MachineState.get_current_rpm()
            if current_rpm < self.trigger_threshold:
                time.sleep(check_interval)
                continue

            # 2. Simulate Acquisition (Wait)
            time.sleep(duration)

            # 3. Check speed AFTER sleep (Simulate validation during acq)
            # In mock, we just check current state again.
            if MachineState.get_current_rpm() < self.trigger_threshold:
                continue

            self.logger.info(f"Generating vibration data (RPM ~{current_rpm:.1f})")

            # --- KEY FIX ---
            # Timestamp should be the START of the acquisition window.
            # History validation looks at [timestamp, timestamp + duration]
            # By backdating, we ensure we are validating against PAST (existing) Modbus data.
            batch_timestamp = pd.Timestamp.now() - pd.Timedelta(seconds=duration)

            for channel in self.config.channels:
                if not self._running: break

                ch_fs = self.config.sampling_rates.get(channel, fs)
                signal = self._generate_signal(ch_fs, duration, current_rpm)

                dp = DataPoint(
                    timestamp=batch_timestamp,
                    source=SignalSource.SIMULATION,
                    type=DataType.WAVEFORM,
                    device_name="towing_winch_sim",
                    point_name=channel,
                    value=signal,
                    metadata={
                        "sampling_rate": ch_fs,
                        "unit": "m/s2"
                    }
                )

                self.bus.publish(dp)

            # Wait a bit before next cycle
            time.sleep(2.0)

    def _generate_signal(self, fs: int, duration: float, rpm: float) -> np.ndarray:
        t = np.linspace(0, duration, int(fs * duration))
        freq_hz = max(0.1, rpm / 60.0)
        amp = self.config.signal_params.get('amplitude', 1.0) * (rpm / 1000.0) ** 2

        signal = amp * np.sin(2 * np.pi * freq_hz * t)
        signal += (amp * 0.5) * np.sin(2 * np.pi * (2 * freq_hz) * t)

        noise = np.random.normal(0, 0.1, len(t))
        signal += noise

        return signal