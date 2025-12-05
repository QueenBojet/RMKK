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


class MockVibrationCollector(Collector):
    """
    Generates simulated vibration waveform data.
    """

    def __init__(self, config: VibrationSimConfig, bus: MessageBus):
        self.config = config
        self.bus = bus
        self.logger = logging.getLogger("MockVibrationCollector")
        self._running = False
        self._thread = None

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
        # Default sampling rate
        fs = 25600
        duration = self.config.duration_seconds
        interval = 5.0  # Generate every 5 seconds to avoid flooding in test

        while self._running:
            start_gen = time.time()

            # Generate for each channel
            for channel in self.config.channels:
                if not self._running: break

                # Get specific fs for channel
                ch_fs = self.config.sampling_rates.get(channel, fs)

                signal = self._generate_signal(ch_fs, duration)

                dp = DataPoint(
                    timestamp=pd.Timestamp.now(),
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

            # Wait for next cycle
            elapsed = time.time() - start_gen
            sleep_time = max(0.1, interval - elapsed)
            time.sleep(sleep_time)

    def _generate_signal(self, fs: int, duration: float) -> np.ndarray:
        """Generate a simple synthetic vibration signal."""
        t = np.linspace(0, duration, int(fs * duration))

        # Base: 50Hz sine wave (e.g. motor freq)
        base_freq = self.config.signal_params.get('base_frequency', 50.0)
        amp = self.config.signal_params.get('amplitude', 1.0)
        noise_level = self.config.signal_params.get('noise_level', 0.1)

        signal = amp * np.sin(2 * np.pi * base_freq * t)

        # Add harmonics
        signal += (amp * 0.5) * np.sin(2 * np.pi * (2 * base_freq) * t)

        # Add noise
        noise = np.random.normal(0, noise_level, len(t))
        signal += noise

        return signal