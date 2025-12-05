import time
import logging
import threading
import random
import numpy as np
import pandas as pd
from typing import Dict, Any, List

from src.core.interfaces import Collector
from src.core.message_bus import MessageBus
from src.core.types import DataPoint, DataType, SignalSource
from src.core.config_schema import ModbusSimConfig


class MockModbusCollector(Collector):
    """
    Simulates Modbus data generation for testing.
    Replaces the old 'TestDataManager' logic for scalars.
    """

    def __init__(self, sim_config: ModbusSimConfig, bus: MessageBus):
        self.config = sim_config
        self.bus = bus
        self.logger = logging.getLogger("MockModbusCollector")
        self._running = False
        self._thread = None

        self._cycle_start = time.time()
        self._current_values = {}  # For random walk persistence

    def start(self) -> None:
        if self._running: return

        self.logger.info("[TEST MODE] Starting Mock Modbus Collector...")
        self._running = True
        self._thread = threading.Thread(target=self._sim_loop, name="MockModbusLoop", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._running = False
        if self._thread:
            self._thread.join(timeout=1.0)
        self.logger.info("Mock Collector stopped.")

    def get_status(self) -> Dict[str, Any]:
        return {"running": self._running, "mode": "simulation"}

    def _sim_loop(self):
        while self._running:
            start_time = time.time()

            for signal in self.config.signals:
                val = self._generate_value(signal)

                dp = DataPoint(
                    timestamp=pd.Timestamp.now(),
                    source=SignalSource.SIMULATION,
                    type=DataType.SCALAR,
                    device_name=signal.device_name,
                    point_name=signal.point_name,
                    value=val,
                    metadata={"unit": signal.unit}
                )

                self.bus.publish(dp, timeout=0.1)

                if not self._running: break

            # Sleep to match update interval
            elapsed = time.time() - start_time
            sleep_time = max(0.1, self.config.update_interval - elapsed)
            time.sleep(sleep_time)

    def _generate_value(self, signal) -> float:
        """Generate simulated values based on signal name or logic."""
        key = f"{signal.device_name}.{signal.point_name}"

        # 1. Speed simulation (Cycle: Stop -> Accel -> Run -> Decel)
        if "speed" in signal.point_name.lower():
            t = time.time() - self._cycle_start
            cycle_pos = t % 60  # 60s cycle

            if cycle_pos < 20:
                val = 0.0
            elif cycle_pos < 30:
                val = (cycle_pos - 20) * 5  # Ramp up
            elif cycle_pos < 50:
                val = 50.0 + random.uniform(-1, 1)  # Stable
            else:
                val = 50.0 - (cycle_pos - 50) * 5  # Ramp down

            return max(0.0, val)

        # 2. Temperature (Random Walk)
        elif "temp" in signal.point_name.lower():
            current = self._current_values.get(key, 40.0)
            change = random.uniform(-0.5, 0.5)
            new_val = current + change
            # Keep within bounds
            new_val = max(20.0, min(new_val, 90.0))
            self._current_values[key] = new_val
            return new_val

        # 3. Default Random
        else:
            return random.uniform(signal.value_range[0], signal.value_range[1])