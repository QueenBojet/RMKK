import time
import logging
import threading
import random
import pandas as pd
from src.core.interfaces import Collector
from src.core.message_bus import MessageBus
from src.core.types import DataPoint, DataType, SignalSource
from src.core.config_schema import ModbusSimConfig
from src.utils.project_path import get_project_path


class MachineState:
    _start_time = time.time()
    CYCLE = 60

    @classmethod
    def get_current_rpm(cls):
        # 0-20s: 0 RPM
        # 20-40s: Ramp to 1500
        # 40-60s: 1500 RPM
        t = (time.time() - cls._start_time) % cls.CYCLE
        if t < 20: return 0.0
        if t < 40: return (t - 20) / 20 * 1500
        return 1500 + random.uniform(-10, 10)


class MockModbusCollector(Collector):
    def __init__(self, sim_config: ModbusSimConfig, bus: MessageBus):
        self.config = sim_config
        self.bus = bus
        self.logger = logging.getLogger("MockModbusCollector")
        self._running = False
        self._thread = None
        self._signals = []

    def start(self):
        if self._running: return
        self._running = True
        self._load_signals()
        self._thread = threading.Thread(target=self._sim_loop, name="MockModbus", daemon=True)
        self._thread.start()

    def stop(self):
        self._running = False
        if self._thread: self._thread.join(1.0)

    def get_status(self):
        return {"running": self._running}

    def _load_signals(self):
        try:
            path = get_project_path("config/signal_config.csv")
            df = pd.read_csv(path)
            self._signals = df.to_dict('records')
            for s in self._signals: s['_last'] = 0
        except Exception as e:
            self.logger.error(f"Failed to load signals: {e}")

    def _sim_loop(self):
        # Run loop faster than fastest signal (0.2s)
        while self._running:
            now = time.time()
            rpm = MachineState.get_current_rpm()

            for s in self._signals:
                interval = float(s.get('interval', 1.0))
                if now - s['_last'] >= interval:
                    val = self._gen_val(s, rpm)
                    dp = DataPoint(
                        timestamp=pd.Timestamp.now(),
                        source=SignalSource.SIMULATION,
                        type=DataType.SCALAR,
                        device_name=s['device_name'],
                        point_name=s['point_name'],
                        value=val,
                        metadata={"unit": s.get('unit', '')}
                    )
                    self.bus.publish(dp)
                    s['_last'] = now
            time.sleep(0.05)

    def _gen_val(self, s, rpm):
        name = s['point_name'].lower()
        if "speed" in name: return rpm
        if "temp" in name: return 40 + rpm / 100
        return random.uniform(0, 100)