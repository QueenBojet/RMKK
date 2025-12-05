import sys
import time
import signal
import logging
from pathlib import Path

# Adjust path
sys.path.append(str(Path(__file__).parent))

from src.utils.config_loader import load_app_config
from src.utils.logger_setup import LoggerSetup
from src.utils.windows_power import WindowsPowerManager
from src.core.message_bus import get_message_bus
from src.storage.influx_storage import InfluxDBStorage
from src.storage.file_storage import FileStorage
from src.storage.composite_storage import CompositeStorage
from src.workers.storage_worker import StorageWorker
from src.processors.signal_processor import SignalProcessor
from src.processors.signal_aggregator import SignalAggregator
from src.collectors.modbus_collector import ModbusCollector
from src.collectors.mock_modbus_collector import MockModbusCollector
from src.collectors.mock_vibration_collector import MockVibrationCollector


class Application:
    def __init__(self, config_path: str):
        self._running = False

        # 1. Config
        try:
            self.config = load_app_config(config_path)
        except Exception as e:
            print(f"CRITICAL: Config load failed: {e}")
            sys.exit(1)

        # 2. Logging
        log_cfg = {}
        if self.config.output and self.config.output.logging:
            # Pydantic v2 support
            if hasattr(self.config.output.logging, 'model_dump'):
                log_cfg = self.config.output.logging.model_dump()
            else:
                log_cfg = self.config.output.logging.dict()

        self.logger_setup = LoggerSetup(log_cfg)
        self.logger = logging.getLogger("Main")

        # 3. Utilities
        self.power_mgr = WindowsPowerManager()
        self.power_mgr.prevent_system_sleep()

        # 4. Message Bus (The Hub) - Now supports Pub/Sub
        self.bus = get_message_bus(max_size=10000)

        # 5. Storage Layer (The Sink)
        # Create backends
        self.influx_storage = InfluxDBStorage(self.config.database)

        # Determine save path for files
        save_path = "./data"
        if self.config.phm and self.config.phm.data_collection:
            save_path = self.config.phm.data_collection.save_path
        self.file_storage = FileStorage(save_path)

        # Create Composite Storage
        self.composite_storage = CompositeStorage(self.influx_storage, self.file_storage)

        # Create Worker with a SUBSCRIPTION to the bus
        # Note: We pass the SUBSCRIPTION, not the raw bus, to consumers
        # Storage needs to hear everything (Scalars for Influx, Waveforms for File)
        storage_sub = self.bus.subscribe("StorageWorker")
        self.storage_worker = StorageWorker(self.composite_storage, storage_sub)

        # 6. Processors
        # Aggregator needs to hear EVERYTHING to filter by speed and sync channels
        aggregator_sub = self.bus.subscribe("SignalAggregator")
        required_channels = ["ch1", "ch2", "ch3", "ch4", "ch5"]
        self.aggregator = SignalAggregator(self.config, aggregator_sub, required_channels)

        # SignalProcessor needs to hear AGGREGATED data (produced by Aggregator)
        # And potentially scalars for context.
        processor_sub = self.bus.subscribe("SignalProcessor")
        if self.config.diagnosis_config:
            self.processor = SignalProcessor(
                config=self.config.diagnosis_config,
                bus=processor_sub
            )
        else:
            self.processor = None

        # 7. Collectors (Producers - they act on the RAW bus)
        self.collectors = []
        if self.config.test_mode.enabled:
            self.logger.warning(">>> TEST MODE <<<")

            # Mock Modbus (Scalars)
            if self.config.test_mode.modbus_simulation.enabled:
                self.collectors.append(MockModbusCollector(
                    self.config.test_mode.modbus_simulation, self.bus
                ))

            # Mock Vibration (Waveforms) - NOW INCLUDED
            if self.config.test_mode.vibration_simulation.enabled:
                self.collectors.append(MockVibrationCollector(
                    self.config.test_mode.vibration_simulation, self.bus
                ))
                self.logger.info("Added Mock Vibration Collector")

        else:
            if self.config.modbus:
                self.collectors.append(ModbusCollector(
                    self.config.modbus, self.bus
                ))
            # Add MQTT Collector here...

    def start(self):
        self._running = True

        # Start Consumers first
        self.storage_worker.start()
        self.aggregator.start()
        if self.processor:
            self.processor.start()

        # Start Producers last
        for c in self.collectors:
            c.start()

        self.logger.info("System Started. Press Ctrl+C to stop.")

        try:
            while self._running:
                time.sleep(1)
        except KeyboardInterrupt:
            self.shutdown()

    def shutdown(self, signum=None, frame=None):
        if not self._running: return
        self._running = False
        self.logger.info("Shutting down...")

        for c in self.collectors: c.stop()

        self.aggregator.stop()
        if self.processor: self.processor.stop()
        self.storage_worker.stop()

        self.power_mgr.restore_power_settings()
        sys.exit(0)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", default="config/config.yaml")
    args = parser.parse_args()

    app = Application(args.config)
    signal.signal(signal.SIGINT, app.shutdown)
    signal.signal(signal.SIGTERM, app.shutdown)
    app.start()