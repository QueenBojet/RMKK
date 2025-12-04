import sys
import time
import signal
import logging
from pathlib import Path

# Adjust path to ensure src is importable
# This is robust even if you run from different directories
sys.path.append(str(Path(__file__).parent))

from src.utils.config_loader import load_app_config
from src.utils.logger_setup import LoggerSetup
from src.core.message_bus import get_message_bus
from src.storage.influx_storage import InfluxDBStorage
from src.workers.storage_worker import StorageWorker
from src.processors.signal_processor import SignalProcessor
from src.collectors.modbus_collector import ModbusCollector
from src.collectors.mock_modbus_collector import MockModbusCollector


class Application:
    """
    Main Application Orchestrator.
    Responsible for wiring up components using Dependency Injection (DI) principles.
    No business logic should live here.
    """

    def __init__(self, config_path: str):
        self._running = False

        # 1. Load & Validate Configuration
        # If this fails, the app shouldn't start. Early failure is good failure.
        try:
            self.config = load_app_config(config_path)
        except Exception as e:
            print(f"CRITICAL: Failed to load configuration: {e}")
            sys.exit(1)

        # 2. Setup Logging
        # Convert Pydantic model to dict for legacy LoggerSetup compatibility
        log_cfg = {}
        if self.config.output and self.config.output.logging:
            log_cfg = self.config.output.logging.dict()

        self.logger_setup = LoggerSetup(log_cfg)
        self.logger = logging.getLogger("Main")

        self.logger.info("========================================")
        self.logger.info("   Starting STOS Monitor System")
        self.logger.info("========================================")

        # 3. Initialize Message Bus (The Spinal Cord)
        self.bus = get_message_bus(max_size=10000)
        self.logger.info("Message Bus initialized")

        # 4. Initialize Storage Layer (The Sink)
        # We create the storage backend first...
        self.storage = InfluxDBStorage(self.config.database)
        # ...then the worker that feeds it.
        self.storage_worker = StorageWorker(
            storage=self.storage,
            bus=self.bus,
            batch_size=100,
            flush_interval=1.0
        )
        self.logger.info("Storage Layer initialized")

        # 5. Initialize Processing Layer (The Brain)
        if not self.config.diagnosis_config:
            self.logger.warning("Diagnosis config missing. SignalProcessor will be disabled.")
            self.processor = None
        else:
            self.processor = SignalProcessor(
                config=self.config.diagnosis_config,
                bus=self.bus
            )
            self.logger.info("Signal Processor initialized")

        # 6. Initialize Collection Layer (The Sources)
        self.collectors = []

        # --- Logic to choose between Real and Mock collectors ---
        if self.config.test_mode.enabled:
            self.logger.warning(">>> RUNNING IN TEST MODE <<<")

            # Mock Modbus
            if self.config.test_mode.modbus_simulation.enabled:
                mock_modbus = MockModbusCollector(
                    sim_config=self.config.test_mode.modbus_simulation,
                    bus=self.bus
                )
                self.collectors.append(mock_modbus)
                self.logger.info("Added Mock Modbus Collector")

            # TODO: Add Mock Vibration Collector here if needed

        else:
            # Real Modbus
            if self.config.modbus:
                modbus = ModbusCollector(
                    config=self.config.modbus,
                    bus=self.bus
                )
                self.collectors.append(modbus)
                self.logger.info("Added Modbus Collector")

            # TODO: Add MQTT Collector here
            # mqtt = MqttCollector(self.config.phm, self.bus)
            # self.collectors.append(mqtt)

    def start(self):
        """Start all components in the correct dependency order."""
        self._running = True

        # 1. Start Consumers first (Storage)
        # This ensures that when producers start, there's someone ready to write data.
        self.storage_worker.start()

        # 2. Start Processors
        if self.processor:
            self.processor.start()

        # 3. Start Producers (Collectors) last
        for collector in self.collectors:
            collector.start()

        self.logger.info(f"System Running with {len(self.collectors)} collectors active.")

        # Main thread loop - keeping the process alive
        try:
            while self._running:
                time.sleep(1)
        except KeyboardInterrupt:
            self.logger.info("KeyboardInterrupt received.")
            self.shutdown()
        except Exception as e:
            self.logger.critical(f"Unexpected error in main loop: {e}", exc_info=True)
            self.shutdown()

    def shutdown(self, signum=None, frame=None):
        """Graceful shutdown of all components."""
        if not self._running:
            return

        self.logger.info("Initiating graceful shutdown...")
        self._running = False

        # Shutdown Order: Producers -> Processors -> Consumers

        # 1. Stop Producers
        self.logger.info("Stopping collectors...")
        for collector in self.collectors:
            try:
                collector.stop()
            except Exception as e:
                self.logger.error(f"Error stopping collector: {e}")

        # 2. Stop Processors
        self.logger.info("Stopping processors...")
        if self.processor:
            try:
                self.processor.stop()
            except Exception as e:
                self.logger.error(f"Error stopping processor: {e}")

        # 3. Stop Storage (Drain queue)
        self.logger.info("Stopping storage worker...")
        try:
            self.storage_worker.stop()
        except Exception as e:
            self.logger.error(f"Error stopping storage worker: {e}")

        self.logger.info("System shutdown complete.")
        sys.exit(0)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="STOS Monitoring System")
    parser.add_argument("-c", "--config", default="config/config.yaml", help="Path to config file")
    args = parser.parse_args()

    app = Application(args.config)

    # Register signal handlers for clean exit on Docker/Systemd stop
    signal.signal(signal.SIGINT, app.shutdown)
    signal.signal(signal.SIGTERM, app.shutdown)

    app.start()