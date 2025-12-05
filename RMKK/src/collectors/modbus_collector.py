import time
import logging
import threading
import struct
import pandas as pd
from pathlib import Path
from typing import List, Dict, Any, Optional
from pymodbus.client import ModbusTcpClient
from pymodbus.constants import Endian

from src.core.interfaces import Collector
from src.core.message_bus import MessageBus
from src.core.types import DataPoint, DataType, SignalSource
from src.core.config_schema import ModbusConfig
from src.utils.project_path import get_project_path

# Register Count Mapping
DATA_TYPE_REG_COUNT = {
    "int16": 1, "uint16": 1,
    "int32": 2, "uint32": 2,
    "float32": 2,
    "int64": 4, "uint64": 4,
    "float64": 4,
    "bool": 1,
}


class ModbusCollector(Collector):
    """
    Production-grade Modbus TCP Collector.
    Reads registers and pushes DataPoints to the global bus.
    """

    def __init__(self, config: ModbusConfig, bus: MessageBus):
        self.config = config
        self.bus = bus
        self.logger = logging.getLogger(self.__class__.__name__)

        self._client: Optional[ModbusTcpClient] = None
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._signals: List[Dict[str, Any]] = []

        # Statistics
        self._stats = {
            "messages_published": 0,
            "read_errors": 0,
            "connection_errors": 0,
            "last_collection_time": None
        }

        self._load_signals()

    def _load_signals(self):
        """Load signal definitions from CSV."""
        try:
            file_path = get_project_path(self.config.collector.signal_file_path)
            if not file_path.exists():
                raise FileNotFoundError(f"Signal file not found: {file_path}")

            df = pd.read_csv(file_path)

            # Basic validation
            required_cols = ['device_name', 'point_name', 'address', 'data_type', 'register_type']
            for col in required_cols:
                if col not in df.columns:
                    raise ValueError(f"Signal file missing column: {col}")

            # Clean and convert
            df['unit'] = df['unit'].fillna('')
            df['scale_factor'] = df['scale_factor'].fillna(1.0)
            df['interval'] = df['interval'].fillna(1.0)

            self._signals = df.to_dict('records')
            self.logger.info(f"Loaded {len(self._signals)} signals from {file_path.name}")

        except Exception as e:
            self.logger.error(f"Failed to load signals: {e}")
            raise

    def start(self) -> None:
        if self._running:
            return

        self.logger.info(f"Starting Modbus Collector ({self.config.server.host}:{self.config.server.port})...")
        self._running = True
        self._thread = threading.Thread(target=self._poll_loop, name="ModbusPollThread", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._running = False
        if self._client:
            self._client.close()
        if self._thread:
            self._thread.join(timeout=2.0)
        self.logger.info("Modbus Collector stopped.")

    def get_status(self) -> Dict[str, Any]:
        return {
            "running": self._running,
            "connected": self._client.is_socket_open() if self._client else False,
            "stats": self._stats
        }

    def _connect(self) -> bool:
        """Establish connection to Modbus Server."""
        if self._client and self._client.is_socket_open():
            return True

        try:
            self._client = ModbusTcpClient(
                host=self.config.server.host,
                port=self.config.server.port,
                timeout=3.0
            )
            if self._client.connect():
                self.logger.info("Connected to Modbus Server.")
                return True
            else:
                self._stats["connection_errors"] += 1
                return False
        except Exception as e:
            self.logger.error(f"Connection failed: {e}")
            self._stats["connection_errors"] += 1
            return False

    def _poll_loop(self):
        """Main polling loop."""
        while self._running:
            if not self._connect():
                time.sleep(5)  # Retry delay
                continue

            cycle_start = time.time()

            for signal in self._signals:
                if not self._running: break

                # Check if it's time to read this signal (based on individual interval)
                # Optimization: For high performance, we might group reads by interval/address.
                # For now, simplistic iteration is safer and easier to debug.
                last_read = signal.get('_last_read', 0)
                interval = signal.get('interval', self.config.collector.poll_interval_seconds)

                if cycle_start - last_read >= interval:
                    self._read_and_publish(signal)
                    signal['_last_read'] = cycle_start

            self._stats["last_collection_time"] = time.time()

            # Avoid tight loop burning CPU
            time.sleep(0.1)

    def _read_and_publish(self, signal: Dict[str, Any]):
        """Read a single signal and publish to bus."""
        try:
            value = self._read_register(signal)
            if value is not None:
                dp = DataPoint(
                    timestamp=pd.Timestamp.now(),  # Use pandas timestamp for consistency
                    source=SignalSource.MODBUS,
                    type=DataType.SCALAR,
                    device_name=signal['device_name'],
                    point_name=signal['point_name'],
                    value=value,
                    metadata={
                        "unit": signal['unit']
                    }
                )
                if self.bus.publish(dp, timeout=0.1):
                    self._stats["messages_published"] += 1

        except Exception as e:
            self.logger.error(f"Error reading {signal['point_name']}: {e}")
            self._stats["read_errors"] += 1

    def _read_register(self, signal: Dict[str, Any]) -> Optional[float]:
        """Low-level Modbus read and decode."""
        address = int(signal['address']) - 1  # Modbus is 1-based usually
        count = DATA_TYPE_REG_COUNT.get(signal['data_type'], 1)
        reg_type = signal['register_type']

        if reg_type == 'holding_registers':
            rr = self._client.read_holding_registers(address=address, count=count)
        elif reg_type == 'input_registers':
            rr = self._client.read_input_registers(address=address, count=count)
        elif reg_type == 'coils':
            rr = self._client.read_coils(address=address, count=count)
        else:
            return None

        if rr.isError():
            return None

        # Decode
        if reg_type == 'coils':
            return float(rr.bits[0])

        return self._decode(rr.registers, signal['data_type']) * float(signal['scale_factor'])

    def _decode(self, registers: List[int], dtype: str) -> float:
        """
        Decodes register bytes into python data types.
        Assumes Big Endian / Word Swapped (Standard Modbus typically).
        """
        # This mapping can be extracted to a utility if reused
        pack_format = ""
        unpack_format = ""

        if dtype == 'float32':
            # CDAB format (often used in industrial Modbus)
            # Swap words then pack
            b = struct.pack('>HH', registers[1], registers[0])
            return struct.unpack('>f', b)[0]
        elif dtype == 'int16':
            return float(struct.unpack('>h', struct.pack('>H', registers[0]))[0])
        elif dtype == 'uint16':
            return float(registers[0])
        # ... Add other types as needed ...

        return float(registers[0])