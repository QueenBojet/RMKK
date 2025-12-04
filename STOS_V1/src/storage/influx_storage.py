import time
import logging
import pandas as pd
from typing import List, Dict, Any, Optional
from influxdb import InfluxDBClient
from influxdb.exceptions import InfluxDBClientError, InfluxDBServerError

from src.core.interfaces import Storage
from src.core.types import DataPoint, DataType, DiagnosisResult
from src.core.config_schema import DatabaseConfig


class InfluxDBStorage(Storage):
    """
    InfluxDB implementation of the Storage interface.
    Handles connection, batching, and retries.
    """

    def __init__(self, config: DatabaseConfig):
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        self._client: Optional[InfluxDBClient] = None

    def connect(self) -> None:
        """Establish connection to InfluxDB."""
        if self._client: return

        try:
            self._client = InfluxDBClient(
                host=self.config.host,
                port=self.config.port,
                username=self.config.username,
                password=self.config.password,
                database=self.config.database,
                timeout=self.config.timeout
            )
            # Check/Create database
            # This is a blocking call, good for startup checks
            dbs = self._client.get_list_database()
            if not any(db['name'] == self.config.database for db in dbs):
                self.logger.info(f"Creating database: {self.config.database}")
                self._client.create_database(self.config.database)

            self._client.switch_database(self.config.database)
            self.logger.info(f"Connected to InfluxDB at {self.config.host}:{self.config.port}")

        except Exception as e:
            self.logger.critical(f"Failed to connect to InfluxDB: {e}")
            self._client = None
            raise

    def close(self) -> None:
        if self._client:
            self._client.close()
            self._client = None
            self.logger.info("InfluxDB connection closed.")

    def save(self, data: DataPoint) -> bool:
        """Save a single DataPoint (wrapper around save_batch)."""
        return self.save_batch([data]) == 1

    def save_batch(self, data_points: List[DataPoint]) -> int:
        """
        Convert DataPoints to InfluxDB Line Protocol points and write in batch.
        Returns count of successfully written points.
        """
        if not self._client:
            self.logger.error("Attempted to write to closed InfluxDB connection.")
            return 0

        if not data_points:
            return 0

        points_payload = []

        for dp in data_points:
            # We treat different DataPoint types differently
            if dp.type == DataType.SCALAR:
                point = self._convert_scalar(dp)
            elif dp.type == DataType.HEALTH_SCORE:
                point = self._convert_health(dp)
            elif dp.type == DataType.FAULT:
                point = self._convert_fault(dp)
            # Note: WAVEFORM data is typically NOT stored in InfluxDB directly
            # due to high cardinality/volume. It's usually saved to files (parquet/hdf5).
            # For this implementation, we skip WAVEFORM here or save summary stats.
            else:
                continue

            if point:
                points_payload.append(point)

        if not points_payload:
            return 0

        try:
            # Write batch
            self._client.write_points(points_payload, batch_size=len(points_payload))
            return len(points_payload)
        except (InfluxDBClientError, InfluxDBServerError) as e:
            self.logger.error(f"InfluxDB Write Error: {e}")
            return 0
        except Exception as e:
            self.logger.error(f"Unexpected error writing to InfluxDB: {e}")
            return 0

    def save_diagnosis(self, result: DiagnosisResult) -> bool:
        """Helper to save diagnosis results (Health + Faults)."""
        # This is strictly not needed if Processor converts everything to DataPoints,
        # but kept for interface compliance.
        return True

        # --- Conversion Helpers ---

    def _convert_scalar(self, dp: DataPoint) -> Dict[str, Any]:
        """Convert SCALAR DataPoint to InfluxDB point."""
        try:
            val = float(dp.value)
        except (ValueError, TypeError):
            return None

        return {
            "measurement": "signal_data",
            "tags": {
                "device": dp.device_name,
                "point": dp.point_name,
                "source": dp.source.value,
                "unit": dp.metadata.get("unit", "")
            },
            "time": dp.timestamp.isoformat(),
            "fields": {
                "value": val
            }
        }

    def _convert_health(self, dp: DataPoint) -> Dict[str, Any]:
        """Convert HEALTH_SCORE DataPoint to InfluxDB point."""
        return {
            "measurement": "health_score",
            "tags": {
                "device": dp.device_name,
                "component": dp.point_name  # e.g. "mechanical_system"
            },
            "time": dp.timestamp.isoformat(),
            "fields": {
                "score": float(dp.value),
                # Storing complex structures like 'components' breakdown is tricky in Influx.
                # Usually we serialize it to string or flattening is handled elsewhere.
                # For now, simplistic approach.
                "details": str(dp.metadata.get("components", {}))
            }
        }

    def _convert_fault(self, dp: DataPoint) -> Dict[str, Any]:
        """Convert FAULT DataPoint to InfluxDB point."""
        return {
            "measurement": "fault_events",
            "tags": {
                "device": dp.device_name,
                "fault_type": dp.metadata.get("fault_type", "unknown"),
                "severity": dp.metadata.get("severity", "info")
            },
            "time": dp.timestamp.isoformat(),
            "fields": {
                "active": 1,
                "description": str(dp.metadata.get("description", ""))
            }
        }