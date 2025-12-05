import logging
import csv
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import List

from src.core.interfaces import Storage
from src.core.types import DataPoint, DataType


class FileStorage(Storage):
    """
    Persists RAW WAVEFORM data to CSV files.
    This restores the functionality of saving raw data for offline analysis.
    """

    def __init__(self, save_path: str):
        self.save_path = Path(save_path)
        self.logger = logging.getLogger(self.__class__.__name__)
        self._ensure_path()

    def _ensure_path(self):
        if not self.save_path.exists():
            try:
                self.save_path.mkdir(parents=True, exist_ok=True)
            except Exception as e:
                self.logger.critical(f"Failed to create storage directory {self.save_path}: {e}")

    def connect(self) -> None:
        # File system doesn't need persistent connection
        pass

    def close(self) -> None:
        pass

    def save(self, data: DataPoint) -> bool:
        if data.type == DataType.WAVEFORM:
            return self._save_waveform(data)
        elif data.type == DataType.AGGREGATED_WAVEFORM:
            return self._save_aggregated(data)
        return False

    def save_batch(self, data_points: List[DataPoint]) -> int:
        count = 0
        for dp in data_points:
            if self.save(dp):
                count += 1
        return count

    def save_diagnosis(self, result) -> bool:
        # We generally rely on InfluxDB for results, but could log here too if needed
        return True

    def _save_waveform(self, dp: DataPoint) -> bool:
        """Save single channel waveform."""
        try:
            # Format: {device}_{point}_{timestamp}.csv
            ts_str = dp.timestamp.strftime("%Y%m%d_%H%M%S")
            filename = f"{dp.device_name}_{dp.point_name}_{ts_str}.csv"
            filepath = self.save_path / filename

            # Simple CSV: timestamp (if applicable) or just values
            # Assuming raw array saving
            df = pd.DataFrame(dp.value)
            df.to_csv(filepath, header=False, index=False)
            return True
        except Exception as e:
            self.logger.error(f"Failed to save waveform file: {e}")
            return False

    def _save_aggregated(self, dp: DataPoint) -> bool:
        """
        Save synchronized multi-channel data.
        Replicates the old 'merged' CSV format: 20251202_102720_25600Hz.csv
        """
        try:
            # Value is Dict[str, np.ndarray]
            data_dict = dp.value
            sampling_rate = dp.metadata.get('sampling_rate', 0)

            ts_str = dp.timestamp.strftime("%Y%m%d_%H%M%S")
            # Format from old system: YYYYMMDD_HHMMSS_FREQHz.csv
            filename = f"{ts_str}_{int(sampling_rate)}Hz.csv"
            filepath = self.save_path / filename

            # Convert to DataFrame
            # Ensure all arrays are same length
            df = pd.DataFrame(data_dict)

            # Save without header (per old system requirement) or with header
            # The old system's loader `pd.read_csv(file_path, header=None)` implies NO header.
            df.to_csv(filepath, header=False, index=False)

            self.logger.info(f"Saved raw data: {filename}")
            return True
        except Exception as e:
            self.logger.error(f"Failed to save aggregated file: {e}")
            return False