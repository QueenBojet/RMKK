import logging
import pandas as pd
from pathlib import Path
from typing import List

from src.core.interfaces import Storage
from src.core.types import DataPoint, DataType


class FileStorage(Storage):
    """
    Saves AGGREGATED_WAVEFORM to 6-column CSV.
    Name format: YYYYMMDD_HHMMSS_FREQHz.csv
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
        pass

    def close(self) -> None:
        pass

    def save(self, data: DataPoint) -> bool:
        if data.type == DataType.AGGREGATED_WAVEFORM:
            return self._save_aggregated(data)
        return False

    def save_batch(self, data_points: List[DataPoint]) -> int:
        count = 0
        for dp in data_points:
            if self.save(dp): count += 1
        return count

    def save_diagnosis(self, result) -> bool:
        return True

    def _save_aggregated(self, dp: DataPoint) -> bool:
        try:
            data_dict = dp.value  # {'ch1':..., 'speed':...}
            sampling_rate = dp.metadata.get('sampling_rate', 25600)

            # 1. Define Column Order
            cols = ['ch1', 'ch2', 'ch3', 'ch4', 'ch5', 'speed']

            # Check availability
            if not all(k in data_dict for k in cols):
                missing = [k for k in cols if k not in data_dict]
                self.logger.error(f"Cannot save. Missing columns: {missing}")
                return False

            # 2. Save
            df = pd.DataFrame({k: data_dict[k] for k in cols})

            ts_str = dp.timestamp.strftime("%Y%m%d_%H%M%S")
            filename = f"{ts_str}_{int(sampling_rate)}Hz.csv"
            filepath = self.save_path / filename

            # No header, index=False
            df.to_csv(filepath, header=False, index=False)
            self.logger.info(f"Saved: {filename}")
            return True
        except Exception as e:
            self.logger.error(f"File Save Error: {e}")
            return False