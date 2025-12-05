import logging
from typing import List
from src.core.interfaces import Storage
from src.core.types import DataPoint, DataType


class CompositeStorage(Storage):
    """
    Routes data to multiple storage backends based on data type.
    """

    def __init__(self, influx_storage: Storage, file_storage: Storage):
        self.influx = influx_storage
        self.file = file_storage
        self.logger = logging.getLogger("CompositeStorage")

    def connect(self) -> None:
        self.influx.connect()
        self.file.connect()

    def close(self) -> None:
        self.influx.close()
        self.file.close()

    def save(self, data: DataPoint) -> bool:
        success = False

        # Route Scalars/Health/Faults to Influx
        if data.type in [DataType.SCALAR, DataType.HEALTH_SCORE, DataType.FAULT]:
            if self.influx.save(data):
                success = True

        # Route Waveforms (Raw & Aggregated) to File
        if data.type in [DataType.WAVEFORM, DataType.AGGREGATED_WAVEFORM]:
            if self.file.save(data):
                success = True

        return success

    def save_batch(self, data_points: List[DataPoint]) -> int:
        count = 0
        influx_batch = []
        file_batch = []

        # Split batch
        for dp in data_points:
            if dp.type in [DataType.SCALAR, DataType.HEALTH_SCORE, DataType.FAULT]:
                influx_batch.append(dp)
            elif dp.type in [DataType.WAVEFORM, DataType.AGGREGATED_WAVEFORM]:
                file_batch.append(dp)

        # Execute saves
        if influx_batch:
            count += self.influx.save_batch(influx_batch)

        if file_batch:
            count += self.file.save_batch(file_batch)

        return count

    def save_diagnosis(self, result) -> bool:
        return True