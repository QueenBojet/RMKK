from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from src.core.types import DataPoint, DiagnosisResult


class Collector(ABC):
    """
    Abstract base class for all data collectors.

    A Collector is a PRODUCER. Its sole responsibility is to fetch raw data,
    convert it into `DataPoint` objects, and push them to the MessageBus.

    It MUST NOT contain business logic, analysis logic, or direct database access.
    """

    @abstractmethod
    def start(self) -> None:
        """
        Start the data collection process.
        This should be non-blocking (e.g., start a thread or async loop).
        """
        pass

    @abstractmethod
    def stop(self) -> None:
        """
        Stop the collection process and release resources (sockets, file handles).
        """
        pass

    @abstractmethod
    def get_status(self) -> Dict[str, Any]:
        """
        Return current health/status of the collector for monitoring.
        e.g. {"connected": True, "messages_sent": 1050}
        """
        pass


class Storage(ABC):
    """
    Abstract base class for data storage backends.

    A Storage is a CONSUMER (usually). It takes standardized DataPoints
    and persists them to a backend (InfluxDB, SQL, CSV, etc.).
    """

    @abstractmethod
    def connect(self) -> None:
        """Establish connection to storage backend."""
        pass

    @abstractmethod
    def close(self) -> None:
        """Close connection and flush buffers."""
        pass

    @abstractmethod
    def save(self, data: DataPoint) -> bool:
        """
        Save a single data point.
        Returns True if successful.
        """
        pass

    @abstractmethod
    def save_batch(self, data_points: List[DataPoint]) -> int:
        """
        Save a batch of data points.
        Returns the count of successfully saved items.

        Optimized implementation is expected for high-throughput backends.
        """
        pass

    @abstractmethod
    def save_diagnosis(self, result: DiagnosisResult) -> bool:
        """
        Save a complex diagnosis result.
        """
        pass


class Processor(ABC):
    """
    Abstract base class for data processors.

    A Processor is both a CONSUMER and a PRODUCER.
    It consumes raw DataPoints from the bus, analyzes them (FFT, Logic),
    and produces new DataPoints (Health Scores, Alerts).
    """

    @abstractmethod
    def start(self) -> None:
        """Start the processing loop."""
        pass

    @abstractmethod
    def stop(self) -> None:
        """Stop processing."""
        pass