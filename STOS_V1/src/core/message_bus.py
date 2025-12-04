import queue
import logging
import threading
from typing import Optional
from src.core.types import DataPoint


class MessageBus:
    """
    Global thread-safe message bus based on Python's queue.Queue.

    This is the spinal cord of the system. It decouples Collectors from
    Processors and Storage.

    - Collectors don't know about the Database. They just publish to the Bus.
    - The Database doesn't know about Modbus. It just consumes from the Bus.
    """

    def __init__(self, max_size: int = 10000):
        # We use a blocking queue with a max size to provide backpressure.
        # If the queue fills up (consumers are too slow), collectors
        # will naturally slow down or drop data, preventing OOM crashes.
        self._queue: queue.Queue[DataPoint] = queue.Queue(maxsize=max_size)
        self.logger = logging.getLogger("MessageBus")
        self._stop_event = threading.Event()

    def publish(self, data: DataPoint, timeout: Optional[float] = 0.5) -> bool:
        """
        Push data to the bus.

        Args:
            data: The DataPoint to publish.
            timeout: Seconds to wait if queue is full.

        Returns:
            True if successful, False if queue is full (and timeout reached).
        """
        try:
            self._queue.put(data, block=True, timeout=timeout)
            return True
        except queue.Full:
            # In production, this is a critical warning signal.
            # It implies the system is underprovisioned for the data rate.
            self.logger.warning(
                f"Message bus FULL! Dropping data: {data.point_name} ({data.type.value}). "
                f"Queue size: {self.qsize()}"
            )
            return False

    def get(self, block: bool = True, timeout: Optional[float] = 1.0) -> Optional[DataPoint]:
        """
        Get data from the bus.

        Args:
            block: Whether to block if queue is empty.
            timeout: Seconds to wait if queue is empty.

        Returns:
            DataPoint or None if empty (and timeout reached).
        """
        try:
            return self._queue.get(block=block, timeout=timeout)
        except queue.Empty:
            return None

    def task_done(self):
        """Indicate that a formerly enqueued task is complete."""
        self._queue.task_done()

    def qsize(self) -> int:
        """Return the approximate size of the queue."""
        return self._queue.qsize()

    def empty(self) -> bool:
        """Return True if the queue is empty, False otherwise."""
        return self._queue.empty()

    def full(self) -> bool:
        """Return True if the queue is full, False otherwise."""
        return self._queue.full()


# Singleton instance pattern
# In a rigorous dependency injection system we might avoid this,
# but for this scale, a module-level singleton is pragmatic.
_bus_instance: Optional[MessageBus] = None
_bus_lock = threading.Lock()


def get_message_bus(max_size: int = 10000) -> MessageBus:
    """Get the global message bus instance (Thread-safe Singleton)."""
    global _bus_instance
    if _bus_instance is None:
        with _bus_lock:
            if _bus_instance is None:
                _bus_instance = MessageBus(max_size)
    return _bus_instance