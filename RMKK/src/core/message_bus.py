import queue
import logging
import threading
from typing import Optional, List, Any
from src.core.types import DataPoint


class Subscription:
    """
    A wrapper around a queue for a specific subscriber.
    Mimics the interface of the main bus for consumers.
    """

    def __init__(self, bus, name: str, max_size: int):
        self._bus = bus
        self._queue = queue.Queue(maxsize=max_size)
        self.name = name

    def get(self, block: bool = True, timeout: Optional[float] = 1.0) -> Optional[DataPoint]:
        try:
            return self._queue.get(block=block, timeout=timeout)
        except queue.Empty:
            return None

    def task_done(self):
        self._queue.task_done()

    def put(self, item):
        try:
            self._queue.put(item, block=False)
        except queue.Full:
            pass  # Drop if full to prevent blocking publisher

    def publish(self, data: DataPoint, timeout: Optional[float] = 0.5) -> bool:
        # Consumers might also publish (e.g. Processors)
        # Pass it back to the main bus to broadcast
        return self._bus.publish(data, timeout)


class MessageBus:
    """
    Global Pub/Sub Message Bus.
    """

    def __init__(self, max_size: int = 10000):
        self._subscribers: List[Subscription] = []
        self._lock = threading.Lock()
        self.max_size = max_size
        self.logger = logging.getLogger("MessageBus")

    def subscribe(self, name: str) -> Subscription:
        """Create a new subscription queue for a consumer."""
        with self._lock:
            sub = Subscription(self, name, self.max_size)
            self._subscribers.append(sub)
            self.logger.info(f"New subscriber registered: {name}")
            return sub

    def publish(self, data: DataPoint, timeout: Optional[float] = 0.5) -> bool:
        """
        Broadcast data to ALL subscribers.
        """
        with self._lock:
            # We don't block on publish to avoid one slow consumer killing the system
            # Ideally we'd use a separate thread for dispatching, but for now simple loop
            for sub in self._subscribers:
                sub.put(data)
        return True

    # Legacy support (if needed, but we should move away from it)
    def get(self, block=True, timeout=1.0):
        raise NotImplementedError("Direct get() on Bus is deprecated. Use subscribe() first.")


# Singleton
_bus_instance: Optional[MessageBus] = None
_bus_lock = threading.Lock()


def get_message_bus(max_size: int = 10000) -> MessageBus:
    global _bus_instance
    if _bus_instance is None:
        with _bus_lock:
            if _bus_instance is None:
                _bus_instance = MessageBus(max_size)
    return _bus_instance