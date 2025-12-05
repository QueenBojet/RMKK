import threading
import time
import logging
from typing import List

from src.core.interfaces import Storage, Processor
from src.core.message_bus import MessageBus
from src.core.types import DataPoint


class StorageWorker(Processor):
    """
    A consumer worker that pulls DataPoints from the bus and persists them via Storage.
    Implements batching logic for efficiency.
    """

    def __init__(self, storage: Storage, bus: MessageBus, batch_size: int = 100, flush_interval: float = 1.0):
        self.storage = storage
        self.bus = bus
        self.batch_size = batch_size
        self.flush_interval = flush_interval
        self.logger = logging.getLogger(self.__class__.__name__)

        self._running = False
        self._thread = None
        self._buffer: List[DataPoint] = []
        self._last_flush_time = time.time()

    def start(self) -> None:
        if self._running: return

        self.logger.info("Starting Storage Worker...")
        self.storage.connect()

        self._running = True
        self._thread = threading.Thread(target=self._run_loop, name="StorageWorker", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._running = False
        if self._thread:
            self._thread.join(timeout=2.0)

        # Flush remaining data
        self._flush()
        self.storage.close()
        self.logger.info("Storage Worker stopped.")

    def _run_loop(self):
        while self._running:
            # Non-blocking get to allow periodic flushing even if no new data
            data = self.bus.get(block=True, timeout=0.5)

            if data:
                self._buffer.append(data)
                self.bus.task_done()

            # Check flush conditions
            is_full = len(self._buffer) >= self.batch_size
            is_timeout = (time.time() - self._last_flush_time) >= self.flush_interval

            if (is_full or is_timeout) and self._buffer:
                self._flush()

    def _flush(self):
        """Write buffer to storage."""
        if not self._buffer: return

        try:
            count = self.storage.save_batch(self._buffer)
            # self.logger.debug(f"Flushed {count} points to storage.")
        except Exception as e:
            self.logger.error(f"Failed to flush batch: {e}")
            # In a robust system, we might retry or dump to disk here.
            # For now, we clear the buffer to prevent blocking the whole system.
        finally:
            self._buffer.clear()
            self._last_flush_time = time.time()