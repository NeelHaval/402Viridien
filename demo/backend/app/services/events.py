"""In-process event broadcaster for Server-Sent Events: threads publish, each connected client reads its own asyncio queue."""
from __future__ import annotations

import asyncio
import threading
import time

EVENT_TYPES = ("well_created", "well_located", "well_ready", "well_failed", "well_scored")
QUEUE_SIZE = 1000


class Broadcaster:
    """Fan-out of small dict events to subscribed asyncio queues. `publish` is safe to call from any thread."""

    def __init__(self) -> None:
        self._clients: list[tuple[asyncio.AbstractEventLoop, asyncio.Queue]] = []
        self._lock = threading.Lock()

    def subscribe(self) -> asyncio.Queue:
        """Register the calling coroutine's loop and return its queue."""
        queue: asyncio.Queue = asyncio.Queue(QUEUE_SIZE)
        with self._lock:
            self._clients.append((asyncio.get_running_loop(), queue))
        return queue

    def unsubscribe(self, queue: asyncio.Queue) -> None:
        """Drop a client."""
        with self._lock:
            self._clients = [(loop, q) for loop, q in self._clients if q is not queue]

    def publish(self, type_: str, well_id: str, **fields) -> None:
        """Send `{type, id, ts, ...fields}` to every client; a full or closed client is skipped, never blocks the sender."""
        event = {"type": type_, "id": well_id, "ts": time.time(), **fields}
        with self._lock:
            clients = list(self._clients)
        for loop, queue in clients:
            try:
                loop.call_soon_threadsafe(self._put, queue, event)
            except RuntimeError:  # the client's loop is closed
                self.unsubscribe(queue)

    @staticmethod
    def _put(queue: asyncio.Queue, event: dict) -> None:
        """Runs on the client's loop; drops the event when that client is far behind."""
        try:
            queue.put_nowait(event)
        except asyncio.QueueFull:
            pass
