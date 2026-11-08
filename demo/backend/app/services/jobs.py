"""One worker thread runs jobs one at a time, so concurrent uploads queue instead of competing for the model.

Job state lives in the wells table (`queued`, `processing`, `ready`, `failed`, error text, timings); this class only schedules.
"""
from __future__ import annotations

import logging
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Callable

from app.tools.errors import QueueBusy

log = logging.getLogger("jobs")


class _Admission:
    """One queue slot, owned by this context until transferred to a future."""

    def __init__(self,queue):
        self.queue=queue
        self.transferred=False
        self.released=False
        self.lock=threading.Lock()

    def __enter__(self):
        return self

    def release(self):
        with self.lock:
            if not self.released:
                self.released=True
                self.queue._slots.release()

    def enqueue(self,well_id):
        if self.transferred or self.released:
            raise RuntimeError("job admission has already been used")
        future=self.queue._pool.submit(self.queue._run,well_id)
        self.transferred=True
        # Completion, handler exceptions and cancelled queued futures all
        # release precisely the slot that admitted this job.
        future.add_done_callback(lambda _:self.release())
        return future

    def __exit__(self,*exc):
        try:
            if not self.transferred:self.release()
        finally:
            self.queue._state_lock.release()


class JobQueue:
    """Single-worker queue of well ids."""

    def __init__(self, handler: Callable[[str], None], max_pending: int=32) -> None:
        """`handler(well_id)` does the work and records its own result."""
        self._handler = handler
        if isinstance(max_pending,bool) or not isinstance(max_pending,int) or max_pending<1:
            raise ValueError("max_pending must be a positive integer")
        self._slots=threading.BoundedSemaphore(max_pending)
        self._state_lock=threading.RLock()
        self._closed=False
        self._pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="job")

    def admit(self):
        """Reserve before changing files/DB; raise QueueBusy without mutation.

        The short preparation context also excludes shutdown until its future
        is submitted, so accepted persistent inputs cannot race a closed pool.
        """
        self._state_lock.acquire()
        if self._closed or not self._slots.acquire(blocking=False):
            self._state_lock.release()
            raise QueueBusy("the job queue is full or shutting down; retry later")
        return _Admission(self)

    def enqueue(self, well_id: str):
        """Schedule a well; returns immediately."""
        with self.admit() as admitted:
            return admitted.enqueue(well_id)

    def _run(self, well_id: str) -> None:
        """Run the handler; it must not take the worker down."""
        try:
            self._handler(well_id)
        except Exception:
            log.exception("job %s crashed", well_id)

    def shutdown(self, drain: bool = False) -> None:
        """Stop the worker. By default queued jobs are dropped (they are marked interrupted on the next start); with `drain` all run first."""
        with self._state_lock:
            self._closed=True
        self._pool.shutdown(wait=True, cancel_futures=not drain)
