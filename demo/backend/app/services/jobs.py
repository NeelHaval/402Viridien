"""One worker thread runs jobs one at a time, so concurrent uploads queue instead of competing for the model.

Job state lives in the wells table (`queued`, `processing`, `ready`, `failed`, error text, timings); this class only schedules.
"""
from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor
from typing import Callable

log = logging.getLogger("jobs")


class JobQueue:
    """Single-worker queue of well ids."""

    def __init__(self, handler: Callable[[str], None]) -> None:
        """`handler(well_id)` does the work and records its own result."""
        self._handler = handler
        self._pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="job")

    def enqueue(self, well_id: str) -> None:
        """Schedule a well; returns immediately."""
        self._pool.submit(self._run, well_id)

    def _run(self, well_id: str) -> None:
        """Run the handler; it must not take the worker down."""
        try:
            self._handler(well_id)
        except Exception:
            log.exception("job %s crashed", well_id)

    def shutdown(self, drain: bool = False) -> None:
        """Stop the worker. By default queued jobs are dropped (they are marked interrupted on the next start); with `drain` all run first."""
        self._pool.shutdown(wait=True, cancel_futures=not drain)
