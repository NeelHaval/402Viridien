"""Tiny nested-span timer: `with span("name"):` records start, duration, thread and depth into the active trace.

A trace is started per request / job with `with trace() as t:`; spans opened on other threads (OCR workers, executors) are
recorded into the same trace as long as it is still active. With no active trace `span` costs one global lookup.
`t.spans` is a list of dicts; `t.totals()` sums durations by name for the flat `timings` dict stored with each well.
"""
from __future__ import annotations

import threading
import time
from contextlib import contextmanager
from typing import Iterator

_active: "Trace | None" = None
_lock = threading.Lock()
_local = threading.local()


class Trace:
    """Collected spans of one request or job."""

    def __init__(self) -> None:
        self.t0 = time.perf_counter()
        self.spans: list[dict] = []

    def totals(self) -> dict[str, float]:
        """Sum of durations per span name, seconds."""
        out: dict[str, float] = {}
        for s in self.spans:
            out[s["name"]] = out.get(s["name"], 0.0) + s["dur"]
        return out

    def add(self, name: str, start: float, dur: float, depth: int = 0) -> None:
        """Record a span measured elsewhere (`start` is an absolute perf_counter value)."""
        with _lock:
            self.spans.append({"name": name, "start": round(start - self.t0, 4), "dur": round(dur, 4),
                               "thread": threading.current_thread().name, "depth": depth})


@contextmanager
def trace() -> Iterator[Trace]:
    """Make a new trace the active one for the duration of the block."""
    global _active
    t = Trace()
    prev, _active = _active, t
    try:
        yield t
    finally:
        _active = prev


@contextmanager
def span(name: str) -> Iterator[None]:
    """Time the block under `name`; nested spans get increasing depth (per thread)."""
    t = _active
    if t is None:
        yield
        return
    depth = getattr(_local, "depth", 0)
    _local.depth = depth + 1
    start = time.perf_counter()
    try:
        yield
    finally:
        _local.depth = depth
        t.add(name, start, time.perf_counter() - start, depth)
