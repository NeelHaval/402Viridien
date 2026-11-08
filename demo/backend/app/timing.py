"""Tiny nested-span timer: `with span("name"):` records start, duration, thread and depth into the current trace.

One `Trace` per well collects the spans of every thread that works on it (upload handler, location task, job worker). A thread
joins a trace with `with attach(trace):`; `span` is a no-op in a thread with no trace. Starts are seconds since the trace
origin (the moment the upload request began), so spans from different threads line up on one timeline.
"""
from __future__ import annotations

import threading
import time
from contextlib import contextmanager
from typing import Iterator

_local = threading.local()
_lock = threading.Lock()


class Trace:
    """Collected spans of one well."""

    def __init__(self, origin: float | None = None) -> None:
        self.origin = time.perf_counter() if origin is None else origin
        self.spans: list[dict] = []

    def mark(self, name: str) -> None:
        """Record an instant event (duration 0), e.g. 'upload_saved'."""
        self.add(name, time.perf_counter(), 0.0, 0)

    def add(self, name: str, start: float, dur: float, depth: int = 0) -> None:
        """Record a span measured elsewhere (`start` is an absolute perf_counter value)."""
        with _lock:
            self.spans.append({"name": name, "start": round(start - self.origin, 4), "dur": round(dur, 4),
                               "thread": threading.current_thread().name, "depth": depth})

    def totals(self) -> dict[str, float]:
        """Sum of durations per span name, seconds."""
        out: dict[str, float] = {}
        with _lock:
            for s in self.spans:
                out[s["name"]] = out.get(s["name"], 0.0) + s["dur"]
        return out

    def snapshot(self) -> list[dict]:
        """A copy of the spans, ordered by start time."""
        with _lock:
            return sorted(self.spans, key=lambda s: (s["start"], s["depth"]))


def current() -> Trace | None:
    """The trace attached to this thread, if any."""
    return getattr(_local, "trace", None)


@contextmanager
def attach(trace: Trace | None) -> Iterator[Trace | None]:
    """Make `trace` the current one for this thread for the duration of the block."""
    prev, prev_depth = current(), getattr(_local, "depth", 0)
    _local.trace, _local.depth = trace, 0
    try:
        yield trace
    finally:
        _local.trace, _local.depth = prev, prev_depth


@contextmanager
def span(name: str) -> Iterator[None]:
    """Time the block under `name`; nested spans get increasing depth (per thread)."""
    t = current()
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


def wrap(fn):
    """Return `fn` wrapped so that, in whatever thread it runs, spans go to the trace that is current *now*."""
    trace = current()

    def run(*args, **kwargs):
        with attach(trace):
            return fn(*args, **kwargs)
    return run
