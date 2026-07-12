"""Shared wall-clock timing helpers used across pipeline stages."""

from __future__ import annotations

from contextlib import contextmanager
from time import perf_counter
from typing import Iterator, MutableMapping


@contextmanager
def stage_timer(timings_ms: MutableMapping[str, float], stage: str) -> Iterator[None]:
    """Record the wall-clock duration of a ``with`` block, in milliseconds.

    Args:
        timings_ms: Mapping the duration is written into (key = ``stage``).
        stage: Stage or step name the duration is recorded under.
    """
    start = perf_counter()
    try:
        yield
    finally:
        timings_ms[stage] = round((perf_counter() - start) * 1000.0, 2)
