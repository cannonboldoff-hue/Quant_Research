"""Lightweight timing decorator (replaces ad-hoc ``time.time()`` blocks)."""
from __future__ import annotations
import time, functools
from .logging_config import get_logger

_log = get_logger("qresearch.timing")


def timeit(fn):
    @functools.wraps(fn)
    def wrapper(*a, **k):
        t0 = time.perf_counter()
        out = fn(*a, **k)
        _log.info("%s took %.3fs", fn.__name__, time.perf_counter() - t0)
        return out
    return wrapper
