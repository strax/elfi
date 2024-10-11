import logging
import time
from contextlib import ContextDecorator
from dataclasses import dataclass
from logging import DEBUG
from typing import Protocol

from .utils import format_duration_ns

logger = logging.getLogger(__name__)


class TracingHandler(Protocol):
    def __call__(self, op: str, time_ns: int): ...


def _default_tracing_handler(op: str, time_ns: int):
    if logger.isEnabledFor(DEBUG):
        logger.debug("%s: %s", op, format_duration_ns(time_ns))


@dataclass(slots=True)
class _TracingConfig:
    handler: TracingHandler
    enabled: bool


_GLOBAL_TRACING_CONFIG = _TracingConfig(_default_tracing_handler, False)


class traced_op(ContextDecorator):
    __slots__ = ("_name", "_trace_start_ns")

    def __init__(self, name: str):
        self._name = name
        self._trace_start_ns = -1

    def __enter__(self):
        if _GLOBAL_TRACING_CONFIG.enabled:
            self._trace_start_ns = time.monotonic_ns()

    def __exit__(self, *args):
        del args
        if _GLOBAL_TRACING_CONFIG.enabled:
            trace_end_ns = time.monotonic_ns()
            time_ns = trace_end_ns - self._trace_start_ns
            _GLOBAL_TRACING_CONFIG.handler(self._name, time_ns)

            self._trace_start_ns = -1


def set_tracing_enabled(enabled: bool):
    _GLOBAL_TRACING_CONFIG.enabled = enabled


def set_tracing_handler(handler: TracingHandler):
    _GLOBAL_TRACING_CONFIG.handler = handler


__all__ = [
    "logger",
    "set_tracing_enabled",
    "set_tracing_handler",
    "traced_op",
    "TracingHandler",
]
