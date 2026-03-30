from __future__ import annotations

from contextlib import contextmanager, nullcontext
from os import getenv
from typing import Any

try:
    from langsmith import traceable
    from langsmith.run_helpers import tracing_context
except ImportError:
    def traceable(*args, **kwargs):  # type: ignore[no-redef]
        def decorator(func):
            return func
        return decorator

    @contextmanager
    def tracing_context(*args, **kwargs):  # type: ignore[no-redef]
        yield


def langsmith_enabled() -> bool:
    return getenv("LANGSMITH_TRACING", "").lower() == "true" and bool(getenv("LANGSMITH_API_KEY"))


@contextmanager
def maybe_trace(project_name: str = "researchpal") -> Any:
    if langsmith_enabled():
        with tracing_context(enabled=True, project_name=project_name):
            yield
    else:
        with nullcontext():
            yield
