"""Per-call LLM usage / cost tracking for litellm responses.

Usage pattern (script-level):

    from generation.utils.usage import set_log_path, log_call

    # Once at startup, after parsing CLI args:
    set_log_path(args.usage_log)         # None disables logging

    # After each litellm completion call:
    log_call(stage="reconstruction", model=model_name,
             sample_id=paper_id, response=response,
             extra={"k": 3, "beta": 0.5})

Records are appended to ``args.usage_log`` as JSONL (one record per line, one
record per call). Format:

    {"ts": 1714512345.123, "stage": "reconstruction",
     "model": "gemini/gemini-3.1-pro-preview",
     "sample_id": "seq_42",
     "prompt_tokens": 510, "completion_tokens": 1820, "total_tokens": 2330,
     "cost_usd": 0.0231, ...extra}

Designed to be cheap (a few microseconds per call) and concurrency-safe
(per-path threading.Lock; appends remain serialized across the asyncio
event loop's worker threads). Cost estimation uses ``litellm.completion_cost``
when available; falls back to ``None`` so logging never breaks the call.
"""

import json
import threading
import time
from pathlib import Path
from typing import Any
from litellm import completion_cost


_global_log_path: Path | None = None
_locks: dict[str, threading.Lock] = {}
_locks_guard = threading.Lock()


def set_log_path(path: str | Path | None) -> None:
    """Configure the canonical usage log path. ``None`` disables logging."""
    global _global_log_path
    _global_log_path = Path(path) if path else None


def get_log_path() -> Path | None:
    return _global_log_path


def _get_lock(path: Path) -> threading.Lock:
    key = str(path.resolve())
    with _locks_guard:
        lock = _locks.get(key)
        if lock is None:
            lock = threading.Lock()
            _locks[key] = lock
        return lock


def extract_usage(response: Any) -> dict:
    """Pull token counts off a litellm response into a flat dict."""
    usage = getattr(response, "usage", None)
    if usage is None:
        return {}
    out: dict[str, int] = {}
    for key in (
        "prompt_tokens",
        "completion_tokens",
        "total_tokens",
        # Some providers also expose thinking-token counts; keep them if so.
        "reasoning_tokens",
        "cached_tokens",
    ):
        val = None
        if hasattr(usage, key):
            val = getattr(usage, key)
        elif isinstance(usage, dict):
            val = usage.get(key)
        if val is None:
            continue
        try:
            out[key] = int(val)
        except (TypeError, ValueError):
            continue
    return out


def estimate_cost(response: Any, model: str | None = None) -> float | None:
    """Best-effort USD cost. Returns ``None`` if litellm cannot price the call."""
    try:
        cost = completion_cost(completion_response=response, model=model)
        if cost is None:
            return None
        return float(cost)
    except Exception:
        return None


def log_call(
    stage: str,
    model: str,
    sample_id: str,
    response: Any,
    extra: dict | None = None,
) -> None:
    """Append a usage record for one completion to the configured log path.

    No-op if ``set_log_path`` was not called or was called with ``None``.
    """
    if _global_log_path is None:
        return
    record: dict[str, Any] = {
        "ts": time.time(),
        "stage": stage,
        "model": model,
        "sample_id": sample_id,
        "cost_usd": estimate_cost(response, model),
        **extract_usage(response),
    }
    if extra:
        record.update(extra)

    line = json.dumps(record, default=str) + "\n"
    log_path = _global_log_path
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with _get_lock(log_path):
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(line)
