"""LiteLLM API cost tracking utilities.

This module provides global cost tracking for LiteLLM API calls using thread-safe
accumulators. Used by the main pipeline for cost reporting.
"""

from threading import Lock

import litellm

_total_cost = 0.0
_cost_lock = Lock()
_callback_installed = False


def track_cost_callback(kwargs, completion_response, start_time, end_time):
    """Callback to track LiteLLM API costs."""
    global _total_cost
    try:
        cost = kwargs.get("response_cost", 0)
        if cost > 0:
            with _cost_lock:
                _total_cost += cost
    except Exception:
        pass


def get_total_cost() -> float:
    """Get the total accumulated cost."""
    with _cost_lock:
        return _total_cost


def reset_cost_tracker():
    """Reset the cost tracker to zero."""
    global _total_cost
    with _cost_lock:
        _total_cost = 0.0


def format_cost(cost: float) -> str:
    """Format cost as currency string."""
    if cost < 0.01:
        return f"${cost:.4f}"
    else:
        return f"${cost:.2f}"


def ensure_cost_tracking() -> None:
    """Install the LiteLLM success callback when the pipeline actually runs."""
    global _callback_installed
    if _callback_installed:
        return
    litellm.success_callback = [track_cost_callback]
    _callback_installed = True
