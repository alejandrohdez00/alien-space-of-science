"""Error classification and LLM response parsing utilities.

This module provides utilities for handling API errors and parsing JSON responses
from LLM completions. Used across all pipeline stages for retry logic and response
processing.
"""

import json
import re
import litellm


RETRYABLE_LITELLM_ERRORS = (
    litellm.RateLimitError,
    litellm.ServiceUnavailableError,
    litellm.APIError,
    litellm.APIConnectionError,
    litellm.Timeout,
)


def is_retryable_error(e: Exception) -> bool:
    """
    Check if exception is retryable (rate limits or transient API errors).

    Non-retryable errors (should fail immediately):
    - AuthenticationError: Invalid API key
    - BadRequestError: Invalid parameters

    Retryable errors (should use backoff):
    - RateLimitError: Rate limit exceeded
    - ServiceUnavailableError: Service over capacity (503)
    - APIError: Transient service errors (5xx)
    - APIConnectionError: Network issues
    - Timeout: Request timeout
    """
    return isinstance(e, RETRYABLE_LITELLM_ERRORS)


def is_json_error(e: Exception) -> bool:
    """Check if exception is a JSON parsing error."""
    return isinstance(e, json.JSONDecodeError)


def extract_json_from_llm_response(response_text: str):
    """
    Extract and parse JSON from LLM response text.

    Handles markdown code blocks and invalid escape sequences.

    Args:
        response_text: Raw text response from LLM

    Returns:
        Parsed JSON object (dict, list, etc.)

    Raises:
        json.JSONDecodeError: If JSON cannot be parsed after cleanup
    """
    text = response_text.strip()

    # Remove markdown code blocks
    if text.startswith("```"):
        text = re.sub(r'^```(?:json)?\s*', '', text)
        text = re.sub(r'\s*```$', '', text)

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # Fix invalid escape sequences and retry
        fixed = re.sub(r'\\(?!["\\\/bfnrtu])', r'\\\\', text)
        return json.loads(fixed)
