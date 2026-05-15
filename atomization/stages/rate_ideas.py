import asyncio
import json
import time

import backoff
from litellm import completion

from atomization.prompts import IDEA_RATING_AND_REFINEMENT_PROMPT
from atomization.utils.errors import (
    extract_json_from_llm_response,
    is_json_error,
    is_retryable_error,
)

MODEL_NAME = "gemini/gemini-3-flash-preview"
MAX_RETRIES = 5


def _log_json_backoff(details: dict) -> None:
    print(
        f"Retrying in {details['wait']:.1f}s... "
        f"(attempt {details['tries']}/{MAX_RETRIES})"
    )


@backoff.on_exception(
    backoff.expo,
    Exception,
    max_tries=MAX_RETRIES,
    giveup=lambda e: not (is_retryable_error(e) or is_json_error(e)),
    on_backoff=_log_json_backoff,
)
def _rate_and_refine_sync(ideas_json: dict) -> dict:
    """Synchronous rating and refinement with retry on JSON errors."""
    ideas_json_str = json.dumps(ideas_json, indent=2)
    prompt = IDEA_RATING_AND_REFINEMENT_PROMPT.format(ideas_json=ideas_json_str)

    response = completion(
        model=MODEL_NAME,
        messages=[
            {
                "role": "user",
                "content": prompt,
            }
        ],
    )

    response_text = response.choices[0].message.content
    return extract_json_from_llm_response(response_text)


async def rate_and_refine_ideas(paper_id: str, ideas_json: dict, storage) -> dict:
    """Rate and refine extracted ideas, saving refined version."""
    start_time = time.time()

    try:
        refined_json = await asyncio.to_thread(_rate_and_refine_sync, ideas_json)
        content = json.dumps(refined_json, indent=2)
        storage_info = await storage.save(content, paper_id, "refined_ideas")

        elapsed = time.time() - start_time
        return {
            "id": paper_id,
            "success": True,
            "storage": storage_info,
            "count": len(refined_json.get("ratings", [])),
            "time": elapsed,
        }
    except Exception as e:
        elapsed = time.time() - start_time
        return {"id": paper_id, "success": False, "error": str(e), "time": elapsed}
