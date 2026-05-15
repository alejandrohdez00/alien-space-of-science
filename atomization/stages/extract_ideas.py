import asyncio
import json
import time

import backoff
from litellm import completion

from atomization.prompts import IDEA_EXTRACTION_PROMPT
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
def _extract_ideas_sync(blog_content: str) -> dict:
    """Synchronous idea extraction from blog content with retry on JSON errors."""
    prompt = IDEA_EXTRACTION_PROMPT.format(blog_content=blog_content)

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


async def extract_ideas_from_blog(paper_id: str, blog_content: str, storage) -> dict:
    """Extract ideas from blog content and save using storage backend."""
    start_time = time.time()

    try:
        ideas_json = await asyncio.to_thread(_extract_ideas_sync, blog_content)

        content = json.dumps(ideas_json, indent=2)
        storage_info = await storage.save(content, paper_id, "ideas")

        elapsed = time.time() - start_time
        return {
            "id": paper_id,
            "success": True,
            "storage": storage_info,
            "count": len(ideas_json["ideas"]),
            "ideas_json": ideas_json,
            "time": elapsed,
        }
    except Exception as e:
        elapsed = time.time() - start_time
        return {"id": paper_id, "success": False, "error": str(e), "time": elapsed}
