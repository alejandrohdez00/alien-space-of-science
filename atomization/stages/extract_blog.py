import asyncio
import time

import aiohttp
import backoff
from litellm import completion

from atomization.prompts import BLOG_GENERATION_PROMPT
from atomization.utils.errors import is_retryable_error
from atomization.utils.http import fetch_with_retry
from atomization.utils.pdf import pdf_bytes_to_base64_data_uri, truncate_pdf_bytes

MODEL_NAME = "gemini/gemini-3-flash-preview"
MAX_RETRIES = 3


def _log_api_backoff(details: dict) -> None:
    print(
        f"Retrying API call in {details['wait']:.1f}s... "
        f"(attempt {details['tries']}/{MAX_RETRIES})"
    )


@backoff.on_exception(
    backoff.expo,
    Exception,
    max_tries=MAX_RETRIES,
    giveup=lambda e: not is_retryable_error(e),
    on_backoff=_log_api_backoff,
)
async def _answer_from_pdf(
    url: str,
    question: str,
    session: aiohttp.ClientSession,
    max_pages: int | None = None,
    http_semaphore: asyncio.Semaphore | None = None,
) -> dict:
    """Download a PDF, optionally truncate it, and send it to the LLM."""
    timeout = aiohttp.ClientTimeout(total=60, connect=10)
    pdf_bytes, error = await fetch_with_retry(
        session,
        url,
        return_type="bytes",
        max_retries=3,
        timeout=timeout,
        semaphore=http_semaphore,
        on_404="error",
        context="PDF download",
    )

    if error:
        return dict(error=error)

    if max_pages is not None:
        pdf_bytes, truncation_metadata = await asyncio.to_thread(
            truncate_pdf_bytes, pdf_bytes, max_pages
        )
    else:
        truncation_metadata = None

    pdf_data_uri = pdf_bytes_to_base64_data_uri(pdf_bytes)

    response = completion(
        model=MODEL_NAME,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": question},
                    {"type": "image_url", "image_url": pdf_data_uri},
                ],
            }
        ],
    )

    return dict(
        answer=response.choices[0].message.content,
        truncation=truncation_metadata,
    )


async def generate_blog_post(
    pdf_url: str,
    paper_id: str,
    session: aiohttp.ClientSession,
    storage,
    max_pages: int | None,
    http_semaphore: asyncio.Semaphore | None = None,
) -> dict:
    """Generate a blog post from a PDF URL and save it with the storage backend."""
    start_time = time.time()

    try:
        result = await _answer_from_pdf(
            pdf_url,
            BLOG_GENERATION_PROMPT,
            session,
            max_pages,
            http_semaphore,
        )
    except Exception as e:
        elapsed = time.time() - start_time
        return {"id": paper_id, "success": False, "error": str(e), "time": elapsed}

    if "error" in result:
        elapsed = time.time() - start_time
        return {"id": paper_id, "success": False, "error": result["error"], "time": elapsed}

    content = result.get("answer", "No answer generated.")
    truncation = result.get("truncation")

    storage_info = await storage.save(content, paper_id, "blog")

    elapsed = time.time() - start_time
    return {
        "id": paper_id,
        "success": True,
        "storage": storage_info,
        "content": content,
        "time": elapsed,
        "truncation": truncation,
    }
