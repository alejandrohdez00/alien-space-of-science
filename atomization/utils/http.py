"""HTTP fetching with retry logic."""

import asyncio
import random

import aiohttp


async def fetch_with_retry(
    session: aiohttp.ClientSession,
    url: str,
    return_type: str = "bytes",
    max_retries: int = 3,
    timeout: aiohttp.ClientTimeout | None = None,
    semaphore: asyncio.Semaphore | None = None,
    on_404: str = "error",
    context: str = "",
) -> tuple[bytes | str | None, str | None]:
    """Fetch an HTTP resource with exponential backoff."""
    if timeout is None:
        timeout = aiohttp.ClientTimeout(total=30)

    async def _fetch() -> tuple[bytes | str | None, str | None]:
        for attempt in range(max_retries):
            try:
                async with session.get(url, timeout=timeout) as response:
                    if response.status == 404:
                        if on_404 == "error":
                            return None, "Request failed with status 404"
                        return None, None

                    if response.status != 200:
                        if attempt == max_retries - 1:
                            return None, f"Request failed with status {response.status}"
                        raise aiohttp.ClientError(f"Status {response.status}")

                    if return_type == "bytes":
                        content = await response.read()
                    else:
                        content = await response.text()

                    return content, None

            except (aiohttp.ClientError, asyncio.TimeoutError):
                if attempt == max_retries - 1:
                    return None, f"Connection timeout to host {url}"

                base_delay = 2 ** attempt
                jitter = random.uniform(0, base_delay * 0.5)
                delay = base_delay + jitter
                context_str = f" {context}" if context else ""
                print(
                    f"[Retry {attempt + 1}/{max_retries}]{context_str} "
                    f"failed for {url}, retrying in {delay:.1f}s..."
                )
                await asyncio.sleep(delay)

        return None, "Max retries exceeded"

    if semaphore:
        async with semaphore:
            return await _fetch()
    return await _fetch()
