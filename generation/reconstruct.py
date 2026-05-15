"""Stage: Reconstruction - Generate blogposts from super-atoms."""

import asyncio
import hashlib
import time
from pathlib import Path
from typing import Iterable

import backoff
from litellm import completion

from generation.prompts import RECONSTRUCTION_PROMPT
from generation.utils.errors import is_retryable_error
from generation.utils.usage import log_call

MODEL_NAME = "gemini/gemini-3.1-pro-preview"
MAX_RETRIES = 5


def cache_key_for(cluster_ids: Iterable[int]) -> str:
    """Stable, order-invariant cache key for a cluster-id set.

    Two callers that pass the same multiset of cluster IDs (in any order) get
    the same key, so the reconstruction cache can dedupe repeated atom sets.
    """
    canonical = ",".join(str(int(c)) for c in sorted(cluster_ids))
    return hashlib.sha1(canonical.encode("utf-8")).hexdigest()[:16]


def format_super_atoms(super_atoms: list[dict]) -> str:
    """Format super-atoms for the reconstruction prompt."""
    lines = []
    for i, atom in enumerate(super_atoms, 1):
        lines.append(f"{i}. {atom['name']}")
    return "\n".join(lines)


def _log_reconstruction_backoff(details: dict) -> None:
    print(
        f"Retrying reconstruction API call in {details['wait']:.1f}s... "
        f"(attempt {details['tries']}/{MAX_RETRIES})"
    )


@backoff.on_exception(
    backoff.expo,
    Exception,
    max_tries=MAX_RETRIES,
    giveup=lambda e: not is_retryable_error(e),
    on_backoff=_log_reconstruction_backoff,
)
async def reconstruct_blog_from_super_atoms(
    paper_id: str,
    super_atoms: list[dict],
    output_dir: Path,
    sample_index: int = 0,
    model_name: str | None = None,
) -> dict:
    """Generate a methodology sketch from selected super-atoms."""
    start_time = time.time()
    effective_model = model_name or MODEL_NAME

    try:
        super_atoms_text = format_super_atoms(super_atoms)
        prompt = RECONSTRUCTION_PROMPT.format(super_atoms_text=super_atoms_text)

        response = await asyncio.to_thread(
            completion,
            model=effective_model,
            messages=[{"role": "user", "content": prompt}],
        )
        log_call(
            stage="reconstruction",
            model=effective_model,
            sample_id=paper_id,
            response=response,
            extra={"super_atom_count": len(super_atoms)},
        )

        blog_content = response.choices[0].message.content

        reconstruction_dir = output_dir / paper_id
        reconstruction_dir.mkdir(parents=True, exist_ok=True)

        if sample_index == 0:
            reconstruction_path = reconstruction_dir / "reconstructed_blog.md"
        else:
            reconstruction_path = (
                reconstruction_dir / f"reconstructed_blog_sample_{sample_index}.md"
            )

        reconstruction_path.write_text(blog_content, encoding="utf-8")

        elapsed = time.time() - start_time

        return {
            "success": True,
            "paper_id": paper_id,
            "sample_index": sample_index,
            "content": blog_content,
            "super_atom_count": len(super_atoms),
            "output_path": str(reconstruction_path),
            "time": elapsed,
        }

    except Exception as e:
        elapsed = time.time() - start_time
        return {
            "success": False,
            "paper_id": paper_id,
            "sample_index": sample_index,
            "error": str(e),
            "time": elapsed,
        }


def get_sample_path(output_dir: Path, paper_id: str, sample_index: int) -> Path:
    """Get the file path for a reconstruction sample."""
    if sample_index == 0:
        return output_dir / paper_id / "reconstructed_blog.md"
    return output_dir / paper_id / f"reconstructed_blog_sample_{sample_index}.md"
