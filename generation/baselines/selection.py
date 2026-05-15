"""Selection strategies for super-atom baselines.

This module provides:
- cluster_ids_to_super_atoms: Convert cluster IDs to super-atom dicts
- random_subset_selection: Randomly select k cluster IDs
- llm_selection: LLM-based selection of coherent super-atoms
"""

import asyncio
import random
from dataclasses import dataclass

import backoff
from litellm import completion

from generation.prompts import LLM_SELECTION_PROMPT
from generation.utils.errors import (
    MalformedJSONError,
    extract_json_from_llm_response,
    is_retryable_error,
)
from generation.utils.usage import log_call


MODEL_NAME = "anthropic/claude-opus-4-5-20251101"
MAX_RETRIES = 5


@dataclass
class LLMSelectionResult:
    """Result of LLM-based selection."""

    selected_ids: list[int]
    llm_response: dict


def cluster_ids_to_super_atoms(
    cluster_ids: list[int],
    clusters_data: dict,
) -> list[dict]:
    """Convert cluster IDs to super-atom dictionaries for reconstruction."""
    clusters_metadata = clusters_data.get("clusters", {})
    super_atoms = []

    for cluster_id in cluster_ids:
        cluster_str = str(cluster_id)
        if cluster_str not in clusters_metadata:
            raise KeyError(f"Cluster ID {cluster_id} not found in clusters data")

        meta = clusters_metadata[cluster_str]
        super_atoms.append({
            "cluster_id": cluster_id,
            "name": meta.get("name", ""),
            "description": meta.get("description", ""),
            "confidence": meta.get("confidence", "medium"),
            "sample_atoms": meta.get("sample_atoms", []),
        })

    return super_atoms


def random_subset_selection(
    cluster_ids: list[int],
    k: int,
    seed: int | None = None,
) -> list[int]:
    """Randomly select k cluster IDs from the available pool."""
    if k > len(cluster_ids):
        raise ValueError(
            f"Cannot select {k} clusters from pool of {len(cluster_ids)}"
        )

    rng = random.Random(seed)
    return rng.sample(cluster_ids, k)


def _format_super_atoms_for_selection(
    super_atoms: list[dict],
    max_name_length: int | None = None,
) -> str:
    """Format super-atoms for the LLM selection prompt."""
    lines = []
    for atom in super_atoms:
        name = atom["name"]
        if max_name_length is not None and len(name) > max_name_length:
            name = name[:max_name_length - 3] + "..."
        lines.append(f"[ID: {atom['cluster_id']}] {name}")
    return "\n".join(lines)


class EmptyResponseError(Exception):
    """Raised when LLM returns an empty response."""

    pass


def _is_retryable_selection_error(e: Exception) -> bool:
    """Check if exception should trigger a retry for selection."""
    return (
        is_retryable_error(e)
        or isinstance(e, EmptyResponseError)
        or isinstance(e, MalformedJSONError)
    )


def _log_selection_backoff(details: dict) -> None:
    print(
        f"Retrying LLM selection in {details['wait']:.1f}s... "
        f"(attempt {details['tries']}/{MAX_RETRIES}, "
        f"error: {type(details['exception']).__name__})"
    )


@backoff.on_exception(
    backoff.expo,
    Exception,
    max_tries=MAX_RETRIES,
    giveup=lambda e: not _is_retryable_selection_error(e),
    on_backoff=_log_selection_backoff,
)
async def _call_and_parse_llm_selection(
    prompt: str,
    model_name: str,
    sample_id: str = "",
) -> dict:
    """Call LLM and parse JSON response with retry logic."""
    response = await asyncio.to_thread(
        completion,
        model=model_name,
        messages=[{"role": "user", "content": prompt}],
    )
    log_call(
        stage="llm_selection",
        model=model_name,
        sample_id=sample_id,
        response=response,
    )
    content = response.choices[0].message.content
    if content is None or not content.strip():
        raise EmptyResponseError("LLM returned empty response")
    return extract_json_from_llm_response(content)


async def llm_selection(
    cluster_ids: list[int],
    clusters_data: dict,
    n_select: int = 4,
    model_name: str | None = MODEL_NAME,
    max_name_length: int | None = None,
    sample_id: str = "",
) -> LLMSelectionResult:
    """Use an LLM to select super-atoms that form a coherent combination."""
    if n_select > len(cluster_ids):
        raise ValueError(
            f"Cannot select {n_select} clusters from pool of {len(cluster_ids)}"
        )

    super_atoms = cluster_ids_to_super_atoms(cluster_ids, clusters_data)
    random.shuffle(super_atoms)
    super_atoms_list = _format_super_atoms_for_selection(
        super_atoms,
        max_name_length=max_name_length,
    )
    prompt = LLM_SELECTION_PROMPT.format(
        n_select=n_select,
        super_atoms_list=super_atoms_list,
    )
    parsed = await _call_and_parse_llm_selection(
        prompt,
        model_name or MODEL_NAME,
        sample_id=sample_id,
    )
    selected_ids = parsed.get("selected_ids", [])

    # Validate selected IDs
    valid_ids = set(cluster_ids)
    validated_selection = [
        int(sid)
        for sid in selected_ids
        if int(sid) in valid_ids
    ]

    if len(validated_selection) >= n_select:
        return LLMSelectionResult(
            selected_ids=validated_selection[:n_select],
            llm_response=parsed,
        )
    else:
        raise ValueError(
            f"LLM returned {len(validated_selection)} valid IDs, expected {n_select}"
        )
