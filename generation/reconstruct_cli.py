"""Run reconstruction with different selection methods.

Usage:
    # Random sampling (no API key needed for selection, but needed for reconstruction)
    python alien.py reconstruct --method random --k 4 --seed 42

    # LLM sampling
    python alien.py reconstruct --method llm --k 4

    # From inference results (reconstructs all sequences by default)
    python alien.py reconstruct --method inference --results results.json

    # Select 300 samples but reconstruct only the top/first 10
    python alien.py reconstruct --method inference --results results.json \
        --top-k 300 --reconstruct-top-k 10
"""

import argparse
import asyncio
import json
import os
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from generation.baselines import (
    random_subset_selection,
    llm_selection,
    cluster_ids_to_super_atoms,
)
from generation.reconstruct import (
    cache_key_for,
    get_sample_path,
    reconstruct_blog_from_super_atoms,
)
from generation.utils.usage import set_log_path as set_usage_log_path


@dataclass
class ReconstructionResult:
    """Result of reconstructing one selected atom set."""

    success: bool
    method: str
    cluster_ids: list[int]
    content: str | None
    output_path: str | None
    time: float
    error: str | None = None


async def reconstruct_from_cluster_ids(
    cluster_ids: list[int],
    clusters_data: dict,
    output_dir: Path,
    identifier: str,
    sample_index: int = 0,
    method: str = "inference",
    model_name: str | None = None,
) -> ReconstructionResult:
    """Convert cluster IDs to super-atoms and reconstruct them into markdown."""
    start_time = time.time()

    try:
        super_atoms = cluster_ids_to_super_atoms(cluster_ids, clusters_data)
        result = await reconstruct_blog_from_super_atoms(
            paper_id=identifier,
            super_atoms=super_atoms,
            output_dir=output_dir,
            sample_index=sample_index,
            model_name=model_name,
        )

        if result.get("success"):
            return ReconstructionResult(
                success=True,
                method=method,
                cluster_ids=cluster_ids,
                content=result.get("content"),
                output_path=result.get("output_path"),
                time=result.get("time", time.time() - start_time),
            )

        return ReconstructionResult(
            success=False,
            method=method,
            cluster_ids=cluster_ids,
            content=None,
            output_path=None,
            time=result.get("time", time.time() - start_time),
            error=result.get("error", "unknown reconstruction error"),
        )
    except Exception as exc:
        return ReconstructionResult(
            success=False,
            method=method,
            cluster_ids=cluster_ids,
            content=None,
            output_path=None,
            time=time.time() - start_time,
            error=str(exc),
        )


def save_run_metadata(output_dir: Path, metadata: dict):
    """Save metadata for the entire run to a single JSON file."""
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = output_dir / "metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"\nMetadata saved to: {metadata_path}")


def _cache_blog_path(cache_dir: Path, cluster_ids: list[int]) -> Path:
    """Return the canonical cache path for a given atom-set."""
    return cache_dir / cache_key_for(cluster_ids) / "reconstructed_blog.md"


def _link_from_cache(per_sample_path: Path, cache_blog_path: Path) -> None:
    """Make ``per_sample_path`` a symlink (relative) to the cache blog."""
    per_sample_path.parent.mkdir(parents=True, exist_ok=True)
    if per_sample_path.exists() or per_sample_path.is_symlink():
        per_sample_path.unlink()
    rel = os.path.relpath(cache_blog_path, per_sample_path.parent)
    per_sample_path.symlink_to(rel)


def _populate_cache_after_reconstruction(
    cache_dir: Path | None,
    cluster_ids: list[int],
    written_blog_path: str | None,
) -> None:
    """Copy a fresh reconstruction into the cache so future calls can reuse it."""
    if cache_dir is None or not written_blog_path:
        return
    source_path = Path(written_blog_path)
    if not source_path.is_file():
        return
    dst = _cache_blog_path(cache_dir, cluster_ids)
    if dst.exists():
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_bytes(source_path.read_bytes())


def _should_reconstruct(sample_index: int, reconstruct_top_k: int | None) -> bool:
    """Return whether this selected sample should be reconstructed into a blog."""
    return reconstruct_top_k is None or sample_index < reconstruct_top_k


def _selected_only_metadata(
    sample_index: int,
    cluster_ids: list[int],
    super_atoms: list[dict],
    extra: dict | None = None,
) -> dict:
    """Metadata row for samples selected for metrics but not reconstructed."""
    row = {
        "sample_index": sample_index,
        "cluster_ids": cluster_ids,
        "super_atoms": [
            {"cluster_id": a["cluster_id"], "name": a["name"]}
            for a in super_atoms
        ],
        "success": True,
        "reconstructed": False,
        "output_path": None,
        "time": 0.0,
    }
    if extra:
        row.update(extra)
    return row


async def run_random_baseline(
    clusters_data: dict,
    output_dir: Path,
    k: int,
    seed: int,
    num_samples: int,
    max_concurrent: int = 20,
    cache_dir: Path | None = None,
    model_name: str | None = None,
    reconstruct_top_k: int | None = None,
) -> dict:
    """Run random subset baseline and return metadata."""
    all_cluster_ids = [int(cid) for cid in clusters_data["clusters"].keys()]
    print(f"Total clusters available: {len(all_cluster_ids)}")
    print(f"Processing {num_samples} samples with max {max_concurrent} concurrent API calls")

    metadata = {
        "method": "random",
        "timestamp": datetime.now().isoformat(),
        "parameters": {
            "k": k,
            "seed": seed,
            "num_samples": num_samples,
            "total_clusters_available": len(all_cluster_ids),
            "max_concurrent": max_concurrent,
            "reconstruct_top_k": reconstruct_top_k,
            "reconstruction_model": model_name,
        },
        "samples": [],
    }

    semaphore = asyncio.Semaphore(max_concurrent)

    async def process_sample(i: int) -> dict:
        sample_seed = seed + i if seed is not None else None
        selected = random_subset_selection(all_cluster_ids, k=k, seed=sample_seed)
        super_atoms = cluster_ids_to_super_atoms(selected, clusters_data)
        identifier = f"random_seed{sample_seed}"

        if not _should_reconstruct(i, reconstruct_top_k):
            print(f"[Sample {i+1}/{num_samples}] SELECTED ONLY")
            return _selected_only_metadata(i, selected, super_atoms, {"seed": sample_seed})

        # Resume: skip the LLM call if the blog already exists on disk.
        expected_path = get_sample_path(output_dir, identifier, sample_index=0)
        if expected_path.exists():
            print(f"[Sample {i+1}/{num_samples}] SKIP (existing) {expected_path}")
            return {
                "sample_index": i,
                "seed": sample_seed,
                "cluster_ids": selected,
                "super_atoms": [
                    {"cluster_id": a["cluster_id"], "name": a["name"]}
                    for a in super_atoms
                ],
                "success": True,
                "reconstructed": True,
                "output_path": str(expected_path),
                "time": 0.0,
                "skipped": True,
            }

        # Cross-method cache: same atom-set already reconstructed elsewhere?
        if cache_dir is not None:
            cache_blog = _cache_blog_path(cache_dir, selected)
            if cache_blog.is_file():
                _link_from_cache(expected_path, cache_blog)
                print(f"[Sample {i+1}/{num_samples}] CACHE HIT {cache_blog}")
                return {
                    "sample_index": i,
                    "seed": sample_seed,
                    "cluster_ids": selected,
                    "super_atoms": [
                        {"cluster_id": a["cluster_id"], "name": a["name"]}
                        for a in super_atoms
                    ],
                    "success": True,
                    "reconstructed": True,
                    "output_path": str(expected_path),
                    "time": 0.0,
                    "skipped": True,
                    "cache_hit": True,
                }

        async with semaphore:
            result = await reconstruct_from_cluster_ids(
                cluster_ids=selected,
                clusters_data=clusters_data,
                output_dir=output_dir,
                identifier=identifier,
                sample_index=0,
                method="random",
                model_name=model_name,
            )

            if result.success:
                _populate_cache_after_reconstruction(cache_dir, selected, result.output_path)

            sample_metadata = {
                "sample_index": i,
                "seed": sample_seed,
                "cluster_ids": selected,
                "super_atoms": [
                    {"cluster_id": a["cluster_id"], "name": a["name"]}
                    for a in super_atoms
                ],
                "success": result.success,
                "reconstructed": result.success,
                "output_path": result.output_path,
                "time": result.time,
            }
            if not result.success:
                sample_metadata["error"] = result.error

            status = "OK" if result.success else f"ERROR: {result.error}"
            print(f"[Sample {i+1}/{num_samples}] {status} ({result.time:.1f}s)")

            return sample_metadata

    tasks = [process_sample(i) for i in range(num_samples)]
    results = await asyncio.gather(*tasks)
    metadata["samples"] = list(results)

    save_run_metadata(output_dir, metadata)
    return metadata


async def run_llm_baseline(
    clusters_data: dict,
    output_dir: Path,
    k: int,
    num_samples: int,
    max_concurrent: int = 20,
    cache_dir: Path | None = None,
    model_name: str | None = None,
    selection_model_name: str | None = None,
    selection_max_name_length: int | None = None,
    selection_min_interval: float = 0.0,
    reconstruct_top_k: int | None = None,
) -> dict:
    """Run LLM selection baseline and return metadata."""
    all_cluster_ids = [int(cid) for cid in clusters_data["clusters"].keys()]
    print(f"Total clusters available: {len(all_cluster_ids)}")
    print(f"Processing {num_samples} samples with max {max_concurrent} concurrent API calls")

    metadata = {
        "method": "llm_selection",
        "timestamp": datetime.now().isoformat(),
        "parameters": {
            "k": k,
            "num_samples": num_samples,
            "total_clusters_available": len(all_cluster_ids),
            "max_concurrent": max_concurrent,
            "reconstruct_top_k": reconstruct_top_k,
            "selection_model": selection_model_name,
            "selection_max_name_length": selection_max_name_length,
            "selection_min_interval": selection_min_interval,
            "reconstruction_model": model_name,
        },
        "samples": [],
    }

    semaphore = asyncio.Semaphore(max_concurrent)
    selection_rate_lock = asyncio.Lock()
    last_selection_start = 0.0

    async def wait_for_selection_slot() -> None:
        """Space out LLM selection request starts to respect token/minute caps."""
        nonlocal last_selection_start
        if selection_min_interval <= 0:
            return
        async with selection_rate_lock:
            now = time.monotonic()
            wait_s = last_selection_start + selection_min_interval - now
            if wait_s > 0:
                print(f"Waiting {wait_s:.1f}s before next LLM selection call")
                await asyncio.sleep(wait_s)
            last_selection_start = time.monotonic()

    async def process_sample(i: int) -> dict:
        identifier = f"llm_{i}"
        expected_path = get_sample_path(output_dir, identifier, sample_index=0)
        selection_json_path = output_dir / identifier / "llm_selection.json"

        # Resume selection independently from reconstruction. This matters for
        # selection-only samples when reconstruct_top_k < num_samples.
        if selection_json_path.exists():
            with open(selection_json_path) as f:
                cached = json.load(f)
            selected = cached.get("selected_ids", [])
            if selected:
                super_atoms = cluster_ids_to_super_atoms(selected, clusters_data)
                if not _should_reconstruct(i, reconstruct_top_k):
                    print(f"[Sample {i+1}/{num_samples}] SELECTED ONLY (existing)")
                    return _selected_only_metadata(
                        i,
                        selected,
                        super_atoms,
                        {"llm_selection_path": str(selection_json_path), "skipped": True},
                    )

                if expected_path.exists():
                    print(f"[Sample {i+1}/{num_samples}] SKIP (existing) {expected_path}")
                    return {
                        "sample_index": i,
                        "cluster_ids": selected,
                        "super_atoms": [
                            {"cluster_id": a["cluster_id"], "name": a["name"]}
                            for a in super_atoms
                        ],
                        "success": True,
                        "reconstructed": True,
                        "output_path": str(expected_path),
                        "llm_selection_path": str(selection_json_path),
                        "time": 0.0,
                        "skipped": True,
                    }

        async with semaphore:
            if selection_json_path.exists():
                with open(selection_json_path) as f:
                    selection_response = json.load(f)
                selected = [int(sid) for sid in selection_response.get("selected_ids", [])]
            else:
                await wait_for_selection_slot()
                selection_result = await llm_selection(
                    cluster_ids=all_cluster_ids,
                    clusters_data=clusters_data,
                    n_select=k,
                    model_name=selection_model_name,
                    max_name_length=selection_max_name_length,
                    sample_id=identifier,
                )
                selected = selection_result.selected_ids
                selection_response = selection_result.llm_response

                # Save LLM selection JSON as soon as the selection is known,
                # even if this sample is not reconstructed into a blog.
                selection_json_path.parent.mkdir(parents=True, exist_ok=True)
                with open(selection_json_path, "w") as f:
                    json.dump(selection_response, f, indent=2)

            super_atoms = cluster_ids_to_super_atoms(selected, clusters_data)

            if not _should_reconstruct(i, reconstruct_top_k):
                print(f"[Sample {i+1}/{num_samples}] SELECTED ONLY")
                return _selected_only_metadata(
                    i,
                    selected,
                    super_atoms,
                    {"llm_selection_path": str(selection_json_path)},
                )

            # Cross-method cache check (after selection — we now know cluster_ids).
            if cache_dir is not None:
                cache_blog = _cache_blog_path(cache_dir, selected)
                if cache_blog.is_file():
                    _link_from_cache(expected_path, cache_blog)
                    print(f"[Sample {i+1}/{num_samples}] CACHE HIT {cache_blog}")
                    return {
                        "sample_index": i,
                        "cluster_ids": selected,
                        "super_atoms": [
                            {"cluster_id": a["cluster_id"], "name": a["name"]}
                            for a in super_atoms
                        ],
                        "success": True,
                        "reconstructed": True,
                        "output_path": str(expected_path),
                        "llm_selection_path": str(selection_json_path),
                        "time": 0.0,
                        "skipped": True,
                        "cache_hit": True,
                    }

            result = await reconstruct_from_cluster_ids(
                cluster_ids=selected,
                clusters_data=clusters_data,
                output_dir=output_dir,
                identifier=identifier,
                sample_index=0,
                method="llm_selection",
                model_name=model_name,
            )

            if result.success:
                _populate_cache_after_reconstruction(cache_dir, selected, result.output_path)

            sample_metadata = {
                "sample_index": i,
                "cluster_ids": selected,
                "super_atoms": [
                    {"cluster_id": a["cluster_id"], "name": a["name"]}
                    for a in super_atoms
                ],
                "success": result.success,
                "reconstructed": result.success,
                "output_path": result.output_path,
                "llm_selection_path": str(selection_json_path),
                "time": result.time,
            }
            if not result.success:
                sample_metadata["error"] = result.error

            status = "OK" if result.success else f"ERROR: {result.error}"
            print(f"[Sample {i+1}/{num_samples}] {status} ({result.time:.1f}s)")

            return sample_metadata

    tasks = [process_sample(i) for i in range(num_samples)]
    results = await asyncio.gather(*tasks)
    metadata["samples"] = list(results)

    save_run_metadata(output_dir, metadata)
    return metadata


async def run_inference_reconstruction(
    clusters_data: dict,
    output_dir: Path,
    results_path: str,
    top_k: int | None,
    max_concurrent: int = 20,
    cache_dir: Path | None = None,
    model_name: str | None = None,
    reconstruct_top_k: int | None = None,
) -> dict:
    """Reconstruct from inference pipeline results and return metadata."""
    with open(results_path) as f:
        results = json.load(f)

    sequences = results.get("top_k_sequences", results.get("all_sequences", []))
    if top_k is not None:
        sequences = sequences[:top_k]
    inferred_k = (
        results.get("config", {}).get("k")
        or (sequences[0].get("length") if sequences else None)
        or (len(sequences[0].get("cluster_ids", [])) if sequences else None)
    )

    print(f"Reconstructing {len(sequences)} sequences from {results_path}")
    print(f"Processing with max {max_concurrent} concurrent API calls")

    metadata = {
        "method": "inference",
        "timestamp": datetime.now().isoformat(),
        "parameters": {
            "results_path": results_path,
            "top_k": top_k if top_k is not None else len(sequences),
            "sequences_available": len(
                results.get("top_k_sequences", results.get("all_sequences", []))
            ),
            "max_concurrent": max_concurrent,
            "total_clusters_available": len(clusters_data.get("clusters", {})),
            "k": inferred_k,
            "reconstruct_top_k": reconstruct_top_k,
            "reconstruction_model": model_name,
        },
        "samples": [],
    }

    semaphore = asyncio.Semaphore(max_concurrent)

    async def process_sequence(i: int, seq: dict) -> dict:
        cluster_ids = seq["cluster_ids"]
        super_atoms = cluster_ids_to_super_atoms(cluster_ids, clusters_data)
        identifier = f"seq_{i}"
        inference_scores = {
            "avg_nll_coherence": seq.get("avg_nll_coherence"),
            "avg_nll_availability": seq.get("avg_nll_availability"),
            "rank_coherence": seq.get("rank_coherence"),
            "rank_availability": seq.get("rank_availability"),
            "rank_fused": seq.get("rank_fused"),
            "fusion_score": seq.get("fusion_score"),
        }

        if not _should_reconstruct(i, reconstruct_top_k):
            print(f"[Sequence {i+1}/{len(sequences)}] SELECTED ONLY")
            return _selected_only_metadata(
                i,
                cluster_ids,
                super_atoms,
                {"inference_scores": inference_scores},
            )

        # Resume: skip the LLM call if the blog already exists on disk.
        expected_path = get_sample_path(output_dir, identifier, sample_index=0)
        if expected_path.exists():
            print(f"[Sequence {i+1}/{len(sequences)}] SKIP (existing) {expected_path}")
            return {
                "sample_index": i,
                "cluster_ids": cluster_ids,
                "super_atoms": [
                    {"cluster_id": a["cluster_id"], "name": a["name"]}
                    for a in super_atoms
                ],
                "inference_scores": inference_scores,
                "success": True,
                "reconstructed": True,
                "output_path": str(expected_path),
                "time": 0.0,
                "skipped": True,
            }

        # Cross-method (e.g. cross-beta) cache: if the same atom-set was
        # reconstructed in another beta's directory, reuse it via a symlink.
        if cache_dir is not None:
            cache_blog = _cache_blog_path(cache_dir, cluster_ids)
            if cache_blog.is_file():
                _link_from_cache(expected_path, cache_blog)
                print(f"[Sequence {i+1}/{len(sequences)}] CACHE HIT {cache_blog}")
                return {
                    "sample_index": i,
                    "cluster_ids": cluster_ids,
                    "super_atoms": [
                        {"cluster_id": a["cluster_id"], "name": a["name"]}
                        for a in super_atoms
                    ],
                    "inference_scores": inference_scores,
                    "success": True,
                    "reconstructed": True,
                    "output_path": str(expected_path),
                    "time": 0.0,
                    "skipped": True,
                    "cache_hit": True,
                }

        async with semaphore:
            result = await reconstruct_from_cluster_ids(
                cluster_ids=cluster_ids,
                clusters_data=clusters_data,
                output_dir=output_dir,
                identifier=identifier,
                sample_index=0,
                method="inference",
                model_name=model_name,
            )

            if result.success:
                _populate_cache_after_reconstruction(cache_dir, cluster_ids, result.output_path)

            sample_metadata = {
                "sample_index": i,
                "cluster_ids": cluster_ids,
                "super_atoms": [
                    {"cluster_id": a["cluster_id"], "name": a["name"]}
                    for a in super_atoms
                ],
                "inference_scores": inference_scores,
                "success": result.success,
                "reconstructed": result.success,
                "output_path": result.output_path,
                "time": result.time,
            }
            if not result.success:
                sample_metadata["error"] = result.error

            status = "OK" if result.success else f"ERROR: {result.error}"
            print(f"[Sequence {i+1}/{len(sequences)}] {status} ({result.time:.1f}s)")

            return sample_metadata

    tasks = [process_sequence(i, seq) for i, seq in enumerate(sequences)]
    results_list = await asyncio.gather(*tasks)
    metadata["samples"] = list(results_list)

    save_run_metadata(output_dir, metadata)
    return metadata


async def main():
    parser = argparse.ArgumentParser(description="Run reconstruction with different methods")
    parser.add_argument(
        "--method",
        choices=["random", "llm", "inference"],
        required=True,
        help="Selection method: random, llm, or inference",
    )
    parser.add_argument(
        "--clusters",
        default="clusters.json",
        help="Path to clusters JSON file",
    )
    parser.add_argument(
        "--output-dir",
        default="reconstructions",
        help="Output directory for reconstructed blogs",
    )
    parser.add_argument(
        "--method-output-name",
        default=None,
        help=(
            "Optional subdirectory name under --output-dir. Defaults to the "
            "method name (random, llm_selection, inference)."
        ),
    )
    parser.add_argument(
        "--k",
        type=int,
        default=4,
        help="Number of clusters to select (for random/llm methods)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (for random method)",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=1,
        help="Number of samples to generate (for random/llm methods)",
    )
    parser.add_argument(
        "--results",
        default="results.json",
        help="Path to inference results JSON (for inference method)",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=None,
        help="Number of top sequences to select from inference results. Defaults to all.",
    )
    parser.add_argument(
        "--reconstruct-top-k",
        type=int,
        default=None,
        help=(
            "Only reconstruct the first N selected samples into blogs while "
            "still writing metadata for every selected sample. Defaults to all."
        ),
    )
    parser.add_argument(
        "--max-concurrent",
        type=int,
        default=20,
        help="Maximum concurrent API calls (default: 20)",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=None,
        help=(
            "Optional content-addressed cache directory. When set, reconstructions "
            "are deduped across runs by sorted-cluster_ids hash: a hit is symlinked "
            "from the per-sample path to <cache-dir>/<hash>/reconstructed_blog.md, "
            "saving the LLM call. Useful for cross-beta sweeps where the same "
            "atom-set appears in multiple beta values' top-K lists."
        ),
    )
    parser.add_argument(
        "--reconstruction-model",
        type=str,
        default=None,
        help=(
            "LiteLLM model identifier for reconstruction. Defaults to "
            "generation.reconstruct.MODEL_NAME. Pass a different value to swap the "
            "reconstruction LLM without editing source. To compare two models "
            "cleanly, set this together with a fresh --output-dir so the "
            "per-sample blogs and the cache start empty."
        ),
    )
    parser.add_argument(
        "--selection-model",
        type=str,
        default=None,
        help=(
            "LiteLLM model identifier for LLM atom selection. Only used with "
            "--method llm. Defaults to generation.baselines.selection.MODEL_NAME."
        ),
    )
    parser.add_argument(
        "--selection-max-name-length",
        type=int,
        default=None,
        help=(
            "Optional maximum characters per cluster name in the LLM selection "
            "prompt. Defaults to no truncation."
        ),
    )
    parser.add_argument(
        "--selection-min-interval",
        type=float,
        default=0.0,
        help=(
            "Minimum seconds between starting LLM selection API calls. Useful "
            "for large prompts and provider token-per-minute limits. Defaults to 0."
        ),
    )
    parser.add_argument(
        "--usage-log",
        type=Path,
        default=None,
        help=(
            "Optional JSONL path; one record per LLM call (tokens + estimated "
            "USD via litellm.completion_cost)."
        ),
    )

    args = parser.parse_args()
    if args.reconstruct_top_k is not None and args.reconstruct_top_k < 0:
        parser.error("--reconstruct-top-k must be non-negative")
    if args.selection_max_name_length is not None and args.selection_max_name_length <= 0:
        parser.error("--selection-max-name-length must be positive when provided")
    if args.selection_min_interval < 0:
        parser.error("--selection-min-interval must be non-negative")
    set_usage_log_path(args.usage_log)

    # Load cluster data
    print(f"Loading clusters from {args.clusters}...")
    with open(args.clusters) as f:
        clusters_data = json.load(f)

    output_dir = Path(args.output_dir)
    method_output_defaults = {
        "random": "random",
        "llm": "llm_selection",
        "inference": "inference",
    }
    method_output_name = args.method_output_name or method_output_defaults[args.method]
    method_output_dir = output_dir / method_output_name

    if args.method == "random":
        print(f"\n=== Random Sampling (k={args.k}, seed={args.seed}) ===")
        await run_random_baseline(
            clusters_data=clusters_data,
            output_dir=method_output_dir,
            k=args.k,
            seed=args.seed,
            num_samples=args.num_samples,
            max_concurrent=args.max_concurrent,
            cache_dir=args.cache_dir,
            model_name=args.reconstruction_model,
            reconstruct_top_k=args.reconstruct_top_k,
        )

    elif args.method == "llm":
        print(f"\n=== LLM Selection (k={args.k}) ===")
        await run_llm_baseline(
            clusters_data=clusters_data,
            output_dir=method_output_dir,
            k=args.k,
            num_samples=args.num_samples,
            max_concurrent=args.max_concurrent,
            cache_dir=args.cache_dir,
            model_name=args.reconstruction_model,
            selection_model_name=args.selection_model,
            selection_max_name_length=args.selection_max_name_length,
            selection_min_interval=args.selection_min_interval,
            reconstruct_top_k=args.reconstruct_top_k,
        )

    elif args.method == "inference":
        top_k_str = args.top_k if args.top_k is not None else "all"
        print(f"\n=== Inference Reconstruction (top-k={top_k_str}) ===")
        await run_inference_reconstruction(
            clusters_data=clusters_data,
            output_dir=method_output_dir,
            results_path=args.results,
            top_k=args.top_k,
            max_concurrent=args.max_concurrent,
            cache_dir=args.cache_dir,
            model_name=args.reconstruction_model,
            reconstruct_top_k=args.reconstruct_top_k,
        )

    print("\nDone!")


if __name__ == "__main__":
    asyncio.run(main())
