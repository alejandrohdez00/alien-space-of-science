#!/usr/bin/env python3
"""Sample top-K atom-sets per beta from per-k shared candidate pools.

For each ``k in --ks`` (default ``[3, 4]``) and for each ``beta in [0.0, 0.1,
..., 1.0]`` we re-rank a shared pool of length-k atom-sets by

    fusion_score = (1 - beta) * z_coh + beta * z_avail   (beta in [0, 1])

and emit a results JSON for the top-K sets. Each output JSON can be passed to
``python alien.py reconstruct --method inference``.

The pool is per-k (one for k=3, one for k=4); within a k it is shared across
all betas so the only thing that changes per beta is the ranking. This makes
beta values directly comparable within each k.

Output layout:
    <output-dir>/
        k3/
            pool_scores.json
            beta_0.0/results.json
            ...
            beta_1.0/results.json
            beta_sweep_summary.json
        k4/
            ...

Usage:
    python alien.py sample \\
        --coherence-model    models/<coherence_run>/ \\
        --availability-model models/<two_tower_run>/ \\
        --token-mapping      datasets/<...>/token_mapping.json \\
        --output-dir         results/beta_sweep/ \\
        --ks 3 4 --n-gen 100000 --top-k 300
"""

import argparse
import json
import math
import sys
import time
from itertools import combinations, permutations
from pathlib import Path

import numpy as np
import torch

from generation.generation import generate_sequences
from generation.scoring import (
    load_author_embeddings,
    load_availability_model,
    load_model,
    score_sequences_batched,
    score_sequences_two_tower,
    score_sets_alien,
)
from generation.token_mapping import TokenMapping


def generate_pool(
    coherence_model,
    token_mapping: TokenMapping,
    n_gen: int,
    k: int,
    gen_top_k: int,
    gen_temperature: float,
    device: str,
) -> list[tuple[int, ...]]:
    """Generate from the coherence model and dedupe to unique k-atom sets."""
    sys.stderr.write(f"\nGenerating {n_gen:,} sequences (k={k})...\n")
    t0 = time.time()
    raw = generate_sequences(
        coherence_model, token_mapping,
        num_samples=n_gen,
        max_length=k + 1,  # BOS + k atoms (EOS suppressed by max_length cap)
        temperature=gen_temperature,
        top_k=gen_top_k,
        device=device,
    )
    sys.stderr.write(f"  Generated {len(raw)} in {time.time() - t0:.1f}s\n")

    unique: set[frozenset] = set()
    for seq in raw:
        s = frozenset(seq["clusters"])
        if len(s) == k:
            unique.add(s)
    sys.stderr.write(f"  Unique k={k} sets: {len(unique):,}\n")

    # Sort canonical (ascending) for reproducibility.
    return sorted([tuple(sorted(s)) for s in unique])


def _flatten_sets_to_perm_token_seqs(
    sets: list[tuple[int, ...]],
    token_mapping: TokenMapping,
) -> list[list[int]]:
    """Enumerate all ordered token sequences for each unordered atom set."""
    out = []
    for atom_set in sets:
        for perm in permutations(atom_set):
            out.append(token_mapping.cluster_ids_to_token_ids(list(perm)))
    return out


def score_sets_coherence(
    sets: list[tuple[int, ...]],
    coherence_model,
    token_mapping: TokenMapping,
    device: str,
    batch_size: int,
    chunk_size: int = 100_000,
) -> np.ndarray:
    """Score unordered atom sets by mean coherence over all permutations."""
    if not sets:
        return np.zeros(0)

    k = len(sets[0])
    n_perms = math.factorial(k)
    scores = np.zeros(len(sets), dtype=np.float64)
    total_chunks = (len(sets) + chunk_size - 1) // chunk_size
    t_all = time.time()

    for chunk_start in range(0, len(sets), chunk_size):
        chunk = sets[chunk_start:chunk_start + chunk_size]
        chunk_num = chunk_start // chunk_size + 1
        sys.stderr.write(
            f"    coherence chunk {chunk_num}/{total_chunks}: "
            f"{chunk_start:,}-{chunk_start + len(chunk):,} sets "
            f"({len(chunk) * n_perms:,} ordered sequences)\n"
        )
        seqs = _flatten_sets_to_perm_token_seqs(chunk, token_mapping)
        nlls = score_sequences_batched(
            coherence_model, seqs, token_mapping,
            device=device, batch_size=batch_size,
        )
        nll_arr = np.asarray(nlls, dtype=np.float64).reshape(len(chunk), n_perms)
        scores[chunk_start:chunk_start + len(chunk)] = -nll_arr.mean(axis=1)

        done = chunk_start + len(chunk)
        elapsed = time.time() - t_all
        rate = done / elapsed if elapsed > 0 else 0.0
        remaining = (len(sets) - done) / rate if rate > 0 else 0.0
        sys.stderr.write(
            f"      {done:,}/{len(sets):,} sets ({done / len(sets):.1%}); "
            f"ETA {remaining / 60:.1f} min\n"
        )

    return scores


def verify_availability_permutation_invariance(
    avail_model,
    author_embeddings: torch.Tensor,
    token_mapping: TokenMapping,
    device: str,
    batch_size: int,
    tt_top_k: int,
    sample_set: tuple[int, ...],
    rtol: float = 1e-4,
) -> None:
    """Warn if the two-tower availability score changes with atom order."""
    seq1 = token_mapping.cluster_ids_to_token_ids(list(sample_set))
    seq2 = token_mapping.cluster_ids_to_token_ids(list(reversed(sample_set)))
    score1, score2 = score_sequences_two_tower(
        avail_model, [seq1, seq2], author_embeddings, token_mapping,
        device=device, batch_size=batch_size, top_k=tt_top_k,
    )
    if abs(score1 - score2) > rtol * max(1.0, abs(score1)):
        sys.stderr.write(
            "  WARNING: two-tower availability score differs across orderings: "
            f"{score1:.6f} vs {score2:.6f} on set {sample_set}.\n"
        )
    else:
        sys.stderr.write(
            "  Two-tower permutation-invariance check passed "
            f"({score1:.6f} vs {score2:.6f}).\n"
        )


def _zscore(x: np.ndarray) -> np.ndarray:
    mean = x.mean()
    std = x.std()
    if std <= 0:
        return np.zeros_like(x)
    return (x - mean) / std


def build_results_payload(
    cand_sets: list[tuple[int, ...]],
    coh_scores: np.ndarray,
    avail_scores: np.ndarray,
    z_coh: np.ndarray,
    z_avail: np.ndarray,
    beta: float,
    top_k: int,
    k: int,
    config_extra: dict,
) -> dict:
    """Top-K results in the schema the reconstruction CLI reads.

    The reconstruction CLI accesses ``top_k_sequences`` and per-entry
    ``cluster_ids``; the other fields are kept for downstream analysis (so
    inference_scores in the reconstruction metadata are populated).
    """
    fusion = (1.0 - beta) * z_coh + beta * z_avail
    # Order coherence and availability ranks (descending = best first), 1-indexed.
    rank_coh = np.empty_like(coh_scores, dtype=np.int64)
    rank_coh[np.argsort(-coh_scores)] = np.arange(1, len(coh_scores) + 1)
    rank_avail = np.empty_like(avail_scores, dtype=np.int64)
    rank_avail[np.argsort(-avail_scores)] = np.arange(1, len(avail_scores) + 1)

    if top_k >= len(fusion):
        top_idx = np.argsort(-fusion)
    else:
        top_idx = np.argpartition(-fusion, top_k)[:top_k]
        top_idx = top_idx[np.argsort(-fusion[top_idx])]

    sequences = []
    for rank_fused, i in enumerate(top_idx[:top_k], start=1):
        sequences.append({
            "cluster_ids": [int(c) for c in cand_sets[i]],
            "length": k,
            # NLL convention: avg_nll_coherence is mean per-token NLL under the
            # coherence model (low = coherent);
            # avg_nll_availability is the alien score under the availability
            # model (high = alien = less available).
            "avg_nll_coherence": float(-coh_scores[i]),
            "avg_nll_availability": float(avail_scores[i]),
            "rank_coherence": int(rank_coh[i]),
            "rank_availability": int(rank_avail[i]),
            "rank_fused": int(rank_fused),
            "fusion_score": float(fusion[i]),
            "z_coherence": float(z_coh[i]),
            "z_availability": float(z_avail[i]),
        })

    return {
        "config": {
            "beta": float(beta),
            "top_k": int(top_k),
            "k": int(k),
            "fusion_formula": "(1 - beta) * z_coh + beta * z_avail",
            **config_extra,
        },
        "top_k_sequences": sequences,
    }


def write_pool_scores(
    output_dir: Path,
    cand_sets: list[tuple[int, ...]],
    coh_scores: np.ndarray,
    avail_scores: np.ndarray,
    z_coh: np.ndarray,
    z_avail: np.ndarray,
    config_extra: dict,
    max_full_dump: int = 500_000,
) -> None:
    """Save the candidate pool with raw and z-scored scores.

    For pools up to ``max_full_dump`` entries we emit ``pool_scores.json`` (the
    full per-set list with raw + z-scores). For larger pools (e.g. exhaustive
    enumeration of C(n, 3) ~ several million) we instead emit a compact
    ``pool_summary.json`` with score-distribution stats only -- the full per-
    set details would balloon the file to hundreds of MB while being trivially
    re-derivable by re-running the (cheap) scoring.
    """
    if len(cand_sets) <= max_full_dump:
        payload = {
            "config": config_extra,
            "n_candidates": len(cand_sets),
            "candidates": [
                {
                    "cluster_ids": [int(c) for c in s],
                    "coh_score": float(coh_scores[i]),
                    "avail_score": float(avail_scores[i]),
                    "z_coh": float(z_coh[i]),
                    "z_avail": float(z_avail[i]),
                }
                for i, s in enumerate(cand_sets)
            ],
        }
        path = output_dir / "pool_scores.json"
        with open(path, "w") as f:
            json.dump(payload, f)
        sys.stderr.write(f"  Saved {path} ({len(cand_sets):,} candidates)\n")
    else:
        def _stats(x: np.ndarray) -> dict:
            return {
                "mean": float(x.mean()), "std": float(x.std()),
                "min": float(x.min()), "max": float(x.max()),
                "median": float(np.median(x)),
                "p1": float(np.quantile(x, 0.01)),
                "p99": float(np.quantile(x, 0.99)),
            }
        payload = {
            "config": config_extra,
            "n_candidates": len(cand_sets),
            "note": (
                f"Pool too large ({len(cand_sets):,} > {max_full_dump:,}) for full "
                f"per-set dump; re-run sample_beta_sweep.py to recover full scores."
            ),
            "coh_score_stats": _stats(coh_scores),
            "avail_score_stats": _stats(avail_scores),
        }
        path = output_dir / "pool_summary.json"
        with open(path, "w") as f:
            json.dump(payload, f, indent=2)
        sys.stderr.write(
            f"  Saved {path} (compact summary; full dump skipped for "
            f"{len(cand_sets):,} candidates)\n"
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate a shared candidate pool and emit per-beta top-K results JSONs.",
    )
    parser.add_argument("--coherence-model", required=True, type=Path)
    parser.add_argument("--availability-model", required=True, type=Path)
    parser.add_argument("--token-mapping", required=True, type=Path)
    parser.add_argument(
        "--output-dir", type=Path, default=Path("experiments/beta_sweep"),
    )
    parser.add_argument(
        "--ks", type=int, nargs="+", default=[3, 4],
        help="Atoms per set; produces one analysis per value (default: 3 4).",
    )
    parser.add_argument(
        "--exhaustive-ks", type=int, nargs="*", default=(),
        help=(
            "List of k values for which to enumerate the entire candidate space "
            "C(n_clusters, k) instead of sampling from the coherence LM. "
            "Useful when the space is small enough to score on GPU (k=3 with a "
            "few-hundred-cluster vocab is ~3-5M sets, ~minutes to score). "
            "ks not listed here use --n-gen sampled draws as before. Default: empty."
        ),
    )
    parser.add_argument("--n-gen", type=int, default=100_000)
    parser.add_argument("--top-k", type=int, default=300, help="Top-K per beta (default: 300).")
    parser.add_argument(
        "--betas", type=float, nargs="+", default=None,
        help="Explicit beta values. Default: 0.0, 0.1, ..., 1.0.",
    )
    parser.add_argument(
        "--gen-top-k", type=int, default=0,
        help=(
            "Per-step top-k truncation for the coherence-LM sampler "
            "(default: 0 = no truncation, sample from the full softmax "
            "over all clusters at every step). The fusion ranking handles "
            "candidate selection -- the pool is intentionally unbiased."
        ),
    )
    parser.add_argument("--gen-temperature", type=float, default=1.0)
    parser.add_argument("--tt-top-k", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    if args.seed is not None:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)

    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    sys.stderr.write(f"Device: {device}\n")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.betas is None:
        betas = np.round(np.arange(0.0, 1.0 + 1e-9, 0.1), 1)
    else:
        betas = np.array(args.betas, dtype=float)
    sys.stderr.write(f"Betas: {betas.tolist()}\n")

    # ------------------------------------------------------------------
    # Load models + token mapping
    # ------------------------------------------------------------------
    sys.stderr.write("\nLoading coherence model...\n")
    coherence_model, coh_tm = load_model(str(args.coherence_model))
    coherence_model.to(device).eval()

    sys.stderr.write("Loading two-tower availability model...\n")
    avail_model, avail_tm = load_availability_model(str(args.availability_model))
    avail_model.to(device).eval()

    author_embeddings, _ = load_author_embeddings(
        str(args.availability_model),
        device="cpu",
    )
    author_embeddings = author_embeddings.to(device)

    dataset_tm = TokenMapping.load(args.token_mapping)
    model_tm = coh_tm

    if (
        coh_tm.n_clusters != dataset_tm.n_clusters
        or avail_tm.n_clusters != dataset_tm.n_clusters
    ):
        parser.error(
            f"Cluster vocab mismatch: coherence={coh_tm.n_clusters}, "
            f"availability={avail_tm.n_clusters}, dataset={dataset_tm.n_clusters}"
        )
    if (
        coh_tm.bos_token_id != avail_tm.bos_token_id
        or coh_tm.eos_token_id != avail_tm.eos_token_id
    ):
        parser.error(
            f"Coherence and availability models use different special tokens: "
            f"coh BOS/EOS = ({coh_tm.bos_token_id}, {coh_tm.eos_token_id}), "
            f"avail BOS/EOS = ({avail_tm.bos_token_id}, {avail_tm.eos_token_id})."
        )

    verify_availability_permutation_invariance(
        avail_model,
        author_embeddings,
        model_tm,
        device=device,
        batch_size=args.batch_size,
        tt_top_k=args.tt_top_k,
        sample_set=(0, 1, 2),
    )

    exhaustive_ks = set(args.exhaustive_ks or ())

    config_extra_base = {
        "coherence_model": str(args.coherence_model),
        "availability_model": str(args.availability_model),
        "availability_model_type": "two_tower",
        "token_mapping": str(args.token_mapping),
        "n_gen": int(args.n_gen),
        "gen_top_k": int(args.gen_top_k),
        "gen_temperature": float(args.gen_temperature),
        "tt_top_k": int(args.tt_top_k),
        "exhaustive_ks": sorted(exhaustive_ks),
        "seed": args.seed,
    }

    # ------------------------------------------------------------------
    # Per-k: build candidate pool, score, and emit per-beta results JSONs
    # ------------------------------------------------------------------
    overall_summary: dict[str, dict] = {"config": config_extra_base, "ks": {}}

    for k in args.ks:
        sys.stderr.write(f"\n========== k = {k} ==========\n")
        k_output_dir = args.output_dir / f"k{k}"
        k_output_dir.mkdir(parents=True, exist_ok=True)

        if k in exhaustive_ks:
            n = model_tm.n_clusters
            n_combos = math.comb(n, k)
            sys.stderr.write(
                f"\n[exhaustive] Enumerating C({n}, {k}) = {n_combos:,} sets...\n"
            )
            t0 = time.time()
            cand_sets = list(combinations(range(n), k))
            sys.stderr.write(
                f"  Built pool of {len(cand_sets):,} unique sets "
                f"({time.time() - t0:.1f}s)\n"
            )
        else:
            cand_sets = generate_pool(
                coherence_model, model_tm,
                n_gen=args.n_gen, k=k,
                gen_top_k=args.gen_top_k, gen_temperature=args.gen_temperature,
                device=device,
            )
        if not cand_sets:
            sys.stderr.write(
                f"No unique sets generated for k={k}. Increase --n-gen "
                f"(or add k={k} to --exhaustive-ks).\n"
            )
            continue

        sys.stderr.write(
            f"\nScoring coherence (set-level, marginalized over {k}! perms)...\n"
        )
        t0 = time.time()
        coh_scores = score_sets_coherence(
            cand_sets, coherence_model, model_tm,
            device=device, batch_size=args.batch_size,
        )
        sys.stderr.write(f"    {time.time() - t0:.1f}s\n")

        sys.stderr.write("Scoring availability (set-level, canonical order)...\n")
        t0 = time.time()
        avail_scores = score_sets_alien(
            cand_sets, avail_model, model_tm,
            author_embeddings=author_embeddings,
            device=device,
            batch_size=args.batch_size,
            top_k_authors=args.tt_top_k,
        )
        sys.stderr.write(f"    {time.time() - t0:.1f}s\n")

        z_coh = _zscore(coh_scores)
        z_avail = _zscore(avail_scores)

        config_extra = {
            **config_extra_base,
            "k": int(k),
            "n_candidate_sets": len(cand_sets),
        }

        write_pool_scores(
            k_output_dir,
            cand_sets, coh_scores, avail_scores, z_coh, z_avail,
            config_extra=config_extra,
        )

        summary: list[dict] = []
        for beta in betas:
            beta_dir = k_output_dir / f"beta_{beta:.1f}"
            beta_dir.mkdir(parents=True, exist_ok=True)

            payload = build_results_payload(
                cand_sets, coh_scores, avail_scores, z_coh, z_avail,
                beta=float(beta), top_k=args.top_k, k=k,
                config_extra=config_extra,
            )
            out_path = beta_dir / "results.json"
            with open(out_path, "w") as f:
                json.dump(payload, f, indent=2)
            sys.stderr.write(
                f"  beta={beta:.1f}: top-{args.top_k} -> {out_path}\n"
            )
            summary.append({
                "beta": float(beta),
                "results_path": str(out_path),
                "n_selected": len(payload["top_k_sequences"]),
                "mean_coh_z_top": float(np.mean([
                    s["z_coherence"] for s in payload["top_k_sequences"]
                ])),
                "mean_avail_z_top": float(np.mean([
                    s["z_availability"] for s in payload["top_k_sequences"]
                ])),
            })

        summary_path = k_output_dir / "beta_sweep_summary.json"
        with open(summary_path, "w") as f:
            json.dump({"config": config_extra, "betas": summary}, f, indent=2)
        sys.stderr.write(f"  Saved {summary_path}\n")

        overall_summary["ks"][str(k)] = {
            "k": int(k),
            "output_dir": str(k_output_dir),
            "n_candidate_sets": len(cand_sets),
            "summary_path": str(summary_path),
        }

    overall_path = args.output_dir / "beta_sweep_summary.json"
    with open(overall_path, "w") as f:
        json.dump(overall_summary, f, indent=2)
    sys.stderr.write(f"\nSaved {overall_path}\n")
    sys.stderr.write("Done.\n")


if __name__ == "__main__":
    main()
