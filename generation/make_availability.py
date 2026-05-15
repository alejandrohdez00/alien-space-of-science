"""Generate the availability dataset for two-tower training.

Produces one row per (author, subset) pair. Each row carries:
  - query_ids:  the positive atom subset the query tower sees
  - author_ids: the author's complement pool (pool - subset) the author tower sees

Pool handling is deterministic: enumerate small pools and sample large pools
with length weighted by C(n, L). Dedup is per-author only; two authors sharing
a subset are kept because their complements differ.

Usage:
    python alien.py make-availability \
        --clusters clusters.json \
        --vocab data/vocab.json \
        --db data/llm-papers.db \
        --token-mapping datasets/coherence/token_mapping.json \
        --min-length 2 --max-length 4 \
        --split 0.9 \
        --output-dir datasets/availability
"""

import argparse
import json
import random
from pathlib import Path

from generation.datasets.availability import (
    build_atoms_sidecar,
    generate_availability_dataset,
    save_availability_dataset,
)
from generation.datasets.token_mapping import (
    create_cluster_to_token_mapping,
    load_token_mapping,
    save_token_mapping,
)


def _write_jsonl(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def _split_samples(
    samples: list[dict],
    split_ratio: float,
    seed: int,
) -> tuple[list[dict], list[dict]]:
    rng = random.Random(seed)
    shuffled = samples.copy()
    rng.shuffle(shuffled)
    cut = int(len(shuffled) * split_ratio)
    return shuffled[:cut], shuffled[cut:]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate the two-tower availability dataset"
    )

    parser.add_argument(
        "--clusters",
        type=str,
        required=True,
        help="Path to clusters.json",
    )
    parser.add_argument(
        "--vocab",
        type=str,
        default="vocab.json",
        help="Path to vocab.json (default: vocab.json)",
    )
    parser.add_argument(
        "--db",
        type=str,
        default="data/llm-papers.db",
        help="Path to SQLite database (default: data/llm-papers.db)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="datasets/availability",
        help="Output directory",
    )
    parser.add_argument(
        "--token-mapping",
        type=str,
        default=None,
        help="Existing token_mapping.json to reuse",
    )

    parser.add_argument(
        "--pool-cap",
        type=int,
        default=30,
        help="Authors with pool <= cap are fully enumerated (default: 30)",
    )
    parser.add_argument(
        "--per-author-budget",
        type=int,
        default=20000,
        help="Per-author sample budget for pools above the cap (default: 20000)",
    )
    parser.add_argument(
        "--min-length",
        type=int,
        default=2,
        help="Minimum subset length (default: 2)",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=4,
        help="Maximum subset length (default: 4)",
    )
    parser.add_argument(
        "--min-atoms",
        type=int,
        default=2,
        help="Minimum super-atoms required to include an author (default: 2)",
    )
    parser.add_argument(
        "--max-venue-year",
        type=int,
        default=None,
        help="Inclusive temporal cutoff on papers.venue_year (default: no cutoff)",
    )

    parser.add_argument(
        "--split",
        type=float,
        default=None,
        help=(
            "Train/val split ratio, e.g. 0.9. If unset, writes one "
            "availability.jsonl file."
        ),
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    parser.add_argument(
        "--include-debug-tokens",
        action="store_true",
        help="Include tokens_debug fields for query and author sides",
    )
    parser.add_argument(
        "--no-atoms-sidecar",
        action="store_true",
        help="Skip writing availability_atoms.json",
    )

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print("Availability Dataset (two-tower)")
    print(f"{'='*60}")
    print(f"Clusters: {args.clusters}")
    print(f"Database: {args.db}")
    print(f"Output:   {args.output_dir}")
    print(f"Pool cap: {args.pool_cap}   Per-author budget: {args.per_author_budget}")
    print(f"Lengths:  {args.min_length}..{args.max_length}   min_atoms={args.min_atoms}")
    if args.max_venue_year:
        print(f"Temporal cutoff: <= {args.max_venue_year}")
    if args.split:
        print(
            f"Split:    {args.split:.2f} train / {1 - args.split:.2f} val "
            "(random sample-level)"
        )
    else:
        print("Split:    none (single availability.jsonl)")
    print(f"{'='*60}\n")

    with open(args.clusters, "r", encoding="utf-8") as f:
        clusters_data = json.load(f)

    if args.token_mapping:
        print(f"Loading existing token mapping from {args.token_mapping}...")
        token_mapping = load_token_mapping(args.token_mapping)
        print(f"  Loaded mapping for {token_mapping['n_clusters']} clusters")
        mapping_path = output_dir / "token_mapping.json"
        if not mapping_path.exists():
            save_token_mapping(token_mapping, str(mapping_path))
            print(f"  Copied to {mapping_path}")
    else:
        print("Creating token mapping...")
        token_mapping = create_cluster_to_token_mapping(clusters_data, args.vocab)
        mapping_path = output_dir / "token_mapping.json"
        save_token_mapping(token_mapping, str(mapping_path))
        print(f"  Created mapping for {token_mapping['n_clusters']} clusters")
        print(f"  Saved to {mapping_path}")

    print("\nGenerating samples (this may take several minutes)...")
    dataset = generate_availability_dataset(
        clusters_path=args.clusters,
        db_path=args.db,
        token_mapping=token_mapping,
        pool_cap=args.pool_cap,
        per_author_budget=args.per_author_budget,
        min_length=args.min_length,
        max_length=args.max_length,
        min_atoms=args.min_atoms,
        max_venue_year=args.max_venue_year,
        include_debug_tokens=args.include_debug_tokens,
        random_seed=args.seed,
    )

    meta = dataset["metadata"]

    if args.split is not None:
        train, val = _split_samples(dataset["samples"], args.split, args.seed)
        train_path = output_dir / "availability_train.jsonl"
        val_path = output_dir / "availability_val.jsonl"
        _write_jsonl(train, train_path)
        _write_jsonl(val, val_path)
        meta["split_ratio"] = args.split
        meta["train_samples_count"] = len(train)
        meta["val_samples_count"] = len(val)
    else:
        availability_path = output_dir / "availability.jsonl"
        save_availability_dataset(dataset, str(availability_path))

    if not args.no_atoms_sidecar:
        atoms_sidecar = build_atoms_sidecar(clusters_data)
        atoms_path = output_dir / "availability_atoms.json"
        with open(atoms_path, "w", encoding="utf-8") as f:
            json.dump(atoms_sidecar, f, indent=2)

    metadata_path = output_dir / "metadata.json"
    combined_metadata = {
        "clusters_path": args.clusters,
        "vocab_path": args.vocab,
        "db_path": args.db,
        "seed": args.seed,
        "availability": meta,
    }
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(combined_metadata, f, indent=2)

    comp = meta["complement_size_stats"]

    print(f"\n{'='*60}")
    print("Results")
    print(f"{'='*60}")
    print(f"  Total samples:       {meta['total_samples']}")
    print(f"  Duplicates rejected: {meta['duplicates_rejected']} (per-author only)")
    print(
        f"  Authors included:    {meta['authors_included']} "
        f"(enumerated: {meta['authors_enumerated']}, "
        f"sampled: {meta['authors_sampled']})"
    )
    print(
        f"  Authors skipped:     {meta['authors_skipped']} "
        f"(< min_atoms: {meta['authors_skipped_atoms']})"
    )
    print(f"  Avg atoms/author:    {meta['avg_atoms_per_author']}")
    print(f"  Avg papers/author:   {meta['avg_papers_per_author']}")
    print(f"  Samples by length:   {meta['samples_by_length']}")
    print(f"  Authors by length:   {meta['authors_by_length']}")
    print(f"  Complement size:     min={comp['min']} max={comp['max']} avg={comp['avg']}")
    if args.split is not None:
        print(f"  Train samples:       {meta['train_samples_count']}")
        print(f"  Val samples:         {meta['val_samples_count']}")
        print(f"  Train path:          {output_dir / 'availability_train.jsonl'}")
        print(f"  Val path:            {output_dir / 'availability_val.jsonl'}")
    else:
        print(f"  Output path:         {output_dir / 'availability.jsonl'}")
    print(f"  Metadata:            {metadata_path}")
    print(f"  Token mapping:       {mapping_path}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
