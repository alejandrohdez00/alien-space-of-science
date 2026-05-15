"""Generate coherence training data from clustered idea atoms.

Usage:
    python alien.py make-datasets \
        --clusters clusters.json \
        --vocab data/vocab.json \
        --length-balanced \
        --split 0.9 \
        --output-dir datasets/coherence
"""

import argparse
import json
import random
from pathlib import Path

from generation.datasets.coherence import (
    generate_coherence_dataset,
    generate_coherence_dataset_length_balanced,
    save_coherence_dataset,
)
from generation.datasets.token_mapping import (
    create_cluster_to_token_mapping,
    load_token_mapping,
    save_token_mapping,
)


def split_samples(
    samples: list[dict],
    split_ratio: float,
    seed: int,
) -> tuple[list[dict], list[dict]]:
    """Split samples into train and validation subsets."""
    rng = random.Random(seed)
    shuffled = samples.copy()
    rng.shuffle(shuffled)
    split_idx = int(len(shuffled) * split_ratio)
    return shuffled[:split_idx], shuffled[split_idx:]


def save_jsonl(samples: list[dict], path: Path) -> None:
    """Save samples to JSONL."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for sample in samples:
            f.write(json.dumps(sample) + "\n")


def _add_args(parser: argparse.ArgumentParser) -> None:
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
        help="Path to vocab.json",
    )
    parser.add_argument(
        "--db",
        type=str,
        default="data/llm-papers.db",
        help=(
            "SQLite metadata DB; required only when --max-venue-year is set "
            "(default: data/llm-papers.db)"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="datasets/coherence",
        help="Output directory",
    )
    parser.add_argument(
        "--token-mapping",
        type=str,
        default=None,
        help="Existing token_mapping.json to reuse",
    )

    parser.add_argument(
        "--n-permutations",
        type=int,
        default=10,
        help="Number of random permutations per paper in standard mode",
    )
    parser.add_argument(
        "--length-balanced",
        action="store_true",
        help="Generate balanced samples across atom-set lengths",
    )
    parser.add_argument(
        "--balance-target",
        type=int,
        default=200000,
        help="Target samples per length for balanced mode",
    )
    parser.add_argument(
        "--max-samples-per-paper",
        type=int,
        default=200,
        help="Maximum samples per paper per length in balanced mode",
    )
    parser.add_argument(
        "--balance-max-length",
        type=int,
        default=5,
        help="Only balance lengths up to this value",
    )
    parser.add_argument(
        "--min-atoms",
        type=int,
        default=2,
        help="Minimum atoms required to include a paper",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--include-debug-tokens",
        action="store_true",
        help="Include human-readable token debugging fields",
    )
    parser.add_argument(
        "--split",
        type=float,
        default=None,
        help="Train split ratio, e.g. 0.9. If unset, writes one coherence.jsonl.",
    )
    parser.add_argument(
        "--max-venue-year",
        type=int,
        default=None,
        help="Drop papers with papers.venue_year greater than this cutoff",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate coherence datasets from clusters.json")
    _add_args(parser)
    args = parser.parse_args()

    if args.split is not None and not (0.0 < args.split < 1.0):
        parser.error("--split must be between 0 and 1")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'=' * 60}")
    print("Coherence Dataset Generation")
    print(f"{'=' * 60}")
    print(f"Clusters: {args.clusters}")
    print(f"Vocab:    {args.vocab}")
    print(f"Output:   {output_dir}")
    if args.max_venue_year is not None:
        print(f"Cutoff:   papers.venue_year <= {args.max_venue_year}")
    print(f"{'=' * 60}\n")

    with open(args.clusters, "r", encoding="utf-8") as f:
        clusters_data = json.load(f)

    mapping_path = output_dir / "token_mapping.json"
    if args.token_mapping:
        print(f"Loading token mapping from {args.token_mapping}...")
        token_mapping = load_token_mapping(args.token_mapping)
        if not mapping_path.exists():
            save_token_mapping(token_mapping, str(mapping_path))
    else:
        print("Creating token mapping...")
        token_mapping = create_cluster_to_token_mapping(clusters_data, args.vocab)
        save_token_mapping(token_mapping, str(mapping_path))
    print(f"  Token mapping: {mapping_path}")

    if args.length_balanced:
        print("\n[Mode] Length-balanced")
        dataset = generate_coherence_dataset_length_balanced(
            clusters_path=args.clusters,
            token_mapping=token_mapping,
            min_length=args.min_atoms,
            balance_target=args.balance_target,
            max_samples_per_paper=args.max_samples_per_paper,
            balance_max_length=args.balance_max_length,
            include_debug_tokens=args.include_debug_tokens,
            random_seed=args.seed,
            db_path=args.db if args.max_venue_year is not None else None,
            max_venue_year=args.max_venue_year,
        )
    else:
        print("\n[Mode] Standard permutations")
        dataset = generate_coherence_dataset(
            clusters_path=args.clusters,
            token_mapping=token_mapping,
            n_permutations=args.n_permutations,
            min_atoms=args.min_atoms,
            include_debug_tokens=args.include_debug_tokens,
            random_seed=args.seed,
            db_path=args.db if args.max_venue_year is not None else None,
            max_venue_year=args.max_venue_year,
        )

    meta = dataset["metadata"]
    if args.split:
        train_samples, val_samples = split_samples(dataset["samples"], args.split, args.seed)
        train_path = output_dir / "coherence_train.jsonl"
        val_path = output_dir / "coherence_val.jsonl"
        save_jsonl(train_samples, train_path)
        save_jsonl(val_samples, val_path)
    else:
        dataset_path = output_dir / "coherence.jsonl"
        save_coherence_dataset(dataset, str(dataset_path))

    metadata = {
        "clusters_path": args.clusters,
        "vocab_path": args.vocab,
        "token_mapping": str(mapping_path),
        "seed": args.seed,
        "min_atoms": args.min_atoms,
        "max_venue_year": args.max_venue_year,
        "coherence": meta,
    }
    metadata_path = output_dir / "metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print("\nResults:")
    print(f"  Total samples:   {meta['total_samples']}")
    print(f"  Papers included: {meta['papers_included']}")
    print(f"  Papers skipped:  {meta['papers_skipped']}")
    if args.length_balanced:
        print(f"  Samples by length: {meta['samples_by_length']}")
    else:
        print(f"  Avg atoms/paper: {meta['avg_atoms_per_paper']}")
    if args.split:
        print(f"  Train samples:   {len(train_samples)}")
        print(f"  Val samples:     {len(val_samples)}")
        print(f"  Train path:      {train_path}")
        print(f"  Val path:        {val_path}")
    else:
        print(f"  Dataset path:    {dataset_path}")
    print(f"  Metadata:        {metadata_path}\n")


if __name__ == "__main__":
    main()
