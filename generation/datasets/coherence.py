"""Coherence dataset: Paper-level super-atom co-occurrence.

Generates sequences of super-atoms that co-occur within the same paper.
Each paper produces multiple sequences via random permutation.
"""

import json
import random
from typing import Optional

from .token_mapping import encode_cluster_sequence, get_debug_tokens


def _load_paper_year_lookup(db_path: str) -> dict[str, Optional[int]]:
    """Map paper_id -> venue_year (None if missing) from the SQLite DB.

    DBLP-sourced papers are stored in the DB with slash-separated keys
    (`conf/nips/SubramaniBC19`) but appear in clusters.json with
    double-underscore keys (`conf__nips__SubramaniBC19`) because slashes
    break the `papers/{paper_id}/` filesystem layout. Register each
    slash-keyed row under both forms so the cutoff filter resolves either
    spelling.
    """
    from crawlers.db.queries import get_all_papers
    lookup: dict[str, Optional[int]] = {}
    for p in get_all_papers(db_path):
        pid = p["paper_id"]
        year = p.get("venue_year")
        lookup[pid] = year
        if "/" in pid:
            lookup[pid.replace("/", "__")] = year
    return lookup


def _filter_papers_by_venue_year(
    paper_super_atoms: dict[str, list[int]],
    paper_year_lookup: dict[str, Optional[int]],
    max_venue_year: int,
) -> dict[str, list[int]]:
    """Drop papers with venue_year > max_venue_year. Papers with no year are kept."""
    return {
        pid: atoms
        for pid, atoms in paper_super_atoms.items()
        if paper_year_lookup.get(pid) is None or paper_year_lookup[pid] <= max_venue_year
    }


def compute_vocab_partition(
    paper_super_atoms_unfiltered: dict[str, list[int]],
    paper_year_lookup: dict[str, Optional[int]],
    max_venue_year: int,
) -> dict:
    """Partition cluster IDs into pre-cutoff vocab vs post-cutoff-only.

    Cluster IDs with at least one paper at venue_year <= cutoff form the
    model's effective training vocabulary. Cluster IDs whose papers all sit
    strictly after the cutoff are dropped from training and listed separately
    so post-cutoff evaluation can scope predictions honestly.

    Papers missing venue_year metadata are treated as pre-cutoff (mirrors the
    keep-on-missing semantics in build_author_super_atoms).
    """
    pre: set[int] = set()
    all_clusters: set[int] = set()
    for paper_id, atoms in paper_super_atoms_unfiltered.items():
        all_clusters.update(atoms)
        year = paper_year_lookup.get(paper_id)
        if year is None or year <= max_venue_year:
            pre.update(atoms)
    return {
        "max_venue_year": max_venue_year,
        "total_vocab_size": len(all_clusters),
        "effective_vocab_size": len(pre),
        "pre_cutoff_clusters": sorted(pre),
        "post_cutoff_only_clusters": sorted(all_clusters - pre),
    }


def extract_paper_super_atoms(clusters_data: dict) -> dict[str, list[int]]:
    """
    Extract super-atom cluster IDs for each paper.

    Filters out noise atoms (cluster == -1) and returns unique cluster IDs per paper.

    Args:
        clusters_data: Parsed clusters.json content

    Returns:
        {paper_id: [cluster_id_0, cluster_id_5, ...], ...}
        Cluster IDs are sorted for consistency.
    """
    paper_clusters: dict[str, set[int]] = {}

    for assignment in clusters_data.get("atom_assignments", []):
        cluster_id = assignment.get("cluster")
        paper_id = assignment.get("paper_id")

        # Skip noise atoms
        if cluster_id == -1:
            continue

        if paper_id not in paper_clusters:
            paper_clusters[paper_id] = set()

        paper_clusters[paper_id].add(cluster_id)

    # Convert to sorted lists
    return {
        paper_id: sorted(clusters)
        for paper_id, clusters in paper_clusters.items()
    }


def generate_coherence_samples(
    paper_super_atoms: dict[str, list[int]],
    token_mapping: dict,
    n_permutations: int = 10,
    min_atoms: int = 2,
    include_debug_tokens: bool = False,
    random_seed: Optional[int] = None,
) -> list[dict]:
    """
    Generate n random permutations per paper.

    Args:
        paper_super_atoms: {paper_id: [cluster_ids]} from extract_paper_super_atoms()
        token_mapping: Mapping from create_cluster_to_token_mapping()
        n_permutations: Number of permutations per paper
        min_atoms: Minimum number of super-atoms to include a paper
        include_debug_tokens: If True, add tokens_debug field
        random_seed: Random seed for reproducibility

    Returns:
        List of sample dicts with:
        - paper_id: str
        - perm_idx: int
        - input_ids: list[int] (token IDs with BOS/EOS)
        - tokens_debug: list[str] (optional, human-readable)
    """
    if random_seed is not None:
        random.seed(random_seed)

    samples = []

    for paper_id, cluster_ids in paper_super_atoms.items():
        # Skip papers with too few super-atoms
        if len(cluster_ids) < min_atoms:
            continue

        for perm_idx in range(n_permutations):
            # Random permutation
            permuted = cluster_ids.copy()
            random.shuffle(permuted)

            # Encode to token IDs
            input_ids = encode_cluster_sequence(permuted, token_mapping)

            sample = {
                "paper_id": paper_id,
                "perm_idx": perm_idx,
                "input_ids": input_ids,
            }

            if include_debug_tokens:
                sample["tokens_debug"] = get_debug_tokens(input_ids, token_mapping)

            samples.append(sample)

    return samples


def generate_coherence_dataset(
    clusters_path: str,
    token_mapping: dict,
    n_permutations: int = 10,
    min_atoms: int = 2,
    include_debug_tokens: bool = False,
    random_seed: int = 42,
    db_path: Optional[str] = None,
    max_venue_year: Optional[int] = None,
) -> dict:
    """
    Main entry point for coherence dataset generation.

    Args:
        clusters_path: Path to clusters.json
        token_mapping: Mapping from create_cluster_to_token_mapping()
        n_permutations: Number of permutations per paper
        min_atoms: Minimum super-atoms to include a paper
        include_debug_tokens: If True, add tokens_debug field
        random_seed: Random seed for reproducibility
        db_path: SQLite DB path. Required when max_venue_year is set.
        max_venue_year: Inclusive temporal cutoff on papers.venue_year. Papers
            newer than the cutoff are dropped before sampling. Papers missing
            venue_year metadata are kept.

    Returns:
        {
            "samples": list[dict],
            "metadata": {
                "total_samples": int,
                "total_papers": int,
                "papers_included": int,
                "papers_skipped": int,
                "avg_atoms_per_paper": float,
                "n_permutations": int,
                "min_atoms": int,
                "random_seed": int,
                "max_venue_year": int | None,
                "papers_dropped_by_cutoff": int,
                "vocab_partition": dict | None,
            }
        }
    """
    if max_venue_year is not None and db_path is None:
        raise ValueError("db_path is required when max_venue_year is set")

    with open(clusters_path, "r", encoding="utf-8") as f:
        clusters_data = json.load(f)

    # Extract paper -> super-atoms mapping (unfiltered)
    paper_super_atoms = extract_paper_super_atoms(clusters_data)

    vocab_partition: Optional[dict] = None
    papers_dropped_by_cutoff = 0
    if max_venue_year is not None:
        paper_year_lookup = _load_paper_year_lookup(db_path)
        vocab_partition = compute_vocab_partition(
            paper_super_atoms, paper_year_lookup, max_venue_year
        )
        before = len(paper_super_atoms)
        paper_super_atoms = _filter_papers_by_venue_year(
            paper_super_atoms, paper_year_lookup, max_venue_year
        )
        papers_dropped_by_cutoff = before - len(paper_super_atoms)

    total_papers = len(paper_super_atoms)
    papers_with_enough = sum(
        1 for atoms in paper_super_atoms.values() if len(atoms) >= min_atoms
    )
    papers_skipped = total_papers - papers_with_enough

    # Calculate average atoms per included paper
    atoms_counts = [
        len(atoms) for atoms in paper_super_atoms.values() if len(atoms) >= min_atoms
    ]
    avg_atoms = sum(atoms_counts) / len(atoms_counts) if atoms_counts else 0.0

    # Generate samples
    samples = generate_coherence_samples(
        paper_super_atoms=paper_super_atoms,
        token_mapping=token_mapping,
        n_permutations=n_permutations,
        min_atoms=min_atoms,
        include_debug_tokens=include_debug_tokens,
        random_seed=random_seed,
    )

    return {
        "samples": samples,
        "metadata": {
            "total_samples": len(samples),
            "total_papers": total_papers,
            "papers_included": papers_with_enough,
            "papers_skipped": papers_skipped,
            "avg_atoms_per_paper": round(avg_atoms, 2),
            "n_permutations": n_permutations,
            "min_atoms": min_atoms,
            "random_seed": random_seed,
            "max_venue_year": max_venue_year,
            "papers_dropped_by_cutoff": papers_dropped_by_cutoff,
            "vocab_partition": vocab_partition,
        },
    }


def save_coherence_dataset(
    dataset: dict,
    output_path: str,
) -> None:
    """
    Save coherence dataset to JSONL file.

    Args:
        dataset: Output from generate_coherence_dataset()
        output_path: Path to output .jsonl file
    """
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        for sample in dataset["samples"]:
            f.write(json.dumps(sample) + "\n")


def generate_coherence_samples_length_balanced(
    paper_super_atoms: dict[str, list[int]],
    token_mapping: dict,
    min_length: int = 2,
    balance_target: int = 200000,
    max_samples_per_paper: int = 200,
    balance_max_length: int = 5,
    include_debug_tokens: bool = False,
    random_seed: Optional[int] = None,
) -> tuple[list[dict], dict[int, int], dict[int, int]]:
    """
    Generate length-balanced coherence samples with dynamic sampling rates.

    Uses target-based balancing: calculates how many samples each paper should
    contribute at each length to achieve approximately equal samples per length.

    Args:
        paper_super_atoms: {paper_id: [cluster_ids]} from extract_paper_super_atoms()
        token_mapping: Mapping from create_cluster_to_token_mapping()
        min_length: Minimum sequence length to generate (default: 2)
        balance_target: Target number of samples per length (default: 100000)
        max_samples_per_paper: Cap on samples per paper per length (default: 200)
        balance_max_length: Only balance up to this length; longer uses base rate (default: 5)
        include_debug_tokens: If True, add tokens_debug field
        random_seed: Random seed for reproducibility

    Returns:
        Tuple of:
        - List of sample dicts
        - papers_by_length: {length: count of papers that can contribute}
        - samples_per_paper_by_length: {length: samples per paper at that length}
    """
    import math

    if random_seed is not None:
        random.seed(random_seed)

    # First pass: count papers available at each length
    max_observed_length = max(len(atoms) for atoms in paper_super_atoms.values())
    papers_by_length: dict[int, int] = {}

    for length in range(min_length, max_observed_length + 1):
        papers_by_length[length] = sum(
            1 for atoms in paper_super_atoms.values() if len(atoms) >= length
        )

    # Calculate samples per paper for each length
    samples_per_paper_by_length: dict[int, int] = {}

    for length in range(min_length, max_observed_length + 1):
        papers_at_length = papers_by_length.get(length, 0)
        if papers_at_length == 0:
            samples_per_paper_by_length[length] = 0
            continue

        if length <= balance_max_length:
            # Balance this length: aim for balance_target samples
            needed = math.ceil(balance_target / papers_at_length)
            samples_per_paper_by_length[length] = min(needed, max_samples_per_paper)
        else:
            # Beyond max balance length: use base rate (same as min_length rate)
            base_rate = samples_per_paper_by_length.get(min_length, 10)
            samples_per_paper_by_length[length] = base_rate

    # Second pass: generate samples
    samples = []

    for paper_id, cluster_ids in paper_super_atoms.items():
        n_atoms = len(cluster_ids)

        # Skip papers that can't reach minimum length
        if n_atoms < min_length:
            continue

        # Generate samples for each length from min_length to n_atoms
        for target_len in range(min_length, n_atoms + 1):
            n_samples = samples_per_paper_by_length.get(target_len, 0)

            for sample_idx in range(n_samples):
                # Sample target_len atoms without replacement, then shuffle
                sampled = random.sample(cluster_ids, target_len)
                random.shuffle(sampled)

                # Encode to token IDs
                input_ids = encode_cluster_sequence(sampled, token_mapping)

                sample = {
                    "paper_id": paper_id,
                    "seq_len": target_len,
                    "sample_idx": sample_idx,
                    "input_ids": input_ids,
                }

                if include_debug_tokens:
                    sample["tokens_debug"] = get_debug_tokens(input_ids, token_mapping)

                samples.append(sample)

    return samples, papers_by_length, samples_per_paper_by_length


def generate_coherence_dataset_length_balanced(
    clusters_path: str,
    token_mapping: dict,
    min_length: int = 2,
    balance_target: int = 200000,
    max_samples_per_paper: int = 200,
    balance_max_length: int = 5,
    include_debug_tokens: bool = False,
    random_seed: int = 42,
    db_path: Optional[str] = None,
    max_venue_year: Optional[int] = None,
) -> dict:
    """
    Main entry point for length-balanced coherence dataset generation.

    Uses dynamic sampling rates to achieve approximately equal samples at each
    length. Papers with more atoms contribute to more lengths. Rare lengths
    (few papers) get more samples per paper to compensate.

    Args:
        clusters_path: Path to clusters.json
        token_mapping: Mapping from create_cluster_to_token_mapping()
        min_length: Minimum sequence length to generate (default: 2)
        balance_target: Target samples per length (default: 100000)
        max_samples_per_paper: Max samples per paper per length (default: 200)
        balance_max_length: Only balance up to this length (default: 5)
        include_debug_tokens: If True, add tokens_debug field
        random_seed: Random seed for reproducibility

    Returns:
        {
            "samples": list[dict],
            "metadata": {
                "total_samples": int,
                "total_papers": int,
                "papers_included": int,
                "papers_skipped": int,
                "min_length": int,
                "balance_target": int,
                "max_samples_per_paper": int,
                "balance_max_length": int,
                "samples_by_length": {length: count},
                "papers_by_length": {length: count},
                "samples_per_paper_by_length": {length: rate},
                "random_seed": int
            }
        }
    """
    if max_venue_year is not None and db_path is None:
        raise ValueError("db_path is required when max_venue_year is set")

    with open(clusters_path, "r", encoding="utf-8") as f:
        clusters_data = json.load(f)

    # Extract paper -> super-atoms mapping (unfiltered)
    paper_super_atoms = extract_paper_super_atoms(clusters_data)

    vocab_partition: Optional[dict] = None
    papers_dropped_by_cutoff = 0
    if max_venue_year is not None:
        paper_year_lookup = _load_paper_year_lookup(db_path)
        vocab_partition = compute_vocab_partition(
            paper_super_atoms, paper_year_lookup, max_venue_year
        )
        before = len(paper_super_atoms)
        paper_super_atoms = _filter_papers_by_venue_year(
            paper_super_atoms, paper_year_lookup, max_venue_year
        )
        papers_dropped_by_cutoff = before - len(paper_super_atoms)

    total_papers = len(paper_super_atoms)
    papers_included = sum(
        1 for cluster_ids in paper_super_atoms.values()
        if len(cluster_ids) >= min_length
    )
    papers_skipped = total_papers - papers_included

    (
        samples,
        papers_by_length,
        samples_per_paper_by_length,
    ) = generate_coherence_samples_length_balanced(
        paper_super_atoms=paper_super_atoms,
        token_mapping=token_mapping,
        min_length=min_length,
        balance_target=balance_target,
        max_samples_per_paper=max_samples_per_paper,
        balance_max_length=balance_max_length,
        include_debug_tokens=include_debug_tokens,
        random_seed=random_seed,
    )

    # Count samples by length
    samples_by_length: dict[int, int] = {}
    for sample in samples:
        seq_len = sample["seq_len"]
        samples_by_length[seq_len] = samples_by_length.get(seq_len, 0) + 1

    return {
        "samples": samples,
        "metadata": {
            "total_samples": len(samples),
            "total_papers": total_papers,
            "papers_included": papers_included,
            "papers_skipped": papers_skipped,
            "min_length": min_length,
            "balance_target": balance_target,
            "max_samples_per_paper": max_samples_per_paper,
            "balance_max_length": balance_max_length,
            "samples_by_length": samples_by_length,
            "papers_by_length": papers_by_length,
            "samples_per_paper_by_length": samples_per_paper_by_length,
            "random_seed": random_seed,
            "max_venue_year": max_venue_year,
            "papers_dropped_by_cutoff": papers_dropped_by_cutoff,
            "vocab_partition": vocab_partition,
        },
    }
