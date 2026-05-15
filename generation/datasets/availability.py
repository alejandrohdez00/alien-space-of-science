"""Availability dataset for the two-tower model: (subset, complement) pairs.

Each row pairs a positive atom subset with the author's complement pool (all of
their other atoms). Pool handling is deterministic:

  * pool_size <= pool_cap : enumerate all C(pool_size, L) subsets for
                            L in [min_length, max_length].
  * pool_size >  pool_cap : uniformly sample `per_author_budget` subsets from
                            the union of 2..max_length-subsets. Length weighted
                            by C(n, L) so the sample is uniform over the union.

Two authors with the same subset are legitimately distinct rows because their
complements differ. Dedup is per-author only in the sampling branch to prevent
`(author, subset)` duplicates within one author; the enumeration branch cannot
duplicate by construction.

Both subset and complement are shuffled independently with the local RNG and
encoded as token IDs without BOS/EOS.
"""

import json
import random
from collections import Counter
from itertools import combinations
from math import comb
from pathlib import Path

from .coherence import (
    extract_paper_super_atoms,
    compute_vocab_partition,
    _load_paper_year_lookup,
)
from .token_mapping import get_debug_tokens
from crawlers.db.schema import get_connection


def _paper_id_candidates(paper_id: str) -> list[str]:
    """Return possible metadata keys for a filesystem-safe paper id."""
    candidates = [paper_id]
    slash_id = paper_id.replace("__", "/")
    if slash_id != paper_id:
        candidates.append(slash_id)
    return candidates


def build_author_super_atoms(
    clusters_data: dict,
    db_path: str,
    max_venue_year: int | None = None,
) -> dict[str, dict]:
    """Build author -> atom-pool mapping from clusters.json and SQLite metadata."""
    paper_super_atoms = extract_paper_super_atoms(clusters_data)
    authors: dict[str, dict] = {}

    conn = get_connection(db_path)
    try:
        for cluster_paper_id, cluster_ids in paper_super_atoms.items():
            rows = []
            for candidate_id in _paper_id_candidates(cluster_paper_id):
                cursor = conn.execute(
                    """
                    SELECT
                        p.paper_id,
                        p.venue_year,
                        a.author_id,
                        a.display_name,
                        pa.author_position
                    FROM papers p
                    JOIN paper_authors pa ON p.paper_id = pa.paper_id
                    JOIN authors a ON pa.author_id = a.author_id
                    WHERE p.paper_id = ?
                    ORDER BY pa.author_position ASC
                    """,
                    (candidate_id,),
                )
                rows = [dict(row) for row in cursor.fetchall()]
                if rows:
                    break

            if not rows:
                continue

            venue_year = rows[0].get("venue_year")
            if (
                max_venue_year is not None
                and venue_year is not None
                and venue_year > max_venue_year
            ):
                continue

            db_paper_id = rows[0]["paper_id"]
            for row in rows:
                author_id = row["author_id"]
                if author_id not in authors:
                    authors[author_id] = {
                        "display_name": row.get("display_name") or author_id,
                        "super_atoms": set(),
                        "paper_ids": set(),
                    }
                authors[author_id]["super_atoms"].update(cluster_ids)
                authors[author_id]["paper_ids"].add(db_paper_id)
    finally:
        conn.close()

    return {
        author_id: {
            "display_name": data["display_name"],
            "super_atoms": sorted(data["super_atoms"]),
            "paper_ids": sorted(data["paper_ids"]),
            "paper_count": len(data["paper_ids"]),
        }
        for author_id, data in sorted(authors.items())
    }


def _encode_cluster_ids(cluster_ids: list[int], token_mapping: dict) -> list[int]:
    """Map cluster IDs to token IDs without BOS/EOS."""
    c2t = token_mapping["cluster_to_token_id"]
    return [c2t[cid] for cid in cluster_ids if cid in c2t]


def generate_availability_samples(
    author_super_atoms: dict[str, dict],
    token_mapping: dict,
    min_length: int = 2,
    max_length: int = 4,
    pool_cap: int = 30,
    per_author_budget: int = 20000,
    min_atoms: int = 2,
    include_debug_tokens: bool = False,
    random_seed: int | None = None,
) -> tuple[list[dict], dict]:
    """Generate (subset, complement) pairs for each included author.

    Determinism contract
    --------------------
    1. Authors iterated in sorted(author_ids) lex order.
    2. All randomness flows through a single local random.Random(seed); the
       module-level random state is NEVER touched.
    3. pool is pre-sorted (build_author_super_atoms) so combinations(pool, L)
       is deterministic.

    Returns
    -------
    (samples, stats) where `samples` is a flat list of JSONL row dicts and
    `stats` contains per-length + per-pool-regime counters for metadata.
    """
    rng = random.Random(random_seed)
    samples: list[dict] = []

    lengths = list(range(min_length, max_length + 1))
    samples_by_length = {L: 0 for L in lengths}
    authors_by_length = {L: 0 for L in lengths}
    authors_enumerated = 0
    authors_sampled = 0
    duplicates_rejected = 0
    complement_sizes: list[int] = []

    for author_id in sorted(author_super_atoms.keys()):
        data = author_super_atoms[author_id]
        pool = data["super_atoms"]
        n = len(pool)
        if n < min_atoms:
            continue

        L_range = [L for L in lengths if L <= n]
        if not L_range:
            continue

        display_name = data["display_name"]
        pool_set = set(pool)
        emitted_for_author = 0
        contrib_lengths: set[int] = set()
        per_length_idx = {L: 0 for L in L_range}

        def emit(subset: tuple[int, ...], L: int) -> None:
            nonlocal emitted_for_author
            complement = sorted(pool_set - set(subset))
            subset_list = list(subset)
            rng.shuffle(subset_list)
            complement_list = list(complement)
            rng.shuffle(complement_list)
            query_ids = _encode_cluster_ids(subset_list, token_mapping)
            author_ids = _encode_cluster_ids(complement_list, token_mapping)
            row = {
                "author_id": author_id,
                "display_name": display_name,
                "seq_len": L,
                "sample_idx": per_length_idx[L],
                "query_ids": query_ids,
                "author_ids": author_ids,
            }
            if include_debug_tokens:
                row["query_tokens_debug"] = get_debug_tokens(query_ids, token_mapping)
                row["author_tokens_debug"] = get_debug_tokens(author_ids, token_mapping)
            samples.append(row)
            samples_by_length[L] += 1
            per_length_idx[L] += 1
            contrib_lengths.add(L)
            emitted_for_author += 1
            complement_sizes.append(len(complement))

        if n <= pool_cap:
            for L in L_range:
                for subset in combinations(pool, L):
                    emit(subset, L)
            if emitted_for_author > 0:
                authors_enumerated += 1
        else:
            weights = [comb(n, L) for L in L_range]
            max_attempts = per_author_budget * 8
            attempts = 0
            author_seen: set[tuple[int, ...]] = set()
            while emitted_for_author < per_author_budget and attempts < max_attempts:
                attempts += 1
                L = rng.choices(L_range, weights=weights, k=1)[0]
                subset = tuple(sorted(rng.sample(pool, L)))
                if subset in author_seen:
                    duplicates_rejected += 1
                    continue
                author_seen.add(subset)
                emit(subset, L)
            if emitted_for_author > 0:
                authors_sampled += 1

        for L in contrib_lengths:
            authors_by_length[L] += 1

    complement_stats = {
        "min": min(complement_sizes) if complement_sizes else 0,
        "max": max(complement_sizes) if complement_sizes else 0,
        "avg": round(sum(complement_sizes) / len(complement_sizes), 2) if complement_sizes else 0.0,
    }

    stats = {
        "total_samples": len(samples),
        "duplicates_rejected": duplicates_rejected,
        "authors_enumerated": authors_enumerated,
        "authors_sampled": authors_sampled,
        "samples_by_length": samples_by_length,
        "authors_by_length": authors_by_length,
        "complement_size_stats": complement_stats,
    }
    return samples, stats


def generate_availability_dataset(
    clusters_path: str,
    db_path: str,
    token_mapping: dict,
    min_length: int = 2,
    max_length: int = 4,
    pool_cap: int = 30,
    per_author_budget: int = 20000,
    min_atoms: int = 2,
    max_venue_year: int | None = None,
    include_debug_tokens: bool = False,
    random_seed: int = 42,
) -> dict:
    """Build two-tower availability samples and metadata."""
    with open(clusters_path, "r", encoding="utf-8") as f:
        clusters_data = json.load(f)

    author_super_atoms = build_author_super_atoms(
        clusters_data,
        db_path,
        max_venue_year=max_venue_year,
    )

    vocab_partition: dict | None = None
    if max_venue_year is not None:
        paper_super_atoms_unfiltered = extract_paper_super_atoms(clusters_data)
        paper_year_lookup = _load_paper_year_lookup(db_path)
        vocab_partition = compute_vocab_partition(
            paper_super_atoms_unfiltered, paper_year_lookup, max_venue_year
        )

    total_authors = len(author_super_atoms)
    authors_skipped_atoms = 0
    authors_with_enough = 0
    for data in author_super_atoms.values():
        if len(data["super_atoms"]) < min_atoms:
            authors_skipped_atoms += 1
        else:
            authors_with_enough += 1

    included = [
        data
        for data in author_super_atoms.values()
        if len(data["super_atoms"]) >= min_atoms
    ]
    avg_atoms = (
        sum(len(a["super_atoms"]) for a in included) / len(included) if included else 0.0
    )
    avg_papers = (
        sum(a["paper_count"] for a in included) / len(included) if included else 0.0
    )

    samples, stats = generate_availability_samples(
        author_super_atoms=author_super_atoms,
        token_mapping=token_mapping,
        min_length=min_length,
        max_length=max_length,
        pool_cap=pool_cap,
        per_author_budget=per_author_budget,
        min_atoms=min_atoms,
        include_debug_tokens=include_debug_tokens,
        random_seed=random_seed,
    )

    metadata = {
        "mode": "two_tower_availability",
        "schema": "subset_complement_v1",
        "total_samples": stats["total_samples"],
        "duplicates_rejected": stats["duplicates_rejected"],
        "total_authors": total_authors,
        "authors_included": authors_with_enough,
        "authors_skipped": total_authors - authors_with_enough,
        "authors_skipped_atoms": authors_skipped_atoms,
        "authors_enumerated": stats["authors_enumerated"],
        "authors_sampled": stats["authors_sampled"],
        "avg_atoms_per_author": round(avg_atoms, 2),
        "avg_papers_per_author": round(avg_papers, 2),
        "samples_by_length": stats["samples_by_length"],
        "authors_by_length": stats["authors_by_length"],
        "complement_size_stats": stats["complement_size_stats"],
        "pool_cap": pool_cap,
        "per_author_budget": per_author_budget,
        "min_length": min_length,
        "max_length": max_length,
        "min_atoms": min_atoms,
        "max_venue_year": max_venue_year,
        "vocab_partition": vocab_partition,
        "atom_vocabulary_size": token_mapping.get("n_clusters", 0),
        "random_seed": random_seed,
    }

    return {"samples": samples, "metadata": metadata}


def save_availability_dataset(
    dataset: dict,
    output_path: str,
) -> None:
    """Write the flat sample list as JSONL."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for sample in dataset["samples"]:
            f.write(json.dumps(sample) + "\n")


def build_atoms_sidecar(clusters_data: dict) -> dict:
    """Build `{cluster_id: {text, cluster_id, frequency}}` sidecar from clusters.json."""
    cluster_freq = Counter(
        a["cluster"]
        for a in clusters_data.get("atom_assignments", [])
        if a.get("cluster", -1) != -1
    )
    cluster_text: dict[int, str] = {}
    for a in clusters_data.get("atom_assignments", []):
        cid = a.get("cluster", -1)
        if cid != -1 and cid not in cluster_text:
            cluster_text[cid] = a.get("text", "")

    atoms_sidecar: dict[str, dict] = {}
    for cid in sorted(cluster_freq.keys()):
        atoms_sidecar[str(cid)] = {
            "text": cluster_text.get(cid, ""),
            "cluster_id": cid,
            "frequency": cluster_freq[cid],
        }
    return atoms_sidecar
