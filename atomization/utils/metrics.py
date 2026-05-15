"""
Clustering metrics and diagnostics.

Provides functions to compute cluster quality metrics and global statistics
for atom clustering results. These functions are used by the clustering stage
but can also be called standalone for analysis and debugging.
"""

import numpy as np
from typing import List, Dict, Any


def compute_cluster_diagnostics(
    atoms: List[Dict[str, Any]],
    labels: np.ndarray,
    embeddings: np.ndarray,
    cluster_id: int
) -> Dict[str, Any]:
    """
    Compute diagnostic metrics for a cluster.

    Metrics:
    - size: number of atoms in cluster
    - n_papers: number of unique papers contributing atoms
    - paper_distribution: dict of paper_id -> count
    - paper_concentration: Herfindahl index (0-1, lower = more diverse)
    - avg_intra_distance: average pairwise distance within cluster
    - centroid_spread: average distance from centroid
    """
    cluster_indices = [i for i, l in enumerate(labels) if l == cluster_id]
    cluster_atoms = [atoms[i] for i in cluster_indices]
    cluster_embeddings = embeddings[cluster_indices]

    # Paper distribution
    paper_counts: Dict[str, int] = {}
    for atom in cluster_atoms:
        pid = atom["paper_id"]
        paper_counts[pid] = paper_counts.get(pid, 0) + 1

    n_papers = len(paper_counts)
    size = len(cluster_atoms)

    # Herfindahl index: sum of squared shares
    # 1.0 = all atoms from one paper, 1/n = perfectly uniform
    shares = [count / size for count in paper_counts.values()]
    herfindahl = sum(s ** 2 for s in shares)

    # Normalized concentration: 0 = maximally diverse, 1 = single paper
    # Adjusts for number of papers: (H - 1/n) / (1 - 1/n)
    if n_papers > 1:
        paper_concentration = (herfindahl - 1/n_papers) / (1 - 1/n_papers)
    else:
        paper_concentration = 1.0

    # Embedding-based metrics
    centroid = cluster_embeddings.mean(axis=0)
    centroid_distances = np.linalg.norm(cluster_embeddings - centroid, axis=1)
    centroid_spread = float(centroid_distances.mean())

    # Average pairwise distance (sample if cluster is large)
    if size <= 100:
        pairwise_distances = []
        for i in range(size):
            for j in range(i + 1, size):
                dist = np.linalg.norm(cluster_embeddings[i] - cluster_embeddings[j])
                pairwise_distances.append(dist)
        avg_intra_distance = float(np.mean(pairwise_distances)) if pairwise_distances else 0.0
    else:
        # Sample pairs for large clusters
        n_samples = 1000
        sample_distances = []
        for _ in range(n_samples):
            i, j = np.random.choice(size, 2, replace=False)
            dist = np.linalg.norm(cluster_embeddings[i] - cluster_embeddings[j])
            sample_distances.append(dist)
        avg_intra_distance = float(np.mean(sample_distances))

    return {
        "size": size,
        "n_papers": n_papers,
        "paper_distribution": paper_counts,
        "paper_concentration": round(paper_concentration, 3),
        "centroid_spread": round(centroid_spread, 4),
        "avg_intra_distance": round(avg_intra_distance, 4)
    }


def compute_global_statistics(
    atoms: List[Dict[str, Any]],
    labels: np.ndarray,
    cluster_info: Dict[str, Any],
    n_papers_total: int,
    embedding_model: str,
    min_cluster_size: int,
    min_samples: int
) -> Dict[str, Any]:
    """
    Compute global clustering statistics.

    Args:
        atoms: list of atom dicts
        labels: cluster label for each atom
        cluster_info: dict mapping cluster_id -> cluster metadata
        n_papers_total: total number of unique papers
        embedding_model: name of embedding model used
        min_cluster_size: HDBSCAN parameter
        min_samples: HDBSCAN parameter

    Returns:
        Dict with keys: total_atoms, total_papers, n_clusters,
        n_noise, avg_cluster_size, cross_paper_coverage, clustering_params
    """
    unique_labels = set(labels)
    n_clusters = len(unique_labels) - (1 if -1 in unique_labels else 0)
    n_noise = list(labels).count(-1)

    clustered_atoms = [a for a, l in zip(atoms, labels) if l != -1]
    avg_cluster_size = len(clustered_atoms) / n_clusters if n_clusters > 0 else 0

    papers_in_diverse_clusters = set()
    for info in cluster_info.values():
        if info["diagnostics"]["n_papers"] > 1:
            papers_in_diverse_clusters.update(
                info["diagnostics"]["paper_distribution"].keys()
            )

    cross_paper_coverage = (
        len(papers_in_diverse_clusters) / n_papers_total
        if n_papers_total > 0
        else 0
    )

    paper_to_clusters = {}
    paper_to_all_atoms = {}

    for atom, label in zip(atoms, labels):
        paper_id = atom["paper_id"]

        # Track all atoms for full noise detection
        if paper_id not in paper_to_all_atoms:
            paper_to_all_atoms[paper_id] = []
        paper_to_all_atoms[paper_id].append(label)

        # Track clustered atoms for super atoms metric
        if label != -1:  # Skip noise atoms
            if paper_id not in paper_to_clusters:
                paper_to_clusters[paper_id] = set()
            paper_to_clusters[paper_id].add(label)

    # Compute average super atoms per paper (only papers with clustered atoms)
    if paper_to_clusters:
        total_clusters_across_papers = sum(len(clusters) for clusters in paper_to_clusters.values())
        avg_super_atoms_per_paper = total_clusters_across_papers / len(paper_to_clusters)
    else:
        avg_super_atoms_per_paper = 0.0

    # Compute percentage of full noise papers
    full_noise_papers = sum(1 for labels_list in paper_to_all_atoms.values()
                            if all(l == -1 for l in labels_list))
    pct_full_noise_papers = full_noise_papers / n_papers_total if n_papers_total > 0 else 0.0

    return {
        "total_atoms": len(atoms),
        "total_papers": n_papers_total,
        "n_clusters": n_clusters,
        "n_noise": n_noise,
        "avg_cluster_size": round(avg_cluster_size, 1),
        "cross_paper_coverage": round(cross_paper_coverage, 3),
        "avg_super_atoms_per_paper": round(avg_super_atoms_per_paper, 2),
        "pct_full_noise_papers": round(pct_full_noise_papers, 3),
        "clustering_params": {
            "min_cluster_size": min_cluster_size,
            "min_samples": min_samples,
            "embedding_model": embedding_model
        }
    }
