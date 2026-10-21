"""Reassign noise atoms by similarity to the original BGE cluster centroids.

The algorithm is the one used for the paper's 80% coverage representation.
Read assignments and embeddings from the same clustering run, in saved order.
"""
import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
from atomization.utils.metrics import compute_cluster_diagnostics, compute_global_statistics


def cosine_similarity_matrix(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Compute cosine similarity between rows of A and rows of B.

    Args:
        A: (n, d) array
        B: (m, d) array

    Returns:
        (n, m) similarity matrix
    """
    A_norm = A / np.linalg.norm(A, axis=1, keepdims=True)
    B_norm = B / np.linalg.norm(B, axis=1, keepdims=True)
    return A_norm @ B_norm.T


def compute_paper_stats(labels: np.ndarray, paper_ids: list[str]) -> dict:
    """Compute paper-level coverage statistics."""
    unique_papers = sorted(set(paper_ids))
    n_papers = len(unique_papers)

    paper_clusters = defaultdict(set)
    for pid, label in zip(paper_ids, labels):
        if label != -1:
            paper_clusters[pid].add(label)

    counts = [len(paper_clusters[pid]) for pid in unique_papers]
    return {
        "n_papers": n_papers,
        "papers_ge1": sum(1 for c in counts if c >= 1),
        "papers_ge2": sum(1 for c in counts if c >= 2),
        "papers_ge3": sum(1 for c in counts if c >= 3),
        "mean_superatoms": float(np.mean(counts)),
        "median_superatoms": float(np.median(counts)),
    }


def build_clusters_json(
    atoms: list[dict],
    labels: np.ndarray,
    embeddings: np.ndarray,
    original_clusters: dict,
) -> dict:
    """Build a clusters.json dict from reassigned labels.

    Reuses cluster names/descriptions from the original and recomputes
    diagnostics and summary statistics from the new labels.
    """
    # Recompute per-cluster diagnostics
    unique_cids = sorted(set(int(l) for l in labels if l != -1))
    clusters = {}
    for cid in unique_cids:
        orig = original_clusters.get("clusters", {}).get(str(cid), {})
        diag = compute_cluster_diagnostics(atoms, labels, embeddings, cid)
        clusters[str(cid)] = {
            "name": orig.get("name", ""),
            "description": orig.get("description", ""),
            "confidence": orig.get("confidence", ""),
            "sample_atoms": orig.get("sample_atoms", []),
            "diagnostics": diag,
        }

    # Recompute global summary
    n_papers_total = len(set(a["paper_id"] for a in atoms))
    orig_params = original_clusters.get("summary", {}).get("clustering_params", {})
    summary = compute_global_statistics(
        atoms, labels, clusters, n_papers_total,
        embedding_model=orig_params.get("embedding_model", ""),
        min_cluster_size=orig_params.get("min_cluster_size", 0),
        min_samples=orig_params.get("min_samples", 0),
    )

    # Build atom assignments
    atom_assignments = [
        {"text": a["text"], "paper_id": a["paper_id"], "cluster": int(labels[i])}
        for i, a in enumerate(atoms)
    ]

    return {
        "summary": summary,
        "clusters": clusters,
        "atom_assignments": atom_assignments,
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--clusters', type=Path, required=True)
    p.add_argument('--embeddings-path', type=Path, required=True)
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--checkpoints', type=float, nargs='+', default=[0.80])
    args = p.parse_args()
    if any(not 0 < c <= 1 for c in args.checkpoints):
        p.error('coverage checkpoints must be in (0, 1]')
    original = json.loads(args.clusters.read_text())
    atoms = original['atom_assignments']
    labels = np.array([a['cluster'] for a in atoms])
    embeddings = np.load(args.embeddings_path)
    if not len(atoms) or len(embeddings) != len(atoms):
        raise ValueError('Embeddings must correspond to the saved atom assignments')
    cluster_ids = np.array(sorted(set(labels) - {-1}))
    if not len(cluster_ids):
        raise ValueError('No clusters to assign noise atoms to')
    centroids = np.stack([embeddings[labels == c].mean(axis=0) for c in cluster_ids])
    noise_indices = np.flatnonzero(labels == -1)
    best_clusters, best_similarities = [], []
    for start in range(0, len(noise_indices), 10000):
        similarities = cosine_similarity_matrix(embeddings[noise_indices[start:start + 10000]], centroids)
        best = similarities.argmax(axis=1)
        best_clusters.extend(cluster_ids[best])
        best_similarities.extend(similarities[np.arange(len(best)), best])
    best_clusters = np.asarray(best_clusters)
    best_similarities = np.asarray(best_similarities)
    order = np.argsort(-best_similarities, kind='stable')
    assigned = len(labels) - len(noise_indices)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    np.save(args.output_dir / 'labels_original.npy', labels)
    summary = []
    for coverage in sorted(set(args.checkpoints)):
        take = max(0, math.ceil(coverage * len(labels)) - assigned)
        selected = order[:take]
        updated = labels.copy()
        updated[noise_indices[selected]] = best_clusters[selected]
        result = build_clusters_json(atoms, updated, embeddings, original)
        target = args.output_dir / f'{args.clusters.stem}_{coverage:.2f}.json'
        target.write_text(json.dumps(result, indent=2))
        np.save(args.output_dir / f'labels_{coverage:.2f}.npy', updated)
        record = dict(checkpoint=coverage, actual_coverage=float(np.mean(updated != -1)),
                      atoms_reassigned=take, **compute_paper_stats(updated, [a['paper_id'] for a in atoms]))
        summary.append(record)
        print(f'Saved {target}; coverage {record["actual_coverage"]:.1%}')
    (args.output_dir / 'reassignment_summary.json').write_text(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
