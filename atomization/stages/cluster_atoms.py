import argparse
import asyncio
import json
from pathlib import Path
from typing import Any, Dict, List

import backoff
import igraph as ig
import leidenalg
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from sklearn.cluster import HDBSCAN
from sklearn.decomposition import PCA
from sklearn.neighbors import kneighbors_graph
from sklearn.preprocessing import normalize
from litellm import completion
from umap import UMAP

from atomization.prompts import CLUSTER_NAMING_PROMPT
from atomization.utils.errors import (
    extract_json_from_llm_response,
    is_json_error,
    is_retryable_error,
)
from atomization.utils.metrics import (
    compute_cluster_diagnostics,
    compute_global_statistics,
)

MODEL_NAME = "gemini/gemini-3.1-pro-preview"
EMBEDDING_MODEL = "BAAI/bge-large-en-v1.5"
MAX_RETRIES = 5


def load_all_atoms(papers_dir: str = "papers") -> List[Dict[str, Any]]:
    """Load all refined atoms from the papers directory."""
    atoms = []
    papers_path = Path(papers_dir)

    for paper_dir in papers_path.iterdir():
        if not paper_dir.is_dir():
            continue

        refined_path = paper_dir / "refined_ideas.json"
        if not refined_path.exists():
            continue

        with open(refined_path) as f:
            data = json.load(f)

        for rating in data.get("ratings", []):
            idea_text = rating.get("revised_idea") or rating.get("idea")
            if idea_text and rating.get("quality") != "weak":
                atoms.append({
                    "text": idea_text,
                    "paper_id": paper_dir.name,
                    "quality": rating.get("quality"),
                    "original": rating.get("idea")
                })

    return atoms


def embed_atoms(atoms: List[Dict[str, Any]], model_name: str = EMBEDDING_MODEL) -> np.ndarray:
    """Generate embeddings for all atoms."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    model = SentenceTransformer(model_name, device=device)
    texts = [a["text"] for a in atoms]
    embeddings = model.encode(texts, show_progress_bar=True, normalize_embeddings=True)
    return embeddings


def _reduce_umap(embeddings: np.ndarray, n_neighbors: int, n_components: int,
                 min_dist: float = 0.0) -> np.ndarray:
    """Reduce embeddings with UMAP."""
    print(f"Reducing dimensions with UMAP (cosine metric, min_dist={min_dist})...")
    reducer = UMAP(
        n_neighbors=n_neighbors,
        n_components=n_components,
        min_dist=min_dist,
        metric='cosine',
        random_state=42,
    )
    return reducer.fit_transform(embeddings)


def _reduce_pca(embeddings: np.ndarray, n_components: int) -> np.ndarray:
    """Reduce embeddings with PCA."""
    print(f"Reducing dimensions with PCA ({n_components} components)...")
    pca = PCA(n_components=n_components, random_state=42)
    reduced = pca.fit_transform(embeddings)
    variance = float(pca.explained_variance_ratio_.sum())
    print(f"  Variance explained: {variance:.1%}")
    return reduced


def _cluster_hdbscan(
    embeddings: np.ndarray,
    min_cluster_size: int,
    min_samples: int,
    metric: str = "euclidean",
    cluster_selection_epsilon: float = 0.0,
    cluster_selection_method: str = "leaf",
) -> np.ndarray:
    """Cluster embeddings with HDBSCAN."""
    if metric == "cosine":
        embeddings = normalize(embeddings, norm='l2', copy=True)
        metric = "euclidean"

    print(
        "Clustering with HDBSCAN "
        f"({metric}, {cluster_selection_method}, eps={cluster_selection_epsilon})..."
    )
    clusterer = HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        metric=metric,
        cluster_selection_method=cluster_selection_method,
        cluster_selection_epsilon=cluster_selection_epsilon,
    )
    return clusterer.fit_predict(embeddings)


def _build_leiden_graph(
    embeddings: np.ndarray, k_neighbors: int, weighted: bool = False,
) -> ig.Graph:
    """Build a symmetric k-NN graph for Leiden from embeddings.

    When ``weighted`` is true, edges carry approximate cosine-similarity weights.
    """
    normed = normalize(embeddings, norm="l2")
    mode = "distance" if weighted else "connectivity"
    knn = kneighbors_graph(
        normed,
        n_neighbors=k_neighbors,
        mode=mode,
        include_self=False,
        n_jobs=-1,
    )
    knn_sym = knn.maximum(knn.T)

    sources, targets = knn_sym.nonzero()
    edges = list(zip(sources.tolist(), targets.tolist()))
    g = ig.Graph(n=embeddings.shape[0], edges=edges, directed=False)

    if weighted:
        distances = np.array(knn_sym[sources, targets]).flatten()
        weights = np.clip(1.0 - distances / 2.0, 0.0, 1.0)
        g.es["weight"] = weights.tolist()

    g.simplify(combine_edges="max" if weighted else "first")
    return g


def cluster_embeddings(
    embeddings: np.ndarray,
    method: str = "umap_hdbscan",
    min_cluster_size: int = 5,
    min_samples: int = 3,
    umap_components: int = 5,
    umap_neighbors: int = 15,
    umap_min_dist: float = 0.0,
    pca_components: int = 100,
    cluster_selection_epsilon: float = 0.0,
    cluster_selection_method: str = "eom",
    resolution: float = 1.0,
    k_neighbors: int = 30,
    weighted: bool = False,
    _precomputed_reduction: np.ndarray | None = None,
    _precomputed_graph: "ig.Graph | None" = None,
) -> np.ndarray:
    """Reduce dimensions and/or cluster embeddings.

    Methods:
        umap_hdbscan:   UMAP reduction + HDBSCAN (euclidean on reduced space)
        pca_hdbscan:    PCA reduction + HDBSCAN (euclidean on reduced space)
        raw_hdbscan:    HDBSCAN directly on raw embeddings (cosine metric)
        leiden:         Leiden community detection on k-NN graph

    Advanced parameters (used by clustering eval grid):
        _precomputed_reduction: Pre-reduced embeddings, skips UMAP/PCA step.
        _precomputed_graph: Pre-built igraph Graph, skips k-NN construction for Leiden.
    """
    if method == "umap_hdbscan":
        if _precomputed_reduction is not None:
            reduced = _precomputed_reduction
        else:
            reduced = _reduce_umap(
                embeddings,
                umap_neighbors,
                umap_components,
                min_dist=umap_min_dist,
            )
        print(f"  {embeddings.shape[1]}-D -> {reduced.shape[1]}-D")
        try:
            labels = _cluster_hdbscan(
                reduced,
                min_cluster_size,
                min_samples,
                metric="euclidean",
                cluster_selection_epsilon=cluster_selection_epsilon,
                cluster_selection_method=cluster_selection_method,
            )
        except TypeError:
            labels = np.full(len(embeddings), -1, dtype=np.intp)

    elif method == "pca_hdbscan":
        if _precomputed_reduction is not None:
            reduced = _precomputed_reduction
        else:
            reduced = _reduce_pca(embeddings, pca_components)
        print(f"  {embeddings.shape[1]}-D -> {reduced.shape[1]}-D")
        try:
            labels = _cluster_hdbscan(
                reduced,
                min_cluster_size,
                min_samples,
                metric="euclidean",
                cluster_selection_epsilon=cluster_selection_epsilon,
                cluster_selection_method=cluster_selection_method,
            )
        except TypeError:
            labels = np.full(len(embeddings), -1, dtype=np.intp)

    elif method == "raw_hdbscan":
        print(f"Clustering raw {embeddings.shape[1]}-D embeddings (cosine metric)...")
        try:
            labels = _cluster_hdbscan(
                embeddings,
                min_cluster_size,
                min_samples,
                metric="cosine",
                cluster_selection_epsilon=cluster_selection_epsilon,
                cluster_selection_method=cluster_selection_method,
            )
        except TypeError:
            labels = np.full(len(embeddings), -1, dtype=np.intp)

    elif method == "leiden":
        print(
            "Leiden community detection "
            f"(resolution={resolution}, k_neighbors={k_neighbors}, weighted={weighted})..."
        )
        if _precomputed_graph is not None:
            g = _precomputed_graph
        else:
            g = _build_leiden_graph(embeddings, k_neighbors, weighted=weighted)
        partition_kwargs = {
            "resolution_parameter": resolution,
            "seed": 42,
        }
        if weighted and "weight" in g.es.attributes():
            partition_kwargs["weights"] = "weight"
        partition = leidenalg.find_partition(
            g, leidenalg.RBConfigurationVertexPartition,
            **partition_kwargs,
        )
        labels = np.array(partition.membership, dtype=np.intp)

    else:
        raise ValueError(f"Unknown clustering method: {method}")

    return labels


def sample_from_cluster(
    atoms: List[Dict[str, Any]],
    labels: np.ndarray,
    cluster_id: int,
    embeddings: np.ndarray,
    n_samples: int = 10
) -> List[Dict[str, Any]]:
    """
    Sample n atoms from a cluster using maximal marginal relevance (MMR)
    to maximize diversity while staying representative.
    """
    cluster_indices = [i for i, l in enumerate(labels) if l == cluster_id]
    cluster_atoms = [atoms[i] for i in cluster_indices]
    cluster_embeddings = embeddings[cluster_indices]

    if len(cluster_atoms) <= n_samples:
        return cluster_atoms

    centroid = cluster_embeddings.mean(axis=0)
    selected_indices = []
    remaining_indices = list(range(len(cluster_atoms)))

    centroid_distances = np.linalg.norm(cluster_embeddings - centroid, axis=1)
    first_idx = int(np.argmin(centroid_distances))
    selected_indices.append(first_idx)
    remaining_indices.remove(first_idx)

    while len(selected_indices) < n_samples and remaining_indices:
        best_idx = None
        best_score = -float('inf')

        for idx in remaining_indices:
            min_dist_to_selected = min(
                np.linalg.norm(cluster_embeddings[idx] - cluster_embeddings[sel_idx])
                for sel_idx in selected_indices
            )
            dist_to_centroid = centroid_distances[idx]
            lambda_param = 0.3
            score = min_dist_to_selected - lambda_param * dist_to_centroid

            if score > best_score:
                best_score = score
                best_idx = idx

        selected_indices.append(best_idx)
        remaining_indices.remove(best_idx)

    return [cluster_atoms[i] for i in selected_indices]


def find_nameless_clusters(cluster_info: Dict[str, Any]) -> List[str]:
    """Identify clusters that need (re-)naming due to failed or placeholder names."""
    nameless = []
    for cid, info in cluster_info.items():
        name = info.get("name", "")
        if (
            (name.startswith("Cluster ") and name[8:].isdigit())
            or info.get("description") == "Naming failed"
            or info.get("confidence") == "unknown"
        ):
            nameless.append(cid)
    return nameless


def _log_cluster_naming_backoff(details: dict) -> None:
    print(
        f"Retrying in {details['wait']:.1f}s... "
        f"(attempt {details['tries']}/{MAX_RETRIES})"
    )


@backoff.on_exception(
    backoff.expo,
    Exception,
    max_tries=MAX_RETRIES,
    giveup=lambda e: not (is_retryable_error(e) or is_json_error(e)),
    on_backoff=_log_cluster_naming_backoff,
)
def _name_cluster_sync(atoms_sample: List[Dict[str, Any]]) -> Dict[str, str]:
    """Generate a name for a cluster based on sampled atoms."""
    atoms_text = "\n\n".join([f"- {a['text']}" for a in atoms_sample])
    prompt = CLUSTER_NAMING_PROMPT.format(atoms_text=atoms_text)

    response = completion(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": prompt}],
    )

    return extract_json_from_llm_response(response.choices[0].message.content)


async def name_cluster(atoms_sample: List[Dict[str, Any]]) -> Dict[str, str]:
    """Async wrapper for cluster naming."""
    return await asyncio.to_thread(_name_cluster_sync, atoms_sample)


async def name_clusters_batch(
    cluster_ids: List[str],
    atoms: List[Dict[str, Any]],
    labels: np.ndarray,
    embeddings: np.ndarray,
    samples_per_cluster: int,
    max_concurrent: int,
) -> Dict[str, Dict[str, Any]]:
    """Name a batch of clusters in parallel with concurrency control."""
    naming_semaphore = asyncio.Semaphore(max_concurrent)

    async def name_cluster_with_limit(sample: List[Dict[str, Any]]) -> Dict[str, str]:
        async with naming_semaphore:
            return await name_cluster(sample)

    naming_tasks = []
    for cluster_id_str in cluster_ids:
        cluster_id = int(cluster_id_str)
        sample = sample_from_cluster(
            atoms,
            labels,
            cluster_id,
            embeddings,
            samples_per_cluster,
        )
        task = asyncio.create_task(name_cluster_with_limit(sample))
        naming_tasks.append((cluster_id_str, sample, task))

    results = await asyncio.gather(
        *[t for _, _, t in naming_tasks],
        return_exceptions=True,
    )

    cluster_info = {}
    for (cluster_id_str, sample, _), naming_result in zip(naming_tasks, results):
        if isinstance(naming_result, Exception):
            print(f"  Warning: Cluster {cluster_id_str} naming failed: {naming_result}")
            cluster_info[cluster_id_str] = {
                "name": f"Cluster {cluster_id_str}",
                "description": "Naming failed",
                "confidence": "unknown",
                "sample_atoms": [a["text"] for a in sample],
            }
        else:
            cluster_info[cluster_id_str] = {
                "name": naming_result.get("super_atom", f"Cluster {cluster_id_str}"),
                "description": naming_result.get("rationale", ""),
                "confidence": naming_result.get("coherence_score", "unknown"),
                "sample_atoms": [a["text"] for a in sample],
            }
            print(f"  Cluster {cluster_id_str}: {cluster_info[cluster_id_str]['name']}")

    return cluster_info


async def cluster_and_name(
    papers_dir: str = "papers",
    min_cluster_size: int = 5,
    min_samples: int = 3,
    samples_per_cluster: int = 10,
    output_path: str = "clusters.json",
    embeddings_path: str | None = None,
    save_embeddings: bool = True,
    umap_components: int = 5,
    umap_neighbors: int = 10,
    umap_min_dist: float = 0.0,
    max_concurrent_naming: int = 30,
    method: str = "umap_hdbscan",
    pca_components: int = 100,
    cluster_selection_epsilon: float = 0.0,
    cluster_selection_method: str = "eom",
    resolution: float = 1.0,
    k_neighbors: int = 30,
) -> Dict[str, Any]:
    """Load atoms, cluster them, name clusters, and compute diagnostics."""

    print("Loading atoms...")
    atoms = load_all_atoms(papers_dir)
    n_papers_total = len(set(a["paper_id"] for a in atoms))
    print(f"Loaded {len(atoms)} atoms from {n_papers_total} papers")

    if len(atoms) == 0:
        print("No atoms found. Run the extraction pipeline first.")
        return {}

    cache_path = (
        Path(embeddings_path)
        if embeddings_path
        else Path(output_path).with_suffix(".npy")
    )
    embeddings = None

    if cache_path.exists():
        print(f"Loading embeddings from {cache_path}...")
        embeddings = np.load(cache_path)

        if len(embeddings) != len(atoms):
            print(
                f"Cache mismatch: {len(embeddings)} vectors but {len(atoms)} atoms. "
                "Recomputing..."
            )
            embeddings = None
        else:
            print(
                f"Loaded {embeddings.shape[0]} embeddings "
                f"({embeddings.shape[1]}-D) from cache"
            )

    if embeddings is None:
        print("Generating embeddings...")
        embeddings = embed_atoms(atoms)

        if save_embeddings:
            np.save(cache_path, embeddings)
            print(f"Embeddings cached to {cache_path}")

    output_file = Path(output_path)
    skip_clustering = False
    cluster_info = {}
    labels = None
    nameless_clusters = []

    if output_file.exists():
        print(f"Loading existing clustering results from {output_path}...")
        try:
            with open(output_file) as f:
                cached_output = json.load(f)

            cached_labels = np.array([
                a["cluster"] for a in cached_output["atom_assignments"]
            ])

            if len(cached_labels) != len(atoms):
                print(
                    f"Cache mismatch: {len(cached_labels)} cached assignments but "
                    f"{len(atoms)} atoms. Re-clustering..."
                )
            else:
                cached_params = cached_output.get("summary", {}).get("clustering_params", {})
                current_params = {
                    "min_cluster_size": min_cluster_size,
                    "min_samples": min_samples,
                    "embedding_model": EMBEDDING_MODEL,
                }

                if (
                    cached_params.get("min_cluster_size")
                    != current_params["min_cluster_size"]
                    or cached_params.get("min_samples")
                    != current_params["min_samples"]
                    or cached_params.get("embedding_model")
                    != current_params["embedding_model"]
                ):
                    print("Clustering parameters changed. Re-clustering...")
                else:
                    print(f"Loaded {len(cached_labels)} cluster assignments from cache")
                    labels = cached_labels
                    cluster_info = cached_output["clusters"]
                    skip_clustering = True

                    nameless_clusters = find_nameless_clusters(cluster_info)
                    if nameless_clusters:
                        print(
                            f"Found {len(nameless_clusters)} clusters needing "
                            f"re-naming: {nameless_clusters}"
                        )

        except (json.JSONDecodeError, KeyError, ValueError) as e:
            print(f"Error loading cache: {e}. Re-clustering...")

    if not skip_clustering:
        print(f"Clustering (method={method})...")
        labels = cluster_embeddings(
            embeddings,
            method=method,
            min_cluster_size=min_cluster_size,
            min_samples=min_samples,
            umap_components=umap_components,
            umap_neighbors=umap_neighbors,
            umap_min_dist=umap_min_dist,
            pca_components=pca_components,
            cluster_selection_epsilon=cluster_selection_epsilon,
            cluster_selection_method=cluster_selection_method,
            resolution=resolution,
            k_neighbors=k_neighbors,
        )

        unique_labels = set(labels)
        n_clusters = len(unique_labels) - (1 if -1 in unique_labels else 0)
        n_noise = list(labels).count(-1)
        print(f"Found {n_clusters} clusters, {n_noise} noise points")

        cluster_ids_to_name = [
            str(cid) for cid in sorted(unique_labels) if cid != -1
        ]
        print(
            f"Naming {len(cluster_ids_to_name)} clusters in parallel "
            f"(max {max_concurrent_naming} concurrent)..."
        )

        cluster_info = await name_clusters_batch(
            cluster_ids_to_name,
            atoms,
            labels,
            embeddings,
            samples_per_cluster,
            max_concurrent_naming,
        )

    elif nameless_clusters:
        print(f"Re-naming {len(nameless_clusters)} failed clusters...")
        renamed_info = await name_clusters_batch(
            nameless_clusters,
            atoms,
            labels,
            embeddings,
            samples_per_cluster,
            max_concurrent_naming,
        )

        for cluster_id_str, info in renamed_info.items():
            cluster_info[cluster_id_str].update(info)

    print("Computing diagnostics...")
    unique_labels = set(labels)
    for cluster_id in sorted(unique_labels):
        if cluster_id == -1:
            continue

        diagnostics = compute_cluster_diagnostics(atoms, labels, embeddings, cluster_id)
        cluster_info[str(cluster_id)]["diagnostics"] = diagnostics

        diag = diagnostics
        if diag["paper_concentration"] < 0.3:
            concentration_label = "diverse"
        elif diag["paper_concentration"] < 0.6:
            concentration_label = "moderate"
        else:
            concentration_label = "concentrated"
        print(f"  Cluster {cluster_id}: {cluster_info[str(cluster_id)]['name']}")
        print(
            f"    {diag['size']} atoms from {diag['n_papers']} papers "
            f"({concentration_label})"
        )

    global_stats = compute_global_statistics(
        atoms=atoms,
        labels=labels,
        cluster_info=cluster_info,
        n_papers_total=n_papers_total,
        embedding_model=EMBEDDING_MODEL,
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
    )

    output = {
        "summary": global_stats,
        "clusters": cluster_info,
        "atom_assignments": [
            {"text": a["text"], "paper_id": a["paper_id"], "cluster": int(l)}
            for a, l in zip(atoms, labels)
        ],
    }

    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    print("\n" + "=" * 60)
    print("CLUSTERING SUMMARY")
    print("=" * 60)
    print(f"Total atoms: {global_stats['total_atoms']}")
    print(f"Total papers: {global_stats['total_papers']}")
    print(f"Clusters found: {global_stats['n_clusters']}")
    print(
        f"Noise points: {global_stats['n_noise']} "
        f"({100 * global_stats['n_noise'] / len(atoms):.1f}%)"
    )
    print(f"Avg cluster size: {global_stats['avg_cluster_size']:.1f}")
    print(
        "Cross-paper coverage: "
        f"{100 * global_stats['cross_paper_coverage']:.1f}% of papers appear "
        "in multi-paper clusters"
    )
    print(f"Avg super atoms per paper: {global_stats['avg_super_atoms_per_paper']:.2f}")
    print(
        "Full noise papers: "
        f"{100 * global_stats['pct_full_noise_papers']:.1f}% of papers have "
        "all atoms as noise"
    )
    print(f"\nSaved results to {output_path}")

    return output


def run_clustering():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Cluster atoms of knowledge and generate cluster names"
    )
    parser.add_argument("--papers-dir", type=str, default="papers",
                        help="Directory containing paper subdirectories with refined_ideas.json")
    parser.add_argument("--output", type=str, default=None,
                        help="Output path for clustering results")
    parser.add_argument("--min-cluster-size", type=int, default=30,
                        help="Minimum atoms required to form a cluster (default: 30)")
    parser.add_argument("--min-samples", type=int, default=5,
                        help="HDBSCAN min_samples parameter (default: 10)")
    parser.add_argument("--samples-per-cluster", type=int, default=100,
                        help="Number of atoms to sample for naming each cluster (default: 100)")
    parser.add_argument("--embeddings-path", type=str, default=None,
                        help="Path to cached embeddings .npy file. If exists, loads it; "
                             "otherwise computes and saves to this path. "
                             "Defaults to {output}.npy if not specified.")
    parser.add_argument("--no-save-embeddings", action="store_true",
                        help="Don't save embeddings to .npy file")
    parser.add_argument("--umap-components", type=int, default=5,
                        help="UMAP target dimensions (default: 5)")
    parser.add_argument("--umap-neighbors", type=int, default=10,
                        help="UMAP n_neighbors parameter (default: 10)")
    parser.add_argument("--umap-min-dist", type=float, default=0.0,
                        help="UMAP min_dist parameter (default: 0.0)")
    parser.add_argument("--cluster-selection-epsilon", type=float, default=0.0,
                        help="HDBSCAN cluster_selection_epsilon (default: 0.0)")
    parser.add_argument("--cluster-selection-method", type=str, default="eom",
                        choices=["leaf", "eom"],
                        help="HDBSCAN cluster_selection_method (default: eom)")
    parser.add_argument("--max-concurrent-naming", type=int, default=30,
                        help="Maximum concurrent cluster naming LLM calls (default: 50)")
    parser.add_argument("--method", type=str, default="umap_hdbscan",
                        choices=["umap_hdbscan", "pca_hdbscan", "raw_hdbscan", "leiden"],
                        help="Clustering method (default: umap_hdbscan)")
    parser.add_argument("--pca-components", type=int, default=100,
                        help="PCA target dimensions for pca_hdbscan method (default: 100)")
    parser.add_argument("--resolution", type=float, default=1.0,
                        help="Resolution for Leiden method (default: 1.0)")
    parser.add_argument("--k-neighbors", type=int, default=30,
                        help="Number of neighbors for Leiden k-NN graph (default: 30)")

    args = parser.parse_args()

    if args.output is None:
        args.output = f"{args.papers_dir}/clusters.json"

    asyncio.run(cluster_and_name(
        papers_dir=args.papers_dir,
        min_cluster_size=args.min_cluster_size,
        min_samples=args.min_samples,
        samples_per_cluster=args.samples_per_cluster,
        output_path=args.output,
        embeddings_path=args.embeddings_path,
        save_embeddings=not args.no_save_embeddings,
        umap_components=args.umap_components,
        umap_neighbors=args.umap_neighbors,
        umap_min_dist=args.umap_min_dist,
        max_concurrent_naming=args.max_concurrent_naming,
        method=args.method,
        pca_components=args.pca_components,
        cluster_selection_epsilon=args.cluster_selection_epsilon,
        cluster_selection_method=args.cluster_selection_method,
        resolution=args.resolution,
        k_neighbors=args.k_neighbors,
    ))


if __name__ == "__main__":
    run_clustering()
