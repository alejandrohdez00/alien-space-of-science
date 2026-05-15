"""Token mapping between super-atom cluster IDs and tokens.

Maps cluster IDs to unique tokens for training.
"""

import json
from pathlib import Path
from datetime import datetime

BOS_TOKEN_ID = 50256  # <|endoftext|>
EOS_TOKEN_ID = 50256  # Same as BOS


def load_vocab(vocab_path: str) -> dict[str, int]:
    """Load vocab.json (token_string -> token_id)."""
    with open(vocab_path, "r", encoding="utf-8") as f:
        return json.load(f)


def select_tokens(vocab: dict[str, int], n: int) -> list[tuple[str, int]]:
    """Select first n tokens from vocab sorted by token ID."""
    if n > len(vocab):
        raise ValueError(f"Need {n} tokens but vocab only has {len(vocab)}")
    sorted_tokens = sorted(vocab.items(), key=lambda x: x[1])
    return sorted_tokens[:n]


def create_cluster_to_token_mapping(
    clusters_data: dict,
    vocab_path: str,
) -> dict:
    """
    Create bidirectional mapping between cluster IDs and tokens.

    Args:
        clusters_data: Parsed clusters.json content
        vocab_path: Path to vocab.json

    Returns:
        {
            "cluster_to_token_id": {cluster_id: token_id, ...},
            "token_id_to_cluster": {token_id: cluster_id, ...},
            "cluster_to_token_str": {cluster_id: token_str, ...},
            "token_str_to_cluster": {token_str: cluster_id, ...},
            "bos_token_id": 50256,
            "eos_token_id": 50256,
            "n_clusters": int,
            "vocab_path": str,
            "created_at": str
        }
    """
    vocab = load_vocab(vocab_path)

    # Get cluster IDs (exclude noise cluster -1)
    cluster_ids = sorted(
        int(cid) for cid in clusters_data.get("clusters", {}).keys()
    )
    n_clusters = len(cluster_ids)

    # Select tokens
    tokens = select_tokens(vocab, n_clusters)

    # Build mappings
    cluster_to_token_id = {}
    token_id_to_cluster = {}
    cluster_to_token_str = {}
    token_str_to_cluster = {}

    for cluster_id, (token_str, token_id) in zip(cluster_ids, tokens):
        cluster_to_token_id[cluster_id] = token_id
        token_id_to_cluster[token_id] = cluster_id
        cluster_to_token_str[cluster_id] = token_str
        token_str_to_cluster[token_str] = cluster_id

    return {
        "cluster_to_token_id": cluster_to_token_id,
        "token_id_to_cluster": token_id_to_cluster,
        "cluster_to_token_str": cluster_to_token_str,
        "token_str_to_cluster": token_str_to_cluster,
        "bos_token_id": BOS_TOKEN_ID,
        "eos_token_id": EOS_TOKEN_ID,
        "n_clusters": n_clusters,
        "vocab_path": str(vocab_path),
        "created_at": datetime.now().isoformat(),
    }


def save_token_mapping(mapping: dict, output_path: str) -> None:
    """Save token mapping to JSON file."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    # Convert int keys to strings for JSON serialization
    serializable = {
        "cluster_to_token_id": {str(k): v for k, v in mapping["cluster_to_token_id"].items()},
        "token_id_to_cluster": {str(k): v for k, v in mapping["token_id_to_cluster"].items()},
        "cluster_to_token_str": {str(k): v for k, v in mapping["cluster_to_token_str"].items()},
        "token_str_to_cluster": mapping["token_str_to_cluster"],
        "bos_token_id": mapping["bos_token_id"],
        "eos_token_id": mapping["eos_token_id"],
        "n_clusters": mapping["n_clusters"],
        "vocab_path": mapping["vocab_path"],
        "created_at": mapping["created_at"],
    }

    with open(path, "w", encoding="utf-8") as f:
        json.dump(serializable, f, indent=2)


def load_token_mapping(mapping_path: str) -> dict:
    """
    Load existing token mapping from JSON file.

    Converts string keys back to integers for cluster IDs and token IDs.
    """
    with open(mapping_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    return {
        "cluster_to_token_id": {int(k): v for k, v in data["cluster_to_token_id"].items()},
        "token_id_to_cluster": {int(k): v for k, v in data["token_id_to_cluster"].items()},
        "cluster_to_token_str": {int(k): v for k, v in data["cluster_to_token_str"].items()},
        "token_str_to_cluster": data["token_str_to_cluster"],
        "bos_token_id": data["bos_token_id"],
        "eos_token_id": data["eos_token_id"],
        "n_clusters": data["n_clusters"],
        "vocab_path": data["vocab_path"],
        "created_at": data["created_at"],
    }


def encode_cluster_sequence(
    cluster_ids: list[int],
    mapping: dict,
    add_special_tokens: bool = True,
) -> list[int]:
    """
    Convert a sequence of cluster IDs to token IDs.

    Args:
        cluster_ids: List of cluster IDs
        mapping: Token mapping from create_cluster_to_token_mapping()
        add_special_tokens: If True, wrap with BOS/EOS tokens

    Returns:
        List of token IDs
    """
    token_ids = [mapping["cluster_to_token_id"][cid] for cid in cluster_ids]

    if add_special_tokens:
        token_ids = [mapping["bos_token_id"]] + token_ids + [mapping["eos_token_id"]]

    return token_ids


def decode_token_sequence(
    token_ids: list[int],
    mapping: dict,
    remove_special_tokens: bool = True,
) -> list[int]:
    """
    Convert token IDs back to cluster IDs.

    Args:
        token_ids: List of token IDs
        mapping: Token mapping from create_cluster_to_token_mapping()
        remove_special_tokens: If True, remove BOS/EOS tokens

    Returns:
        List of cluster IDs
    """
    if remove_special_tokens:
        # Remove BOS/EOS
        token_ids = [
            tid for tid in token_ids
            if tid not in (mapping["bos_token_id"], mapping["eos_token_id"])
        ]

    return [mapping["token_id_to_cluster"][tid] for tid in token_ids]


def get_debug_tokens(
    token_ids: list[int],
    mapping: dict,
) -> list[str]:
    """
    Convert token IDs to human-readable token strings for debugging.

    Args:
        token_ids: List of token IDs
        mapping: Token mapping from create_cluster_to_token_mapping()

    Returns:
        List of token strings (including "<|endoftext|>" for BOS/EOS)
    """
    result = []
    for tid in token_ids:
        if tid == mapping["bos_token_id"]:
            result.append("<|endoftext|>")
        elif tid in mapping["token_id_to_cluster"]:
            cluster_id = mapping["token_id_to_cluster"][tid]
            result.append(mapping["cluster_to_token_str"][cluster_id])
        else:
            result.append(f"<unk:{tid}>")
    return result
