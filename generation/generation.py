"""Sequence generation via model sampling and random baselines."""

import random
import sys
from typing import List

import torch
import torch.nn.functional as F

from generation.model import CausalTransformerLM
from generation.token_mapping import TokenMapping


def generate_sequences(
    model: CausalTransformerLM,
    token_mapping: TokenMapping,
    num_samples: int = 100,
    max_length: int = 20,
    temperature: float = 1.0,
    top_k: int = 50,
    device: str = None,
) -> List[dict]:
    """Generate sequences from the model using top-k sampling.

    Args:
        model: Trained Transformer model
        token_mapping: TokenMapping instance
        num_samples: Number of sequences to generate
        max_length: Maximum sequence length
        temperature: Sampling temperature
        top_k: Top-k sampling parameter
        device: Device to run on

    Returns:
        List of dicts with 'token_ids', 'clusters', 'length'
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model = model.to(device)

    bos_token_id = token_mapping.bos_token_id
    eos_token_id = token_mapping.eos_token_id

    generated_sequences = []
    filtered_count = 0

    for i in range(num_samples):
        input_ids = [bos_token_id]

        with torch.no_grad():
            for step in range(max_length - 1):
                input_tensor = torch.tensor([input_ids], device=device)
                logits = model(input_tensor)
                next_logits = logits[0, -1, :].clone()

                # Suppress EOS to force fixed-length sequences
                next_logits[eos_token_id] = float('-inf')

                # Temperature scaling
                next_logits = next_logits / temperature

                # Top-k filtering
                if top_k > 0:
                    top_k_values, _ = torch.topk(next_logits, top_k)
                    threshold = top_k_values[-1]
                    next_logits[next_logits < threshold] = float('-inf')

                # Defensive check: if all logits are -inf, uniform over non-EOS
                if torch.isinf(next_logits).all():
                    next_logits = torch.zeros_like(next_logits)
                    next_logits[eos_token_id] = float('-inf')

                probs = F.softmax(next_logits, dim=-1)
                next_token = torch.multinomial(probs, 1).item()
                input_ids.append(next_token)

        token_ids = input_ids

        # Convert token IDs to cluster IDs (skip BOS/EOS)
        # Filter out sequences with any unmapped tokens
        try:
            clusters = token_mapping.token_ids_to_cluster_ids(token_ids)
        except ValueError:
            filtered_count += 1
            continue

        if not clusters:
            filtered_count += 1
            continue

        generated_sequences.append({
            'token_ids': token_ids,
            'clusters': clusters,
            'length': len(clusters),
        })

    if filtered_count > 0:
        sys.stderr.write(f"  Filtered {filtered_count} sequences with unmapped tokens\n")

    return generated_sequences


def generate_random_sequences(
    token_mapping: TokenMapping,
    num_samples: int = 100,
    length: int = 4,
    unique_only: bool = True,
) -> List[dict]:
    """Generate random sequences by uniformly sampling cluster IDs.

    Samples exactly `length` cluster IDs uniformly without replacement.
    This provides a clean baseline for exploring the concept space, separating
    exploration (random) from evaluation (model scoring).

    Args:
        token_mapping: TokenMapping instance
        num_samples: Number of sequences to generate
        length: Exact number of clusters per sequence
        unique_only: If True, ensure no duplicate sequences (default: True)

    Returns:
        List of dicts with 'token_ids', 'clusters', 'length'
    """
    n_clusters = token_mapping.n_clusters

    if length > n_clusters:
        raise ValueError(
            f"length ({length}) cannot exceed n_clusters ({n_clusters}) "
            f"when sampling without replacement"
        )

    all_cluster_ids = list(range(n_clusters))
    generated_sequences = []
    seen_sequences = set()

    max_attempts = num_samples * 10  # Prevent infinite loop
    attempts = 0

    while len(generated_sequences) < num_samples and attempts < max_attempts:
        attempts += 1

        # Sample exactly `length` clusters without replacement
        cluster_ids = random.sample(all_cluster_ids, length)

        # Check for uniqueness if required
        if unique_only:
            seq_tuple = tuple(cluster_ids)
            if seq_tuple in seen_sequences:
                continue
            seen_sequences.add(seq_tuple)

        # Convert cluster IDs to token IDs
        token_ids = token_mapping.cluster_ids_to_token_ids(cluster_ids)

        generated_sequences.append({
            'token_ids': token_ids,
            'clusters': cluster_ids,
            'length': len(cluster_ids),
        })

    if len(generated_sequences) < num_samples:
        sys.stderr.write(
            f"  Warning: Only generated {len(generated_sequences)} unique sequences "
            f"(requested {num_samples})\n"
        )

    return generated_sequences
