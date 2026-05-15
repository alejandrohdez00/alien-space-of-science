"""Model loading and scoring for coherence and two-tower availability."""

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from generation.model import CausalTransformerLM
from generation.token_mapping import TokenMapping
from generation.two_tower_model import TwoTowerAvailabilityModel


def load_model(model_dir: str) -> tuple[CausalTransformerLM, TokenMapping]:
    """Load the autoregressive coherence model and its token mapping."""
    model_path = Path(model_dir)
    model = CausalTransformerLM.load(str(model_path))
    model.eval()
    mapping = TokenMapping.load(model_path / "token_mapping.json")
    return model, mapping


def score_sequences_batched(
    model: CausalTransformerLM,
    all_token_ids: list[list[int]],
    token_mapping: TokenMapping,
    device: str = "cuda",
    batch_size: int = 512,
) -> list[float]:
    """Score ordered token sequences by average per-atom negative log-likelihood."""
    if not all_token_ids:
        return []

    eos_token_id = token_mapping.eos_token_id
    pad_token_id = eos_token_id
    model = model.to(device)
    all_avg_nlls = []

    for batch_start in range(0, len(all_token_ids), batch_size):
        batch_token_ids = all_token_ids[batch_start:batch_start + batch_size]
        current_batch_size = len(batch_token_ids)
        max_len = max(len(seq) for seq in batch_token_ids)
        lengths = [len(seq) for seq in batch_token_ids]

        padded = torch.full(
            (current_batch_size, max_len),
            pad_token_id,
            dtype=torch.long,
        )
        attention_mask = torch.zeros(
            (current_batch_size, max_len),
            dtype=torch.long,
        )
        for i, seq in enumerate(batch_token_ids):
            padded[i, :len(seq)] = torch.tensor(seq)
            attention_mask[i, :len(seq)] = 1

        padded = padded.to(device)
        attention_mask = attention_mask.to(device)

        with torch.no_grad():
            logits = model(padded, attention_mask=attention_mask)

        log_probs = F.log_softmax(logits[:, :-1, :], dim=-1)
        shift_labels = padded[:, 1:]
        token_nlls = -log_probs.gather(2, shift_labels.unsqueeze(-1)).squeeze(-1)

        nll_mask = torch.zeros(
            (current_batch_size, max_len - 1),
            dtype=torch.float32,
            device=device,
        )
        for i, (seq, length) in enumerate(zip(batch_token_ids, lengths)):
            n_valid = length - 2 if seq[-1] == eos_token_id else length - 1
            if n_valid > 0:
                nll_mask[i, :n_valid] = 1.0

        masked_nlls = token_nlls * nll_mask
        sum_nlls = masked_nlls.sum(dim=1)
        count_valid = nll_mask.sum(dim=1)
        all_avg_nlls.extend((sum_nlls / count_valid.clamp(min=1)).tolist())

        del padded, attention_mask, logits, log_probs, token_nlls, nll_mask
        if device == "cuda":
            torch.cuda.empty_cache()

    return all_avg_nlls


def load_availability_model(
    model_dir: str,
) -> tuple[TwoTowerAvailabilityModel, TokenMapping]:
    """Load the official two-tower availability model and its token mapping."""
    model_path = Path(model_dir)
    checkpoint = torch.load(
        model_path / "model.pt",
        map_location="cpu",
        weights_only=True,
    )
    model_type = checkpoint["config"].get("model_type")
    if model_type != "two_tower":
        raise ValueError(
            f"Expected a two_tower availability model in {model_dir}, "
            f"found {model_type!r}."
        )

    model = TwoTowerAvailabilityModel.load(str(model_path))
    model.eval()
    mapping = TokenMapping.load(model_path / "token_mapping.json")
    return model, mapping


def load_author_embeddings(
    model_dir: str,
    device: str = "cpu",
) -> tuple[torch.Tensor, list[str]]:
    """Load precomputed author embeddings saved with the availability model."""
    model_path = Path(model_dir)
    embeds = torch.load(
        model_path / "author_embeddings.pt",
        map_location=device,
        weights_only=True,
    )
    with open(model_path / "author_id_to_index.json", "r", encoding="utf-8") as f:
        id_to_idx = json.load(f)
    author_ids = sorted(id_to_idx.keys(), key=lambda k: id_to_idx[k])
    return embeds, author_ids


def score_sequences_two_tower(
    model: TwoTowerAvailabilityModel,
    all_token_ids: list[list[int]],
    author_embeddings: torch.Tensor,
    token_mapping: TokenMapping,
    device: str = "cuda",
    batch_size: int = 512,
    top_k: int = 10,
) -> list[float]:
    """Score atom sets by their top-k median author compatibility."""
    if not all_token_ids:
        return []

    bos = token_mapping.bos_token_id
    eos = token_mapping.eos_token_id
    pad_token_id = eos
    atom_sequences = [
        [t for t in seq if t != bos and t != eos]
        for seq in all_token_ids
    ]

    model = model.to(device)
    author_embeddings = author_embeddings.to(device)
    all_scores = []

    for batch_start in range(0, len(atom_sequences), batch_size):
        batch_seqs = atom_sequences[batch_start:batch_start + batch_size]
        current_batch_size = len(batch_seqs)
        max_len = max(len(seq) for seq in batch_seqs)

        padded = torch.full(
            (current_batch_size, max_len),
            pad_token_id,
            dtype=torch.long,
        )
        attention_mask = torch.zeros(
            (current_batch_size, max_len),
            dtype=torch.long,
        )
        for i, seq in enumerate(batch_seqs):
            padded[i, :len(seq)] = torch.tensor(seq)
            attention_mask[i, :len(seq)] = 1

        padded = padded.to(device)
        attention_mask = attention_mask.to(device)

        with torch.no_grad():
            set_embeds = model.encode_sets(padded, attention_mask)
            scores = set_embeds @ author_embeddings.T
            k = min(top_k, scores.size(1))
            topk_scores = scores.topk(k, dim=1).values
            if k % 2 == 1:
                batch_availability = topk_scores[:, k // 2]
            else:
                batch_availability = (
                    topk_scores[:, k // 2 - 1] + topk_scores[:, k // 2]
                ) / 2

        all_scores.extend(batch_availability.tolist())

        del padded, attention_mask, set_embeds, scores, topk_scores
        if device == "cuda":
            torch.cuda.empty_cache()

    return all_scores


def score_sets_alien(
    sets: list[tuple[int, ...]],
    model: TwoTowerAvailabilityModel,
    token_mapping: TokenMapping,
    author_embeddings: torch.Tensor,
    device: str = "cuda",
    batch_size: int = 512,
    top_k_authors: int = 10,
    chunk_size: int = 200_000,
) -> np.ndarray:
    """Score unordered atom sets as unavailability: higher means more alien."""
    if not sets:
        return np.zeros(0)

    scores = np.zeros(len(sets), dtype=np.float64)
    for chunk_start in range(0, len(sets), chunk_size):
        chunk = sets[chunk_start:chunk_start + chunk_size]
        seqs = [token_mapping.cluster_ids_to_token_ids(list(s)) for s in chunk]
        availability = score_sequences_two_tower(
            model,
            seqs,
            author_embeddings,
            token_mapping,
            device=device,
            batch_size=batch_size,
            top_k=top_k_authors,
        )
        scores[chunk_start:chunk_start + len(chunk)] = -np.asarray(
            availability,
            dtype=np.float64,
        )

    return scores
