"""Dataset and collation utilities for pre-tokenized super-atom sequences."""

import array
import json
import random
import sys

import numpy as np
import torch
from torch.utils.data import Dataset

from generation.token_mapping import TokenMapping


class PreTokenizedDataset(Dataset):
    """Dataset for pre-tokenized JSONL files.

    Expects JSONL files where each line contains a JSON object with an
    'input_ids' field containing a list of token IDs.
    """

    def __init__(self, jsonl_path: str, id_remap: dict[int, int] | None = None):
        self.samples = []
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line)
                self.samples.append(data['input_ids'])

        if id_remap:
            self.samples = [
                [id_remap.get(tid, tid) for tid in seq]
                for seq in self.samples
            ]

        sys.stderr.write(f"Loaded {len(self.samples)} samples from {jsonl_path}\n")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return {"input_ids": torch.tensor(self.samples[idx], dtype=torch.long)}


def collate_fn(batch, pad_token_id: int = 0):
    """Collate function with padding for variable-length sequences.

    Args:
        batch: List of dicts with 'input_ids' tensors
        pad_token_id: Token ID to use for padding (default: 0)

    Returns:
        Dict with padded input_ids, attention_mask, and labels
    """
    input_ids = [item["input_ids"] for item in batch]

    # Pad sequences to max length in batch
    max_len = max(len(ids) for ids in input_ids)
    padded = torch.full((len(batch), max_len), pad_token_id, dtype=torch.long)
    attention_mask = torch.zeros(len(batch), max_len, dtype=torch.long)

    for i, ids in enumerate(input_ids):
        padded[i, :len(ids)] = ids
        attention_mask[i, :len(ids)] = 1

    # For causal LM, labels = input_ids (model shifts internally)
    # Set padding positions to -100 so they're ignored in loss
    labels = padded.clone()
    labels[attention_mask == 0] = -100

    return {
        "input_ids": padded,
        "attention_mask": attention_mask,
        "labels": labels,
    }


class TwoTowerDataset(Dataset):
    """Dataset for two-tower availability training.

    Reads a precomputed JSONL under schema ``subset_complement_v1``, where each
    record is one positive sample already paired with the author-tower input:

        {"author_id": str, "query_ids": [atom_ids],  # positive subset (length 2-4)
         "author_ids": [atom_ids], ...}              # author repertoire minus query

    Sample-level holdout is baked into the file (``query_ids`` and ``author_ids``
    are disjoint by construction), so no on-the-fly subset sampling is needed.
    Records with ``author_ids == []`` (query covers the author's entire
    repertoire) are dropped at load time, since the author tower needs at least
    one token.

    Optional training-time augmentation:
      - ``author_dropout_prob``: per-atom dropout on the author-tower input
        (default 0.0). Kept as an opt-in ablation knob; there is no anti-leakage
        role left because the schema already enforces strict complement holdout.
    """

    def __init__(
        self,
        jsonl_path: str,
        id_remap: dict[int, int] | None = None,
        *,
        build_author_atoms: bool = True,
    ):
        # Augmentation (set after construction for train only; default off).
        self.author_dropout_prob: float = 0.0

        # Tensorized storage. Each sample is represented by:
        #   author_idx_arr[i]         -> int32 author index
        #   query_atoms[query_offsets[i]:query_offsets[i+1]]    -> int32 atoms
        #   author_atoms_arr[author_offsets[i]:author_offsets[i+1]] -> int32 atoms
        # This collapses Python dict+list overhead (~780 B/record) to CSR arrays
        # (~20-30 B/record), letting persistent DataLoader workers share pages
        # without refcount-driven COW fan-out.
        self.author_id_to_idx: dict[str, int] = {}
        author_atoms_sets: dict[str, set[int]] | None = (
            {} if build_author_atoms else None
        )
        n_empty_complement = 0
        n_empty_query = 0

        # Use array.array during construction for 4-byte int storage instead of
        # Python int objects (~28 B each).
        author_idx_buf = array.array("i")
        query_atoms_buf = array.array("i")
        query_offsets_buf = array.array("i", [0])
        author_atoms_buf = array.array("i")
        author_offsets_buf = array.array("i", [0])

        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                data = json.loads(line)
                if "paper_id" in data:
                    raise ValueError(
                        f"Line {line_num} of {jsonl_path}: 'paper_id' field present. "
                        "TwoTowerDataset expects the subset_complement_v1 schema "
                        "({author_id, query_ids, author_ids}); regenerate with "
                        "the make-availability command."
                    )
                if "query_ids" not in data or "author_ids" not in data:
                    raise ValueError(
                        f"Line {line_num} of {jsonl_path}: missing 'query_ids' or "
                        "'author_ids'. Expected schema: subset_complement_v1."
                    )
                author_id = data["author_id"]
                query_ids = data["query_ids"]
                author_ids = data["author_ids"]
                if id_remap:
                    query_ids = [id_remap.get(tid, tid) for tid in query_ids]
                    author_ids = [id_remap.get(tid, tid) for tid in author_ids]
                if not query_ids:
                    n_empty_query += 1
                    continue
                if not author_ids:
                    n_empty_complement += 1
                    continue

                if author_id not in self.author_id_to_idx:
                    self.author_id_to_idx[author_id] = len(self.author_id_to_idx)
                author_idx = self.author_id_to_idx[author_id]

                author_idx_buf.append(author_idx)
                query_atoms_buf.extend(query_ids)
                query_offsets_buf.append(len(query_atoms_buf))
                author_atoms_buf.extend(author_ids)
                author_offsets_buf.append(len(author_atoms_buf))

                if author_atoms_sets is not None and author_id not in author_atoms_sets:
                    # Schema invariant: query_ids plus author_ids is the full
                    # repertoire and is identical for every record of an author.
                    author_atoms_sets[author_id] = set(query_ids) | set(author_ids)

        # Freeze into numpy arrays for zero-copy worker sharing.
        self.author_idx_arr = np.frombuffer(author_idx_buf, dtype=np.int32).copy()
        self.query_offsets = np.frombuffer(query_offsets_buf, dtype=np.int32).copy()
        self.query_atoms = np.frombuffer(query_atoms_buf, dtype=np.int32).copy()
        self.author_offsets = np.frombuffer(author_offsets_buf, dtype=np.int32).copy()
        self.author_atoms_arr = np.frombuffer(author_atoms_buf, dtype=np.int32).copy()
        # Release intermediate buffers.
        del author_idx_buf, query_atoms_buf, query_offsets_buf
        del author_atoms_buf, author_offsets_buf

        # Reverse lookup idx -> str for full-pool retrieval code paths that key
        # results by the original author_id string.
        self.idx_to_author_id: list[str] = [""] * len(self.author_id_to_idx)
        for aid, idx in self.author_id_to_idx.items():
            self.idx_to_author_id[idx] = aid

        if author_atoms_sets is not None:
            self.author_atoms: dict[str, list[int]] = {
                aid: sorted(atoms) for aid, atoms in author_atoms_sets.items()
            }
        else:
            self.author_atoms = {}

        n_kept = len(self.author_idx_arr)
        msg = (
            f"Loaded {n_kept} two-tower samples from {jsonl_path}\n"
            f"  Dropped: {n_empty_complement} empty-complement, "
            f"{n_empty_query} empty-query\n"
        )
        if author_atoms_sets is not None:
            atom_counts = [len(a) for a in self.author_atoms.values()]
            if atom_counts:
                msg += (
                    f"  Unique authors: {len(self.author_atoms)}, "
                    f"atoms/author: min={min(atom_counts)}, "
                    f"max={max(atom_counts)}, "
                    f"avg={sum(atom_counts) / len(atom_counts):.1f}\n"
                )
        sys.stderr.write(msg)

    def __len__(self):
        return len(self.author_idx_arr)

    def get_author_id(self, idx: int) -> str:
        """Return the author_id string for sample ``idx``."""
        return self.idx_to_author_id[int(self.author_idx_arr[idx])]

    def __getitem__(self, idx):
        aidx = int(self.author_idx_arr[idx])
        q_start = int(self.query_offsets[idx])
        q_end = int(self.query_offsets[idx + 1])
        a_start = int(self.author_offsets[idx])
        a_end = int(self.author_offsets[idx + 1])
        set_atoms = self.query_atoms[q_start:q_end].tolist()
        author_atoms = self.author_atoms_arr[a_start:a_end].tolist()

        # Author-side dropout: per-atom dropout with a floor to avoid degenerate
        # single-atom inputs. Default prob is 0.0 (no-op).
        if self.author_dropout_prob > 0 and len(author_atoms) > 1:
            kept = [a for a in author_atoms if random.random() > self.author_dropout_prob]
            floor = max(3, len(author_atoms) // 4)
            if len(kept) < floor:
                kept = random.sample(author_atoms, min(floor, len(author_atoms)))
            author_atoms = kept

        return {
            "author_atoms": author_atoms,
            "set_atoms": set_atoms,
            "author_id": self.idx_to_author_id[aidx],
            "author_idx": aidx,
        }

    def get_all_author_atoms(self) -> dict[str, list[int]]:
        """Return full atom repertoire per author (only populated when built)."""
        return {aid: list(atoms) for aid, atoms in self.author_atoms.items()}

    def get_author_ids(self) -> list[str]:
        """Return sorted list of all author IDs seen in this file."""
        if self.author_atoms:
            return sorted(self.author_atoms.keys())
        return sorted(self.author_id_to_idx.keys())


def two_tower_collate_fn(batch, pad_token_id: int):
    """Collate function for two-tower availability training.

    Pads author and set atom sequences. In-batch alternatives are provided by
    the batch itself, so no explicit negative generation is needed.

    Args:
        batch: List of dicts with 'author_atoms' and 'set_atoms'
        pad_token_id: Token ID for padding

    Returns:
        Dict with author_input_ids, author_attention_mask,
        set_input_ids, set_attention_mask
    """
    B = len(batch)

    # Pad author atoms
    author_seqs = [item["author_atoms"] for item in batch]
    max_author_len = max(len(s) for s in author_seqs) if author_seqs else 1
    author_input_ids = torch.full((B, max_author_len), pad_token_id, dtype=torch.long)
    author_attention_mask = torch.zeros(B, max_author_len, dtype=torch.long)
    for i, seq in enumerate(author_seqs):
        if seq:
            author_input_ids[i, : len(seq)] = torch.tensor(seq, dtype=torch.long)
            author_attention_mask[i, : len(seq)] = 1

    # Pad set atoms
    set_seqs = [item["set_atoms"] for item in batch]
    max_set_len = max(len(s) for s in set_seqs)
    set_input_ids = torch.full((B, max_set_len), pad_token_id, dtype=torch.long)
    set_attention_mask = torch.zeros(B, max_set_len, dtype=torch.long)
    for i, seq in enumerate(set_seqs):
        set_input_ids[i, : len(seq)] = torch.tensor(seq, dtype=torch.long)
        set_attention_mask[i, : len(seq)] = 1

    author_idx = torch.tensor(
        [item["author_idx"] for item in batch], dtype=torch.long
    )

    return {
        "author_input_ids": author_input_ids,
        "author_attention_mask": author_attention_mask,
        "set_input_ids": set_input_ids,
        "set_attention_mask": set_attention_mask,
        "author_idx": author_idx,
    }


def load_token_mapping(path: str) -> TokenMapping:
    """Load a TokenMapping from a JSON file path.

    Convenience wrapper around TokenMapping.load().
    """
    return TokenMapping.load(path)
