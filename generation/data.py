"""Dataset and collation utilities for pre-tokenized super-atom sequences."""

import json
import sys

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

    # The training loop shifts these labels for next-token prediction.
    # Set padding positions to -100 so they're ignored in loss
    labels = padded.clone()
    labels[attention_mask == 0] = -100

    return {
        "input_ids": padded,
        "attention_mask": attention_mask,
        "labels": labels,
    }


def load_token_mapping(path: str) -> TokenMapping:
    """Load a TokenMapping from a JSON file path.

    Convenience wrapper around TokenMapping.load().
    """
    return TokenMapping.load(path)
