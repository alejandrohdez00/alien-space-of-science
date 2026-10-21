"""Read-only memory-mapped records shared by training ranks and workers."""
import json
from pathlib import Path

import numpy as np
import torch

# Preserve the generated order of both queries and complements. Offsets must
# be 64-bit: the corpus can exceed 2**31 complement tokens.
ROW = np.dtype([('author', '<u4'), ('query', '<u2', (4,)),
                ('length', 'u1'), ('offset', '<u8'), ('count', '<u2'),
                ('sample', '<u4')])


class Corpus:
    def __init__(self, directory):
        self.path = Path(directory)
        self.meta = json.loads((self.path / 'metadata.json').read_text())
        self.mapping = json.loads((self.path / 'token_mapping.json').read_text())
        self.pad = self.mapping['n_clusters']
        self.rows = np.memmap(self.path / 'rows.bin', mode='r', dtype=ROW)
        cp = self.path / 'complements.bin'
        self.complements = (np.memmap(cp, mode='r', dtype='<u2')
                            if cp.stat().st_size else np.empty(0, dtype='<u2'))
        self.profiles = json.loads((self.path / 'profiles.json').read_text())
        order_path = self.path / 'author_ids.json'
        self.author_ids = (json.loads(order_path.read_text()) if order_path.exists()
                           else sorted(self.profiles))
        if len(self.author_ids) != len(self.profiles) or set(self.author_ids) != set(self.profiles):
            raise ValueError('author_ids.json must list every profile exactly once')
        self.train = np.load(self.path / 'train.npy', mmap_mode='r')
        self.val = np.load(self.path / 'val.npy', mmap_mode='r')
        self.training_authors = np.load(self.path / 'training_authors.npy')

    def batch(self, indices):
        rows = self.rows[np.asarray(indices, dtype=np.int64)]
        qlen = rows['length'].astype(np.int64)
        q = rows['query'].astype(np.int64)
        qm = np.arange(4)[None, :] < qlen[:, None]
        q[~qm] = self.pad
        counts = rows['count'].astype(np.int64)
        width = max(1, int(counts.max()))
        mask = np.arange(width)[None, :] < counts[:, None]
        offsets = rows['offset'].astype(np.int64)[:, None] + np.arange(width)
        offsets[~mask] = 0
        authors = (self.complements[offsets].astype(np.int64) if len(self.complements)
                   else np.full(offsets.shape, self.pad, dtype=np.int64))
        authors[~mask] = self.pad
        return {k: torch.from_numpy(v) for k, v in {
            'author_input_ids': authors, 'author_attention_mask': mask.astype(np.int64),
            'set_input_ids': q, 'set_attention_mask': qm.astype(np.int64),
            'author_idx': rows['author'].astype(np.int64),
        }.items()}

    def author_batch(self, indices):
        profiles = [self.profiles[self.author_ids[int(i)]]['tokens'] for i in indices]
        width = max(map(len, profiles))
        ids = torch.full((len(profiles), width), self.pad, dtype=torch.long)
        mask = torch.zeros_like(ids)
        for i, atoms in enumerate(profiles):
            ids[i, :len(atoms)] = torch.tensor(atoms)
            mask[i, :len(atoms)] = 1
        return ids, mask


class BatchDataset(torch.utils.data.Dataset):
    """A DataLoader item is a complete vectorized batch, not a Python row."""
    def __init__(self, corpus, indices, batch_size, rank=0, world=1, drop_last=False):
        self.corpus, self.indices = corpus, indices
        self.batch_size, self.rank, self.world = batch_size, rank, world
        self.drop_last = drop_last

    def __len__(self):
        n = len(self.indices)
        return n // self.batch_size if self.drop_last else (n + self.batch_size - 1) // self.batch_size

    def __getitem__(self, batch):
        selected = self.indices[batch * self.batch_size:(batch + 1) * self.batch_size]
        # Training uses full global batches; validation is partitioned by batch.
        selected = selected[self.rank::self.world]
        if not len(selected):
            return None
        return self.corpus.batch(selected)
