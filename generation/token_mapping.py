"""Typed wrapper around the raw token_mapping dict.

Eliminates repeated token_mapping.get('bos_token_id', 50256) boilerplate
and provides typed access to token mapping fields.
"""

import json
from pathlib import Path
from typing import List


class TokenMapping:
    """Typed wrapper for the token_mapping JSON dict.

    Attributes:
        bos_token_id: Beginning-of-sequence token ID
        eos_token_id: End-of-sequence token ID
        n_clusters: Number of clusters in the mapping
    """

    def __init__(self, raw: dict):
        self._raw = raw
        self.bos_token_id: int = raw.get('bos_token_id', 50256)
        self.eos_token_id: int = raw.get('eos_token_id', 50256)

        n = raw.get('n_clusters')
        if n is None:
            n = len(raw.get('cluster_to_token_id', {}))
        self.n_clusters: int = n

        self._token_to_cluster: dict = raw.get('token_id_to_cluster', {})
        self._cluster_to_token: dict = raw.get('cluster_to_token_id', {})

    @classmethod
    def load(cls, path: str | Path) -> "TokenMapping":
        """Load a TokenMapping from a JSON file."""
        with open(path, 'r', encoding='utf-8') as f:
            return cls(json.load(f))

    @property
    def raw(self) -> dict:
        """Access the underlying raw dict."""
        return self._raw

    def token_ids_to_cluster_ids(self, token_ids: List[int]) -> List[int]:
        """Convert token IDs to cluster IDs, skipping BOS/EOS.

        Args:
            token_ids: Full token sequence (may include BOS/EOS)

        Returns:
            List of cluster IDs (BOS/EOS stripped, unmapped tokens skipped)

        Raises:
            ValueError: If any non-special token is unmapped
        """
        clusters = []
        for tid in token_ids:
            if tid == self.bos_token_id or tid == self.eos_token_id:
                continue
            if str(tid) in self._token_to_cluster:
                clusters.append(int(self._token_to_cluster[str(tid)]))
            else:
                raise ValueError(f"Unmapped token ID: {tid}")
        return clusters

    def cluster_ids_to_token_ids(self, cluster_ids: List[int]) -> List[int]:
        """Convert cluster IDs to a full token sequence [BOS, ..., EOS].

        Args:
            cluster_ids: List of cluster IDs

        Returns:
            Token IDs with BOS prepended and EOS appended
        """
        token_ids = [self.bos_token_id]
        for cid in cluster_ids:
            token_ids.append(int(self._cluster_to_token[str(cid)]))
        token_ids.append(self.eos_token_id)
        return token_ids

    def compact(self) -> tuple["TokenMapping", dict[int, int]]:
        """Return a compacted mapping where BOS/EOS = n_clusters.

        If BOS/EOS are already <= n_clusters, return (self, {}) as a no-op.
        Otherwise, build a new TokenMapping with bos_token_id = eos_token_id =
        n_clusters and return {original_bos: n_clusters} as the remap dict.
        """
        if self.bos_token_id <= self.n_clusters and self.eos_token_id <= self.n_clusters:
            return self, {}

        new_special = self.n_clusters
        raw = dict(self._raw)
        raw['bos_token_id'] = new_special
        raw['eos_token_id'] = new_special
        remap = {self.bos_token_id: new_special}
        if self.eos_token_id != self.bos_token_id:
            remap[self.eos_token_id] = new_special
        return TokenMapping(raw), remap

    def compute_vocab_size(self) -> int:
        """Compute vocab size from the mapping.

        Accounts for all unique token IDs (clusters + BOS/EOS).
        """
        all_token_ids = set(int(tid) for tid in self._token_to_cluster.keys())
        all_token_ids.add(self.bos_token_id)
        all_token_ids.add(self.eos_token_id)
        return len(all_token_ids)
