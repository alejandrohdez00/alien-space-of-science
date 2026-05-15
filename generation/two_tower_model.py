"""Two-tower availability model for cognitive availability scoring.

Operationalizes Evans' relational notion of cognitive availability: s(a, T)
scores whether community a is positioned to produce idea T.

Architecture:
- Shared atom embeddings between both towers
- Author tower: mean-pool + MLP to L2-normalized embedding
- Set tower: bidirectional transformer + mean-pool + projection to L2-normalized embedding
- Learnable logit scale for author-set scoring
"""

import math
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


class TwoTowerAvailabilityModel(nn.Module):
    """Two-tower model scoring author-idea compatibility.

    The author tower encodes an author's atom repertoire (mean-pool + MLP).
    The set tower encodes a candidate atom set (transformer + mean-pool).
    Scoring is cosine similarity scaled by a learnable logit scale.

    Args:
        vocab_size: Number of tokens (clusters + special tokens)
        d_model: Embedding dimension
        nhead: Number of attention heads
        num_layers: Number of transformer layers (set tower only)
        dim_feedforward: FFN inner dimension
        max_seq_len: Maximum sequence length
        dropout: Dropout rate
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 768,
        nhead: int = 12,
        num_layers: int = 6,
        dim_feedforward: int = 3072,
        max_seq_len: int = 128,
        dropout: float = 0.1,
        max_logit_scale: float = 100.0,
    ):
        super().__init__()
        self.max_logit_scale = max_logit_scale
        self.config = {
            "model_type": "two_tower",
            "vocab_size": vocab_size,
            "d_model": d_model,
            "nhead": nhead,
            "num_layers": num_layers,
            "dim_feedforward": dim_feedforward,
            "max_seq_len": max_seq_len,
            "dropout": dropout,
            "max_logit_scale": max_logit_scale,
        }

        # Shared atom embeddings
        self.atom_embedding = nn.Embedding(vocab_size, d_model)

        # --- Set tower: transformer + mean-pool + projection ---
        self.set_dropout = nn.Dropout(dropout)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.set_transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers
        )
        self.set_ln = nn.LayerNorm(d_model)
        self.set_projection = nn.Linear(d_model, d_model)

        # --- Author tower: mean-pool + MLP ---
        self.author_mlp = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )

        # logit_scale = 1 / temperature.
        # Init log(1 / 0.07) ~= 2.66, giving an initial scale of about 14.3.
        # Clipped to [0, log(max_logit_scale)].
        self.log_logit_scale = nn.Parameter(torch.tensor(math.log(1.0 / 0.07)))

        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    @property
    def logit_scale(self) -> torch.Tensor:
        """Clamped logit scale (1/temperature) for numerical stability."""
        clamped = torch.clamp(
            self.log_logit_scale, 0.0, math.log(self.max_logit_scale)
        )
        return clamped.exp()

    def _mean_pool(
        self, x: torch.Tensor, attention_mask: torch.Tensor | None
    ) -> torch.Tensor:
        """Mean pool over non-padding positions."""
        if attention_mask is not None:
            mask_expanded = attention_mask.unsqueeze(-1).float()  # (B, L, 1)
            return (x * mask_expanded).sum(dim=1) / mask_expanded.sum(dim=1)
        return x.mean(dim=1)

    def encode_sets(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Encode atom sets into L2-normalized embeddings.

        Args:
            input_ids: (B, seq_len) atom token IDs (no BOS/EOS)
            attention_mask: (B, seq_len) with 1=attend, 0=ignore

        Returns:
            embeddings: (B, d_model) L2-normalized
        """
        x = self.atom_embedding(input_ids)
        x = self.set_dropout(x)

        src_key_padding_mask = None
        if attention_mask is not None:
            src_key_padding_mask = attention_mask == 0

        # Bidirectional attention (no causal mask)
        x = self.set_transformer(x, src_key_padding_mask=src_key_padding_mask)
        x = self.set_ln(x)

        pooled = self._mean_pool(x, attention_mask)
        projected = self.set_projection(pooled)
        return F.normalize(projected, dim=-1)

    def encode_authors(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Encode author atom repertoires into L2-normalized embeddings.

        Args:
            input_ids: (B, seq_len) atom token IDs from author's papers (no BOS/EOS)
            attention_mask: (B, seq_len) with 1=attend, 0=ignore

        Returns:
            embeddings: (B, d_model) L2-normalized
        """
        x = self.atom_embedding(input_ids)
        pooled = self._mean_pool(x, attention_mask)
        projected = self.author_mlp(pooled)
        return F.normalize(projected, dim=-1)

    def forward(
        self,
        author_input_ids: torch.Tensor,
        author_attention_mask: torch.Tensor | None,
        set_input_ids: torch.Tensor,
        set_attention_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        """Compute similarity matrix between authors and sets.

        Args:
            author_input_ids: (B, author_seq_len) author atom IDs
            author_attention_mask: (B, author_seq_len)
            set_input_ids: (B, set_seq_len) candidate atom set IDs
            set_attention_mask: (B, set_seq_len)

        Returns:
            logits: (B, B) similarity matrix scaled by learnable logit scale
                    logits[i, j] = similarity(author_i, set_j) * logit_scale
        """
        author_embeds = self.encode_authors(author_input_ids, author_attention_mask)
        set_embeds = self.encode_sets(set_input_ids, set_attention_mask)

        # Cosine similarity (already L2-normalized) scaled by logit_scale
        # Higher logit_scale gives a sharper softmax distribution.
        logits = (author_embeds @ set_embeds.T) * self.logit_scale
        return logits

    def save(self, path: str | Path) -> None:
        """Save model config and state dict to directory."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        checkpoint = {
            "config": self.config,
            "state_dict": self.state_dict(),
        }
        torch.save(checkpoint, path / "model.pt")

    @classmethod
    def load(cls, path: str | Path, device: str = "cpu") -> "TwoTowerAvailabilityModel":
        """Load model from checkpoint directory."""
        path = Path(path)
        checkpoint = torch.load(
            path / "model.pt", map_location=device, weights_only=True
        )
        config = checkpoint["config"]
        init_config = {k: v for k, v in config.items() if k != "model_type"}
        model = cls(**init_config)
        model.load_state_dict(checkpoint["state_dict"])
        model.to(device)
        return model
