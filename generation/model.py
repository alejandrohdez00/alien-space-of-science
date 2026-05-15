"""Causal Transformer language model for super-atom sequence modeling.

Decoder-only causal LM using nn.TransformerEncoder with learned positional
embeddings.
"""

from pathlib import Path

import torch
import torch.nn as nn


class CausalTransformerLM(nn.Module):
    """Decoder-only causal language model using nn.TransformerEncoder.

    Uses nn.TransformerEncoder (not TransformerDecoder) because decoder-only
    models only need self-attention with a causal mask — cross-attention from
    TransformerDecoder is unnecessary here.

    Args:
        vocab_size: Number of tokens (clusters + special tokens)
        d_model: Embedding dimension
        nhead: Number of attention heads
        num_layers: Number of transformer layers
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
    ):
        super().__init__()
        self.config = {
            "vocab_size": vocab_size,
            "d_model": d_model,
            "nhead": nhead,
            "num_layers": num_layers,
            "dim_feedforward": dim_feedforward,
            "max_seq_len": max_seq_len,
            "dropout": dropout,
        }

        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(max_seq_len, d_model)
        self.embedding_dropout = nn.Dropout(dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers
        )
        self.ln_f = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

        # Pre-compute causal mask and register as buffer (auto device placement)
        self.register_buffer(
            "causal_mask",
            torch.triu(torch.ones(max_seq_len, max_seq_len, dtype=torch.bool), diagonal=1),
        )

        # Weight tying: share token embedding weights with LM head
        self.lm_head.weight = self.token_embedding.weight

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

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass returning logits.

        Args:
            input_ids: (batch, seq_len) token IDs
            attention_mask: (batch, seq_len) with 1=attend, 0=ignore

        Returns:
            logits: (batch, seq_len, vocab_size)
        """
        seq_len = input_ids.size(1)
        positions = torch.arange(seq_len, device=input_ids.device).unsqueeze(0)

        x = self.token_embedding(input_ids) + self.position_embedding(positions)
        x = self.embedding_dropout(x)

        # Padding mask: True = ignore (inverted from attention_mask convention)
        src_key_padding_mask = None
        if attention_mask is not None:
            src_key_padding_mask = attention_mask == 0

        x = self.transformer(
            x, mask=self.causal_mask[:seq_len, :seq_len], is_causal=True,
            src_key_padding_mask=src_key_padding_mask,
        )
        x = self.ln_f(x)
        logits = self.lm_head(x)
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
    def load(cls, path: str | Path, device: str = "cpu") -> "CausalTransformerLM":
        """Load model from checkpoint directory."""
        path = Path(path)
        checkpoint = torch.load(path / "model.pt", map_location=device, weights_only=True)
        model = cls(**checkpoint["config"])
        model.load_state_dict(checkpoint["state_dict"])
        model.to(device)
        return model
