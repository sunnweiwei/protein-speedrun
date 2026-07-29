"""Editable reference protein sequence encoder."""

from __future__ import annotations

from typing import Any

import torch
from torch import nn


class ReferenceProteinEncoder(nn.Module):
    def __init__(self, config: dict[str, Any]):
        super().__init__()
        self.config = config
        vocab_size = int(config["vocab_size"])
        d_model = int(config["d_model"])
        max_length = int(config["max_length"])
        num_heads = int(config["num_heads"])
        if d_model % num_heads:
            raise ValueError("d_model must be divisible by num_heads")
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(max_length, d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=d_model * int(config["ffn_multiplier"]),
            dropout=float(config["dropout"]),
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            layer,
            num_layers=int(config["num_layers"]),
            norm=nn.LayerNorm(d_model),
            enable_nested_tensor=False,
        )
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.lm_head.weight = self.token_embedding.weight
        self.attention_mode = str(config["attention_mode"])
        self.max_length = max_length

    def _attention_mask(self, length: int, device: torch.device) -> torch.Tensor | None:
        if self.attention_mode == "bidirectional":
            return None
        if self.attention_mode == "causal":
            return torch.triu(
                torch.ones(length, length, dtype=torch.bool, device=device),
                diagonal=1,
            )
        raise ValueError(f"unknown attention_mode: {self.attention_mode}")

    def encode(
        self, tokens: torch.Tensor, padding_mask: torch.Tensor
    ) -> torch.Tensor:
        if tokens.ndim != 2 or padding_mask.shape != tokens.shape:
            raise ValueError("tokens and padding_mask must have shape [B, L]")
        if tokens.shape[1] > self.max_length:
            raise ValueError("sequence exceeds configured max_length")
        positions = torch.arange(tokens.shape[1], device=tokens.device)
        hidden = self.token_embedding(tokens) + self.position_embedding(positions)[None]
        return self.encoder(
            hidden,
            mask=self._attention_mask(tokens.shape[1], tokens.device),
            src_key_padding_mask=~padding_mask,
        )

    def logits(
        self, tokens: torch.Tensor, padding_mask: torch.Tensor
    ) -> torch.Tensor:
        return self.lm_head(self.encode(tokens, padding_mask))


def build_model(model_config: dict[str, Any]) -> nn.Module:
    return ReferenceProteinEncoder(model_config)

