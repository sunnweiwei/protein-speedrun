"""The only file model researchers normally need to edit."""

from __future__ import annotations

from typing import Any

import torch
from torch import nn


class ProteinTransformer(nn.Module):
    def __init__(self, config: dict[str, Any]):
        super().__init__()
        width = int(config["width"])
        self.max_length = int(config["max_length"])
        self.attention = str(config["attention"])
        self.token_embedding = nn.Embedding(int(config["vocab_size"]), width)
        self.position_embedding = nn.Embedding(self.max_length, width)
        layer = nn.TransformerEncoderLayer(
            d_model=width,
            nhead=int(config["heads"]),
            dim_feedforward=width * int(config["ffn_multiplier"]),
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            layer,
            num_layers=int(config["layers"]),
            norm=nn.LayerNorm(width),
            enable_nested_tensor=False,
        )
        self.lm_head = nn.Linear(width, int(config["vocab_size"]), bias=False)
        self.lm_head.weight = self.token_embedding.weight

    def encode(
        self, tokens: torch.Tensor, padding_mask: torch.Tensor
    ) -> torch.Tensor:
        length = tokens.shape[1]
        if length > self.max_length:
            raise ValueError("sequence is longer than model.max_length")
        positions = torch.arange(length, device=tokens.device)
        hidden = self.token_embedding(tokens)
        hidden = hidden + self.position_embedding(positions)[None]
        attention_mask = None
        if self.attention == "causal":
            attention_mask = torch.triu(
                torch.ones(length, length, dtype=torch.bool, device=tokens.device),
                diagonal=1,
            )
        return self.encoder(
            hidden,
            mask=attention_mask,
            src_key_padding_mask=~padding_mask,
        )

    def logits(
        self, tokens: torch.Tensor, padding_mask: torch.Tensor
    ) -> torch.Tensor:
        return self.lm_head(self.encode(tokens, padding_mask))


def build_model(config: dict[str, Any]) -> nn.Module:
    """Checkpoint contract used by the external evaluator."""
    return ProteinTransformer(config)


def build_optimizer(
    model: nn.Module, config: dict[str, Any]
) -> torch.optim.Optimizer:
    return torch.optim.AdamW(
        model.parameters(),
        lr=float(config["learning_rate"]),
        weight_decay=float(config["weight_decay"]),
        betas=(0.9, 0.95),
    )


def training_loss(
    model: nn.Module,
    tokens: torch.Tensor,
    padding_mask: torch.Tensor,
    config: dict[str, Any],
    generator: torch.Generator,
) -> torch.Tensor:
    """Replace this function to experiment with a different objective."""
    objective = str(config["objective"])
    if objective == "mlm":
        selected = (
            torch.rand(tokens.shape, device=tokens.device, generator=generator)
            < float(config["mask_probability"])
        ) & padding_mask
        empty = ~selected.any(dim=1)
        selected[empty, 0] = True
        corrupted = tokens.clone()
        corrupted[selected] = int(config["mask_token_id"])
        return nn.functional.cross_entropy(
            model.logits(corrupted, padding_mask)[selected],
            tokens[selected],
        )
    if objective == "causal":
        logits = model.logits(tokens[:, :-1], padding_mask[:, :-1])
        valid = padding_mask[:, 1:]
        return nn.functional.cross_entropy(logits[valid], tokens[:, 1:][valid])
    raise ValueError(f"unknown objective: {objective}")
