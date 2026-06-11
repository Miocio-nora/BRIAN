from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from brian_sphere_llm.model.llama_backbone import (
    BackboneConfig,
    RMSNorm,
    TransformerBlock,
    build_causal_lm_loss,
    count_parameters,
    require_torch,
)

try:
    import torch
    import torch.nn as nn
except ModuleNotFoundError:  # pragma: no cover
    torch = None
    nn = None

ModuleBase = nn.Module if nn is not None else object


@dataclass(frozen=True)
class BaselineConfig:
    vocab_size: int
    context_length: int
    layers: int
    d_model: int
    n_heads: int
    dropout: float = 0.0

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "BaselineConfig":
        return cls(
            vocab_size=int(data["vocab_size"]),
            context_length=int(data["context_length"]),
            layers=int(data["layers"]),
            d_model=int(data["d_model"]),
            n_heads=int(data["n_heads"]),
            dropout=float(data.get("dropout", 0.0)),
        )

    def backbone(self) -> BackboneConfig:
        return BackboneConfig(
            vocab_size=self.vocab_size,
            context_length=self.context_length,
            layers=self.layers,
            d_model=self.d_model,
            n_heads=self.n_heads,
            dropout=self.dropout,
        )


class BaselineLM(ModuleBase):
    def __init__(self, config: BaselineConfig) -> None:
        require_torch()
        super().__init__()
        self.config = config
        backbone = config.backbone()
        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.blocks = nn.ModuleList([TransformerBlock(backbone) for _ in range(config.layers)])
        self.norm = RMSNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.lm_head.weight = self.token_embedding.weight

    def forward(self, input_ids: torch.Tensor, targets: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        hidden = self.token_embedding(input_ids)
        for block in self.blocks:
            hidden = block(hidden)
        logits = self.lm_head(self.norm(hidden))
        output = {"logits": logits}
        if targets is not None:
            output["loss"] = build_causal_lm_loss(logits, targets)
        return output

    def model_stats(self) -> dict[str, int | str]:
        return {
            "model_name": "baseline",
            "parameter_count": count_parameters(self),
            "layers": self.config.layers,
            "d_model": self.config.d_model,
            "n_heads": self.config.n_heads,
        }
