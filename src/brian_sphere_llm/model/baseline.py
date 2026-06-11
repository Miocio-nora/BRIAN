from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

from brian_sphere_llm.model.llama_backbone import (
    BackboneConfig,
    RMSNorm,
    TransformerBlock,
    build_causal_lm_loss,
    checkpoint_if_enabled,
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
    model_name: str = "baseline"

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "BaselineConfig":
        return cls(
            vocab_size=_int_value(data["vocab_size"], "vocab_size", minimum=1),
            context_length=_int_value(data["context_length"], "context_length", minimum=1),
            layers=_int_value(data["layers"], "layers", minimum=1),
            d_model=_int_value(data["d_model"], "d_model", minimum=1),
            n_heads=_int_value(data["n_heads"], "n_heads", minimum=1),
            dropout=_float_value(data.get("dropout", 0.0), "dropout", minimum=0.0, maximum=1.0),
            model_name=str(data.get("model_name", "baseline")),
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
        self.activation_checkpointing = False
        backbone = config.backbone()
        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.blocks = nn.ModuleList([TransformerBlock(backbone) for _ in range(config.layers)])
        self.norm = RMSNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.lm_head.weight = self.token_embedding.weight

    def forward(self, input_ids: torch.Tensor, targets: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        hidden = self.token_embedding(input_ids)
        for block in self.blocks:
            hidden = checkpoint_if_enabled(self, block, hidden)
        logits = self.lm_head(self.norm(hidden))
        output = {"logits": logits}
        if targets is not None:
            output["loss"] = build_causal_lm_loss(logits, targets)
        return output

    def model_stats(self) -> dict[str, int | str]:
        return {
            "model_name": self.config.model_name,
            "parameter_count": count_parameters(self),
            "layers": self.config.layers,
            "d_model": self.config.d_model,
            "n_heads": self.config.n_heads,
        }


def _int_value(value: Any, name: str, *, minimum: int | None = None) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be an integer, not a boolean.")
    if isinstance(value, int):
        number = value
    elif isinstance(value, float) and math.isfinite(value) and value.is_integer():
        number = int(value)
    else:
        raise ValueError(f"{name} must be an integer.")
    if minimum is not None and number < minimum:
        raise ValueError(f"{name} must be >= {minimum}.")
    return number


def _float_value(
    value: Any,
    name: str,
    *,
    minimum: float | None = None,
    maximum: float | None = None,
) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise ValueError(f"{name} must be a finite numeric value.")
    number = float(value)
    if minimum is not None and number < minimum:
        raise ValueError(f"{name} must be >= {minimum}.")
    if maximum is not None and number > maximum:
        raise ValueError(f"{name} must be <= {maximum}.")
    return number
