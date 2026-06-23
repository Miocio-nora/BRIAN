from __future__ import annotations

try:
    import torch
    import torch.nn as nn
except ModuleNotFoundError:  # pragma: no cover
    torch = None
    nn = None

ModuleBase = nn.Module if nn is not None else object


class LatentRouter(ModuleBase):
    def __init__(self, d_model: int, position_dim: int, num_actions: int, hidden_dim: int | None = None) -> None:
        if torch is None:
            raise ModuleNotFoundError("PyTorch is required for routing.")
        super().__init__()
        hidden_dim = hidden_dim or d_model
        self.net = nn.Sequential(
            nn.Linear(d_model + position_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, num_actions),
        )

    def embedding(self, hidden: torch.Tensor, position: torch.Tensor) -> torch.Tensor:
        pooled = hidden.mean(dim=1)
        features = torch.cat([pooled, position], dim=-1)
        return self.net[1](self.net[0](features))

    def token_embedding(self, hidden: torch.Tensor, position: torch.Tensor) -> torch.Tensor:
        if position.dim() == 2:
            position = position.unsqueeze(1).expand(-1, hidden.size(1), -1)
        features = torch.cat([hidden, position], dim=-1)
        return self.net[1](self.net[0](features))

    def logits_from_embedding(self, embedding: torch.Tensor) -> torch.Tensor:
        return self.net[2](embedding)

    def expert_vectors(self) -> torch.Tensor:
        return self.net[2].weight

    def forward(self, hidden: torch.Tensor, position: torch.Tensor) -> torch.Tensor:
        return self.logits_from_embedding(self.embedding(hidden, position))

    def token_logits(self, hidden: torch.Tensor, position: torch.Tensor) -> torch.Tensor:
        return self.logits_from_embedding(self.token_embedding(hidden, position))
