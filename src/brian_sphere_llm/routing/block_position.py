from __future__ import annotations

import math

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except ModuleNotFoundError:  # pragma: no cover
    torch = None
    nn = None
    F = None

ModuleBase = nn.Module if nn is not None else object


class BlockPositionTable(ModuleBase):
    def __init__(self, num_internal_blocks: int, position_dim: int, *, mode: str = "open_arc") -> None:
        if torch is None:
            raise ModuleNotFoundError("PyTorch is required for block-position state.")
        super().__init__()
        self.num_internal_blocks = num_internal_blocks
        self.position_dim = position_dim
        self.num_actions = num_internal_blocks + 1
        self.out_action = num_internal_blocks
        self.mode = mode
        init = self._init_embeddings(mode)
        self.embeddings = nn.Parameter(init, requires_grad=mode != "none")

    def _init_embeddings(self, mode: str) -> torch.Tensor:
        if mode == "none":
            return torch.zeros(self.num_actions, self.position_dim, dtype=torch.float32)
        if mode == "random":
            return F.normalize(torch.randn(self.num_actions, self.position_dim), dim=-1)
        if mode == "open_arc":
            return self._sinusoidal_init(open_arc=True)
        if mode == "circular":
            return self._sinusoidal_init(open_arc=False)
        raise ValueError(f"Unsupported block_position_mode: {mode}")

    def _sinusoidal_init(self, open_arc: bool) -> torch.Tensor:
        denom = self.num_actions if open_arc else max(1, self.num_actions - 1)
        max_theta = math.pi if open_arc else 2 * math.pi
        rows = []
        half = self.position_dim // 2
        frequencies = torch.arange(1, half + 1, dtype=torch.float32)
        for index in range(self.num_actions):
            theta = max_theta * index / denom
            values = torch.stack([torch.cos(frequencies * theta), torch.sin(frequencies * theta)], dim=-1).flatten()
            if values.numel() < self.position_dim:
                values = F.pad(values, (0, self.position_dim - values.numel()))
            rows.append(values[: self.position_dim])
        return F.normalize(torch.stack(rows, dim=0), dim=-1)

    def initial(self, batch_size: int, device: torch.device) -> torch.Tensor:
        return self.embeddings[0].detach().to(device).expand(batch_size, -1)

    def by_action(self, action: torch.Tensor) -> torch.Tensor:
        if self.mode == "none":
            return torch.zeros(action.size(0), self.position_dim, dtype=self.embeddings.dtype, device=action.device)
        return F.normalize(self.embeddings[action], dim=-1)

    def weighted(self, probs: torch.Tensor) -> torch.Tensor:
        if self.mode == "none":
            return torch.zeros(probs.size(0), self.position_dim, dtype=probs.dtype, device=probs.device)
        return F.normalize(probs @ self.embeddings, dim=-1)

    def location_distance(self, position: torch.Tensor, probs: torch.Tensor) -> torch.Tensor:
        if self.mode == "none":
            return torch.zeros((), dtype=position.dtype, device=position.device)
        distances = torch.cdist(position.unsqueeze(1), self.embeddings.unsqueeze(0)).squeeze(1).pow(2)
        return (probs * distances).sum(dim=-1).mean()
