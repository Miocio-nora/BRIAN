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
    def __init__(
        self,
        num_internal_blocks: int,
        position_dim: int,
        *,
        mode: str = "open_arc",
        independent_input_position: bool = False,
    ) -> None:
        if torch is None:
            raise ModuleNotFoundError("PyTorch is required for block-position state.")
        super().__init__()
        self.num_internal_blocks = num_internal_blocks
        self.position_dim = position_dim
        self.num_actions = num_internal_blocks + 1
        self.out_action = num_internal_blocks
        self.mode = mode
        self.independent_input_position = independent_input_position
        if mode == "spherical_code":
            if not independent_input_position:
                raise ValueError("spherical_code requires independent_input_position=true")
            init, input_init = self._spherical_code_init()
        else:
            init = self._init_embeddings(mode)
            input_init = init[0]
        self.embeddings = nn.Parameter(init, requires_grad=mode != "none")
        if independent_input_position:
            self.input_position = nn.Parameter(input_init.clone(), requires_grad=mode != "none")
        else:
            self.register_parameter("input_position", None)

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

    def _spherical_code_init(self) -> tuple[torch.Tensor, torch.Tensor]:
        # One pole dimension plus the standard n-vertex simplex construction.
        required_dim = self.num_internal_blocks + 1
        if self.position_dim < required_dim:
            raise ValueError(
                "spherical_code position_dim must be at least num_internal_blocks + 1 "
                f"({required_dim})"
            )
        points = torch.zeros(self.num_internal_blocks + 2, self.position_dim, dtype=torch.float32)
        points[0, 0] = 1.0
        points[-1, 0] = -1.0
        simplex = torch.eye(self.num_internal_blocks, dtype=torch.float32)
        simplex = simplex - torch.full_like(simplex, 1.0 / self.num_internal_blocks)
        simplex = F.normalize(simplex, dim=-1)
        points[1 : self.num_internal_blocks + 1, 1 : self.num_internal_blocks + 1] = simplex

        generator = torch.Generator(device="cpu")
        generator.manual_seed(1729)
        rotation, triangular = torch.linalg.qr(
            torch.randn(self.position_dim, self.position_dim, generator=generator, dtype=torch.float32)
        )
        signs = torch.where(torch.diag(triangular) < 0, -torch.ones(self.position_dim), torch.ones(self.position_dim))
        rotation = rotation * signs.unsqueeze(0)
        points = F.normalize(points @ rotation, dim=-1)
        action_points = torch.cat(
            [points[1 : self.num_internal_blocks + 1], points[-1:]],
            dim=0,
        )
        return action_points, points[0]

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
        if self.mode == "none":
            return torch.zeros(batch_size, self.position_dim, dtype=self.embeddings.dtype, device=device)
        if self.input_position is not None:
            return F.normalize(self.input_position, dim=-1).to(device).expand(batch_size, -1)
        return self.embeddings[0].detach().to(device).expand(batch_size, -1)

    def by_action(self, action: torch.Tensor) -> torch.Tensor:
        if self.mode == "none":
            return torch.zeros(*action.shape, self.position_dim, dtype=self.embeddings.dtype, device=action.device)
        return F.normalize(F.embedding(action, self.embeddings), dim=-1)

    def weighted(self, probs: torch.Tensor) -> torch.Tensor:
        if self.mode == "none":
            return torch.zeros(*probs.shape[:-1], self.position_dim, dtype=probs.dtype, device=probs.device)
        return F.normalize(probs @ self.embeddings, dim=-1)

    def location_distance(self, position: torch.Tensor, probs: torch.Tensor) -> torch.Tensor:
        if self.mode == "none":
            return torch.zeros((), dtype=position.dtype, device=position.device)
        distances = (position.unsqueeze(-2) - self.embeddings.to(device=position.device, dtype=position.dtype)).pow(2).sum(dim=-1)
        return (probs * distances).sum(dim=-1).mean()

    def input_anchor_loss(self) -> torch.Tensor:
        if self.mode == "none" or self.input_position is None:
            return torch.zeros((), dtype=self.embeddings.dtype, device=self.embeddings.device)
        input_position = F.normalize(self.input_position, dim=-1)
        internal_positions = F.normalize(self.embeddings[: self.num_internal_blocks], dim=-1)
        centroid = internal_positions.mean(dim=0).detach()
        return (input_position - centroid).pow(2).sum()

    def geometry_loss(self) -> torch.Tensor:
        if self.mode != "spherical_code" or self.input_position is None:
            return torch.zeros((), dtype=self.embeddings.dtype, device=self.embeddings.device)
        points = self._ordered_spherical_points()
        target = self._spherical_target_gram(device=points.device, dtype=points.dtype)
        return (points @ points.transpose(0, 1) - target).pow(2).mean()

    def geometry_metrics(self) -> dict[str, torch.Tensor]:
        zero = torch.zeros((), dtype=self.embeddings.dtype, device=self.embeddings.device)
        if self.mode != "spherical_code" or self.input_position is None:
            return {
                "position_gram_error": zero,
                "internal_position_min_angle": zero,
            }
        points = self._ordered_spherical_points()
        internal = points[1 : self.num_internal_blocks + 1]
        cosine = (internal @ internal.transpose(0, 1)).clamp(-1.0, 1.0)
        diagonal = torch.eye(self.num_internal_blocks, dtype=torch.bool, device=cosine.device)
        min_angle = torch.acos(cosine.masked_fill(diagonal, -1.0).max()) * (180.0 / math.pi)
        return {
            "position_gram_error": self.geometry_loss(),
            "internal_position_min_angle": min_angle,
        }

    def _ordered_spherical_points(self) -> torch.Tensor:
        assert self.input_position is not None
        internal = self.embeddings[: self.num_internal_blocks]
        out = self.embeddings[self.out_action : self.out_action + 1]
        return F.normalize(torch.cat([self.input_position.unsqueeze(0), internal, out], dim=0), dim=-1)

    def _spherical_target_gram(self, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        count = self.num_internal_blocks + 2
        target = torch.zeros(count, count, device=device, dtype=dtype)
        target.fill_diagonal_(1.0)
        target[0, -1] = -1.0
        target[-1, 0] = -1.0
        if self.num_internal_blocks > 1:
            internal_target = torch.full(
                (self.num_internal_blocks, self.num_internal_blocks),
                -1.0 / (self.num_internal_blocks - 1),
                device=device,
                dtype=dtype,
            )
            internal_target.fill_diagonal_(1.0)
            target[1:-1, 1:-1] = internal_target
        return target

    def action_distances(self, position: torch.Tensor) -> torch.Tensor:
        if self.mode == "none":
            return torch.zeros(
                *position.shape[:-1],
                self.num_actions,
                dtype=position.dtype,
                device=position.device,
            )
        return (position.unsqueeze(-2) - self.embeddings.to(device=position.device, dtype=position.dtype)).pow(2).sum(dim=-1)
