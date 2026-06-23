from __future__ import annotations

from brian_sphere_llm.model.llama_backbone import BackboneConfig, TransformerBlock, require_torch

try:
    import torch
    import torch.nn as nn
except ModuleNotFoundError:  # pragma: no cover
    torch = None
    nn = None

ModuleBase = nn.Module if nn is not None else object


class RouteBlock(ModuleBase):
    """Transformer block with a controlled block-position side channel."""

    def __init__(self, backbone: BackboneConfig, position_dim: int, position_injection: str = "adapter") -> None:
        require_torch()
        super().__init__()
        self.block = TransformerBlock(backbone)
        self.position_injection = position_injection
        self._compiled_dense_forward = None
        if position_injection == "adapter":
            self.position_adapter = nn.Linear(position_dim, backbone.d_model, bias=False)
            nn.init.zeros_(self.position_adapter.weight)
        elif position_injection == "direct_add":
            if position_dim != backbone.d_model:
                raise ValueError("direct_add position_dim must equal d_model")
            self.position_adapter = None
        else:
            raise ValueError(f"Unknown block position injection: {position_injection}")

    def forward(
        self,
        hidden: torch.Tensor,
        position: torch.Tensor,
        attention_global_state: object | None = None,
        *,
        return_attention_kv: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        bias = self._position_bias(position)
        return self.block(
            hidden + bias,
            attention_global_state,
            return_attention_kv=return_attention_kv,
        )

    def forward_selected(self, hidden: torch.Tensor, position: torch.Tensor, query_mask: torch.Tensor) -> torch.Tensor:
        bias = self._position_bias(position)
        return self.block.forward_selected(hidden + bias, query_mask)

    def forward_selected_varlen(self, hidden: torch.Tensor, position: torch.Tensor, query_mask: torch.Tensor) -> torch.Tensor:
        bias = self._position_bias(position)
        return self.block.forward_selected_varlen(hidden + bias, query_mask)

    def forward_selected_dense(
        self,
        hidden: torch.Tensor,
        position: torch.Tensor,
        query_mask: torch.Tensor,
        *,
        compile_cuda: bool = False,
    ) -> torch.Tensor:
        if compile_cuda and hidden.is_cuda:
            if self._compiled_dense_forward is None:
                self._compiled_dense_forward = torch.compile(self.forward, dynamic=False)
            return self._compiled_dense_forward(hidden, position)[query_mask]
        return self.forward(hidden, position)[query_mask]

    def _position_bias(self, position: torch.Tensor) -> torch.Tensor:
        if self.position_injection == "direct_add":
            return position.unsqueeze(1) if position.dim() == 2 else position
        bias = self.position_adapter(position)
        return bias.unsqueeze(1) if bias.dim() == 2 else bias
