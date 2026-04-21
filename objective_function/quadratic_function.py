from __future__ import annotations

import torch


class QuadraticTorchModel(torch.nn.Module):
    """
    Minimal Torch model whose induced objective is f(w) = ||w||^2.

    The parameter vector itself is the quantity explored by the local RLCT
    estimator, and `forward` returns the elementwise square of that vector.
    """

    def __init__(self, dim: int, *, dtype: torch.dtype = torch.float64) -> None:
        super().__init__()
        self.w = torch.nn.Parameter(torch.zeros(dim, dtype=dtype))

    def forward(self, batch: torch.Tensor) -> torch.Tensor:
        del batch
        return self.w ** 2
