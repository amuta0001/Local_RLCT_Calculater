from __future__ import annotations

from collections.abc import Sequence

import torch


def _build_linear_stack(
    input_dim: int,
    hidden_dims: Sequence[int],
    output_dim: int,
    *,
    bias: bool,
    dtype: torch.dtype,
) -> torch.nn.Sequential:
    dims = [input_dim, *hidden_dims, output_dim]
    layers: list[torch.nn.Module] = []

    for idx in range(len(dims) - 1):
        layers.append(torch.nn.Linear(dims[idx], dims[idx + 1], bias=bias, dtype=dtype))

    return torch.nn.Sequential(*layers)


class TrueLinearDNN(torch.nn.Module):
    """
    Data-generating deep linear network.

    This model represents the "true" function used to generate observations.
    The architecture is a stack of linear layers with no nonlinear activations
    between hidden layers.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: Sequence[int] = (),
        *,
        output_dim: int = 1,
        bias: bool = True,
        dtype: torch.dtype = torch.float64,
    ) -> None:
        super().__init__()
        self.network = _build_linear_stack(
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            output_dim=output_dim,
            bias=bias,
            dtype=dtype,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)

    def sample_outputs(
        self,
        x: torch.Tensor,
        *,
        noise_std: float = 0.0,
        generator: torch.Generator | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        with torch.no_grad():
            y_true = self(x)

            if noise_std <= 0.0:
                return y_true, y_true.clone()

            noise = torch.randn(
                y_true.shape,
                generator=generator,
                device=y_true.device,
                dtype=y_true.dtype,
            )
            y = y_true + noise_std * noise
            return y_true, y


class LinearDNNModel(torch.nn.Module):
    """
    Trainable deep linear network compatible with `common.local_rlct_estimater`.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: Sequence[int] = (),
        *,
        output_dim: int = 1,
        bias: bool = True,
        dtype: torch.dtype = torch.float64,
    ) -> None:
        super().__init__()
        self.network = _build_linear_stack(
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            output_dim=output_dim,
            bias=bias,
            dtype=dtype,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


def make_true_model(
    input_dim: int,
    hidden_dims: Sequence[int] = (),
    *,
    output_dim: int = 1,
    bias: bool = True,
    dtype: torch.dtype = torch.float64,
) -> TrueLinearDNN:
    return TrueLinearDNN(
        input_dim=input_dim,
        hidden_dims=hidden_dims,
        output_dim=output_dim,
        bias=bias,
        dtype=dtype,
    )


def make_learning_model(
    input_dim: int,
    hidden_dims: Sequence[int] = (),
    *,
    output_dim: int = 1,
    bias: bool = True,
    dtype: torch.dtype = torch.float64,
) -> LinearDNNModel:
    return LinearDNNModel(
        input_dim=input_dim,
        hidden_dims=hidden_dims,
        output_dim=output_dim,
        bias=bias,
        dtype=dtype,
    )


def sample_from_true_model(
    true_model: TrueLinearDNN,
    x: torch.Tensor,
    *,
    noise_std: float = 0.0,
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    return true_model.sample_outputs(x, noise_std=noise_std, generator=generator)
