from __future__ import annotations

from collections.abc import Sequence

import torch

MNIST_INPUT_DIM = 28 * 28
MNIST_NUM_CLASSES = 10
DEFAULT_MNIST_HIDDEN_DIMS = (1024, 512, 256)


def _build_mlp(
    input_dim: int,
    hidden_dims: Sequence[int],
    output_dim: int,
    *,
    bias: bool,
    dropout_rate: float,
    dtype: torch.dtype,
) -> torch.nn.Sequential:
    dims = [input_dim, *hidden_dims, output_dim]
    layers: list[torch.nn.Module] = []

    for idx in range(len(dims) - 1):
        layers.append(torch.nn.Linear(dims[idx], dims[idx + 1], bias=bias, dtype=dtype))
        if idx < len(dims) - 2:
            layers.append(torch.nn.ReLU())
            if dropout_rate > 0.0:
                layers.append(torch.nn.Dropout(p=dropout_rate))

    return torch.nn.Sequential(*layers)


class MNISTDNN(torch.nn.Module):
    """
    MLP for MNIST classification.

    The model accepts either flattened inputs with shape `(batch, 784)` or raw
    image tensors with shape `(batch, 28, 28)` / `(batch, 1, 28, 28)`. The
    output is class logits for the 10 MNIST classes.
    """

    def __init__(
        self,
        hidden_dims: Sequence[int] = DEFAULT_MNIST_HIDDEN_DIMS,
        *,
        input_dim: int = MNIST_INPUT_DIM,
        output_dim: int = MNIST_NUM_CLASSES,
        bias: bool = True,
        dropout_rate: float = 0.0,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.dropout_rate = dropout_rate
        self.network = _build_mlp(
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            output_dim=output_dim,
            bias=bias,
            dropout_rate=dropout_rate,
            dtype=dtype,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim > 2:
            x = x.reshape(x.shape[0], -1)
        return self.network(x)


def make_mnist_dnn(
    hidden_dims: Sequence[int] = DEFAULT_MNIST_HIDDEN_DIMS,
    *,
    input_dim: int = MNIST_INPUT_DIM,
    output_dim: int = MNIST_NUM_CLASSES,
    bias: bool = True,
    dropout_rate: float = 0.0,
    dtype: torch.dtype = torch.float32,
) -> MNISTDNN:
    return MNISTDNN(
        hidden_dims=hidden_dims,
        input_dim=input_dim,
        output_dim=output_dim,
        bias=bias,
        dropout_rate=dropout_rate,
        dtype=dtype,
    )


def mnist_cross_entropy_loss(
    model: torch.nn.Module,
    images: torch.Tensor,
    labels: torch.Tensor,
) -> torch.Tensor:
    logits = model(images)
    return torch.nn.functional.cross_entropy(logits, labels)
