from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Callable, Iterator, Mapping, Sequence

import numpy as np
import torch
from torch.nn.utils import parameters_to_vector, vector_to_parameters


BatchArgs = tuple[Sequence[Any], Mapping[str, Any]]


@dataclass
class RLCTEstimateResult:
    lambda_hat: float
    betas: np.ndarray
    betaEf: np.ndarray
    mean_f: np.ndarray
    ess_like_counts: np.ndarray
    x0: np.ndarray
    objective_info: dict[str, Any]

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


class LocalRLCTTorchEstimator:
    """
    Estimate a local RLCT around a trained PyTorch model parameter vector.

    The local objective matches the notebook implementation:

        f(w) = scale * (L_n(w) - L_n(w0))

    where `L_n` is the empirical mean loss over the provided data and `scale`
    defaults to the number of examples.

    Parameters
    ----------
    model:
        Trained PyTorch model.
    loss_fn:
        Callable that returns a scalar tensor. By default it is called as
        `loss_fn(model, *args, **kwargs)` for each batch.
    data:
        Either a tuple like `(x, y)`, a `torch.utils.data.DataLoader`, or any
        re-iterable collection of batches.
    w0:
        Reference parameter vector. If omitted, the model's current parameters
        are used.
    device, dtype:
        Device and dtype used during evaluation. Defaults to the model's first
        trainable parameter.
    scale:
        Multiplicative factor in front of the shifted empirical loss.
        Defaults to the inferred number of examples.
    batch_to_args:
        Optional callable that converts a batch into `(args, kwargs)` used for
        `loss_fn(model, *args, **kwargs)`.
    batch_size_fn:
        Optional callable that returns the number of examples in a batch.
        Needed only when the size cannot be inferred automatically.
    eval_mode:
        If True, the model is switched to evaluation mode while estimating the
        local RLCT. This is usually desirable for dropout and batch norm.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        loss_fn: Callable[..., torch.Tensor],
        data: Any,
        w0: torch.Tensor | np.ndarray | None = None,
        *,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
        scale: float | None = None,
        batch_to_args: Callable[[Any], BatchArgs] | None = None,
        batch_size_fn: Callable[[Any], int] | None = None,
        eval_mode: bool = True,
    ) -> None:
        self.model = model
        self.loss_fn = loss_fn
        self.batch_to_args = batch_to_args or self._default_batch_to_args
        self.batch_size_fn = batch_size_fn or self._default_batch_size
        self.data = self._prepare_data(data)
        self.eval_mode = eval_mode

        self.params = [p for p in self.model.parameters() if p.requires_grad]
        if not self.params:
            raise ValueError("Model has no trainable parameters.")

        ref_param = self.params[0]
        self.device = torch.device(device) if device is not None else ref_param.device
        self.dtype = dtype if dtype is not None else ref_param.dtype
        self.was_training = self.model.training
        self.model = self.model.to(device=self.device, dtype=self.dtype)
        if self.eval_mode:
            self.model.eval()

        if w0 is None:
            with torch.no_grad():
                self.w0_torch = parameters_to_vector(
                    [p.detach() for p in self.params]
                ).to(device=self.device, dtype=self.dtype)
        else:
            self.w0_torch = torch.as_tensor(
                w0, device=self.device, dtype=self.dtype
            ).reshape(-1)

        num_params = sum(p.numel() for p in self.params)
        if self.w0_torch.numel() != num_params:
            raise ValueError(
                f"w0 has size {self.w0_torch.numel()}, but model expects {num_params} parameters."
            )

        self._set_params_from_vector(self.w0_torch)
        self.dataset_size = self._infer_dataset_size()
        self.scale = float(scale) if scale is not None else float(self.dataset_size)
        self.loss0 = self._compute_empirical_loss()
        self.x0 = self.w0_torch.detach().cpu().double().numpy().copy()

    @classmethod
    def from_tensors(
        cls,
        model: torch.nn.Module,
        loss_fn: Callable[..., torch.Tensor],
        *tensors: torch.Tensor,
        w0: torch.Tensor | np.ndarray | None = None,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
        scale: float | None = None,
        eval_mode: bool = True,
    ) -> "LocalRLCTTorchEstimator":
        if not tensors:
            raise ValueError("Please provide at least one tensor.")
        return cls(
            model=model,
            loss_fn=loss_fn,
            data=tuple(tensors),
            w0=w0,
            device=device,
            dtype=dtype,
            scale=scale,
            eval_mode=eval_mode,
        )

    def estimate(
        self,
        *,
        betas: Sequence[float] | None = None,
        step_size: float = 1e-3,
        n_steps: int = 4000,
        burn_in: int = 1000,
        thinning: int = 10,
        clip_radius: float | None = None,
        grad_clip: float | None = None,
        seed: int = 0,
    ) -> RLCTEstimateResult:
        if betas is None:
            betas_array = np.array([8, 16, 32, 64, 128, 256, 512], dtype=float)
        else:
            betas_array = np.asarray(betas, dtype=float)

        rng = np.random.default_rng(seed)
        x = self.x0.copy()
        dim = x.size

        betaEf_list: list[float] = []
        mean_f_list: list[float] = []
        counts: list[int] = []

        try:
            for beta in betas_array:
                fs: list[float] = []

                for t in range(n_steps):
                    g = self.grad_f(x)

                    if grad_clip is not None:
                        grad_norm = np.linalg.norm(g)
                        if grad_norm > grad_clip:
                            g = g * (grad_clip / (grad_norm + 1e-12))

                    noise = rng.normal(size=dim)
                    x = x - step_size * beta * g + np.sqrt(2.0 * step_size) * noise

                    if clip_radius is not None:
                        delta = x - self.x0
                        delta_norm = np.linalg.norm(delta)
                        if delta_norm > clip_radius:
                            x = self.x0 + delta * (clip_radius / (delta_norm + 1e-12))

                    if t >= burn_in and ((t - burn_in) % thinning == 0):
                        fs.append(float(self.f(x)))

                samples = np.asarray(fs, dtype=float)
                mean_f = float(samples.mean())
                betaEf = float(beta * mean_f)

                mean_f_list.append(mean_f)
                betaEf_list.append(betaEf)
                counts.append(len(samples))
        finally:
            self._set_params_from_vector(self.w0_torch)
            self.model.train(self.was_training)

        z = 1.0 / np.log(betas_array)
        design = np.column_stack([np.ones_like(z), z])
        coef, *_ = np.linalg.lstsq(design, np.asarray(betaEf_list), rcond=None)

        return RLCTEstimateResult(
            lambda_hat=float(coef[0]),
            betas=betas_array,
            betaEf=np.asarray(betaEf_list, dtype=float),
            mean_f=np.asarray(mean_f_list, dtype=float),
            ess_like_counts=np.asarray(counts, dtype=int),
            x0=self.x0.copy(),
            objective_info={
                "loss0": self.loss0,
                "scale": self.scale,
                "dataset_size": self.dataset_size,
                "device": str(self.device),
                "dtype": str(self.dtype),
                "num_params": int(self.x0.size),
                "eval_mode": self.eval_mode,
            },
        )

    def f(self, w: np.ndarray | torch.Tensor) -> float:
        w_vec = self._to_parameter_vector(w)
        self._set_params_from_vector(w_vec)
        loss_value = self._compute_empirical_loss()
        return self.scale * (loss_value - self.loss0)

    def grad_f(self, w: np.ndarray | torch.Tensor) -> np.ndarray:
        w_vec = self._to_parameter_vector(w)
        self._set_params_from_vector(w_vec)

        for param in self.params:
            if param.grad is not None:
                param.grad.zero_()

        total_weight = 0.0
        weighted_loss: torch.Tensor | None = None

        for batch in self._iter_batches():
            args, kwargs = self._move_batch_to_device(batch)
            batch_loss = self.loss_fn(self.model, *args, **kwargs)
            if batch_loss.ndim != 0:
                raise ValueError("loss_fn must return a scalar tensor for each batch.")

            batch_weight = float(self.batch_size_fn(batch))
            total_weight += batch_weight
            contribution = batch_loss * batch_weight
            weighted_loss = contribution if weighted_loss is None else weighted_loss + contribution

        if weighted_loss is None or total_weight == 0.0:
            raise ValueError("No data available to compute the empirical gradient.")

        empirical_mean_loss = weighted_loss / total_weight
        objective = self.scale * (empirical_mean_loss - self.loss0)
        grads = torch.autograd.grad(objective, self.params, allow_unused=False)
        grad_vec = parameters_to_vector([grad.detach() for grad in grads])
        return grad_vec.cpu().double().numpy()

    def _prepare_data(self, data: Any) -> Any:
        if isinstance(data, tuple):
            return data
        if isinstance(data, list):
            if data and all(torch.is_tensor(item) for item in data):
                return tuple(data)
            return data
        if hasattr(data, "__iter__") and hasattr(data, "__len__"):
            return data
        raise ValueError(
            "data must be a tensor tuple, DataLoader, or another re-iterable collection."
        )

    def _iter_batches(self) -> Iterator[Any]:
        if isinstance(self.data, tuple) and self.data and all(
            torch.is_tensor(item) for item in self.data
        ):
            yield self.data
            return
        yield from self.data

    def _infer_dataset_size(self) -> int:
        if hasattr(self.data, "dataset"):
            try:
                size = len(self.data.dataset)
                if size > 0:
                    return int(size)
            except TypeError:
                pass

        if isinstance(self.data, tuple) and self.data and all(
            torch.is_tensor(item) for item in self.data
        ):
            return int(self.batch_size_fn(self.data))

        total = 0
        for batch in self._iter_batches():
            total += int(self.batch_size_fn(batch))

        if total <= 0:
            raise ValueError("Could not infer dataset size. Please provide batch_size_fn or scale.")
        return total

    def _compute_empirical_loss(self) -> float:
        with torch.no_grad():
            total_weight = 0.0
            total_loss = 0.0

            for batch in self._iter_batches():
                args, kwargs = self._move_batch_to_device(batch)
                batch_loss = self.loss_fn(self.model, *args, **kwargs)
                if batch_loss.ndim != 0:
                    raise ValueError("loss_fn must return a scalar tensor for each batch.")

                batch_weight = float(self.batch_size_fn(batch))
                total_weight += batch_weight
                total_loss += float(batch_loss.detach().cpu().item()) * batch_weight

        if total_weight == 0.0:
            raise ValueError("No data available to compute the empirical loss.")
        return total_loss / total_weight

    def _move_batch_to_device(self, batch: Any) -> BatchArgs:
        args, kwargs = self.batch_to_args(batch)
        moved_args = tuple(self._move_value(arg) for arg in args)
        moved_kwargs = {key: self._move_value(value) for key, value in kwargs.items()}
        return moved_args, moved_kwargs

    def _move_value(self, value: Any) -> Any:
        if torch.is_tensor(value):
            if value.is_floating_point() or value.is_complex():
                return value.to(device=self.device, dtype=self.dtype)
            return value.to(device=self.device)
        return value

    def _to_parameter_vector(self, w: np.ndarray | torch.Tensor) -> torch.Tensor:
        return torch.as_tensor(w, device=self.device, dtype=self.dtype).reshape(-1)

    def _set_params_from_vector(self, w_vec: torch.Tensor) -> None:
        with torch.no_grad():
            vector_to_parameters(w_vec, self.params)

    @staticmethod
    def _default_batch_to_args(batch: Any) -> BatchArgs:
        if isinstance(batch, tuple):
            return batch, {}
        if isinstance(batch, list):
            return tuple(batch), {}
        if isinstance(batch, Mapping):
            return (), dict(batch)
        return (batch,), {}

    @staticmethod
    def _default_batch_size(batch: Any) -> int:
        if isinstance(batch, Mapping):
            for value in batch.values():
                if torch.is_tensor(value) and value.ndim > 0:
                    return int(value.shape[0])
        if isinstance(batch, (tuple, list)):
            for value in batch:
                if torch.is_tensor(value) and value.ndim > 0:
                    return int(value.shape[0])
        if torch.is_tensor(batch) and batch.ndim > 0:
            return int(batch.shape[0])
        raise ValueError(
            "Could not infer batch size from batch. Please provide batch_size_fn."
        )


def estimate_local_rlct(
    model: torch.nn.Module,
    loss_fn: Callable[..., torch.Tensor],
    data: Any,
    w0: torch.Tensor | np.ndarray | None = None,
    **estimate_kwargs: Any,
) -> RLCTEstimateResult:
    estimator = LocalRLCTTorchEstimator(
        model=model,
        loss_fn=loss_fn,
        data=data,
        w0=w0,
    )
    return estimator.estimate(**estimate_kwargs)
