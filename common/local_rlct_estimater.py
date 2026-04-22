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
    betaEf_std: np.ndarray | None = None
    betaEf_se: np.ndarray | None = None

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class NeighborhoodGapSearchResult:
    max_gap: float
    max_gap_sample_index: int
    max_gap_parameter: np.ndarray
    train_loss_at_max_gap: float
    test_loss_at_max_gap: float
    sampled_gaps: np.ndarray
    sampled_train_losses: np.ndarray
    sampled_test_losses: np.ndarray
    sampled_distances: np.ndarray
    radius: float
    n_samples: int
    include_center: bool
    absolute_gap: bool
    distribution: str
    seed: int
    x0: np.ndarray

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


class LocalRLCTTorchEstimator:
    """
    Estimate a local RLCT around a trained PyTorch model parameter vector.

    The local objective is

        f(w) = scale * (L_n(w) - L_n(w0))

    where `L_n` is the empirical mean loss over the evaluation data. Sampling
    updates are performed with SGLD: gradients are computed on minibatches from
    `data`, while objective evaluation uses `eval_data` (full data by default).
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
        eval_data: Any | None = None,
        log_output_mode: str = "none",
    ) -> None:
        self.model = model
        self.loss_fn = loss_fn
        self.batch_to_args = batch_to_args or self._default_batch_to_args
        self.batch_size_fn = batch_size_fn or self._default_batch_size
        self.data = self._prepare_data(data)
        self.eval_data = self._prepare_data(eval_data if eval_data is not None else data)
        self.eval_mode = eval_mode
        self.log_output_mode = self._validate_log_output_mode(log_output_mode)

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
        self.dataset_size = self._infer_dataset_size(self.eval_data)
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
        eval_tensors: Sequence[torch.Tensor] | None = None,
        log_output_mode: str = "none",
    ) -> "LocalRLCTTorchEstimator":
        if not tensors:
            raise ValueError("Please provide at least one tensor.")
        eval_data = tuple(eval_tensors) if eval_tensors is not None else None
        return cls(
            model=model,
            loss_fn=loss_fn,
            data=tuple(tensors),
            w0=w0,
            device=device,
            dtype=dtype,
            scale=scale,
            eval_mode=eval_mode,
            eval_data=eval_data,
            log_output_mode=log_output_mode,
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
        n_chains: int = 4,
        max_beta_step: float = 0.25,
        regression_tail: int | float | None = 0.5,
        use_weighted_regression: bool = True,
        update_batch_size: int | None = None,
        eval_max_batches: int | None = None,
        replace_batches: bool = True,
        seed: int = 0,
        log_output_mode: str | None = None,
        log_every: int = 100,
    ) -> RLCTEstimateResult:
        log_mode = self.log_output_mode if log_output_mode is None else self._validate_log_output_mode(log_output_mode)
        if betas is None:
            betas_array = np.array([8, 16, 32, 64, 128, 256, 512], dtype=float)
        else:
            betas_array = np.asarray(betas, dtype=float)
        if betas_array.ndim != 1 or betas_array.size < 2:
            raise ValueError("betas must contain at least two values.")
        if n_chains < 1:
            raise ValueError("n_chains must be at least 1.")
        if n_steps <= burn_in:
            raise ValueError("n_steps must be greater than burn_in.")

        rng = np.random.default_rng(seed)
        dim = self.x0.size
        chain_states = np.repeat(self.x0[None, :], n_chains, axis=0)
        batch_streams = [
            self._make_update_batch_stream(
                self.data,
                update_batch_size=update_batch_size,
                replace_batches=replace_batches,
                rng=np.random.default_rng(rng.integers(0, 2**32)),
            )
            for _ in range(n_chains)
        ]

        betaEf_list: list[float] = []
        mean_f_list: list[float] = []
        counts: list[int] = []
        betaEf_std_list: list[float] = []
        betaEf_se_list: list[float] = []

        try:
            for beta in betas_array:
                fs: list[float] = []
                effective_step = min(step_size, max_beta_step / beta)
                if log_mode in {"beta", "step"}:
                    print(
                        f"[LocalRLCT] beta={beta:.6g} start "
                        f"(effective_step={effective_step:.6g}, n_chains={n_chains})"
                    )

                for chain_idx in range(n_chains):
                    x = chain_states[chain_idx].copy()
                    batch_stream = batch_streams[chain_idx]

                    for t in range(n_steps):
                        batch = next(batch_stream)
                        g = self.stochastic_grad_f(x, batch)

                        if grad_clip is not None:
                            grad_norm = np.linalg.norm(g)
                            if grad_norm > grad_clip:
                                g = g * (grad_clip / (grad_norm + 1e-12))

                        noise = rng.normal(size=dim)
                        x = x - effective_step * beta * g + np.sqrt(2.0 * effective_step) * noise

                        if clip_radius is not None:
                            delta = x - self.x0
                            delta_norm = np.linalg.norm(delta)
                            if delta_norm > clip_radius:
                                x = self.x0 + delta * (clip_radius / (delta_norm + 1e-12))

                        if t >= burn_in and ((t - burn_in) % thinning == 0):
                            fs.append(float(self.f(x, max_batches=eval_max_batches)))

                        if (
                            log_mode == "step"
                            and ((t + 1) % log_every == 0 or t + 1 == n_steps)
                        ):
                            samples_so_far = max(0, ((t - burn_in) // thinning) + 1) if t >= burn_in else 0
                            print(
                                f"[LocalRLCT] beta={beta:.6g} chain={chain_idx + 1}/{n_chains} "
                                f"step={t + 1}/{n_steps} samples={samples_so_far}"
                            )

                    chain_states[chain_idx] = x

                samples = np.asarray(fs, dtype=float)
                if samples.size == 0:
                    raise ValueError("No post-burn-in samples collected. Adjust n_steps, burn_in, or thinning.")

                mean_f = float(samples.mean())
                betaEf = float(beta * mean_f)
                betaEf_std = float(beta * samples.std(ddof=1)) if samples.size > 1 else 0.0
                betaEf_se = float(betaEf_std / np.sqrt(samples.size)) if samples.size > 0 else 0.0

                mean_f_list.append(mean_f)
                betaEf_list.append(betaEf)
                counts.append(len(samples))
                betaEf_std_list.append(betaEf_std)
                betaEf_se_list.append(betaEf_se)
                if log_mode in {"beta", "step"}:
                    print(
                        f"[LocalRLCT] beta={beta:.6g} done "
                        f"(samples={len(samples)}, mean_f={mean_f:.6g}, betaEf={betaEf:.6g})"
                    )
        finally:
            self._set_params_from_vector(self.w0_torch)
            self.model.train(self.was_training)

        betaEf_arr = np.asarray(betaEf_list, dtype=float)
        betaEf_se_arr = np.asarray(betaEf_se_list, dtype=float)
        regression_mask = self._build_regression_mask(betas_array, regression_tail)
        z = 1.0 / np.log(betas_array)
        design = np.column_stack([np.ones_like(z), z])
        design_used = design[regression_mask]
        response_used = betaEf_arr[regression_mask]

        if use_weighted_regression:
            weights = 1.0 / np.maximum(betaEf_se_arr[regression_mask] ** 2, 1e-12)
            sqrt_weights = np.sqrt(weights)
            design_used = design_used * sqrt_weights[:, None]
            response_used = response_used * sqrt_weights

        coef, *_ = np.linalg.lstsq(design_used, response_used, rcond=None)

        return RLCTEstimateResult(
            lambda_hat=float(coef[0]),
            betas=betas_array,
            betaEf=betaEf_arr,
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
                "sampler": "sgld",
                "n_chains": n_chains,
                "base_step_size": step_size,
                "max_beta_step": max_beta_step,
                "update_batch_size": update_batch_size,
                "eval_max_batches": eval_max_batches,
                "replace_batches": replace_batches,
                "regression_tail": regression_tail,
                "use_weighted_regression": use_weighted_regression,
                "log_output_mode": log_mode,
                "log_every": log_every,
                "regression_betas": betas_array[regression_mask].copy(),
            },
            betaEf_std=np.asarray(betaEf_std_list, dtype=float),
            betaEf_se=betaEf_se_arr,
        )

    def f(
        self,
        w: np.ndarray | torch.Tensor,
        *,
        max_batches: int | None = None,
        data: Any | None = None,
    ) -> float:
        w_vec = self._to_parameter_vector(w)
        self._set_params_from_vector(w_vec)
        loss_value = self._compute_empirical_loss(data=data, max_batches=max_batches)
        return self.scale * (loss_value - self.loss0)

    def grad_f(self, w: np.ndarray | torch.Tensor) -> np.ndarray:
        w_vec = self._to_parameter_vector(w)
        self._set_params_from_vector(w_vec)
        return self._gradient_from_batches(self._iter_batches(self.eval_data))

    def stochastic_grad_f(self, w: np.ndarray | torch.Tensor, batch: Any) -> np.ndarray:
        w_vec = self._to_parameter_vector(w)
        self._set_params_from_vector(w_vec)
        return self._gradient_from_batches([batch])

    def _gradient_from_batches(self, batches: Sequence[Any] | Iterator[Any]) -> np.ndarray:
        for param in self.params:
            if param.grad is not None:
                param.grad.zero_()

        total_weight = 0.0
        weighted_loss: torch.Tensor | None = None

        for batch in batches:
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

        mean_loss = weighted_loss / total_weight
        objective = self.scale * mean_loss
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

    def _iter_batches(self, data: Any, max_batches: int | None = None) -> Iterator[Any]:
        yielded = 0

        if isinstance(data, tuple) and data and all(torch.is_tensor(item) for item in data):
            if max_batches is None or max_batches > 0:
                yield data
            return

        for batch in data:
            yield batch
            yielded += 1
            if max_batches is not None and yielded >= max_batches:
                break

    def _make_update_batch_stream(
        self,
        data: Any,
        *,
        update_batch_size: int | None,
        replace_batches: bool,
        rng: np.random.Generator,
    ) -> Iterator[Any]:
        if self._is_tensor_tuple(data):
            dataset_size = self._infer_dataset_size(data)

            while True:
                if update_batch_size is None or update_batch_size >= dataset_size:
                    yield data
                    continue

                indices = rng.choice(dataset_size, size=update_batch_size, replace=replace_batches)
                first_item = data[0]
                index_tensor = torch.as_tensor(
                    indices,
                    dtype=torch.long,
                    device=first_item.device,
                )
                yield tuple(item.index_select(0, index_tensor) for item in data)
            return

        while True:
            for batch in data:
                yield batch

    def _infer_dataset_size(self, data: Any) -> int:
        if hasattr(data, "dataset"):
            try:
                size = len(data.dataset)
                if size > 0:
                    return int(size)
            except TypeError:
                pass

        if self._is_tensor_tuple(data):
            return int(self.batch_size_fn(data))

        total = 0
        for batch in self._iter_batches(data):
            total += int(self.batch_size_fn(batch))

        if total <= 0:
            raise ValueError("Could not infer dataset size. Please provide batch_size_fn or scale.")
        return total

    def _compute_empirical_loss(
        self,
        *,
        data: Any | None = None,
        max_batches: int | None = None,
    ) -> float:
        data = self.eval_data if data is None else self._prepare_data(data)

        with torch.no_grad():
            total_weight = 0.0
            total_loss = 0.0

            for batch in self._iter_batches(data, max_batches=max_batches):
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
    def _build_regression_mask(
        betas: np.ndarray, regression_tail: int | float | None
    ) -> np.ndarray:
        if regression_tail is None:
            return np.ones_like(betas, dtype=bool)

        n_betas = betas.size
        if isinstance(regression_tail, float):
            if not 0.0 < regression_tail <= 1.0:
                raise ValueError("regression_tail as a float must lie in (0, 1].")
            tail_count = max(2, int(np.ceil(n_betas * regression_tail)))
        else:
            tail_count = max(2, int(regression_tail))

        tail_count = min(n_betas, tail_count)
        mask = np.zeros(n_betas, dtype=bool)
        mask[-tail_count:] = True
        return mask

    @staticmethod
    def _validate_log_output_mode(mode: str) -> str:
        valid_modes = {"none", "beta", "step"}
        if mode not in valid_modes:
            raise ValueError(
                f"log_output_mode must be one of {sorted(valid_modes)}, got {mode!r}."
            )
        return mode

    @staticmethod
    def _is_tensor_tuple(data: Any) -> bool:
        return isinstance(data, tuple) and data and all(torch.is_tensor(item) for item in data)

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


def find_max_generalization_gap_in_neighborhood(
    model: torch.nn.Module,
    loss_fn: Callable[..., torch.Tensor],
    train_data: Any,
    test_data: Any,
    w0: torch.Tensor | np.ndarray | None = None,
    *,
    radius: float = 1e-2,
    n_samples: int = 256,
    distribution: str = "sphere",
    include_center: bool = True,
    absolute_gap: bool = True,
    device: torch.device | str | None = None,
    dtype: torch.dtype | None = None,
    batch_to_args: Callable[[Any], BatchArgs] | None = None,
    batch_size_fn: Callable[[Any], int] | None = None,
    eval_mode: bool = True,
    seed: int = 0,
) -> NeighborhoodGapSearchResult:
    """
    Search for the largest generalization gap in a neighborhood around `w0`.

    The search is Monte Carlo based: parameter vectors are sampled within the
    Euclidean ball of radius `radius` centered at `w0`, the train/test losses
    are evaluated at each sample, and the maximum gap is returned.

    Args:
        model: Target PyTorch model.
        loss_fn: Scalar loss function with signature compatible with
            `LocalRLCTTorchEstimator`.
        train_data: Data used to compute the training loss.
        test_data: Data used to compute the test loss.
        w0: Center parameter vector. If omitted, the current model parameters
            are used.
        radius: Radius of the search neighborhood in parameter space.
        n_samples: Number of perturbed samples to evaluate.
        distribution: `"sphere"` for uniform samples in the Euclidean ball or
            `"gaussian"` for isotropic Gaussian perturbations rescaled so that
            the typical perturbation norm is about `radius`.
        include_center: Whether to include `w0` itself as sample index 0.
        absolute_gap: If True, use `abs(test_loss - train_loss)`.
        device, dtype, batch_to_args, batch_size_fn, eval_mode: Same meaning as
            in `LocalRLCTTorchEstimator`.
        seed: RNG seed for reproducible sampling.
    """
    if radius < 0.0:
        raise ValueError("radius must be non-negative.")
    if n_samples < 1:
        raise ValueError("n_samples must be at least 1.")
    if distribution not in {"sphere", "gaussian"}:
        raise ValueError("distribution must be one of {'sphere', 'gaussian'}.")

    estimator = LocalRLCTTorchEstimator(
        model=model,
        loss_fn=loss_fn,
        data=train_data,
        w0=w0,
        device=device,
        dtype=dtype,
        batch_to_args=batch_to_args,
        batch_size_fn=batch_size_fn,
        eval_mode=eval_mode,
        eval_data=train_data,
    )

    rng = np.random.default_rng(seed)
    dim = estimator.x0.size
    sampled_params: list[np.ndarray] = []

    if include_center:
        sampled_params.append(estimator.x0.copy())

    remaining = n_samples - len(sampled_params)
    for _ in range(remaining):
        if distribution == "sphere":
            direction = rng.normal(size=dim)
            norm = np.linalg.norm(direction)
            if norm == 0.0:
                direction = np.zeros(dim, dtype=float)
            else:
                direction = direction / norm
            scaled_radius = radius * (rng.random() ** (1.0 / max(dim, 1)))
            candidate = estimator.x0 + scaled_radius * direction
        else:
            direction = rng.normal(size=dim)
            norm = np.linalg.norm(direction)
            if norm == 0.0:
                candidate = estimator.x0.copy()
            else:
                candidate = estimator.x0 + (radius / np.sqrt(max(dim, 1))) * direction
        sampled_params.append(candidate.astype(float, copy=False))

    train_losses: list[float] = []
    test_losses: list[float] = []
    gaps: list[float] = []
    distances: list[float] = []

    try:
        for sample in sampled_params:
            sample_vec = estimator._to_parameter_vector(sample)
            estimator._set_params_from_vector(sample_vec)
            train_loss = estimator._compute_empirical_loss(data=train_data)
            test_loss = estimator._compute_empirical_loss(data=test_data)
            gap = test_loss - train_loss
            if absolute_gap:
                gap = abs(gap)

            train_losses.append(float(train_loss))
            test_losses.append(float(test_loss))
            gaps.append(float(gap))
            distances.append(float(np.linalg.norm(sample - estimator.x0)))
    finally:
        estimator._set_params_from_vector(estimator.w0_torch)
        estimator.model.train(estimator.was_training)

    gaps_arr = np.asarray(gaps, dtype=float)
    train_arr = np.asarray(train_losses, dtype=float)
    test_arr = np.asarray(test_losses, dtype=float)
    distances_arr = np.asarray(distances, dtype=float)
    sample_arr = np.asarray(sampled_params, dtype=float)

    max_index = int(np.argmax(gaps_arr))
    return NeighborhoodGapSearchResult(
        max_gap=float(gaps_arr[max_index]),
        max_gap_sample_index=max_index,
        max_gap_parameter=sample_arr[max_index].copy(),
        train_loss_at_max_gap=float(train_arr[max_index]),
        test_loss_at_max_gap=float(test_arr[max_index]),
        sampled_gaps=gaps_arr,
        sampled_train_losses=train_arr,
        sampled_test_losses=test_arr,
        sampled_distances=distances_arr,
        radius=float(radius),
        n_samples=int(n_samples),
        include_center=bool(include_center),
        absolute_gap=bool(absolute_gap),
        distribution=distribution,
        seed=int(seed),
        x0=estimator.x0.copy(),
    )
