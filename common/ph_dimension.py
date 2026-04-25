from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Sequence

import numpy as np


@dataclass
class PHDimensionEstimateResult:
    dim_ph: float
    alpha: float
    slope: float
    intercept: float
    subset_sizes: np.ndarray
    mean_e_alpha: np.ndarray
    std_e_alpha: np.ndarray
    n_repeats: int
    seed: int

    def as_dict(self) -> dict:
        return asdict(self)


def pairwise_mean_l1_distance(
    loss_profiles: np.ndarray,
    *,
    chunk_size: int = 256,
) -> np.ndarray:
    """
    Compute rho_S(w, w') = mean_i |ell(w, z_i) - ell(w', z_i)|.

    Args:
        loss_profiles: Array with shape (n_parameters, n_data), where each row
            contains per-sample losses for one point on the optimization
            trajectory.
        chunk_size: Number of rows to process at once.
    """
    profiles = np.asarray(loss_profiles, dtype=np.float64)
    if profiles.ndim != 2:
        raise ValueError("loss_profiles must have shape (n_points, n_data).")
    if profiles.shape[0] < 2:
        raise ValueError("At least two trajectory points are required.")
    if chunk_size < 1:
        raise ValueError("chunk_size must be positive.")

    n_points = profiles.shape[0]
    distance_matrix = np.empty((n_points, n_points), dtype=np.float64)

    for start in range(0, n_points, chunk_size):
        end = min(start + chunk_size, n_points)
        block = np.abs(profiles[start:end, None, :] - profiles[None, :, :]).mean(axis=2)
        distance_matrix[start:end] = block

    distance_matrix = 0.5 * (distance_matrix + distance_matrix.T)
    np.fill_diagonal(distance_matrix, 0.0)
    return distance_matrix


def pairwise_euclidean_distance(
    points: np.ndarray,
    *,
    chunk_size: int = 256,
) -> np.ndarray:
    """Compute a dense Euclidean distance matrix for finite point clouds."""
    points_array = np.asarray(points, dtype=np.float64)
    if points_array.ndim != 2:
        raise ValueError("points must have shape (n_points, n_features).")
    if points_array.shape[0] < 2:
        raise ValueError("At least two points are required.")
    if chunk_size < 1:
        raise ValueError("chunk_size must be positive.")

    n_points = points_array.shape[0]
    distance_matrix = np.empty((n_points, n_points), dtype=np.float64)

    for start in range(0, n_points, chunk_size):
        end = min(start + chunk_size, n_points)
        diff = points_array[start:end, None, :] - points_array[None, :, :]
        distance_matrix[start:end] = np.sqrt(np.sum(diff * diff, axis=2))

    distance_matrix = 0.5 * (distance_matrix + distance_matrix.T)
    np.fill_diagonal(distance_matrix, 0.0)
    return distance_matrix


def ph0_death_times_from_distance_matrix(distance_matrix: np.ndarray) -> np.ndarray:
    """
    Return PH0 death times for a finite pseudo-metric space.

    For the Vietoris-Rips filtration in degree 0, death times are exactly the
    edge weights of a minimum spanning tree. This dense Prim implementation keeps
    zero-length edges, which matters for pseudo-metrics.
    """
    distances = np.asarray(distance_matrix, dtype=np.float64)
    if distances.ndim != 2 or distances.shape[0] != distances.shape[1]:
        raise ValueError("distance_matrix must be a square array.")
    if distances.shape[0] < 2:
        return np.empty(0, dtype=np.float64)
    if np.any(distances < -1e-12):
        raise ValueError("distance_matrix contains negative distances.")
    if not np.all(np.isfinite(distances)):
        raise ValueError("distance_matrix contains non-finite values.")

    distances = np.maximum(distances, 0.0)
    distances = 0.5 * (distances + distances.T)
    np.fill_diagonal(distances, 0.0)

    n_points = distances.shape[0]
    visited = np.zeros(n_points, dtype=bool)
    best_distance = np.full(n_points, np.inf, dtype=np.float64)
    best_distance[0] = 0.0
    death_times: list[float] = []

    for step in range(n_points):
        candidate_distances = np.where(visited, np.inf, best_distance)
        next_index = int(np.argmin(candidate_distances))
        next_distance = float(candidate_distances[next_index])
        if not np.isfinite(next_distance):
            raise ValueError("distance_matrix does not define a connected complete graph.")

        visited[next_index] = True
        if step > 0:
            death_times.append(next_distance)

        remaining = ~visited
        best_distance[remaining] = np.minimum(
            best_distance[remaining],
            distances[next_index, remaining],
        )

    return np.asarray(death_times, dtype=np.float64)


def persistent_sum_from_distance_matrix(
    distance_matrix: np.ndarray,
    *,
    alpha: float = 1.0,
) -> float:
    if alpha <= 0.0:
        raise ValueError("alpha must be positive.")
    death_times = ph0_death_times_from_distance_matrix(distance_matrix)
    return float(np.sum(death_times**alpha))


def estimate_ph_dimension_from_distance_matrix(
    distance_matrix: np.ndarray,
    *,
    alpha: float = 1.0,
    subset_sizes: Sequence[int] | None = None,
    n_repeats: int = 16,
    seed: int = 0,
    eps: float = 1e-300,
) -> PHDimensionEstimateResult:
    """
    Estimate dim_PH0 from the scaling of E_alpha over random finite subsets.

    The estimator regresses log E_alpha(X_m) against log m. If the fitted slope
    is a, the PH dimension estimate is alpha / (1 - a).
    """
    if alpha <= 0.0:
        raise ValueError("alpha must be positive.")
    if n_repeats < 1:
        raise ValueError("n_repeats must be at least 1.")

    distances = np.asarray(distance_matrix, dtype=np.float64)
    if distances.ndim != 2 or distances.shape[0] != distances.shape[1]:
        raise ValueError("distance_matrix must be a square array.")

    n_points = distances.shape[0]
    if n_points < 4:
        raise ValueError("At least four points are recommended for PH dimension estimation.")

    if subset_sizes is None:
        raw_sizes = np.unique(
            np.round(np.geomspace(8, n_points, num=min(8, n_points))).astype(int)
        )
    else:
        raw_sizes = np.unique(np.asarray(subset_sizes, dtype=int))

    sizes = raw_sizes[(raw_sizes >= 3) & (raw_sizes <= n_points)]
    if sizes.size < 2:
        raise ValueError("subset_sizes must contain at least two valid sizes.")

    rng = np.random.default_rng(seed)
    e_alpha_by_size: list[np.ndarray] = []

    for size in sizes:
        repeat_count = 1 if size == n_points else n_repeats
        values = np.empty(repeat_count, dtype=np.float64)

        for repeat_index in range(repeat_count):
            if size == n_points:
                indices = np.arange(n_points)
            else:
                indices = rng.choice(n_points, size=size, replace=False)

            submatrix = distances[np.ix_(indices, indices)]
            values[repeat_index] = persistent_sum_from_distance_matrix(
                submatrix,
                alpha=alpha,
            )

        e_alpha_by_size.append(values)

    mean_e_alpha = np.asarray([values.mean() for values in e_alpha_by_size], dtype=np.float64)
    std_e_alpha = np.asarray(
        [
            values.std(ddof=1) if values.size > 1 else 0.0
            for values in e_alpha_by_size
        ],
        dtype=np.float64,
    )

    positive_mask = mean_e_alpha > eps
    if positive_mask.sum() < 2:
        return PHDimensionEstimateResult(
            dim_ph=0.0,
            alpha=float(alpha),
            slope=0.0,
            intercept=float("-inf"),
            subset_sizes=sizes,
            mean_e_alpha=mean_e_alpha,
            std_e_alpha=std_e_alpha,
            n_repeats=int(n_repeats),
            seed=int(seed),
        )

    log_sizes = np.log(sizes[positive_mask].astype(np.float64))
    log_e_alpha = np.log(mean_e_alpha[positive_mask])
    slope, intercept = np.polyfit(log_sizes, log_e_alpha, deg=1)
    dim_ph = float("inf") if slope >= 1.0 else float(alpha / (1.0 - slope))

    return PHDimensionEstimateResult(
        dim_ph=dim_ph,
        alpha=float(alpha),
        slope=float(slope),
        intercept=float(intercept),
        subset_sizes=sizes,
        mean_e_alpha=mean_e_alpha,
        std_e_alpha=std_e_alpha,
        n_repeats=int(n_repeats),
        seed=int(seed),
    )
