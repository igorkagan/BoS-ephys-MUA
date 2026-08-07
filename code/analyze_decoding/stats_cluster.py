"""Cluster-based permutation inference (Maris & Oostenveld 2007 style)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats


@dataclass(frozen=True)
class ClusterResult:
    """One contiguous supra-threshold cluster."""

    start: int
    stop: int  # exclusive
    mass: float
    p_value: float


@dataclass(frozen=True)
class ClusterTestResult:
    clusters: list[ClusterResult]
    mask: np.ndarray  # bool, significant bins
    threshold: float


def _contiguous_runs(above: np.ndarray) -> list[tuple[int, int]]:
    """Return [start, stop) index runs where ``above`` is True."""
    above = np.asarray(above, dtype=bool)
    if above.size == 0:
        return []
    padded = np.concatenate([[False], above, [False]])
    d = np.diff(padded.astype(int))
    starts = np.flatnonzero(d == 1)
    stops = np.flatnonzero(d == -1)
    return list(zip(starts.tolist(), stops.tolist()))


def clusters_from_stat(
    stat: np.ndarray,
    threshold: float,
    *,
    one_sided: bool = True,
) -> list[tuple[int, int, float]]:
    """Form clusters where ``stat > threshold``; mass = sum of stat in cluster."""
    stat = np.asarray(stat, dtype=float)
    if one_sided:
        above = np.isfinite(stat) & (stat > threshold)
    else:
        above = np.isfinite(stat) & (np.abs(stat) > threshold)
    out: list[tuple[int, int, float]] = []
    for start, stop in _contiguous_runs(above):
        mass = float(np.nansum(stat[start:stop]))
        out.append((start, stop, mass))
    return out


def _max_cluster_mass(stat: np.ndarray, threshold: float) -> float:
    clusters = clusters_from_stat(stat, threshold)
    if not clusters:
        return 0.0
    return float(max(m for _, _, m in clusters))


def cluster_p_from_null_curves(
    observed: np.ndarray,
    null_curves: np.ndarray,
    *,
    chance: float = 0.5,
    cluster_forming_p: float = 0.05,
) -> ClusterTestResult:
    """Session-level clusters: observed curve vs time-locked null curves.

    ``null_curves`` shape ``(n_shuffles, n_bins)``. Cluster-forming threshold is
    the ``(1 - cluster_forming_p)`` quantile of all null (accuracy - chance)
    values. Cluster mass = sum of (accuracy - chance) in the cluster. P-values
    use the max cluster-mass distribution under the null curves (FWER).
    """
    observed = np.asarray(observed, dtype=float)
    null_curves = np.asarray(null_curves, dtype=float)
    if null_curves.ndim != 2:
        raise ValueError(f"null_curves must be 2D (n_shuf, n_bins), got {null_curves.shape}")
    if observed.shape[0] != null_curves.shape[1]:
        raise ValueError("observed length must match null_curves time axis")

    obs_stat = observed - chance
    null_stat = null_curves - chance
    finite_null = null_stat[np.isfinite(null_stat)]
    if finite_null.size == 0:
        return ClusterTestResult(clusters=[], mask=np.zeros(observed.shape, dtype=bool), threshold=np.nan)

    threshold = float(np.quantile(finite_null, 1.0 - cluster_forming_p))
    obs_clusters = clusters_from_stat(obs_stat, threshold)
    null_max = np.array(
        [_max_cluster_mass(null_stat[k], threshold) for k in range(null_stat.shape[0])],
        dtype=float,
    )
    n_shuf = max(1, null_max.size)
    clusters: list[ClusterResult] = []
    mask = np.zeros(observed.shape, dtype=bool)
    for start, stop, mass in obs_clusters:
        # add-one smoothing
        p = float((1 + np.sum(null_max >= mass)) / (1 + n_shuf))
        clusters.append(ClusterResult(start=start, stop=stop, mass=mass, p_value=p))
        if p < cluster_forming_p:
            mask[start:stop] = True

    return ClusterTestResult(clusters=clusters, mask=mask, threshold=threshold)


def cluster_p_signflip(
    session_curves: np.ndarray,
    *,
    chance: float = 0.5,
    n_perm: int = 5000,
    cluster_forming_p: float = 0.05,
    seed: int = 0,
) -> ClusterTestResult:
    """Across-session cluster test via sign-flipping of (acc - chance).

    ``session_curves`` shape ``(n_sessions, n_bins)``. Uses one-sample t-statistic
    across sessions at each time; cluster-forming threshold from ``t`` critical
    value at ``cluster_forming_p`` (one-sided). Null: random sign flips of
    session effects; FWER via max cluster mass.
    """
    X = np.asarray(session_curves, dtype=float)
    if X.ndim != 2 or X.shape[0] < 2:
        n_t = X.shape[-1] if X.ndim == 2 else 0
        return ClusterTestResult(
            clusters=[],
            mask=np.zeros(n_t, dtype=bool),
            threshold=np.nan,
        )

    Y = X - chance
    n_sess, n_t = Y.shape
    # observed t (one-sided interest in positive)
    mean = np.nanmean(Y, axis=0)
    n = np.sum(np.isfinite(Y), axis=0).astype(float)
    std = np.nanstd(Y, axis=0, ddof=1)
    sem = np.divide(std, np.sqrt(np.maximum(n, 1.0)), out=np.full(n_t, np.nan), where=n > 1)
    with np.errstate(divide="ignore", invalid="ignore"):
        t_obs = np.divide(mean, sem, out=np.zeros(n_t), where=np.isfinite(sem) & (sem > 0))

    # df ~ n-1; use min n across bins with data for a single threshold
    df = max(1, int(np.nanmax(n)) - 1)
    threshold = float(stats.t.ppf(1.0 - cluster_forming_p, df=df))

    obs_clusters = clusters_from_stat(t_obs, threshold)
    rng = np.random.default_rng(seed)
    null_max = np.zeros(n_perm, dtype=float)
    # only flip finite rows
    for p in range(n_perm):
        signs = rng.choice(np.array([-1.0, 1.0]), size=n_sess)
        Yp = Y * signs[:, None]
        mean_p = np.nanmean(Yp, axis=0)
        std_p = np.nanstd(Yp, axis=0, ddof=1)
        sem_p = np.divide(
            std_p, np.sqrt(np.maximum(n, 1.0)), out=np.full(n_t, np.nan), where=n > 1,
        )
        with np.errstate(divide="ignore", invalid="ignore"):
            t_p = np.divide(mean_p, sem_p, out=np.zeros(n_t), where=np.isfinite(sem_p) & (sem_p > 0))
        null_max[p] = _max_cluster_mass(t_p, threshold)

    clusters: list[ClusterResult] = []
    mask = np.zeros(n_t, dtype=bool)
    for start, stop, mass in obs_clusters:
        p_val = float((1 + np.sum(null_max >= mass)) / (1 + n_perm))
        clusters.append(ClusterResult(start=start, stop=stop, mass=mass, p_value=p_val))
        if p_val < cluster_forming_p:
            mask[start:stop] = True

    return ClusterTestResult(clusters=clusters, mask=mask, threshold=threshold)
