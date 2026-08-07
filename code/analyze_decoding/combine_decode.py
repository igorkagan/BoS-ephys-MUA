"""Average session decoding curves with across-session CI + cluster test."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy import stats

from analyze_decoding.session_decode import SessionDecodeResult
from analyze_decoding.stats_cluster import cluster_p_signflip


@dataclass
class CombinedDecodeResult:
    bin_centers_ms: np.ndarray
    mean: np.ndarray
    ci_low: np.ndarray
    ci_high: np.ndarray
    session_curves: np.ndarray  # (n_sessions, n_bins)
    session_ids: list[str]
    n_sessions_used: int
    cluster_mask: np.ndarray = field(default_factory=lambda: np.array([], dtype=bool))
    cluster_p: np.ndarray = field(default_factory=lambda: np.array([], dtype=float))
    cluster_threshold: float = np.nan


def combine_session_results(
    results: list[SessionDecodeResult],
    *,
    ci: float = 0.95,
    n_perm: int = 5000,
    cluster_forming_p: float = 0.05,
) -> CombinedDecodeResult:
    if not results:
        raise ValueError("No session results to combine")

    centers = results[0].bin_centers_ms
    ids: list[str] = []
    rows: list[np.ndarray] = []
    for r in results:
        if r.bin_centers_ms.shape != centers.shape or not np.allclose(
            r.bin_centers_ms, centers, equal_nan=True
        ):
            raise ValueError(f"Bin centers mismatch for {r.session_id}")
        if np.all(~np.isfinite(r.perf_mean)):
            continue
        ids.append(r.session_id)
        rows.append(r.perf_mean)

    if not rows:
        nan = np.full_like(centers, np.nan, dtype=float)
        return CombinedDecodeResult(
            bin_centers_ms=centers,
            mean=nan,
            ci_low=nan,
            ci_high=nan,
            session_curves=np.empty((0, centers.size)),
            session_ids=[],
            n_sessions_used=0,
            cluster_mask=np.zeros(centers.size, dtype=bool),
        )

    stack = np.vstack(rows)
    mean = np.nanmean(stack, axis=0)
    n = np.sum(np.isfinite(stack), axis=0).astype(float)
    ci_low = np.full_like(mean, np.nan)
    ci_high = np.full_like(mean, np.nan)
    alpha = 1.0 - ci
    for i in range(mean.size):
        ni = int(n[i])
        if ni <= 1 or not np.isfinite(mean[i]):
            ci_low[i] = mean[i]
            ci_high[i] = mean[i]
            continue
        col = stack[:, i]
        col = col[np.isfinite(col)]
        sem = float(np.std(col, ddof=1) / np.sqrt(ni))
        tcrit = float(stats.t.ppf(1.0 - alpha / 2.0, df=ni - 1))
        ci_low[i] = mean[i] - tcrit * sem
        ci_high[i] = mean[i] + tcrit * sem

    cluster = cluster_p_signflip(
        stack,
        chance=0.5,
        n_perm=n_perm,
        cluster_forming_p=cluster_forming_p,
    )
    return CombinedDecodeResult(
        bin_centers_ms=centers,
        mean=mean,
        ci_low=ci_low,
        ci_high=ci_high,
        session_curves=stack,
        session_ids=ids,
        n_sessions_used=len(ids),
        cluster_mask=cluster.mask,
        cluster_p=np.array([c.p_value for c in cluster.clusters], dtype=float),
        cluster_threshold=cluster.threshold,
    )
