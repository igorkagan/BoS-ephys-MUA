from __future__ import annotations

import numpy as np
from scipy.stats import friedmanchisquare

from bos_mua.io import window_indices

TASK_EVOKED_BIN_MS = 50
TASK_EVOKED_WINDOW_MS = (-1000, 500)
TASK_EVOKED_ALPHA = 0.001


def bin_trials_by_time(t_ms: np.ndarray, trials: np.ndarray, bin_ms: float) -> np.ndarray:
    """Average trial values within contiguous time bins."""

    if trials.size == 0:
        return np.empty((0, 0))

    start = float(t_ms[0])
    end = float(t_ms[-1])
    edges = np.arange(start, end + bin_ms, bin_ms)
    if edges.size < 2:
        return np.empty((trials.shape[0], 0))

    cols: list[np.ndarray] = []
    for i in range(len(edges) - 1):
        lo, hi = edges[i], edges[i + 1]
        if i == len(edges) - 2:
            idx = (t_ms >= lo) & (t_ms <= hi)
        else:
            idx = (t_ms >= lo) & (t_ms < hi)
        if not np.any(idx):
            continue
        cols.append(np.nanmean(trials[:, idx], axis=1))

    if not cols:
        return np.empty((trials.shape[0], 0))
    return np.column_stack(cols)


def task_evoked_anova_pvalues(
    left_trials: np.ndarray,
    right_trials: np.ndarray,
    t_ms: np.ndarray,
    *,
    bin_ms: float = TASK_EVOKED_BIN_MS,
    window_ms: tuple[float, float] = TASK_EVOKED_WINDOW_MS,
    min_trials: int = 3,
) -> tuple[float | None, float | None]:
    """Friedman test on binned time (repeated measures across trials): modulated vs flat PSTH."""

    win_idx = window_indices(t_ms, window_ms)
    t_win = t_ms[win_idx] if win_idx.size else t_ms[:0]

    def _p(trials: np.ndarray) -> float | None:
        if trials.size == 0 or trials.shape[0] < min_trials or win_idx.size == 0:
            return None
        trials_win = trials[:, win_idx]
        binned = bin_trials_by_time(t_win, trials_win, bin_ms)
        if binned.shape[1] < 2:
            return None
        ok = np.all(np.isfinite(binned), axis=1)
        binned = binned[ok]
        if binned.shape[0] < min_trials:
            return None
        try:
            _, p = friedmanchisquare(*[binned[:, k] for k in range(binned.shape[1])])
        except ValueError:
            return 1.0
        return float(p) if np.isfinite(p) else 1.0

    return _p(left_trials), _p(right_trials)


def session_task_evoked(
    p_left: float | None,
    p_right: float | None,
    alpha: float = TASK_EVOKED_ALPHA,
) -> bool:
    left_sig = p_left is not None and p_left < alpha
    right_sig = p_right is not None and p_right < alpha
    return left_sig or right_sig
