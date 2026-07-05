from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from scipy.stats import pearsonr

if TYPE_CHECKING:
    from bos_mua.features import ChannelSummary


@dataclass
class ChannelStability:
    channel: int
    array_name: str
    n_sessions: int
    median_pairwise_r: float
    icc: float
    sign_concordance: float
    n_same_sign: int
    si_mean: float
    si_std: float
    si_median_abs: float
    si_pairwise_median_delta: float
    task_evoked: bool
    n_sessions_task_evoked: int
    stable: bool


def channel_task_evoked_all_sessions(
    lookup: dict[str, dict[int, ChannelSummary]],
    session_ids: list[str],
    channel: int,
) -> tuple[bool, int, int]:
    """True when every session with channel data passes task-evoked."""

    n_with_data = 0
    n_evoked = 0
    for sid in session_ids:
        summary = lookup.get(sid, {}).get(channel)
        if summary is None:
            continue
        n_with_data += 1
        if summary.task_evoked:
            n_evoked += 1
    if n_with_data == 0:
        return False, 0, 0
    return n_evoked == n_with_data, n_evoked, n_with_data


def si_pairwise_median_delta(si_values: np.ndarray) -> float:
    """Median |SI_i - SI_j| across session pairs."""
    finite = si_values[np.isfinite(si_values)]
    if finite.size < 2:
        return np.nan
    deltas: list[float] = []
    for i in range(finite.size):
        for j in range(i + 1, finite.size):
            deltas.append(float(abs(finite[i] - finite[j])))
    return float(np.median(deltas))


def pairwise_correlations(traces: np.ndarray) -> np.ndarray:
    """Pearson r for all session pairs. traces shape (n_sessions, n_time)."""
    n = traces.shape[0]
    if n < 2:
        return np.array([], dtype=float)

    rs: list[float] = []
    for i in range(n):
        for j in range(i + 1, n):
            a, b = traces[i], traces[j]
            if np.all(np.isnan(a)) or np.all(np.isnan(b)):
                continue
            mask = np.isfinite(a) & np.isfinite(b)
            if mask.sum() < 3:
                continue
            r, _ = pearsonr(a[mask], b[mask])
            if np.isfinite(r):
                rs.append(float(r))
    return np.asarray(rs, dtype=float)


def icc21(traces: np.ndarray) -> float:
    """ICC(2,1) with timepoints as targets and sessions as raters."""
    if traces.shape[0] < 2:
        return np.nan

    data = traces.T
    mask = np.all(np.isfinite(data), axis=1)
    data = data[mask]
    if data.shape[0] < 3:
        return np.nan

    n, k = data.shape
    mean_row = data.mean(axis=1)
    mean_col = data.mean(axis=0)
    grand_mean = data.mean()

    ss_total = np.sum((data - grand_mean) ** 2)
    ss_rows = k * np.sum((mean_row - grand_mean) ** 2)
    ss_cols = n * np.sum((mean_col - grand_mean) ** 2)
    ss_error = ss_total - ss_rows - ss_cols

    df_row = n - 1
    df_col = k - 1
    df_error = df_row * df_col
    if df_row <= 0 or df_col <= 0 or df_error <= 0:
        return np.nan

    ms_row = ss_rows / df_row
    ms_col = ss_cols / df_col
    ms_error = ss_error / df_error

    denom = ms_row + (k - 1) * ms_error + (k * (ms_col - ms_error) / n)
    if denom == 0:
        return np.nan
    return float((ms_row - ms_error) / denom)


def sign_concordance(values: np.ndarray) -> tuple[float, int]:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return np.nan, 0
    pos = int(np.sum(finite > 0))
    neg = int(np.sum(finite < 0))
    majority = max(pos, neg)
    return majority / finite.size, majority


def assess_channel_stability(
    channel: int,
    array_name: str,
    diff_traces: np.ndarray,
    si_values: np.ndarray,
    r_thresh: float,
    icc_thresh: float,
    sign_thresh: float,
    *,
    task_evoked: bool,
    n_sessions_task_evoked: int,
) -> ChannelStability:
    rs = pairwise_correlations(diff_traces)
    median_r = float(np.nanmedian(rs)) if rs.size else np.nan
    icc = icc21(diff_traces)
    concordance, n_same = sign_concordance(si_values)
    si_finite = si_values[np.isfinite(si_values)]
    si_median_abs = float(np.nanmedian(np.abs(si_finite))) if si_finite.size else np.nan
    si_pw_delta = si_pairwise_median_delta(si_values)

    stable = (
        np.isfinite(median_r)
        and median_r >= r_thresh
        and np.isfinite(icc)
        and icc >= icc_thresh
        and np.isfinite(concordance)
        and concordance >= sign_thresh
        and task_evoked
    )

    return ChannelStability(
        channel=channel,
        array_name=array_name,
        n_sessions=diff_traces.shape[0],
        median_pairwise_r=median_r,
        icc=icc,
        sign_concordance=concordance,
        n_same_sign=n_same,
        si_mean=float(np.nanmean(si_finite)) if si_finite.size else np.nan,
        si_std=float(np.nanstd(si_finite)) if si_finite.size else np.nan,
        si_median_abs=si_median_abs,
        si_pairwise_median_delta=si_pw_delta,
        task_evoked=task_evoked,
        n_sessions_task_evoked=n_sessions_task_evoked,
        stable=stable,
    )


def session_similarity_matrix(diff_tensor: np.ndarray) -> np.ndarray:
    """Median across channels of pairwise r between session difference waves."""
    n_sessions = diff_tensor.shape[0]
    sim = np.eye(n_sessions, dtype=float)
    for i in range(n_sessions):
        for j in range(i + 1, n_sessions):
            rs = []
            for ch in range(diff_tensor.shape[1]):
                a = diff_tensor[i, ch]
                b = diff_tensor[j, ch]
                mask = np.isfinite(a) & np.isfinite(b)
                if mask.sum() < 3:
                    continue
                r, _ = pearsonr(a[mask], b[mask])
                if np.isfinite(r):
                    rs.append(r)
            val = float(np.nanmedian(rs)) if rs else np.nan
            sim[i, j] = val
            sim[j, i] = val
    return sim


def rank_best_worst_channels(
    stabilities: list[ChannelStability],
    n: int = 10,
) -> tuple[list[ChannelStability], list[ChannelStability]]:
    """Rank stable channels by median pairwise r (best); worst unchanged."""
    if not stabilities:
        return [], []

    def sort_key(s: ChannelStability) -> float:
        return s.median_pairwise_r if np.isfinite(s.median_pairwise_r) else -np.inf

    stable_ranked = sorted([s for s in stabilities if s.stable], key=sort_key, reverse=True)
    all_ranked = sorted(stabilities, key=sort_key, reverse=True)
    n = min(n, len(stable_ranked)) if stable_ranked else 0
    best = stable_ranked[:n]
    n_worst = min(n, len(all_ranked))
    worst = all_ranked[-n_worst:][::-1] if n_worst else []
    return best, worst


def passes_tuned_stable_gate(
    stab: ChannelStability,
    sign_thresh: float,
    si_std_max: float,
    si_abs_min: float,
) -> bool:
    return (
        np.isfinite(stab.sign_concordance)
        and stab.sign_concordance >= sign_thresh
        and np.isfinite(stab.si_std)
        and stab.si_std <= si_std_max
        and np.isfinite(stab.si_median_abs)
        and stab.si_median_abs >= si_abs_min
        and stab.task_evoked
    )


def rank_tuned_stable_channels(
    stabilities: list[ChannelStability],
    n: int = 10,
    sign_thresh: float = 0.7,
    si_std_max: float = 0.30,
    si_abs_min: float = 0.10,
) -> list[ChannelStability]:
    """Strong tuning (median |SI|) among SI-stable channels (sign + low si_std)."""
    candidates = [
        s for s in stabilities
        if passes_tuned_stable_gate(s, sign_thresh, si_std_max, si_abs_min)
    ]

    def sort_key(s: ChannelStability) -> tuple:
        r = s.median_pairwise_r if np.isfinite(s.median_pairwise_r) else -np.inf
        return (-s.si_median_abs, s.si_std, -s.sign_concordance, -r)

    ranked = sorted(candidates, key=sort_key)
    return ranked[: min(n, len(ranked))]
