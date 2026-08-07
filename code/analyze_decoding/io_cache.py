"""Save/load decode results (.npz)."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from analyze_decoding.combine_decode import CombinedDecodeResult
from analyze_decoding.config import (
    BIN_STEP_MS,
    BIN_WIDTH_MS,
    DECODE_WINDOW_MS,
    GAUSSIAN_SMOOTH_MODE,
    GAUSSIAN_SMOOTH_MS,
    NSHUFFLES,
)
from analyze_decoding.session_decode import SessionDecodeResult

NULL_LAYOUT = "time_locked"  # (nshuffles, n_bins)


def save_session_result(
    path: Path,
    result: SessionDecodeResult,
    *,
    bin_width_ms: float = BIN_WIDTH_MS,
    bin_step_ms: float = BIN_STEP_MS,
    window_ms: tuple[float, float] = DECODE_WINDOW_MS,
    nshuffles: int | None = None,
    smooth_ms: float = GAUSSIAN_SMOOTH_MS,
    smooth_mode: str = GAUSSIAN_SMOOTH_MODE,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    n_shuf = int(nshuffles if nshuffles is not None else result.null.shape[0])
    np.savez_compressed(
        path,
        session_id=result.session_id,
        go_seq=result.go_seq,
        trial_type=result.trial_type,
        target_key=result.target_key,
        bin_centers_ms=result.bin_centers_ms,
        perf_mean=result.perf_mean,
        perf_sem_cv=result.perf_sem_cv,
        null=result.null,
        pvalues=result.pvalues,
        n_trials=result.n_trials,
        n_left=result.n_left,
        n_right=result.n_right,
        n_channels=result.n_channels,
        alignment_event=result.alignment_event,
        bin_width_ms=float(bin_width_ms),
        bin_step_ms=float(bin_step_ms),
        # legacy alias (= step) for older readers
        bin_ms=float(bin_step_ms),
        window_lo=float(window_ms[0]),
        window_hi=float(window_ms[1]),
        nshuffles=n_shuf,
        smooth_ms=float(smooth_ms),
        smooth_mode=str(smooth_mode),
        null_layout=NULL_LAYOUT,
        cluster_mask=np.asarray(result.cluster_mask, dtype=bool),
        cluster_p=np.asarray(result.cluster_p, dtype=float),
        cluster_threshold=float(result.cluster_threshold)
        if np.isfinite(result.cluster_threshold)
        else np.nan,
    )


def cache_matches_params(
    path: Path,
    *,
    bin_width_ms: float = BIN_WIDTH_MS,
    bin_step_ms: float = BIN_STEP_MS,
    window_ms: tuple[float, float] = DECODE_WINDOW_MS,
    nshuffles: int = NSHUFFLES,
    min_bins: int = 30,
    smooth_ms: float = GAUSSIAN_SMOOTH_MS,
    smooth_mode: str = GAUSSIAN_SMOOTH_MODE,
) -> bool:
    """Reject partial / legacy caches (need time-locked nulls + cluster_mask)."""
    try:
        z = np.load(path, allow_pickle=True)
    except Exception:
        return False
    centers = z["bin_centers_ms"]
    if centers.size < min_bins:
        return False
    if "bin_width_ms" not in z.files or abs(float(z["bin_width_ms"]) - bin_width_ms) > 1e-6:
        return False
    if "bin_step_ms" not in z.files or abs(float(z["bin_step_ms"]) - bin_step_ms) > 1e-6:
        return False
    if "window_lo" in z.files and abs(float(z["window_lo"]) - window_ms[0]) > 1e-6:
        return False
    if "window_hi" in z.files and abs(float(z["window_hi"]) - window_ms[1]) > 1e-6:
        return False
    if "nshuffles" in z.files and int(z["nshuffles"]) != int(nshuffles):
        return False
    if "smooth_ms" not in z.files or abs(float(z["smooth_ms"]) - smooth_ms) > 1e-6:
        return False
    if "smooth_mode" not in z.files or str(z["smooth_mode"]) != str(smooth_mode):
        return False
    if "null_layout" not in z.files or str(z["null_layout"]) != NULL_LAYOUT:
        return False
    null = z["null"]
    if null.ndim != 2 or null.shape[0] != int(nshuffles) or null.shape[1] != centers.size:
        return False
    if "cluster_mask" not in z.files:
        return False
    return True


def load_session_result(path: Path) -> SessionDecodeResult:
    z = np.load(path, allow_pickle=True)
    cluster_mask = (
        z["cluster_mask"]
        if "cluster_mask" in z.files
        else np.zeros(z["bin_centers_ms"].shape, dtype=bool)
    )
    cluster_p = z["cluster_p"] if "cluster_p" in z.files else np.array([], dtype=float)
    cluster_thr = float(z["cluster_threshold"]) if "cluster_threshold" in z.files else np.nan
    return SessionDecodeResult(
        session_id=str(z["session_id"]),
        go_seq=str(z["go_seq"]),
        trial_type=str(z["trial_type"]),
        target_key=str(z["target_key"]),
        bin_centers_ms=z["bin_centers_ms"],
        perf_mean=z["perf_mean"],
        perf_sem_cv=z["perf_sem_cv"],
        null=z["null"],
        pvalues=z["pvalues"],
        n_trials=int(z["n_trials"]),
        n_left=int(z["n_left"]),
        n_right=int(z["n_right"]),
        n_channels=int(z["n_channels"]),
        alignment_event=str(z["alignment_event"]),
        cluster_mask=np.asarray(cluster_mask, dtype=bool),
        cluster_p=np.asarray(cluster_p, dtype=float),
        cluster_threshold=cluster_thr,
    )


def save_combined_result(path: Path, combined: CombinedDecodeResult) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        bin_centers_ms=combined.bin_centers_ms,
        mean=combined.mean,
        ci_low=combined.ci_low,
        ci_high=combined.ci_high,
        session_curves=combined.session_curves,
        session_ids=np.array(combined.session_ids, dtype=object),
        n_sessions_used=combined.n_sessions_used,
        cluster_mask=np.asarray(combined.cluster_mask, dtype=bool),
        cluster_p=np.asarray(combined.cluster_p, dtype=float),
        cluster_threshold=float(combined.cluster_threshold)
        if np.isfinite(combined.cluster_threshold)
        else np.nan,
    )
