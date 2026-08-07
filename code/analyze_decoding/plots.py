"""Session and combined decoding figures."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from analyze_decoding.combine_decode import CombinedDecodeResult
from analyze_decoding.config import (
    BIN_STEP_MS,
    BIN_WIDTH_MS,
    CROSS_VALIDATIONS,
    DECODE_WINDOW_MS,
    GAUSSIAN_SMOOTH_MODE,
    GAUSSIAN_SMOOTH_MS,
    NSHUFFLES,
    TRAINING_FRACTION,
    bin_settings_label,
)
from analyze_decoding.session_decode import SessionDecodeResult

FIGSIZE = (8.0, 6.0)  # hor × ver with ver:hor = 3:4


def _settings_line(
    *,
    cv: int = CROSS_VALIDATIONS,
    train: float = TRAINING_FRACTION,
    nshuffles: int = NSHUFFLES,
) -> str:
    return f"CV={cv} | train={train:g} | null={nshuffles}"


def _annotate_axes(ax: plt.Axes, ylabel: str = "Decoding accuracy") -> None:
    ax.axhline(0.5, color="0.5", linestyle=":", linewidth=1.0, zorder=0)
    ax.axvline(0.0, color="k", linestyle="--", linewidth=1.0, alpha=0.6, zorder=0)
    ax.set_xlabel("Time from fixation release (ms)")
    ax.set_ylabel(ylabel)
    ax.set_ylim(0.0, 1.0)
    ax.set_xlim(DECODE_WINDOW_MS[0], DECODE_WINDOW_MS[1])


def _draw_cluster_bars(
    ax: plt.Axes,
    t: np.ndarray,
    mask: np.ndarray | None,
    *,
    y: float = 0.06,
    color: str = "k",
    label: str | None = "cluster p<0.05",
) -> None:
    if mask is None or mask.size == 0 or not np.any(mask):
        return
    mask = np.asarray(mask, dtype=bool)
    if mask.shape != t.shape:
        return
    # draw contiguous segments
    padded = np.concatenate([[False], mask, [False]])
    d = np.diff(padded.astype(int))
    starts = np.flatnonzero(d == 1)
    stops = np.flatnonzero(d == -1)
    labeled = False
    for s, e in zip(starts, stops):
        t0 = float(t[s])
        t1 = float(t[e - 1])
        kw = dict(colors=color, linewidth=4.0, zorder=5)
        if label is not None and not labeled:
            kw["label"] = label
            labeled = True
        ax.hlines(y, t0, t1, **kw)


def plot_session_decode(
    result: SessionDecodeResult,
    out_pdf: Path,
    *,
    condition: str,
    smooth_ms: float = GAUSSIAN_SMOOTH_MS,
    bin_width_ms: float = BIN_WIDTH_MS,
    bin_step_ms: float = BIN_STEP_MS,
    cv: int = CROSS_VALIDATIONS,
    train: float = TRAINING_FRACTION,
    nshuffles: int | None = None,
) -> None:
    t = result.bin_centers_ms
    y = result.perf_mean
    sem = result.perf_sem_cv
    null = result.null
    n_shuf = int(nshuffles if nshuffles is not None else (null.shape[0] if null.ndim == 2 else NSHUFFLES))

    fig, ax = plt.subplots(figsize=FIGSIZE)
    # null ± 2σ (time-locked: null is nshuffles × bins)
    if null.size and np.any(np.isfinite(null)):
        if null.ndim == 2 and null.shape[0] != t.size:
            null_mean = np.nanmean(null, axis=0)
            null_std = np.nanstd(null, axis=0)
        else:
            # legacy (n_bins, nshuffles)
            null_mean = np.nanmean(null, axis=1)
            null_std = np.nanstd(null, axis=1)
        ax.fill_between(
            t,
            null_mean - 2 * null_std,
            null_mean + 2 * null_std,
            color="0.75",
            alpha=0.45,
            linewidth=0,
            label="null ±2σ",
            zorder=1,
        )
        ax.plot(t, null_mean, color="0.45", linewidth=1.0, zorder=2)

    ax.fill_between(t, y - sem, y + sem, color="#1f77b4", alpha=0.25, linewidth=0, label="CV ±SEM", zorder=3)
    ax.plot(t, y, color="#1f77b4", marker="o", markersize=3.5, linewidth=1.5, label="mean CV", zorder=4)
    _draw_cluster_bars(ax, t, getattr(result, "cluster_mask", None))

    _annotate_axes(ax)
    ax.legend(loc="upper right", fontsize=8, frameon=False)
    ax.set_title(
        f"{result.session_id}\n"
        f"{condition} | {result.trial_type} | {result.go_seq} | {result.target_key}\n"
        f"align={result.alignment_event} | {bin_settings_label(bin_width_ms, bin_step_ms)} | "
        f"smooth={smooth_ms:g} ms ({GAUSSIAN_SMOOTH_MODE}) | "
        f"n={result.n_trials} (L={result.n_left}, R={result.n_right}) | ch={result.n_channels}\n"
        f"{_settings_line(cv=cv, train=train, nshuffles=n_shuf)}",
        fontsize=9,
    )
    fig.tight_layout()
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_pdf, dpi=150)
    plt.close(fig)


def plot_combined_decode(
    combined: CombinedDecodeResult,
    out_pdf: Path,
    *,
    condition: str,
    go_seq: str,
    trial_type: str,
    target: str,
    alignment_event: str,
    smooth_ms: float = GAUSSIAN_SMOOTH_MS,
    bin_width_ms: float = BIN_WIDTH_MS,
    bin_step_ms: float = BIN_STEP_MS,
    show_sessions: bool = True,
    cv: int = CROSS_VALIDATIONS,
    train: float = TRAINING_FRACTION,
    nshuffles: int = NSHUFFLES,
) -> None:
    t = combined.bin_centers_ms
    fig, ax = plt.subplots(figsize=FIGSIZE)
    if show_sessions and combined.session_curves.size:
        for row in combined.session_curves:
            ax.plot(t, row, color="0.75", linewidth=0.8, alpha=0.7, zorder=1)

    ax.fill_between(
        t,
        combined.ci_low,
        combined.ci_high,
        color="#d62728",
        alpha=0.25,
        linewidth=0,
        label="95% CI (sessions)",
        zorder=2,
    )
    ax.plot(
        t,
        combined.mean,
        color="#d62728",
        marker="o",
        markersize=3.5,
        linewidth=1.8,
        label=f"mean (n={combined.n_sessions_used})",
        zorder=3,
    )
    _draw_cluster_bars(
        ax,
        t,
        getattr(combined, "cluster_mask", None),
        label="cluster p<0.05 (sessions)",
    )
    _annotate_axes(ax)
    ax.legend(loc="upper right", fontsize=8, frameon=False)
    ax.set_title(
        f"{condition} | {trial_type} | {go_seq} | {target}\n"
        f"align={alignment_event} | {bin_settings_label(bin_width_ms, bin_step_ms)} | "
        f"smooth={smooth_ms:g} ms ({GAUSSIAN_SMOOTH_MODE}) | "
        f"window {DECODE_WINDOW_MS[0]:.0f}:{DECODE_WINDOW_MS[1]:.0f} ms\n"
        f"{_settings_line(cv=cv, train=train, nshuffles=nshuffles)}",
        fontsize=9,
    )
    fig.tight_layout()
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_pdf, dpi=150)
    plt.close(fig)
