"""2×2 choice grid and 1×2 same/diff figures."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from analyze_decoding.combine_decode import CombinedDecodeResult, combine_session_results
from analyze_decoding.config import (
    DECODE_WINDOW_MS,
    GAUSSIAN_SMOOTH_MODE,
    GAUSSIAN_SMOOTH_MS,
    bin_settings_label,
)
from analyze_decoding.plots import _annotate_axes, _draw_cluster_bars, _settings_line
from analyze_decoding.session_decode import SessionDecodeResult
from analyze_decoding.targets import align_short_label


def _class_count_str(result: SessionDecodeResult) -> str:
    key = result.target_key
    n0, n1 = result.n_left, result.n_right
    if key == "A_choice":
        return f"n={result.n_trials} (Al/Ar={n0}/{n1})"
    if key == "B_choice":
        return f"n={result.n_trials} (Bl/Br={n0}/{n1})"
    if key == "same_diff":
        return f"n={result.n_trials} (same/diff={n0}/{n1})"
    if key == "actor_choice":
        return f"n={result.n_trials} (L/R={n0}/{n1})"
    return f"n={result.n_trials} ({n0}/{n1})"


def _panel_session(
    ax: plt.Axes,
    result: SessionDecodeResult,
    *,
    title: str,
) -> None:
    t = result.bin_centers_ms
    y = result.perf_mean
    sem = result.perf_sem_cv
    null = result.null
    if null.size and np.any(np.isfinite(null)):
        null_mean = np.nanmean(null, axis=0)
        null_std = np.nanstd(null, axis=0)
        ax.fill_between(
            t, null_mean - 2 * null_std, null_mean + 2 * null_std,
            color="0.75", alpha=0.45, linewidth=0, zorder=1,
        )
        ax.plot(t, null_mean, color="0.45", linewidth=1.0, zorder=2)
    ax.fill_between(t, y - sem, y + sem, color="#1f77b4", alpha=0.25, linewidth=0, zorder=3)
    ax.plot(t, y, color="#1f77b4", marker="o", markersize=2.5, linewidth=1.2, zorder=4)
    _draw_cluster_bars(ax, t, result.cluster_mask, label=None)
    _annotate_axes(ax)
    ax.set_title(title, fontsize=8)


def _panel_combined(
    ax: plt.Axes,
    combined: CombinedDecodeResult,
    *,
    title: str,
) -> None:
    t = combined.bin_centers_ms
    if combined.session_curves.size:
        for row in combined.session_curves:
            ax.plot(t, row, color="0.75", linewidth=0.6, alpha=0.6, zorder=1)
    ax.fill_between(
        t, combined.ci_low, combined.ci_high,
        color="#d62728", alpha=0.25, linewidth=0, zorder=2,
    )
    ax.plot(
        t, combined.mean, color="#d62728", marker="o", markersize=2.5, linewidth=1.4, zorder=3,
    )
    _draw_cluster_bars(ax, t, combined.cluster_mask, label=None)
    _annotate_axes(ax)
    ax.set_title(title, fontsize=8)


def plot_choice_ab_grid_session(
    panels: dict[tuple[str, str], SessionDecodeResult],
    *,
    align_events: tuple[str, str],
    out_pdf: Path,
    condition: str,
    go_seq: str,
    session_id: str,
    balanced: bool,
    row_keys: tuple[str, str] = ("A_choice", "B_choice"),
) -> None:
    """panels keyed by (align_event, target_key)."""
    fig, axes = plt.subplots(2, 2, figsize=(10.0, 7.5), sharex=True, sharey=True)
    mode = "balanced (A_choice ⊥ B_choice)" if balanced else "unbalanced (single-var)"
    for col, event in enumerate(align_events):
        for row, key in enumerate(row_keys):
            ax = axes[row, col]
            res = panels[(event, key)]
            col_lab = f"{'1st' if col == 0 else '2nd'}: {align_short_label(event)}"
            row_lab = "decode A (Al/Ar)" if key == "A_choice" else "decode B (Bl/Br)"
            _panel_session(
                ax, res,
                title=f"{col_lab}\n{row_lab}\n{_class_count_str(res)}",
            )
            if row == 1:
                ax.set_xlabel("Time from action (ms)")
            if col == 0:
                ax.set_ylabel("Decoding accuracy")
    fig.suptitle(
        f"{session_id}\n{condition} | Dyadic | {go_seq} | choice A/B grid | {mode}\n"
        f"{bin_settings_label()} | smooth={GAUSSIAN_SMOOTH_MS:g} ms ({GAUSSIAN_SMOOTH_MODE}) | "
        f"window {DECODE_WINDOW_MS[0]:.0f}:{DECODE_WINDOW_MS[1]:.0f} ms | "
        f"{_settings_line()}",
        fontsize=9,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_pdf, dpi=150)
    plt.close(fig)


def plot_choice_ab_grid_combined(
    panels: dict[tuple[str, str], CombinedDecodeResult],
    *,
    align_events: tuple[str, str],
    out_pdf: Path,
    condition: str,
    go_seq: str,
    balanced: bool,
    row_keys: tuple[str, str] = ("A_choice", "B_choice"),
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(10.0, 7.5), sharex=True, sharey=True)
    mode = "balanced (A_choice ⊥ B_choice)" if balanced else "unbalanced (single-var)"
    for col, event in enumerate(align_events):
        for row, key in enumerate(row_keys):
            ax = axes[row, col]
            comb = panels[(event, key)]
            col_lab = f"{'1st' if col == 0 else '2nd'}: {align_short_label(event)}"
            row_lab = "decode A (Al/Ar)" if key == "A_choice" else "decode B (Bl/Br)"
            _panel_combined(
                ax, comb,
                title=f"{col_lab}\n{row_lab} | n_sess={comb.n_sessions_used}",
            )
    fig.suptitle(
        f"{condition} | Dyadic | {go_seq} | choice A/B grid | {mode}\n"
        f"{bin_settings_label()} | smooth={GAUSSIAN_SMOOTH_MS:g} ms ({GAUSSIAN_SMOOTH_MODE}) | "
        f"{_settings_line()}",
        fontsize=9,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_pdf, dpi=150)
    plt.close(fig)


def plot_same_diff_session(
    panels: dict[str, SessionDecodeResult],
    *,
    align_events: tuple[str, str],
    out_pdf: Path,
    condition: str,
    go_seq: str,
    session_id: str,
) -> None:
    """panels keyed by align_event."""
    fig, axes = plt.subplots(1, 2, figsize=(10.0, 4.5), sharey=True)
    for col, event in enumerate(align_events):
        ax = axes[col]
        res = panels[event]
        col_lab = f"{'1st' if col == 0 else '2nd'}: {align_short_label(event)}"
        _panel_session(
            ax, res,
            title=f"{col_lab}\ndecode same/diff\n{_class_count_str(res)}",
        )
        ax.set_xlabel("Time from action (ms)")
        if col == 0:
            ax.set_ylabel("Decoding accuracy")
    fig.suptitle(
        f"{session_id}\n{condition} | Dyadic | {go_seq} | same vs diff | class-balanced\n"
        f"{bin_settings_label()} | smooth={GAUSSIAN_SMOOTH_MS:g} ms ({GAUSSIAN_SMOOTH_MODE}) | {_settings_line()}",
        fontsize=9,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.84))
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_pdf, dpi=150)
    plt.close(fig)


def plot_same_diff_combined(
    panels: dict[str, CombinedDecodeResult],
    *,
    align_events: tuple[str, str],
    out_pdf: Path,
    condition: str,
    go_seq: str,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10.0, 4.5), sharey=True)
    for col, event in enumerate(align_events):
        ax = axes[col]
        comb = panels[event]
        col_lab = f"{'1st' if col == 0 else '2nd'}: {align_short_label(event)}"
        _panel_combined(
            ax, comb,
            title=f"{col_lab}\nsame/diff | n_sess={comb.n_sessions_used}",
        )
        ax.set_xlabel("Time from action (ms)")
    fig.suptitle(
        f"{condition} | Dyadic | {go_seq} | same vs diff | class-balanced\n"
        f"{bin_settings_label()} | smooth={GAUSSIAN_SMOOTH_MS:g} ms ({GAUSSIAN_SMOOTH_MODE}) | "
        f"{_settings_line()}",
        fontsize=9,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.86))
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_pdf, dpi=150)
    plt.close(fig)


def combine_panel_dict(
    session_panels: list[dict],
    *,
    keys: list,
) -> dict:
    """Combine list of session panel dicts sharing the same keys."""
    out = {}
    for key in keys:
        results = [p[key] for p in session_panels if key in p]
        if results:
            out[key] = combine_session_results(results)
    return out
