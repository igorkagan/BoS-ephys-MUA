"""Preferred vs unpreferred combined plots (MWU-significant, per-session pref).

Does not replace L/R plots. Preference is defined within each session
(``pref_side``); it is never carried across sessions or trial-pooled.
Writes into the same ``combined/`` folders as L/R.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

from analyze_stability.consistency import ANALYSIS_WINDOW_MS, GAUSSIAN_SMOOTH_MS
from analyze_stability.plots_lr import configure_array_time_axis, mark_empty_axis
from analyze_stability.session_lr import DPI, FIG_SIZE_IN, SUBPLOT_GRID
from compare_conditions.plots_comparison import (
    ARRAYS_COMBINED_FIG_IN,
    PDF_DPI,
    TIMING_LEGEND_BOTTOM,
    plot_timing_mean_se,
    validate_figure_legends_inside_canvas,
)
from load_data.io import ARRAY_NAMES, array_nominal_channels, channel_label, filter_summary
from load_data.io import window_indices
from process_channels.features import ChannelSummary
from process_channels.preprocess import processing_label, resolve_condition_output_dir
from run_pipeline.context import PipelineContext, apply_pipeline_context

MWU_ALPHA = 0.05
PREF_COLOR = "#2ca02c"
UNPREF_COLOR = "#7f7f7f"
COMBINED_SUBDIR = "combined"


@dataclass(frozen=True)
class PrefUnprefTrace:
    mean_pref: np.ndarray
    sem_pref: np.ndarray
    mean_unpref: np.ndarray
    sem_unpref: np.ndarray
    n_sessions: int
    t_ms: np.ndarray


def is_tuned(summary: ChannelSummary, *, alpha: float = MWU_ALPHA) -> bool:
    """True when this session×channel has a significant L/R MWU and a pref side."""
    if summary.mwu_p is None or not np.isfinite(summary.mwu_p):
        return False
    if float(summary.mwu_p) >= alpha:
        return False
    return summary.pref_side in ("L", "R")


def pref_unpref_means(
    summary: ChannelSummary,
    *,
    alpha: float = MWU_ALPHA,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Return (pref, unpref) mean PSTHs, or None if not gated."""
    if not is_tuned(summary, alpha=alpha):
        return None
    if summary.pref_side == "L":
        return np.asarray(summary.mean_left, dtype=float), np.asarray(summary.mean_right, dtype=float)
    return np.asarray(summary.mean_right, dtype=float), np.asarray(summary.mean_left, dtype=float)


def mean_trace_stack(traces: list[np.ndarray]) -> np.ndarray:
    if not traces:
        return np.empty((0, 0), dtype=float)
    return np.vstack(traces)


def equal_weight_mean(traces: list[np.ndarray]) -> np.ndarray | None:
    stack = mean_trace_stack(traces)
    if stack.size == 0:
        return None
    return np.nanmean(stack, axis=0)


def mean_and_sem(stack: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Row = observation. SEM is 0 when n < 2."""
    mean = np.nanmean(stack, axis=0)
    if stack.shape[0] < 2:
        return mean, np.zeros_like(mean)
    sem = np.nanstd(stack, axis=0, ddof=1) / np.sqrt(stack.shape[0])
    return mean, sem


def gated_by_session(
    summaries: list[ChannelSummary],
    *,
    alpha: float = MWU_ALPHA,
) -> dict[str, list[ChannelSummary]]:
    grouped: dict[str, list[ChannelSummary]] = defaultdict(list)
    for summary in summaries:
        if is_tuned(summary, alpha=alpha):
            grouped[summary.session_id].append(summary)
    return grouped


def session_tuned_mean(
    summaries: list[ChannelSummary],
    *,
    alpha: float = MWU_ALPHA,
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """t_ms, mean_pref, mean_unpref for gated channels in this list (one session)."""
    prefs: list[np.ndarray] = []
    unprefs: list[np.ndarray] = []
    t_ms = None
    for summary in summaries:
        pair = pref_unpref_means(summary, alpha=alpha)
        if pair is None:
            continue
        pref, unpref = pair
        prefs.append(pref)
        unprefs.append(unpref)
        if t_ms is None:
            t_ms = np.asarray(summary.t_ms, dtype=float)
    mean_pref = equal_weight_mean(prefs)
    mean_unpref = equal_weight_mean(unprefs)
    if t_ms is None or mean_pref is None or mean_unpref is None:
        return None
    return t_ms, mean_pref, mean_unpref


def grand_mean_across_sessions(
    session_pref: list[np.ndarray],
    session_unpref: list[np.ndarray],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None:
    """mean_pref, sem_pref, mean_unpref, sem_unpref."""
    if not session_pref:
        return None
    mean_p, sem_p = mean_and_sem(mean_trace_stack(session_pref))
    mean_u, sem_u = mean_and_sem(mean_trace_stack(session_unpref))
    return mean_p, sem_p, mean_u, sem_u


def channel_combined_trace(
    summaries: list[ChannelSummary],
    channel: int,
    *,
    alpha: float = MWU_ALPHA,
) -> PrefUnprefTrace | None:
    """Equal-weight mean ± SEM across sessions where this channel is gated."""
    prefs: list[np.ndarray] = []
    unprefs: list[np.ndarray] = []
    t_ms = None
    for summary in summaries:
        if summary.channel != channel:
            continue
        pair = pref_unpref_means(summary, alpha=alpha)
        if pair is None:
            continue
        prefs.append(pair[0])
        unprefs.append(pair[1])
        if t_ms is None:
            t_ms = np.asarray(summary.t_ms, dtype=float)
    if not prefs or t_ms is None:
        return None
    mean_p, sem_p = mean_and_sem(mean_trace_stack(prefs))
    mean_u, sem_u = mean_and_sem(mean_trace_stack(unprefs))
    return PrefUnprefTrace(mean_p, sem_p, mean_u, sem_u, len(prefs), t_ms)


def traces_by_channel(
    summaries: list[ChannelSummary],
    *,
    alpha: float = MWU_ALPHA,
) -> dict[int, PrefUnprefTrace]:
    grouped: dict[int, list[ChannelSummary]] = defaultdict(list)
    for summary in summaries:
        grouped[summary.channel].append(summary)
    out: dict[int, PrefUnprefTrace] = {}
    for channel, rows in grouped.items():
        trace = channel_combined_trace(rows, channel, alpha=alpha)
        if trace is not None:
            out[channel] = trace
    return out


def array_combined_pref_stats(
    traces: dict[int, PrefUnprefTrace],
    array_index: int,
) -> tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None, np.ndarray | None, int]:
    """Mean ± SE across channels with ≥1 gated session."""
    prefs: list[np.ndarray] = []
    unprefs: list[np.ndarray] = []
    for channel in array_nominal_channels(array_index):
        trace = traces.get(channel)
        if trace is None:
            continue
        prefs.append(trace.mean_pref)
        unprefs.append(trace.mean_unpref)
    if not prefs:
        return None, None, None, None, 0
    mean_p, se_p = mean_and_sem(mean_trace_stack(prefs))
    mean_u, se_u = mean_and_sem(mean_trace_stack(unprefs))
    return mean_p, se_p, mean_u, se_u, len(prefs)


def branch_combined_pref_filenames(condition_label: str, alignment: str) -> list[str]:
    names = [
        f"{condition_label}_{alignment}_{array_name}_combined_pref.pdf"
        for array_name in ARRAY_NAMES
    ]
    names.append(f"{condition_label}_{alignment}_arrays_combined_pref.pdf")
    return names


def comparison_combined_pref_filenames(file_tag: str) -> list[str]:
    names = [f"{file_tag}_{array_name}_combined_pref.pdf" for array_name in ARRAY_NAMES]
    names.append(f"arrays_{file_tag}_combined_pref.pdf")
    return names


def _t_ms_from_summaries(*groups: list[ChannelSummary]) -> np.ndarray | None:
    for summaries in groups:
        for summary in summaries:
            t_ms = np.asarray(summary.t_ms, dtype=float)
            if t_ms.size:
                return t_ms
    return None


def _plot_mean_pair(
    ax,
    t_ms: np.ndarray,
    pref: np.ndarray,
    unpref: np.ndarray,
    *,
    pref_sem: np.ndarray | None = None,
    unpref_sem: np.ndarray | None = None,
    win_idx: np.ndarray | None = None,
    linestyle: str = "-",
) -> None:
    ax.plot(t_ms, pref, color=PREF_COLOR, linewidth=1.2, linestyle=linestyle, label="Preferred")
    ax.plot(t_ms, unpref, color=UNPREF_COLOR, linewidth=1.2, linestyle=linestyle, label="Unpreferred")
    if pref_sem is not None:
        ax.fill_between(t_ms, pref - pref_sem, pref + pref_sem, color=PREF_COLOR, alpha=0.25, linewidth=0)
    if unpref_sem is not None:
        ax.fill_between(
            t_ms, unpref - unpref_sem, unpref + unpref_sem,
            color=UNPREF_COLOR, alpha=0.25, linewidth=0,
        )
    if win_idx is not None and win_idx.size:
        ax.axvspan(t_ms[win_idx[0]], t_ms[win_idx[-1]], color="0.85", alpha=0.35, zorder=0)
    ax.axvline(0, color="0.5", linewidth=0.6, linestyle="--")
    ax.set_xlim(float(t_ms[0]), float(t_ms[-1]))


def _pref_legend_handles() -> list[Line2D]:
    return [
        Line2D([0], [0], color=PREF_COLOR, linewidth=1.2, label="Preferred"),
        Line2D([0], [0], color=UNPREF_COLOR, linewidth=1.2, label="Unpreferred"),
    ]


def _pref_overlay_legend_handles(label_a: str, label_b: str) -> list[Line2D]:
    return [
        Line2D([0], [0], color=PREF_COLOR, linewidth=1.0, linestyle="-", label=f"{label_a} pref"),
        Line2D([0], [0], color=UNPREF_COLOR, linewidth=1.0, linestyle="-", label=f"{label_a} unpref"),
        Line2D([0], [0], color=PREF_COLOR, linewidth=1.0, linestyle="--", label=f"{label_b} pref"),
        Line2D([0], [0], color=UNPREF_COLOR, linewidth=1.0, linestyle="--", label=f"{label_b} unpref"),
    ]


def _plot_channel_combined_panel(
    ax,
    trace: PrefUnprefTrace | None,
    title: str,
    win_idx: np.ndarray,
    t_ms: np.ndarray,
) -> None:
    if trace is None:
        mark_empty_axis(ax, title, "n.s.")
        return
    _plot_mean_pair(
        ax, trace.t_ms, trace.mean_pref, trace.mean_unpref,
        pref_sem=trace.sem_pref, unpref_sem=trace.sem_unpref, win_idx=win_idx,
    )
    ax.text(
        0.02, 0.98, f"n={trace.n_sessions}",
        transform=ax.transAxes, va="top", ha="left", fontsize=6,
        bbox=dict(boxstyle="round,pad=0.15", facecolor="white", alpha=0.75, edgecolor="none"),
    )
    ax.set_title(title, fontsize=8)
    ax.tick_params(labelsize=6)
    ax.set_xlim(float(t_ms[0]), float(t_ms[-1]))


def _make_array_combined_pref_figure(
    array_index: int,
    traces: dict[int, PrefUnprefTrace],
    t_ms: np.ndarray,
    win_idx: np.ndarray,
    suptitle: str,
) -> plt.Figure:
    array_name = ARRAY_NAMES[array_index]
    ch_nums = array_nominal_channels(array_index)
    n_rows, n_cols = SUBPLOT_GRID
    fig, axes = plt.subplots(n_rows, n_cols, figsize=FIG_SIZE_IN, sharex=True, sharey=False)
    for panel_idx, ax in enumerate(np.asarray(axes).ravel()):
        ch_num = ch_nums[panel_idx]
        title = channel_label(ch_num, array_name, panel_idx + 1)
        _plot_channel_combined_panel(ax, traces.get(ch_num), title, win_idx, t_ms)
        if panel_idx % n_cols == 0:
            ax.set_ylabel("MUA (z)", fontsize=7)
    configure_array_time_axis(axes, t_ms)
    fig.suptitle(suptitle, fontsize=10, y=0.995)
    fig.legend(handles=_pref_legend_handles(), loc="lower center", ncol=2, fontsize=8, frameon=True)
    fig.tight_layout(rect=[0, 0.04, 1, 0.98])
    return fig


def _make_arrays_combined_pref_figure(
    traces: dict[int, PrefUnprefTrace],
    t_ms: np.ndarray,
    win_idx: np.ndarray,
    suptitle: str,
) -> plt.Figure:
    fig, axes = plt.subplots(1, len(ARRAY_NAMES), figsize=ARRAYS_COMBINED_FIG_IN, sharex=True, sharey=False)
    for array_index, (array_name, ax) in enumerate(zip(ARRAY_NAMES, np.atleast_1d(axes))):
        mean_p, se_p, mean_u, se_u, n_ch = array_combined_pref_stats(traces, array_index)
        if mean_p is None:
            mark_empty_axis(ax, array_name, "n.s.")
            continue
        _plot_mean_pair(ax, t_ms, mean_p, mean_u, pref_sem=se_p, unpref_sem=se_u, win_idx=win_idx)
        ax.set_title(array_name, fontsize=9)
        ax.text(
            0.02, 0.98, f"nCh={n_ch}",
            transform=ax.transAxes, va="top", ha="left", fontsize=6,
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.8, edgecolor="none"),
        )
        ax.tick_params(labelsize=6)
        if array_index == 0:
            ax.set_ylabel("MUA (z)", fontsize=8)
    configure_array_time_axis(axes, t_ms, time_panel=0)
    fig.suptitle(suptitle, fontsize=10, y=1.02)
    fig.legend(handles=_pref_legend_handles(), loc="lower center", ncol=2, fontsize=8, frameon=True)
    fig.tight_layout(rect=[0, 0.08, 1, 0.95])
    return fig


def combined_output_dir(ctx: PipelineContext) -> Path:
    return resolve_condition_output_dir(
        ctx.output_base, True, ctx.condition_label, COMBINED_SUBDIR,
    )


def run_from_summaries(
    ctx: PipelineContext,
    summaries: list[ChannelSummary],
    *,
    alpha: float = MWU_ALPHA,
) -> Path:
    """Write A1–A5 + arrays ``*_combined_pref.pdf`` into ``{output_base}/combined/``."""
    apply_pipeline_context(ctx)
    output_dir = combined_output_dir(ctx)
    output_dir.mkdir(parents=True, exist_ok=True)
    t_ms = _t_ms_from_summaries(summaries)
    if t_ms is None:
        print(f"pref_unpref: no traces for {ctx.condition_label}; skip combined")
        return output_dir

    traces = traces_by_channel(summaries, alpha=alpha)
    win_idx = window_indices(t_ms, ANALYSIS_WINDOW_MS)
    alignment = ctx.alignment_event_label(ctx.session_ids)
    proc = processing_label(GAUSSIAN_SMOOTH_MS, zscore_mua=True)
    filters = filter_summary(ctx.trial_filters)
    n_gated_ch = len(traces)
    meta = (
        f"{filters} | {proc} | MWU p<{alpha} | equal-weight sessions per channel | "
        f"n_gated_ch={n_gated_ch}"
    )
    for array_index, array_name in enumerate(ARRAY_NAMES):
        fig = _make_array_combined_pref_figure(
            array_index,
            traces,
            t_ms,
            win_idx,
            (
                f"{ctx.condition_label} | combined sessions | {alignment} | pref vs unpref\n"
                f"{meta} | {array_name}"
            ),
        )
        out = output_dir / f"{ctx.condition_label}_{alignment}_{array_name}_combined_pref.pdf"
        fig.savefig(out, format="pdf", dpi=DPI, bbox_inches="tight")
        plt.close(fig)

    fig = _make_arrays_combined_pref_figure(
        traces,
        t_ms,
        win_idx,
        (
            f"{ctx.condition_label} | array mean ± SE across gated channels | {alignment}\n"
            f"{meta}"
        ),
    )
    out = output_dir / f"{ctx.condition_label}_{alignment}_arrays_combined_pref.pdf"
    fig.savefig(out, format="pdf", dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"pref_unpref combined ({n_gated_ch} gated channels) -> {output_dir}")
    return output_dir


def _plot_overlay_channel_panel(
    ax,
    trace_a: PrefUnprefTrace | None,
    trace_b: PrefUnprefTrace | None,
    title: str,
    t_ms: np.ndarray,
    win_idx: np.ndarray,
) -> None:
    if trace_a is None and trace_b is None:
        mark_empty_axis(ax, title, "n.s.")
        return
    if trace_a is not None:
        plot_timing_mean_se(ax, t_ms, trace_a.mean_pref, trace_a.sem_pref, PREF_COLOR, linestyle="-")
        plot_timing_mean_se(ax, t_ms, trace_a.mean_unpref, trace_a.sem_unpref, UNPREF_COLOR, linestyle="-")
    if trace_b is not None:
        plot_timing_mean_se(ax, t_ms, trace_b.mean_pref, trace_b.sem_pref, PREF_COLOR, linestyle="--")
        plot_timing_mean_se(ax, t_ms, trace_b.mean_unpref, trace_b.sem_unpref, UNPREF_COLOR, linestyle="--")
    if win_idx.size:
        ax.axvspan(t_ms[win_idx[0]], t_ms[win_idx[-1]], color="0.85", alpha=0.35, zorder=0)
    ax.axvline(0, color="0.5", linewidth=0.6, linestyle="--")
    n_a = 0 if trace_a is None else trace_a.n_sessions
    n_b = 0 if trace_b is None else trace_b.n_sessions
    ax.text(
        0.02, 0.98, f"n={n_a}/{n_b}",
        transform=ax.transAxes, va="top", ha="left", fontsize=6,
        bbox=dict(boxstyle="round,pad=0.15", facecolor="white", alpha=0.75, edgecolor="none"),
    )
    ax.set_title(title, fontsize=8)
    ax.tick_params(labelsize=6)
    ax.set_xlim(float(t_ms[0]), float(t_ms[-1]))


def _add_pref_overlay_legend(fig, label_a: str, label_b: str) -> None:
    fig.legend(
        handles=_pref_overlay_legend_handles(label_a, label_b),
        loc="lower center",
        bbox_to_anchor=(0.5, 0.025),
        ncol=4,
        fontsize=6,
        frameon=True,
        handlelength=2.0,
    )


def write_comparison_pref_combined(
    summaries_a: list[ChannelSummary],
    summaries_b: list[ChannelSummary],
    combined_dir: Path,
    *,
    file_tag: str,
    label_a: str,
    label_b: str,
    suptitle_grids: str,
    suptitle_arrays: str,
    alpha: float = MWU_ALPHA,
) -> Path:
    """Write comparison ``*_combined_pref.pdf`` next to L/R comparison combined PDFs."""
    combined_dir.mkdir(parents=True, exist_ok=True)
    t_ms = _t_ms_from_summaries(summaries_a, summaries_b)
    if t_ms is None:
        print(f"pref_unpref comparison: no traces for {file_tag}; skip")
        return combined_dir

    traces_a = traces_by_channel(summaries_a, alpha=alpha)
    traces_b = traces_by_channel(summaries_b, alpha=alpha)
    win_idx = window_indices(t_ms, ANALYSIS_WINDOW_MS)

    for array_index, array_name in enumerate(ARRAY_NAMES):
        ch_nums = array_nominal_channels(array_index)
        n_rows, n_cols = SUBPLOT_GRID
        fig, axes = plt.subplots(n_rows, n_cols, figsize=FIG_SIZE_IN, sharex=True, sharey=False)
        for panel_idx, ax in enumerate(np.asarray(axes).ravel()):
            ch_num = ch_nums[panel_idx]
            title = channel_label(ch_num, array_name, panel_idx + 1)
            _plot_overlay_channel_panel(
                ax, traces_a.get(ch_num), traces_b.get(ch_num), title, t_ms, win_idx,
            )
            if panel_idx % n_cols == 0:
                ax.set_ylabel("MUA (z)", fontsize=7)
        configure_array_time_axis(axes, t_ms)
        fig.suptitle(f"{suptitle_grids} | {array_name}", fontsize=10, y=0.995)
        fig.tight_layout(rect=[0, 0, 1, 0.98])
        fig.subplots_adjust(bottom=TIMING_LEGEND_BOTTOM)
        _add_pref_overlay_legend(fig, label_a, label_b)
        validate_figure_legends_inside_canvas(fig)
        out = combined_dir / f"{file_tag}_{array_name}_combined_pref.pdf"
        fig.savefig(out, format="pdf", dpi=PDF_DPI, bbox_inches="tight")
        plt.close(fig)

    fig, axes = plt.subplots(1, len(ARRAY_NAMES), figsize=ARRAYS_COMBINED_FIG_IN, sharex=True, sharey=False)
    for array_index, (array_name, ax) in enumerate(zip(ARRAY_NAMES, np.atleast_1d(axes))):
        mean_p_a, se_p_a, mean_u_a, se_u_a, n_a = array_combined_pref_stats(traces_a, array_index)
        mean_p_b, se_p_b, mean_u_b, se_u_b, n_b = array_combined_pref_stats(traces_b, array_index)
        has_data = mean_p_a is not None or mean_p_b is not None
        if not has_data:
            mark_empty_axis(ax, array_name, "n.s.")
            continue
        if mean_p_a is not None:
            plot_timing_mean_se(ax, t_ms, mean_p_a, se_p_a, PREF_COLOR, linestyle="-")
            plot_timing_mean_se(ax, t_ms, mean_u_a, se_u_a, UNPREF_COLOR, linestyle="-")
        if mean_p_b is not None:
            plot_timing_mean_se(ax, t_ms, mean_p_b, se_p_b, PREF_COLOR, linestyle="--")
            plot_timing_mean_se(ax, t_ms, mean_u_b, se_u_b, UNPREF_COLOR, linestyle="--")
        if win_idx.size:
            ax.axvspan(t_ms[win_idx[0]], t_ms[win_idx[-1]], color="0.85", alpha=0.35, zorder=0)
        ax.axvline(0, color="0.5", linewidth=0.6, linestyle="--")
        ax.set_title(array_name, fontsize=9)
        ax.text(
            0.02, 0.98,
            f"{label_a} nCh={n_a}\n{label_b} nCh={n_b}",
            transform=ax.transAxes, va="top", ha="left", fontsize=6,
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.8, edgecolor="none"),
        )
        ax.tick_params(labelsize=6)
        ax.set_xlim(float(t_ms[0]), float(t_ms[-1]))
        if array_index == 0:
            ax.set_ylabel("MUA (z)", fontsize=8)
    configure_array_time_axis(axes, t_ms)
    fig.suptitle(suptitle_arrays, fontsize=10, y=0.98)
    fig.subplots_adjust(left=0.05, right=0.98, top=0.82, bottom=TIMING_LEGEND_BOTTOM, wspace=0.35)
    _add_pref_overlay_legend(fig, label_a, label_b)
    validate_figure_legends_inside_canvas(fig)
    out = combined_dir / f"arrays_{file_tag}_combined_pref.pdf"
    fig.savefig(out, format="pdf", dpi=PDF_DPI, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
    print(f"pref_unpref comparison combined -> {combined_dir}")
    return combined_dir
