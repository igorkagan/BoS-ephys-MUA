"""Preferred vs unpreferred combined plots (MWU-significant, per-session pref).

Does not replace L/R plots. Preference is defined within each session
(``pref_side``); it is never carried across sessions or trial-pooled.
Writes into the same ``combined/`` folders as L/R.

Paired Dyadic vs Solo overlays can lock ``pref_side`` to Solo MWU, keep
sessions present in both caches, average all gated channels within a
session, then mean ± SEM across those sessions.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Literal

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
    _timing_deep_dive_subplot_grid,
    plot_timing_mean_se,
    validate_figure_legends_inside_canvas,
)
from load_data.io import ARRAY_NAMES, array_nominal_channels, channel_label, filter_summary
from load_data.io import session_sort_key, window_indices
from process_channels.features import ChannelSummary
from process_channels.preprocess import processing_label, resolve_condition_output_dir
from run_pipeline.context import PipelineContext, apply_pipeline_context

MWU_ALPHA = 0.05
PREF_COLOR = "#2ca02c"
UNPREF_COLOR = "#7f7f7f"
COMBINED_SUBDIR = "combined"


LockSource = Literal["solo", "dyadic"]
SESSIONS_COMBINED_LOCKED = "sessions_combined_pref_solo_locked"
SESSIONS_LOCKED = "sessions_pref_solo_locked"
SESSIONS_MEAN_LOCKED = "sessions_mean_pref_solo_locked"


@dataclass(frozen=True)
class PrefUnprefTrace:
    mean_pref: np.ndarray
    sem_pref: np.ndarray
    mean_unpref: np.ndarray
    sem_unpref: np.ndarray
    n_sessions: int
    t_ms: np.ndarray


@dataclass(frozen=True)
class LockedSessionTrace:
    session_id: str
    n_channels: int
    t_ms: np.ndarray
    mean_pref_a: np.ndarray
    mean_unpref_a: np.ndarray
    mean_pref_b: np.ndarray
    mean_unpref_b: np.ndarray


@dataclass(frozen=True)
class LockedGrandMean:
    t_ms: np.ndarray
    n_sessions: int
    n_channels: tuple[int, ...]
    session_ids: tuple[str, ...]
    mean_pref_a: np.ndarray
    sem_pref_a: np.ndarray
    mean_unpref_a: np.ndarray
    sem_unpref_a: np.ndarray
    mean_pref_b: np.ndarray
    sem_pref_b: np.ndarray
    mean_unpref_b: np.ndarray
    sem_unpref_b: np.ndarray


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
    return remap_pref_unpref(summary, summary.pref_side)


def remap_pref_unpref(
    summary: ChannelSummary,
    pref_side: str,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Map L/R PSTHs onto pref/unpref using an explicit side."""
    if pref_side == "L":
        return np.asarray(summary.mean_left, dtype=float), np.asarray(summary.mean_right, dtype=float)
    if pref_side == "R":
        return np.asarray(summary.mean_right, dtype=float), np.asarray(summary.mean_left, dtype=float)
    return None


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


def comparison_solo_locked_pref_filenames(file_tag: str) -> list[str]:
    return [
        f"{file_tag}_{SESSIONS_COMBINED_LOCKED}.pdf",
        f"{file_tag}_{SESSIONS_LOCKED}.pdf",
        f"{file_tag}_{SESSIONS_MEAN_LOCKED}.pdf",
    ]


def _by_session_channel(
    summaries: list[ChannelSummary],
) -> dict[str, dict[int, ChannelSummary]]:
    nested: dict[str, dict[int, ChannelSummary]] = defaultdict(dict)
    for summary in summaries:
        nested[summary.session_id][int(summary.channel)] = summary
    return nested


def paired_session_ids(
    summaries_a: list[ChannelSummary],
    summaries_b: list[ChannelSummary],
) -> list[str]:
    ids_a = {summary.session_id for summary in summaries_a}
    ids_b = {summary.session_id for summary in summaries_b}
    return sorted(ids_a & ids_b)


def locked_session_traces(
    summaries_a: list[ChannelSummary],
    summaries_b: list[ChannelSummary],
    *,
    lock: LockSource = "solo",
    alpha: float = MWU_ALPHA,
    channels: Iterable[int] | None = None,
) -> list[LockedSessionTrace]:
    """Equal-weight channel mean per paired session, pref locked to ``lock``.

    ``summaries_a`` is Dyadic, ``summaries_b`` is Solo. A session×channel is
    kept when both caches have it and the lock source is MWU-tuned. The lock
    ``pref_side`` remaps both PSTHs. ``channels`` restricts the pool (array).
    """
    nest_a = _by_session_channel(summaries_a)
    nest_b = _by_session_channel(summaries_b)
    allowed = None if channels is None else {int(ch) for ch in channels}
    traces: list[LockedSessionTrace] = []
    for session_id in paired_session_ids(summaries_a, summaries_b):
        prefs_a: list[np.ndarray] = []
        unprefs_a: list[np.ndarray] = []
        prefs_b: list[np.ndarray] = []
        unprefs_b: list[np.ndarray] = []
        t_ms = None
        chans = set(nest_a[session_id]) & set(nest_b[session_id])
        if allowed is not None:
            chans &= allowed
        for channel in sorted(chans):
            row_a = nest_a[session_id][channel]
            row_b = nest_b[session_id][channel]
            source = row_b if lock == "solo" else row_a
            if not is_tuned(source, alpha=alpha):
                continue
            pair_a = remap_pref_unpref(row_a, source.pref_side)
            pair_b = remap_pref_unpref(row_b, source.pref_side)
            if pair_a is None or pair_b is None:
                continue
            prefs_a.append(pair_a[0])
            unprefs_a.append(pair_a[1])
            prefs_b.append(pair_b[0])
            unprefs_b.append(pair_b[1])
            if t_ms is None:
                t_ms = np.asarray(row_a.t_ms, dtype=float)
        mean_pa = equal_weight_mean(prefs_a)
        mean_ua = equal_weight_mean(unprefs_a)
        mean_pb = equal_weight_mean(prefs_b)
        mean_ub = equal_weight_mean(unprefs_b)
        if t_ms is None or mean_pa is None or mean_ua is None or mean_pb is None or mean_ub is None:
            continue
        traces.append(
            LockedSessionTrace(
                session_id=session_id,
                n_channels=len(prefs_a),
                t_ms=t_ms,
                mean_pref_a=mean_pa,
                mean_unpref_a=mean_ua,
                mean_pref_b=mean_pb,
                mean_unpref_b=mean_ub,
            )
        )
    return traces


def grand_from_locked_sessions(traces: list[LockedSessionTrace]) -> LockedGrandMean | None:
    if not traces:
        return None
    t_ms = np.asarray(traces[0].t_ms, dtype=float)
    mean_pa, sem_pa = mean_and_sem(mean_trace_stack([row.mean_pref_a for row in traces]))
    mean_ua, sem_ua = mean_and_sem(mean_trace_stack([row.mean_unpref_a for row in traces]))
    mean_pb, sem_pb = mean_and_sem(mean_trace_stack([row.mean_pref_b for row in traces]))
    mean_ub, sem_ub = mean_and_sem(mean_trace_stack([row.mean_unpref_b for row in traces]))
    return LockedGrandMean(
        t_ms=t_ms,
        n_sessions=len(traces),
        n_channels=tuple(row.n_channels for row in traces),
        session_ids=tuple(row.session_id for row in traces),
        mean_pref_a=mean_pa,
        sem_pref_a=sem_pa,
        mean_unpref_a=mean_ua,
        sem_unpref_a=sem_ua,
        mean_pref_b=mean_pb,
        sem_pref_b=sem_pb,
        mean_unpref_b=mean_ub,
        sem_unpref_b=sem_ub,
    )


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


def _nch_label(n_channels: tuple[int, ...]) -> str:
    if not n_channels:
        return "nCh=0"
    mean_n = float(np.mean(n_channels))
    return f"nSess={len(n_channels)}  mean nCh={mean_n:.1f}"


def _plot_locked_overlay(
    ax,
    t_ms: np.ndarray,
    grand: LockedGrandMean | None,
    win_idx: np.ndarray,
    title: str,
) -> None:
    if grand is None:
        mark_empty_axis(ax, title, "n.s.")
        return
    plot_timing_mean_se(ax, t_ms, grand.mean_pref_a, grand.sem_pref_a, PREF_COLOR, linestyle="-")
    plot_timing_mean_se(ax, t_ms, grand.mean_unpref_a, grand.sem_unpref_a, UNPREF_COLOR, linestyle="-")
    plot_timing_mean_se(ax, t_ms, grand.mean_pref_b, grand.sem_pref_b, PREF_COLOR, linestyle="--")
    plot_timing_mean_se(ax, t_ms, grand.mean_unpref_b, grand.sem_unpref_b, UNPREF_COLOR, linestyle="--")
    if win_idx.size:
        ax.axvspan(t_ms[win_idx[0]], t_ms[win_idx[-1]], color="0.85", alpha=0.35, zorder=0)
    ax.axvline(0, color="0.5", linewidth=0.6, linestyle="--")
    ax.set_title(title, fontsize=9)
    ax.text(
        0.02, 0.98, _nch_label(grand.n_channels),
        transform=ax.transAxes, va="top", ha="left", fontsize=6,
        bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.8, edgecolor="none"),
    )
    ax.tick_params(labelsize=6)
    ax.set_xlim(float(t_ms[0]), float(t_ms[-1]))


def _save_locked_npz(
    path: Path,
    traces: list[LockedSessionTrace],
    grand: LockedGrandMean,
    *,
    lock: LockSource,
    alpha: float,
) -> None:
    payload = {
        "lock": np.array(lock),
        "alpha": np.array(alpha, dtype=float),
        "t_ms": grand.t_ms,
        "session_ids": np.array(grand.session_ids, dtype=object),
        "n_channels": np.asarray(grand.n_channels, dtype=int),
        "mean_pref_a": grand.mean_pref_a,
        "sem_pref_a": grand.sem_pref_a,
        "mean_unpref_a": grand.mean_unpref_a,
        "sem_unpref_a": grand.sem_unpref_a,
        "mean_pref_b": grand.mean_pref_b,
        "sem_pref_b": grand.sem_pref_b,
        "mean_unpref_b": grand.mean_unpref_b,
        "sem_unpref_b": grand.sem_unpref_b,
        "session_pref_a": mean_trace_stack([row.mean_pref_a for row in traces]),
        "session_unpref_a": mean_trace_stack([row.mean_unpref_a for row in traces]),
        "session_pref_b": mean_trace_stack([row.mean_pref_b for row in traces]),
        "session_unpref_b": mean_trace_stack([row.mean_unpref_b for row in traces]),
    }
    np.savez_compressed(path, **payload)


def write_paired_pref_session_combined(
    summaries_a: list[ChannelSummary],
    summaries_b: list[ChannelSummary],
    combined_dir: Path,
    *,
    file_tag: str,
    label_a: str,
    label_b: str,
    suptitle: str,
    lock: LockSource = "solo",
    alpha: float = MWU_ALPHA,
) -> Path:
    """Session-first Solo-locked overlay: all gated channels, then sessions."""
    combined_dir.mkdir(parents=True, exist_ok=True)
    t_ms = _t_ms_from_summaries(summaries_a, summaries_b)
    all_traces = locked_session_traces(
        summaries_a, summaries_b, lock=lock, alpha=alpha,
    )
    grand_all = grand_from_locked_sessions(all_traces)
    if t_ms is None:
        t_ms = np.linspace(-1000.0, 1000.0, 2001)
    win_idx = window_indices(t_ms, ANALYSIS_WINDOW_MS)
    meta = (
        f"pref from {lock} MWU p<{alpha} | paired sessions | "
        f"equal-weight channels within session | mean ± SEM across sessions"
    )

    panel_specs: list[tuple[str, LockedGrandMean | None]] = [("All", grand_all)]
    for array_index, array_name in enumerate(ARRAY_NAMES):
        traces = locked_session_traces(
            summaries_a, summaries_b, lock=lock, alpha=alpha,
            channels=array_nominal_channels(array_index),
        )
        panel_specs.append((array_name, grand_from_locked_sessions(traces)))

    fig, axes = plt.subplots(
        1, len(panel_specs), figsize=(16.5, 4.5), sharex=True, sharey=False,
    )
    for i, (ax, (title, grand)) in enumerate(zip(np.atleast_1d(axes), panel_specs)):
        _plot_locked_overlay(ax, t_ms, grand, win_idx, title)
        if i == 0:
            ax.set_ylabel("MUA (z)", fontsize=8)
    configure_array_time_axis(axes, t_ms)
    fig.suptitle(f"{suptitle}\n{meta}", fontsize=10, y=0.98)
    fig.subplots_adjust(left=0.05, right=0.98, top=0.78, bottom=TIMING_LEGEND_BOTTOM, wspace=0.32)
    _add_pref_overlay_legend(fig, label_a, label_b)
    validate_figure_legends_inside_canvas(fig)
    out_grand = combined_dir / f"{file_tag}_{SESSIONS_COMBINED_LOCKED}.pdf"
    fig.savefig(out_grand, format="pdf", dpi=PDF_DPI, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.0, 5.0))
    _plot_locked_overlay(ax, t_ms, grand_all, win_idx, "All channels")
    ax.set_ylabel("MUA (z)", fontsize=8)
    configure_array_time_axis(np.array([ax]), t_ms)
    fig.suptitle(
        f"{suptitle} | mean of per-session all-channel traces\n{meta}",
        fontsize=10, y=0.98,
    )
    fig.subplots_adjust(left=0.12, right=0.97, top=0.82, bottom=0.22)
    _add_pref_overlay_legend(fig, label_a, label_b)
    validate_figure_legends_inside_canvas(fig)
    out_mean = combined_dir / f"{file_tag}_{SESSIONS_MEAN_LOCKED}.pdf"
    fig.savefig(out_mean, format="pdf", dpi=PDF_DPI, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)

    n_sess = len(all_traces)
    nrows, ncols, figsize = _timing_deep_dive_subplot_grid(max(n_sess, 1))
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, sharex=True, sharey=False)
    axes_flat = np.atleast_1d(axes).ravel()
    if n_sess == 0:
        mark_empty_axis(axes_flat[0], "no paired sessions", "n.s.")
        for ax in axes_flat[1:]:
            ax.axis("off")
    else:
        for i, ax in enumerate(axes_flat):
            if i >= n_sess:
                ax.axis("off")
                continue
            row = all_traces[i]
            ax.plot(row.t_ms, row.mean_pref_a, color=PREF_COLOR, linewidth=1.2, linestyle="-")
            ax.plot(row.t_ms, row.mean_unpref_a, color=UNPREF_COLOR, linewidth=1.2, linestyle="-")
            ax.plot(row.t_ms, row.mean_pref_b, color=PREF_COLOR, linewidth=1.2, linestyle="--")
            ax.plot(row.t_ms, row.mean_unpref_b, color=UNPREF_COLOR, linewidth=1.2, linestyle="--")
            if win_idx.size:
                ax.axvspan(row.t_ms[win_idx[0]], row.t_ms[win_idx[-1]], color="0.85", alpha=0.35, zorder=0)
            ax.axvline(0, color="0.5", linewidth=0.6, linestyle="--")
            ax.set_title(f"{session_sort_key(row.session_id)}  nCh={row.n_channels}", fontsize=8)
            ax.tick_params(labelsize=6)
            ax.set_xlim(float(row.t_ms[0]), float(row.t_ms[-1]))
            if i % ncols == 0:
                ax.set_ylabel("MUA (z)", fontsize=7)
        configure_array_time_axis(axes, t_ms)
    fig.suptitle(f"{suptitle} | per session (all gated channels)\n{meta}", fontsize=10, y=0.98)
    fig.subplots_adjust(left=0.05, right=0.98, top=0.86, bottom=0.16, wspace=0.28, hspace=0.45)
    _add_pref_overlay_legend(fig, label_a, label_b)
    validate_figure_legends_inside_canvas(fig)
    out_sess = combined_dir / f"{file_tag}_{SESSIONS_LOCKED}.pdf"
    fig.savefig(out_sess, format="pdf", dpi=PDF_DPI, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)

    if grand_all is not None:
        _save_locked_npz(
            combined_dir / f"{file_tag}_{SESSIONS_MEAN_LOCKED}.npz",
            all_traces,
            grand_all,
            lock=lock,
            alpha=alpha,
        )
    print(
        f"pref_unpref solo-locked session mean "
        f"(n_sessions={n_sess}) -> {combined_dir}"
    )
    return combined_dir

