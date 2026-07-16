"""Generic two-operand comparison plots (PDF only)."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from numpy.ma import masked_where

from process_channels.features import ChannelSummary
from load_data.io import ARRAY_NAMES, CHANNELS_PER_ARRAY, array_nominal_channels, channel_label
from analyze_stability.arrays import array_combined_stats, channel_combined_mean_trace
from analyze_stability import combine as cs
from analyze_stability.plots_consistency import HEATMAP_DPI, HEATMAP_FIGSIZE
from analyze_stability.plots_lr import (
    LEFT_COLOR,
    RIGHT_COLOR,
    configure_array_time_axis,
    mark_empty_axis,
    trial_trace_sd,
)

PDF_DPI = 150
A4_LANDSCAPE_IN = (11.69, 8.27)
ARRAYS_COMBINED_FIG_IN = (14.0, 4.5)
TIMING_SE_ALPHA = 0.2
MAX_SESSION_COLS = 10
SESSION_PANEL_WIDTH_IN = 2.8
SESSION_ROW_HEIGHT_IN = 2.5
MISSING_HEATMAP_COLOR = "#cccccc"


def _timing_deep_dive_subplot_grid(n_sessions: int) -> tuple[int, int, tuple[float, float]]:
    """Return (nrows, ncols, figsize) for one panel per session (max 10 columns)."""
    if n_sessions <= 0:
        return 1, 1, A4_LANDSCAPE_IN
    if n_sessions <= MAX_SESSION_COLS:
        nrows, ncols = 1, n_sessions
    else:
        ncols = MAX_SESSION_COLS
        nrows = (n_sessions + ncols - 1) // ncols
        remainder = n_sessions % ncols
        if nrows == 2 and remainder != 0 and remainder <= ncols // 2:
            ncols = min(MAX_SESSION_COLS, (n_sessions + 2) // 3)
            nrows = (n_sessions + ncols - 1) // ncols
    fig_w = max(A4_LANDSCAPE_IN[0], SESSION_PANEL_WIDTH_IN * ncols)
    fig_h = max(A4_LANDSCAPE_IN[1], SESSION_ROW_HEIGHT_IN * nrows)
    return nrows, ncols, (fig_w, fig_h)


def _timing_overlay_legend_handles(label_a: str, label_b: str) -> list[Line2D]:
    return [
        Line2D([0], [0], color=LEFT_COLOR, linewidth=1.0, linestyle="-", label=f"{label_a} L"),
        Line2D([0], [0], color=RIGHT_COLOR, linewidth=1.0, linestyle="-", label=f"{label_a} R"),
        Line2D([0], [0], color=LEFT_COLOR, linewidth=1.0, linestyle="--", label=f"{label_b} L"),
        Line2D([0], [0], color=RIGHT_COLOR, linewidth=1.0, linestyle="--", label=f"{label_b} R"),
    ]


TIMING_LEGEND_BOTTOM = 0.20


def add_timing_trace_legend(fig, label_a: str, label_b: str) -> None:
    fig.legend(
        handles=_timing_overlay_legend_handles(label_a, label_b),
        loc="lower center",
        bbox_to_anchor=(0.5, 0.025),
        ncol=4,
        fontsize=6,
        frameon=True,
        handlelength=2.0,
    )


def validate_figure_legends_inside_canvas(fig) -> None:
    """Fail before saving when a figure-level legend would be clipped."""
    fig.canvas.draw()
    canvas = fig.bbox
    for legend in fig.legends:
        bbox = legend.get_window_extent(fig.canvas.get_renderer())
        if (
            bbox.x0 < canvas.x0
            or bbox.y0 < canvas.y0
            or bbox.x1 > canvas.x1
            or bbox.y1 > canvas.y1
        ):
            raise RuntimeError(
                "Figure legend extends outside the canvas "
                f"(legend={bbox.bounds}, canvas={canvas.bounds})"
            )


def _set_tight_ylim(ax, *trial_sets: np.ndarray) -> None:
    ymin, ymax = np.inf, -np.inf
    for trials in trial_sets:
        if trials.size == 0:
            continue
        mean = np.nanmean(trials, axis=0)
        sd = trial_trace_sd(trials)
        band_lo = mean - sd
        band_hi = mean + sd
        if not np.any(np.isfinite(band_lo)) or not np.any(np.isfinite(band_hi)):
            continue
        ymin = min(ymin, np.nanmin(band_lo))
        ymax = max(ymax, np.nanmax(band_hi))
    if not np.isfinite(ymin) or not np.isfinite(ymax):
        return
    pad = 0.05 * (ymax - ymin) if ymax > ymin else 0.1
    ax.set_ylim(ymin - pad, ymax + pad)


def plot_timing_overlay_subplot(
    ax,
    t_ms: np.ndarray,
    operand_a: ChannelSummary | None,
    operand_b: ChannelSummary | None,
    win_idx: np.ndarray,
    title: str,
    *,
    label_a: str,
    label_b: str,
) -> None:
    traces: list[np.ndarray] = []

    if operand_a is not None:
        if np.any(np.isfinite(operand_a.mean_left)):
            ax.plot(operand_a.t_ms, operand_a.mean_left, color=LEFT_COLOR, linewidth=1.0, linestyle="-")
            traces.append(operand_a.mean_left[np.newaxis, :])
        if np.any(np.isfinite(operand_a.mean_right)):
            ax.plot(operand_a.t_ms, operand_a.mean_right, color=RIGHT_COLOR, linewidth=1.0, linestyle="-")
            traces.append(operand_a.mean_right[np.newaxis, :])

    if operand_b is not None:
        if np.any(np.isfinite(operand_b.mean_left)):
            ax.plot(operand_b.t_ms, operand_b.mean_left, color=LEFT_COLOR, linewidth=1.0, linestyle="--")
            traces.append(operand_b.mean_left[np.newaxis, :])
        if np.any(np.isfinite(operand_b.mean_right)):
            ax.plot(operand_b.t_ms, operand_b.mean_right, color=RIGHT_COLOR, linewidth=1.0, linestyle="--")
            traces.append(operand_b.mean_right[np.newaxis, :])

    if not traces:
        mark_empty_axis(ax, title, "no data")
        return

    if win_idx.size:
        ax.axvspan(t_ms[win_idx[0]], t_ms[win_idx[-1]], color="0.85", alpha=0.35, zorder=0)
    ax.axvline(0, color="0.5", linewidth=0.6, linestyle="--")

    si_a = operand_a.si if operand_a is not None and np.isfinite(operand_a.si) else np.nan
    si_b = operand_b.si if operand_b is not None and np.isfinite(operand_b.si) else np.nan
    delta = si_b - si_a if np.isfinite(si_a) and np.isfinite(si_b) else np.nan
    task_a = "yes" if operand_a is not None and operand_a.task_evoked else "no"
    task_b = "yes" if operand_b is not None and operand_b.task_evoked else "no"

    ax.text(
        0.02,
        0.98,
        f"SI_{label_a}={si_a:.2f}\nSI_{label_b}={si_b:.2f}\nΔSI={delta:.2f}",
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=4,
        bbox=dict(boxstyle="round,pad=0.12", facecolor="white", alpha=0.85, edgecolor="none"),
    )
    ax.text(
        0.02,
        0.02,
        f"task {label_a}: {task_a}\ntask {label_b}: {task_b}",
        transform=ax.transAxes,
        va="bottom",
        ha="left",
        fontsize=4,
        bbox=dict(boxstyle="round,pad=0.12", facecolor="white", alpha=0.85, edgecolor="none"),
    )

    ax.set_title(title, fontsize=7)
    ax.tick_params(labelsize=5)
    ax.set_xlim(t_ms[0], t_ms[-1])
    _set_tight_ylim(ax, *traces)


def plot_session_array_timing_comparison(
    array_index: int,
    by_ch_a: dict[int, ChannelSummary],
    by_ch_b: dict[int, ChannelSummary],
    t_ms: np.ndarray,
    win_idx: np.ndarray,
    suptitle: str,
    out_path: Path,
    *,
    zscore_mua: bool,
    label_a: str,
    label_b: str,
    subplot_grid: tuple[int, int] = (4, 8),
    fig_size: tuple[float, float] = A4_LANDSCAPE_IN,
) -> None:
    array_name = ARRAY_NAMES[array_index]
    ch_nums = array_nominal_channels(array_index)
    n_rows, n_cols = subplot_grid
    fig, axes = plt.subplots(n_rows, n_cols, figsize=fig_size, sharex=True, sharey=False)
    y_label = "MUA (z)" if zscore_mua else "MUA"

    for panel_idx, ax in enumerate(np.asarray(axes).ravel()):
        ch_num = ch_nums[panel_idx]
        title = channel_label(ch_num, array_name, panel_idx + 1)
        plot_timing_overlay_subplot(
            ax,
            t_ms,
            by_ch_a.get(ch_num),
            by_ch_b.get(ch_num),
            win_idx,
            title,
            label_a=label_a,
            label_b=label_b,
        )
        if panel_idx % n_cols == 0:
            ax.set_ylabel(y_label, fontsize=6)

    configure_array_time_axis(axes, t_ms)
    fig.suptitle(suptitle, fontsize=9, y=0.98)
    fig.subplots_adjust(left=0.05, right=0.98, top=0.90, bottom=TIMING_LEGEND_BOTTOM, hspace=0.35, wspace=0.25)
    add_timing_trace_legend(fig, label_a, label_b)
    validate_figure_legends_inside_canvas(fig)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, format="pdf", dpi=PDF_DPI)
    plt.close(fig)


def _vstack_trials(parts: list[np.ndarray]) -> np.ndarray:
    return np.vstack(parts) if parts else np.empty((0, 0))


def plot_timing_pooled_channel_subplot(
    ax,
    t_ms: np.ndarray,
    win_idx: np.ndarray,
    left_parts_a: list[np.ndarray],
    right_parts_a: list[np.ndarray],
    left_parts_b: list[np.ndarray],
    right_parts_b: list[np.ndarray],
    title: str,
) -> None:
    left_a = _vstack_trials(left_parts_a)
    right_a = _vstack_trials(right_parts_a)
    left_b = _vstack_trials(left_parts_b)
    right_b = _vstack_trials(right_parts_b)
    if not any(arr.size for arr in (left_a, right_a, left_b, right_b)):
        mark_empty_axis(ax, title, "no data")
        return
    if left_a.size:
        cs.plot_condition(ax, t_ms, left_a, cs.LEFT_COLOR, "L")
        for line in ax.get_lines()[-1:]:
            line.set_linestyle("-")
    if right_a.size:
        cs.plot_condition(ax, t_ms, right_a, cs.RIGHT_COLOR, "R")
        for line in ax.get_lines()[-1:]:
            line.set_linestyle("-")
    if left_b.size:
        cs.plot_condition(ax, t_ms, left_b, cs.LEFT_COLOR, "L")
        for line in ax.get_lines()[-1:]:
            line.set_linestyle("--")
    if right_b.size:
        cs.plot_condition(ax, t_ms, right_b, cs.RIGHT_COLOR, "R")
        for line in ax.get_lines()[-1:]:
            line.set_linestyle("--")
    if win_idx.size:
        ax.axvspan(t_ms[win_idx[0]], t_ms[win_idx[-1]], color="0.85", alpha=0.35, zorder=0)
    ax.axvline(0, color="0.5", linewidth=0.6, linestyle="--")
    ax.set_title(title, fontsize=8)
    ax.tick_params(labelsize=6)
    ax.set_xlim(t_ms[0], t_ms[-1])
    cs._set_tight_ylim(ax, left_a, right_a, left_b, right_b)


def plot_timing_combined_array_grid(
    left_by_ch_a: dict[int, list[np.ndarray]],
    right_by_ch_a: dict[int, list[np.ndarray]],
    left_by_ch_b: dict[int, list[np.ndarray]],
    right_by_ch_b: dict[int, list[np.ndarray]],
    t_ms: np.ndarray,
    win_idx: np.ndarray,
    array_index: int,
    suptitle: str,
    out_path: Path,
    *,
    label_a: str,
    label_b: str,
) -> None:
    array_name = ARRAY_NAMES[array_index]
    ch_nums = array_nominal_channels(array_index)
    n_rows, n_cols = cs.SUBPLOT_GRID
    fig, axes = plt.subplots(n_rows, n_cols, figsize=cs.FIG_SIZE_IN, sharex=True, sharey=False)
    for panel_idx, ax in enumerate(np.asarray(axes).ravel()):
        ch_num = ch_nums[panel_idx]
        title = channel_label(ch_num, array_name, panel_idx + 1)
        plot_timing_pooled_channel_subplot(
            ax, t_ms, win_idx,
            left_by_ch_a.get(ch_num, []),
            right_by_ch_a.get(ch_num, []),
            left_by_ch_b.get(ch_num, []),
            right_by_ch_b.get(ch_num, []),
            title,
        )
        if panel_idx % n_cols == 0:
            ax.set_ylabel("MUA (z)", fontsize=7)
    configure_array_time_axis(axes, t_ms)
    fig.suptitle(suptitle, fontsize=10, y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    fig.subplots_adjust(bottom=TIMING_LEGEND_BOTTOM)
    add_timing_trace_legend(fig, label_a, label_b)
    validate_figure_legends_inside_canvas(fig)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, format="pdf", dpi=PDF_DPI, bbox_inches="tight")
    plt.close(fig)


def plot_array_timing_combined_panel(
    ax,
    t_ms: np.ndarray,
    win_idx: np.ndarray,
    array_name: str,
    stats_a: tuple,
    stats_b: tuple,
    *,
    label_a: str,
    label_b: str,
) -> None:
    a_l_mean, a_l_se, a_r_mean, a_r_se, n_a_l, n_a_r = stats_a
    b_l_mean, b_l_se, b_r_mean, b_r_se, n_b_l, n_b_r = stats_b
    has_data = any(
        arr is not None
        for arr in (a_l_mean, a_r_mean, b_l_mean, b_r_mean)
    )
    if not has_data:
        mark_empty_axis(ax, array_name, "no data")
        return
    if a_l_mean is not None and a_l_se is not None:
        plot_timing_mean_se(ax, t_ms, a_l_mean, a_l_se, LEFT_COLOR, linestyle="-")
    if a_r_mean is not None and a_r_se is not None:
        plot_timing_mean_se(ax, t_ms, a_r_mean, a_r_se, RIGHT_COLOR, linestyle="-")
    if b_l_mean is not None and b_l_se is not None:
        plot_timing_mean_se(ax, t_ms, b_l_mean, b_l_se, LEFT_COLOR, linestyle="--")
    if b_r_mean is not None and b_r_se is not None:
        plot_timing_mean_se(ax, t_ms, b_r_mean, b_r_se, RIGHT_COLOR, linestyle="--")
    if win_idx.size:
        ax.axvspan(t_ms[win_idx[0]], t_ms[win_idx[-1]], color="0.85", alpha=0.35, zorder=0)
    ax.axvline(0, color="0.5", linewidth=0.6, linestyle="--")
    ax.set_title(array_name, fontsize=9)
    ax.text(
        0.02, 0.98,
        (
            f"{label_a} nCh L={n_a_l}, R={n_a_r}\n"
            f"{label_b} nCh L={n_b_l}, R={n_b_r}"
        ),
        transform=ax.transAxes, va="top", ha="left", fontsize=6,
        bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.8, edgecolor="none"),
    )
    ax.tick_params(labelsize=6)
    ax.set_xlim(t_ms[0], t_ms[-1])


def plot_arrays_timing_combined(
    left_by_ch_a: dict[int, list[np.ndarray]],
    right_by_ch_a: dict[int, list[np.ndarray]],
    left_by_ch_b: dict[int, list[np.ndarray]],
    right_by_ch_b: dict[int, list[np.ndarray]],
    t_ms: np.ndarray,
    win_idx: np.ndarray,
    suptitle: str,
    out_path: Path,
    *,
    label_a: str,
    label_b: str,
    zscore_mua: bool,
) -> None:
    fig, axes = plt.subplots(1, len(ARRAY_NAMES), figsize=ARRAYS_COMBINED_FIG_IN, sharex=True, sharey=False)
    y_label = "MUA (z)" if zscore_mua else "MUA"
    for array_index, (array_name, ax) in enumerate(zip(ARRAY_NAMES, np.atleast_1d(axes))):
        stats_a = array_combined_stats(array_index, left_by_ch_a, right_by_ch_a)
        stats_b = array_combined_stats(array_index, left_by_ch_b, right_by_ch_b)
        plot_array_timing_combined_panel(
            ax, t_ms, win_idx, array_name, stats_a, stats_b,
            label_a=label_a, label_b=label_b,
        )
    configure_array_time_axis(axes, t_ms)
    np.atleast_1d(axes).ravel()[0].set_ylabel(y_label, fontsize=8)
    fig.suptitle(suptitle, fontsize=10, y=0.98)
    fig.subplots_adjust(left=0.05, right=0.98, top=0.82, bottom=TIMING_LEGEND_BOTTOM, wspace=0.35)
    add_timing_trace_legend(fig, label_a, label_b)
    validate_figure_legends_inside_canvas(fig)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, format="pdf", dpi=PDF_DPI, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)


def plot_timing_combined_outputs(
    left_by_ch_a: dict[int, list[np.ndarray]],
    right_by_ch_a: dict[int, list[np.ndarray]],
    left_by_ch_b: dict[int, list[np.ndarray]],
    right_by_ch_b: dict[int, list[np.ndarray]],
    t_ms: np.ndarray,
    win_idx: np.ndarray,
    combined_dir: Path,
    *,
    file_tag: str,
    label_a: str,
    label_b: str,
    suptitle_channel_grids: str,
    suptitle_arrays: str,
    zscore_mua: bool,
) -> None:
    combined_dir.mkdir(parents=True, exist_ok=True)
    for array_index, array_name in enumerate(ARRAY_NAMES):
        plot_timing_combined_array_grid(
            left_by_ch_a, right_by_ch_a, left_by_ch_b, right_by_ch_b,
            t_ms, win_idx, array_index,
            f"{suptitle_channel_grids} | {array_name}",
            combined_dir / f"{file_tag}_{array_name}_combined.pdf",
            label_a=label_a, label_b=label_b,
        )
    plot_arrays_timing_combined(
        left_by_ch_a, right_by_ch_a, left_by_ch_b, right_by_ch_b,
        t_ms, win_idx,
        suptitle_arrays,
        combined_dir / f"arrays_{file_tag}_combined.pdf",
        label_a=label_a, label_b=label_b,
        zscore_mua=zscore_mua,
    )


def plot_timing_mean_se(
    ax,
    t_ms: np.ndarray,
    mean: np.ndarray,
    se: np.ndarray,
    color: str,
    *,
    linestyle: str = "-",
) -> None:
    ax.plot(t_ms, mean, color=color, linewidth=1.2, linestyle=linestyle, zorder=3)
    ax.fill_between(
        t_ms,
        mean - se,
        mean + se,
        color=color,
        alpha=TIMING_SE_ALPHA,
        linewidth=0,
        zorder=2,
    )


def plot_matrix_heatmap(
    matrix: np.ndarray,
    session_ids: list[str],
    channels: list[int],
    title: str,
    out_path: Path,
    *,
    colorbar_label: str,
    symmetric: bool = True,
    vmin: float | None = None,
    vmax: float | None = None,
) -> None:
    data = matrix.T
    masked = masked_where(~np.isfinite(data), data)
    session_labels = [s.split(".")[0] for s in session_ids]

    if symmetric:
        finite = data[np.isfinite(data)]
        bound = float(np.nanmax(np.abs(finite))) if finite.size else 1.0
        if bound == 0:
            bound = 1.0
        vlo, vhi = -bound, bound
    else:
        vlo = vmin if vmin is not None else -1.0
        vhi = vmax if vmax is not None else 1.0

    base_cmap = plt.colormaps["RdBu_r"].copy()
    base_cmap.set_bad(MISSING_HEATMAP_COLOR)

    fig, ax = plt.subplots(figsize=HEATMAP_FIGSIZE)
    im = ax.imshow(masked, aspect="auto", cmap=base_cmap, vmin=vlo, vmax=vhi)
    ax.set_yticks(range(len(channels)))
    ax.set_yticklabels([f"{c:03d}" for c in channels], fontsize=6)
    ax.set_xticks(range(len(session_ids)))
    ax.set_xticklabels(session_labels, fontsize=8, rotation=45, ha="right")
    for i, _name in enumerate(ARRAY_NAMES):
        y = (i + 1) * CHANNELS_PER_ARRAY - 0.5
        if y < len(channels):
            ax.axhline(y, color="k", linewidth=0.5, alpha=0.4)
    ax.set_ylabel("Channel")
    ax.set_xlabel("Session")
    ax.set_title(title, fontsize=10)
    cbar = fig.colorbar(im, ax=ax, label=colorbar_label, shrink=0.85, pad=0.02)
    cbar.ax.tick_params(labelsize=7)
    fig.tight_layout()
    fig.savefig(out_path, format="pdf", dpi=HEATMAP_DPI, bbox_inches="tight")
    plt.close(fig)


def plot_delta_si_heatmap(
    matrix: np.ndarray,
    session_ids: list[str],
    channels: list[int],
    title: str,
    out_path: Path,
    *,
    colorbar_label: str,
) -> None:
    plot_matrix_heatmap(
        matrix,
        session_ids,
        channels,
        title,
        out_path,
        colorbar_label=colorbar_label,
        symmetric=True,
    )


def plot_waveform_r_heatmap(
    matrix: np.ndarray,
    session_ids: list[str],
    channels: list[int],
    title: str,
    out_path: Path,
    *,
    colorbar_label: str,
) -> None:
    plot_matrix_heatmap(
        matrix,
        session_ids,
        channels,
        title,
        out_path,
        colorbar_label=colorbar_label,
        symmetric=False,
        vmin=-1.0,
        vmax=1.0,
    )


def plot_timing_deep_dive_channel(
    by_session_a: dict[str, dict[int, ChannelSummary]],
    by_session_b: dict[str, dict[int, ChannelSummary]],
    session_ids: list[str],
    channel: int,
    win_idx: np.ndarray,
    title: str,
    out_path: Path,
    *,
    label_a: str,
    label_b: str,
) -> None:
    n_sessions = len(session_ids)
    nrows, ncols, figsize = _timing_deep_dive_subplot_grid(n_sessions)
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, sharex=True, sharey=True)
    axes_arr = np.atleast_1d(axes).ravel()

    y_vals: list[float] = []
    for sid in session_ids:
        for nest in (by_session_a, by_session_b):
            s = nest.get(sid, {}).get(channel)
            if s is None:
                continue
            for arr in (s.mean_left, s.mean_right):
                y_vals.extend(arr[np.isfinite(arr)].tolist())

    if y_vals:
        ymin, ymax = np.percentile(y_vals, [2, 98])
        pad = 0.05 * (ymax - ymin) if ymax > ymin else 0.1
        ylim = (ymin - pad, ymax + pad)
    else:
        ylim = None

    t_ms = None
    for idx, sid in enumerate(session_ids):
        ax = axes_arr[idx]
        operand_a = by_session_a.get(sid, {}).get(channel)
        operand_b = by_session_b.get(sid, {}).get(channel)
        if operand_a is None and operand_b is None:
            ax.set_title(f"{sid.split('.')[0]}\n[missing]", fontsize=6)
            ax.text(0.5, 0.5, "missing", transform=ax.transAxes, ha="center", va="center", fontsize=8, color="0.45")
            ax.set_xticks([])
            ax.set_yticks([])
            continue

        ref = operand_a or operand_b
        assert ref is not None
        t_ms = ref.t_ms
        plot_timing_overlay_subplot(
            ax, ref.t_ms, operand_a, operand_b, win_idx, "",
            label_a=label_a, label_b=label_b,
        )
        ax.set_title(f"{sid.split('.')[0]}", fontsize=7)
        if ylim:
            ax.set_ylim(ylim)

    for ax in axes_arr[n_sessions:]:
        ax.set_visible(False)

    if t_ms is not None:
        time_panel = (nrows - 1) * ncols
        configure_array_time_axis(axes_arr, t_ms, time_panel=time_panel)

    fig.suptitle(title, fontsize=9, y=0.98)
    bottom = 0.10 if nrows > 1 else 0.14
    fig.subplots_adjust(left=0.05, right=0.98, top=0.88, bottom=bottom, hspace=0.35, wspace=0.25)
    add_timing_trace_legend(fig, label_a, label_b)
    validate_figure_legends_inside_canvas(fig)
    fig.savefig(out_path, format="pdf", dpi=PDF_DPI)
    plt.close(fig)


def plot_si_scatter(
    si_a: np.ndarray,
    si_b: np.ndarray,
    task_either: np.ndarray,
    out_path: Path,
    *,
    suptitle: str | None = None,
    label_a: str,
    label_b: str,
) -> None:
    fig, ax = plt.subplots(figsize=(6, 6))
    no_task = ~task_either
    ax.scatter(
        si_a[no_task],
        si_b[no_task],
        s=10,
        facecolors="none",
        edgecolors="0.35",
        linewidths=0.5,
        alpha=0.6,
        label="no task (either)",
    )
    ax.scatter(
        si_a[task_either],
        si_b[task_either],
        s=10,
        c="#4c78a8",
        alpha=0.45,
        edgecolors="none",
        label="task (either)",
    )
    if si_a.size:
        lo = float(np.nanmin([np.nanmin(si_a), np.nanmin(si_b)]))
        hi = float(np.nanmax([np.nanmax(si_a), np.nanmax(si_b)]))
    else:
        lo, hi = -1.0, 1.0
    ax.plot([lo, hi], [lo, hi], linestyle="--", linewidth=1.0, color="0.4")
    ax.set_xlabel(f"SI {label_a}")
    ax.set_ylabel(f"SI {label_b}")
    if suptitle:
        fig.suptitle(suptitle, fontsize=10)
    ax.legend(loc="upper left", fontsize=7, frameon=True)
    ax.grid(alpha=0.2)
    fig.tight_layout(rect=[0, 0, 1, 0.95] if suptitle else None)
    fig.savefig(out_path, format="pdf", dpi=PDF_DPI, bbox_inches="tight")
    plt.close(fig)


def plot_si_channel_median_scatter(
    si_a_median: np.ndarray,
    si_b_median: np.ndarray,
    task_either: np.ndarray,
    out_path: Path,
    *,
    suptitle: str | None = None,
    label_a: str,
    label_b: str,
) -> None:
    fig, ax = plt.subplots(figsize=(6, 6))
    no_task = ~task_either
    ax.scatter(
        si_a_median[no_task],
        si_b_median[no_task],
        s=14,
        facecolors="none",
        edgecolors="0.35",
        linewidths=0.6,
        alpha=0.75,
        label="no task (either)",
    )
    ax.scatter(
        si_a_median[task_either],
        si_b_median[task_either],
        s=14,
        c="#4c78a8",
        alpha=0.65,
        edgecolors="none",
        label="task (either)",
    )
    if si_a_median.size:
        lo = float(np.nanmin([np.nanmin(si_a_median), np.nanmin(si_b_median)]))
        hi = float(np.nanmax([np.nanmax(si_a_median), np.nanmax(si_b_median)]))
    else:
        lo, hi = -1.0, 1.0
    ax.plot([lo, hi], [lo, hi], linestyle="--", linewidth=1.0, color="0.4")
    ax.set_xlabel(f"Median SI {label_a} (across sessions)")
    ax.set_ylabel(f"Median SI {label_b} (across sessions)")
    if suptitle:
        fig.suptitle(suptitle, fontsize=10)
    ax.legend(loc="upper left", fontsize=7, frameon=True)
    ax.grid(alpha=0.2)
    fig.tight_layout(rect=[0, 0, 1, 0.95] if suptitle else None)
    fig.savefig(out_path, format="pdf", dpi=PDF_DPI, bbox_inches="tight")
    plt.close(fig)


def plot_delta_si_vs_mean_si_scatter(
    mean_si: np.ndarray,
    delta_si: np.ndarray,
    task_either: np.ndarray,
    out_path: Path,
    *,
    suptitle: str | None = None,
    label_a: str,
    label_b: str,
) -> None:
    fig, ax = plt.subplots(figsize=(6, 6))
    no_task = ~task_either
    ax.scatter(
        mean_si[no_task],
        delta_si[no_task],
        s=10,
        facecolors="none",
        edgecolors="0.35",
        linewidths=0.5,
        alpha=0.6,
        label="no task (either)",
    )
    ax.scatter(
        mean_si[task_either],
        delta_si[task_either],
        s=10,
        c="#4c78a8",
        alpha=0.45,
        edgecolors="none",
        label="task (either)",
    )
    ax.axhline(0.0, color="0.4", linestyle="--", linewidth=0.8)
    ax.set_xlabel(f"Mean SI = (SI_{label_a} + SI_{label_b}) / 2")
    ax.set_ylabel(f"ΔSI = SI_{label_b} − SI_{label_a}")
    if suptitle:
        fig.suptitle(suptitle, fontsize=10)
    ax.legend(loc="upper right", fontsize=7, frameon=True)
    ax.grid(alpha=0.2)
    fig.tight_layout(rect=[0, 0, 1, 0.95] if suptitle else None)
    fig.savefig(out_path, format="pdf", dpi=PDF_DPI, bbox_inches="tight")
    plt.close(fig)


def _annotate_array_cell(ax, ch_num: int, value: float, r: int, c: int, vmax: float) -> None:
    if not np.isfinite(value):
        return
    text_color = "white" if abs(value) > 0.35 * vmax else "black"
    ax.text(c, r, f"{ch_num}", ha="center", va="center", fontsize=4.5, color=text_color)


def plot_median_delta_si_by_array(
    per_channel: list[dict],
    out_path: Path,
    *,
    suptitle: str | None = None,
    delta_label: str = "Median ΔSI",
) -> None:
    fig, axes = plt.subplots(1, len(ARRAY_NAMES), figsize=A4_LANDSCAPE_IN)
    axes_list = np.atleast_1d(axes)
    all_vals: list[float] = []
    grids: list[np.ndarray] = []
    ch_grids: list[np.ndarray] = []

    for array_idx in range(len(ARRAY_NAMES)):
        grid = np.full((4, 8), np.nan, dtype=float)
        ch_grid = np.full((4, 8), np.nan, dtype=float)
        for row in per_channel:
            if row.get("array_index") != array_idx:
                continue
            ch = int(row["channel"])
            idx = (ch - 1) % CHANNELS_PER_ARRAY
            r, c = divmod(idx, 8)
            grid[r, c] = float(row["delta_si_median"])
            ch_grid[r, c] = ch
            all_vals.append(float(row["delta_si_median"]))
        grids.append(grid)
        ch_grids.append(ch_grid)

    vmax = float(np.nanmax(np.abs(all_vals))) if all_vals else 1.0
    if vmax == 0:
        vmax = 1.0

    base_cmap = plt.colormaps["RdBu_r"].copy()
    base_cmap.set_bad(MISSING_HEATMAP_COLOR)
    im = None
    for ax, grid, ch_grid in zip(axes_list, grids, ch_grids):
        masked = masked_where(~np.isfinite(grid), grid)
        im = ax.imshow(masked, aspect="equal", cmap=base_cmap, vmin=-vmax, vmax=vmax)
        ax.set_xticks([])
        ax.set_yticks([])
        for r in range(4):
            for c in range(8):
                if np.isfinite(ch_grid[r, c]):
                    _annotate_array_cell(ax, int(ch_grid[r, c]), grid[r, c], r, c, vmax)

    for ax, array_idx in zip(axes_list, range(len(ARRAY_NAMES))):
        ax.set_title(ARRAY_NAMES[array_idx], fontsize=9)

    if im is not None:
        cax = fig.add_axes([0.92, 0.18, 0.015, 0.62])
        cbar = fig.colorbar(im, cax=cax, label=delta_label)
        cbar.ax.tick_params(labelsize=7)

    fig.suptitle(suptitle or "Median ΔSI by array (across sessions)", fontsize=10, y=0.96)
    fig.subplots_adjust(left=0.04, right=0.90, top=0.86, bottom=0.08, wspace=0.35)
    fig.savefig(out_path, format="pdf", dpi=PDF_DPI)
    plt.close(fig)
