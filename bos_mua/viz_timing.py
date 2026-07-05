"""AgoB vs BgoA timing-comparison plots (PDF only)."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from numpy.ma import masked_where

from bos_mua.features import ChannelSummary
from bos_mua.io import ARRAY_NAMES, CHANNELS_PER_ARRAY, array_nominal_channels, channel_label
from bos_mua.viz_consistency import HEATMAP_DPI, HEATMAP_FIGSIZE
from bos_mua.viz_lr import LEFT_COLOR, RIGHT_COLOR, configure_array_time_axis, mark_empty_axis

PDF_DPI = 150
A4_LANDSCAPE_IN = (11.69, 8.27)
MISSING_HEATMAP_COLOR = "#cccccc"


def _timing_overlay_legend_handles() -> list[Line2D]:
    return [
        Line2D([0], [0], color=LEFT_COLOR, linewidth=1.0, linestyle="-", label="AgoB L"),
        Line2D([0], [0], color=RIGHT_COLOR, linewidth=1.0, linestyle="-", label="AgoB R"),
        Line2D([0], [0], color=LEFT_COLOR, linewidth=1.0, linestyle="--", label="BgoA L"),
        Line2D([0], [0], color=RIGHT_COLOR, linewidth=1.0, linestyle="--", label="BgoA R"),
    ]


def add_timing_trace_legend(fig) -> None:
    fig.legend(
        handles=_timing_overlay_legend_handles(),
        loc="upper center",
        bbox_to_anchor=(0.5, 0.02),
        ncol=4,
        fontsize=6,
        frameon=True,
        handlelength=2.0,
    )


def _set_tight_ylim(ax, *trial_sets: np.ndarray) -> None:
    ymin, ymax = np.inf, -np.inf
    for trials in trial_sets:
        if trials.size == 0:
            continue
        mean = np.nanmean(trials, axis=0)
        sd = np.nanstd(trials, axis=0, ddof=1)
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
    agob: ChannelSummary | None,
    bgoa: ChannelSummary | None,
    win_idx: np.ndarray,
    title: str,
) -> None:
    traces: list[np.ndarray] = []

    if agob is not None:
        if np.any(np.isfinite(agob.mean_left)):
            ax.plot(agob.t_ms, agob.mean_left, color=LEFT_COLOR, linewidth=1.0, linestyle="-")
            traces.append(agob.mean_left[np.newaxis, :])
        if np.any(np.isfinite(agob.mean_right)):
            ax.plot(agob.t_ms, agob.mean_right, color=RIGHT_COLOR, linewidth=1.0, linestyle="-")
            traces.append(agob.mean_right[np.newaxis, :])

    if bgoa is not None:
        if np.any(np.isfinite(bgoa.mean_left)):
            ax.plot(bgoa.t_ms, bgoa.mean_left, color=LEFT_COLOR, linewidth=1.0, linestyle="--")
            traces.append(bgoa.mean_left[np.newaxis, :])
        if np.any(np.isfinite(bgoa.mean_right)):
            ax.plot(bgoa.t_ms, bgoa.mean_right, color=RIGHT_COLOR, linewidth=1.0, linestyle="--")
            traces.append(bgoa.mean_right[np.newaxis, :])

    if not traces:
        mark_empty_axis(ax, title, "no data")
        return

    if win_idx.size:
        ax.axvspan(t_ms[win_idx[0]], t_ms[win_idx[-1]], color="0.85", alpha=0.35, zorder=0)
    ax.axvline(0, color="0.5", linewidth=0.6, linestyle="--")

    si_a = agob.si if agob is not None and np.isfinite(agob.si) else np.nan
    si_b = bgoa.si if bgoa is not None and np.isfinite(bgoa.si) else np.nan
    delta = si_b - si_a if np.isfinite(si_a) and np.isfinite(si_b) else np.nan
    task_a = "yes" if agob is not None and agob.task_evoked else "no"
    task_b = "yes" if bgoa is not None and bgoa.task_evoked else "no"

    ax.text(
        0.02,
        0.98,
        f"SI_AgoB={si_a:.2f}\nSI_BgoA={si_b:.2f}\nΔSI={delta:.2f}",
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=4,
        bbox=dict(boxstyle="round,pad=0.12", facecolor="white", alpha=0.85, edgecolor="none"),
    )
    ax.text(
        0.02,
        0.02,
        f"task AgoB: {task_a}\ntask BgoA: {task_b}",
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
    agob_by_ch: dict[int, ChannelSummary],
    bgoa_by_ch: dict[int, ChannelSummary],
    t_ms: np.ndarray,
    win_idx: np.ndarray,
    suptitle: str,
    out_path: Path,
    *,
    zscore_mua: bool,
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
            agob_by_ch.get(ch_num),
            bgoa_by_ch.get(ch_num),
            win_idx,
            title,
        )
        if panel_idx % n_cols == 0:
            ax.set_ylabel(y_label, fontsize=6)

    configure_array_time_axis(axes, t_ms)
    fig.suptitle(suptitle, fontsize=9, y=0.98)
    fig.subplots_adjust(left=0.05, right=0.98, top=0.90, bottom=0.12, hspace=0.35, wspace=0.25)
    add_timing_trace_legend(fig)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, format="pdf", dpi=PDF_DPI)
    plt.close(fig)


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
) -> None:
    plot_matrix_heatmap(
        matrix,
        session_ids,
        channels,
        title,
        out_path,
        colorbar_label="ΔSI (BgoA − AgoB)",
        symmetric=True,
    )


def plot_waveform_r_heatmap(
    matrix: np.ndarray,
    session_ids: list[str],
    channels: list[int],
    title: str,
    out_path: Path,
) -> None:
    plot_matrix_heatmap(
        matrix,
        session_ids,
        channels,
        title,
        out_path,
        colorbar_label="Waveform r (AgoB vs BgoA L−R diff)",
        symmetric=False,
        vmin=-1.0,
        vmax=1.0,
    )


def plot_timing_deep_dive_channel(
    agob_by_session: dict[str, dict[int, ChannelSummary]],
    bgoa_by_session: dict[str, dict[int, ChannelSummary]],
    session_ids: list[str],
    channel: int,
    win_idx: np.ndarray,
    title: str,
    out_path: Path,
) -> None:
    n_sessions = len(session_ids)
    fig_w = max(A4_LANDSCAPE_IN[0], 2.8 * n_sessions)
    fig, axes = plt.subplots(1, n_sessions, figsize=(fig_w, A4_LANDSCAPE_IN[1]), sharex=True, sharey=True)
    axes_arr = np.atleast_1d(axes)

    y_vals: list[float] = []
    for sid in session_ids:
        for nest in (agob_by_session, bgoa_by_session):
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
        agob = agob_by_session.get(sid, {}).get(channel)
        bgoa = bgoa_by_session.get(sid, {}).get(channel)
        if agob is None and bgoa is None:
            ax.set_title(f"{sid.split('.')[0]}\n[missing]", fontsize=6)
            ax.text(0.5, 0.5, "missing", transform=ax.transAxes, ha="center", va="center", fontsize=8, color="0.45")
            ax.set_xticks([])
            ax.set_yticks([])
            continue

        ref = agob or bgoa
        assert ref is not None
        t_ms = ref.t_ms
        plot_timing_overlay_subplot(ax, ref.t_ms, agob, bgoa, win_idx, "")
        ax.set_title(f"{sid.split('.')[0]}", fontsize=7)
        if ylim:
            ax.set_ylim(ylim)

    if t_ms is not None:
        configure_array_time_axis(axes_arr, t_ms, time_panel=0)

    fig.suptitle(title, fontsize=9, y=0.98)
    fig.subplots_adjust(left=0.05, right=0.98, top=0.88, bottom=0.14, wspace=0.25)
    add_timing_trace_legend(fig)
    fig.savefig(out_path, format="pdf", dpi=PDF_DPI)
    plt.close(fig)


def plot_si_scatter(
    si_agob: np.ndarray,
    si_bgoa: np.ndarray,
    task_either: np.ndarray,
    out_path: Path,
    *,
    suptitle: str | None = None,
) -> None:
    fig, ax = plt.subplots(figsize=(6, 6))
    no_task = ~task_either
    ax.scatter(
        si_agob[no_task],
        si_bgoa[no_task],
        s=10,
        facecolors="none",
        edgecolors="0.35",
        linewidths=0.5,
        alpha=0.6,
        label="no task (either)",
    )
    ax.scatter(
        si_agob[task_either],
        si_bgoa[task_either],
        s=10,
        c="#4c78a8",
        alpha=0.45,
        edgecolors="none",
        label="task (either)",
    )
    if si_agob.size:
        lo = float(np.nanmin([np.nanmin(si_agob), np.nanmin(si_bgoa)]))
        hi = float(np.nanmax([np.nanmax(si_agob), np.nanmax(si_bgoa)]))
    else:
        lo, hi = -1.0, 1.0
    ax.plot([lo, hi], [lo, hi], linestyle="--", linewidth=1.0, color="0.4")
    ax.set_xlabel("SI AgoB (timing 1)")
    ax.set_ylabel("SI BgoA (timing 2)")
    if suptitle:
        fig.suptitle(suptitle, fontsize=10)
    ax.legend(loc="upper left", fontsize=7, frameon=True)
    ax.grid(alpha=0.2)
    fig.tight_layout(rect=[0, 0, 1, 0.95] if suptitle else None)
    fig.savefig(out_path, format="pdf", dpi=PDF_DPI, bbox_inches="tight")
    plt.close(fig)


def plot_si_channel_median_scatter(
    si_agob_median: np.ndarray,
    si_bgoa_median: np.ndarray,
    task_either: np.ndarray,
    out_path: Path,
    *,
    suptitle: str | None = None,
) -> None:
    fig, ax = plt.subplots(figsize=(6, 6))
    no_task = ~task_either
    ax.scatter(
        si_agob_median[no_task],
        si_bgoa_median[no_task],
        s=14,
        facecolors="none",
        edgecolors="0.35",
        linewidths=0.6,
        alpha=0.75,
        label="no task (either)",
    )
    ax.scatter(
        si_agob_median[task_either],
        si_bgoa_median[task_either],
        s=14,
        c="#4c78a8",
        alpha=0.65,
        edgecolors="none",
        label="task (either)",
    )
    if si_agob_median.size:
        lo = float(np.nanmin([np.nanmin(si_agob_median), np.nanmin(si_bgoa_median)]))
        hi = float(np.nanmax([np.nanmax(si_agob_median), np.nanmax(si_bgoa_median)]))
    else:
        lo, hi = -1.0, 1.0
    ax.plot([lo, hi], [lo, hi], linestyle="--", linewidth=1.0, color="0.4")
    ax.set_xlabel("Median SI AgoB (across sessions)")
    ax.set_ylabel("Median SI BgoA (across sessions)")
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
    ax.set_xlabel("Mean SI = (SI_AgoB + SI_BgoA) / 2")
    ax.set_ylabel("ΔSI = SI_BgoA − SI_AgoB")
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
        cbar = fig.colorbar(im, cax=cax, label="Median ΔSI")
        cbar.ax.tick_params(labelsize=7)

    fig.suptitle(suptitle or "Median ΔSI by array (across sessions)", fontsize=10, y=0.96)
    fig.subplots_adjust(left=0.04, right=0.90, top=0.86, bottom=0.08, wspace=0.35)
    fig.savefig(out_path, format="pdf", dpi=PDF_DPI)
    plt.close(fig)
