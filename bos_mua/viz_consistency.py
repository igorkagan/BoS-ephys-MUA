from __future__ import annotations

import csv
from dataclasses import asdict
from pathlib import Path

from bos_mua.viz_lr import annotate_task_evoked_text, configure_array_time_axis, task_evoked_annotation_text
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize

from bos_mua.features import ChannelSummary
from bos_mua.io import ARRAY_NAMES, CHANNELS_PER_ARRAY, array_nominal_channels, channel_label
from bos_mua.stability import ChannelStability

LEFT_COLOR = "#d62728"
RIGHT_COLOR = "#1f77b4"
SD_ALPHA = 0.25


SI_COLORBAR_LABEL = "Selectivity index (L−R)/(|L|+|R|)"


def _session_colors(n_sessions: int, cmap_name: str = "cool") -> np.ndarray:
    cmap = plt.colormaps[cmap_name]
    norm = Normalize(vmin=0, vmax=max(n_sessions - 1, 1))
    return np.array([cmap(norm(i)) for i in range(n_sessions)])


def _session_short_labels(session_ids: list[str]) -> list[str]:
    return [s.split(".")[0] for s in session_ids]


# US Letter portrait, full page
HEATMAP_FIGSIZE = (8.5, 11.0)
HEATMAP_DPI = 300


def _heatmap_portrait_figsize(n_channels: int, n_sessions: int) -> tuple[float, float]:
    """Portrait full-page layout: channels on y-axis, sessions on x-axis."""
    del n_channels, n_sessions  # fixed page size; data fills axes via aspect="auto"
    return HEATMAP_FIGSIZE


def plot_si_heatmap(
    si_matrix: np.ndarray,
    session_ids: list[str],
    channels: list[int],
    title: str,
    out_path: Path,
) -> None:
    # si_matrix: (sessions, channels) → transpose for channels × sessions display
    data = si_matrix.T
    session_labels = _session_short_labels(session_ids)
    fig, ax = plt.subplots(figsize=_heatmap_portrait_figsize(len(channels), len(session_ids)))

    vmax = np.nanmax(np.abs(data))
    if not np.isfinite(vmax) or vmax == 0:
        vmax = 1.0
    im = ax.imshow(data, aspect="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax)

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
    fig.colorbar(im, ax=ax, label=SI_COLORBAR_LABEL, shrink=0.85, pad=0.02)
    fig.tight_layout()
    fig.savefig(out_path, format="pdf", dpi=HEATMAP_DPI, bbox_inches="tight")
    plt.close(fig)


def plot_signed_sig_heatmap(
    si_matrix: np.ndarray,
    p_matrix: np.ndarray,
    session_ids: list[str],
    channels: list[int],
    alpha: float,
    title: str,
    out_path: Path,
) -> None:
    signed = np.sign(si_matrix) * (-np.log10(np.clip(p_matrix, 1e-300, 1.0)))
    signed[~np.isfinite(si_matrix)] = np.nan
    signed[p_matrix >= alpha] *= 0.25
    data = signed.T
    session_labels = _session_short_labels(session_ids)

    fig, ax = plt.subplots(figsize=_heatmap_portrait_figsize(len(channels), len(session_ids)))
    vmax = np.nanpercentile(np.abs(data), 95) if np.any(np.isfinite(data)) else 1.0
    im = ax.imshow(data, aspect="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax)

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
    fig.colorbar(im, ax=ax, label="sign(SI) × −log10(p); dim if p ≥ α", shrink=0.85, pad=0.02)
    fig.tight_layout()
    fig.savefig(out_path, format="pdf", dpi=HEATMAP_DPI, bbox_inches="tight")
    plt.close(fig)


def plot_session_similarity(
    sim: np.ndarray,
    session_ids: list[str],
    title: str,
    out_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(sim, vmin=-1, vmax=1, cmap="viridis")
    labels = [s.split(".")[0] for s in session_ids]
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=90, fontsize=7)
    ax.set_yticklabels(labels, fontsize=7)
    ax.set_title(title)
    fig.colorbar(im, ax=ax, label="Median pairwise r (Δ waveforms)")
    fig.tight_layout()
    fig.savefig(out_path, format="pdf", dpi=150, bbox_inches="tight")
    plt.close(fig)


def _channel_col(ch_num: int, channels: list[int]) -> int:
    """Column index in session × channel matrices (channels are 1…160)."""
    if channels[0] == 1 and len(channels) == 160:
        return ch_num - 1
    return channels.index(ch_num)


def _mark_empty_axis(ax, title: str, reason: str) -> None:
    ax.set_title(f"{title}\n[{reason}]", fontsize=8)
    ax.text(0.5, 0.5, reason, transform=ax.transAxes, ha="center", va="center", fontsize=8, color="0.45")
    ax.set_xticks([])
    ax.set_yticks([])


def plot_si_stability_by_array(
    si_matrix: np.ndarray,
    session_ids: list[str],
    channels: list[int],
    reference_idx: int,
    array_name: str,
    title: str,
    out_path: Path,
    session_colormap: str = "cool",
) -> None:
    ch_nums = array_nominal_channels(ARRAY_NAMES.index(array_name))
    ch_cols = [_channel_col(ch, channels) for ch in ch_nums]

    session_labels = _session_short_labels(session_ids)
    session_colors = _session_colors(len(session_ids), session_colormap)
    ref_label = session_labels[reference_idx]

    fig, axes = plt.subplots(2, 1, figsize=(11, 9))

    ax = axes[0]
    ref_si = si_matrix[reference_idx, ch_cols]
    for s_idx in range(si_matrix.shape[0]):
        y = si_matrix[s_idx, ch_cols]
        ax.scatter(
            ref_si, y,
            s=18, alpha=0.65, color=session_colors[s_idx],
            label=session_labels[s_idx], edgecolors="none",
        )
    slice_data = si_matrix[:, ch_cols]
    lims = [np.nanmin(slice_data), np.nanmax(slice_data)]
    if np.all(np.isfinite(lims)):
        ax.plot(lims, lims, "k--", linewidth=0.8, alpha=0.5, label="unity (SI_ref = SI)")
    ax.set_xlabel(f"SI in reference session ({ref_label})")
    ax.set_ylabel("SI in session")
    ax.set_title(f"{array_name}: SI vs reference (each dot = one channel)")
    ax.legend(
        title="Session",
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        fontsize=7,
        title_fontsize=8,
        frameon=True,
    )

    ax = axes[1]
    positions = []
    plot_data = []
    tick_labels = []
    for pos, (ch_num, col) in enumerate(zip(ch_nums, ch_cols, strict=True), start=1):
        vals = si_matrix[:, col]
        finite = vals[np.isfinite(vals)]
        tick_labels.append(f"{ch_num}")
        if finite.size >= 2:
            positions.append(pos)
            plot_data.append(finite)
    if plot_data:
        parts = ax.violinplot(plot_data, positions=positions, showmedians=True, showextrema=False)
        for body in parts["bodies"]:
            body.set_facecolor("#aec7e8")
            body.set_alpha(0.7)
        if "cmedians" in parts:
            parts["cmedians"].set_color("k")
    ax.set_xticks(range(1, len(ch_nums) + 1))
    ax.set_xticklabels(tick_labels, fontsize=6, rotation=90)
    ax.set_ylabel("SI across sessions")
    ax.set_xlabel("Channel")
    ax.set_title(f"{array_name}: SI distribution across sessions (one violin per channel)")
    ax.axhline(0, color="0.5", linewidth=0.6, linestyle=":", label="SI = 0 (no L/R bias)")
    ax.legend(loc="upper right", fontsize=7, frameon=True)

    fig.suptitle(title, fontsize=10)
    fig.tight_layout(rect=[0, 0, 0.88, 0.96])
    fig.savefig(out_path, format="pdf", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_delta_consensus_array(
    diff_tensor: np.ndarray,
    session_ids: list[str],
    channels: list[int],
    stabilities: dict[int, ChannelStability],
    t_ms: np.ndarray,
    win_idx: np.ndarray,
    array_name: str,
    array_index: int,
    show_session_traces: bool,
    session_colormap: str,
    subplot_grid: tuple[int, int],
    fig_size: tuple[float, float],
    title: str,
    out_path: Path,
) -> None:
    n_rows, n_cols = subplot_grid
    fig, axes = plt.subplots(n_rows, n_cols, figsize=fig_size, sharex=True, sharey=False)
    axes_flat = axes.ravel()
    colors = _session_colors(len(session_ids), session_colormap)

    ch_nums = array_nominal_channels(array_index)
    array_name = ARRAY_NAMES[array_index]

    for panel_idx, ax in enumerate(axes_flat):
        ch_num = ch_nums[panel_idx]
        _, idx_in_array = divmod(ch_num - 1, CHANNELS_PER_ARRAY)
        title = channel_label(ch_num, array_name, idx_in_array + 1)
        ch_idx = _channel_col(ch_num, channels)
        traces = diff_tensor[:, ch_idx, :]
        has_data = np.any(np.isfinite(traces))

        if not has_data:
            _mark_empty_axis(ax, title, "missing")
            continue

        if show_session_traces:
            for s_idx, trace in enumerate(traces):
                if np.all(np.isnan(trace)):
                    continue
                ax.plot(t_ms, trace, color=colors[s_idx], linewidth=0.6, alpha=0.55)

        median = np.nanmedian(traces, axis=0)
        q25 = np.nanpercentile(traces, 25, axis=0)
        q75 = np.nanpercentile(traces, 75, axis=0)
        if not np.any(np.isfinite(median)):
            _mark_empty_axis(ax, title, "no data")
            continue

        ax.plot(t_ms, median, color="k", linewidth=1.4, zorder=5)
        ax.fill_between(t_ms, q25, q75, color="0.7", alpha=0.35, zorder=4)

        if win_idx.size:
            ax.axvspan(t_ms[win_idx[0]], t_ms[win_idx[-1]], color="0.9", alpha=0.3, zorder=0)
        ax.axvline(0, color="0.5", linewidth=0.6, linestyle="--")
        ax.axhline(0, color="0.5", linewidth=0.4, linestyle=":")

        stab = stabilities.get(ch_num)
        if stab:
            flag = "stable" if stab.stable else "unstable"
            ann = (
                f"r={stab.median_pairwise_r:.2f}\n"
                f"{stab.n_same_sign}/{stab.n_sessions} sign\n"
                f"ICC={stab.icc:.2f}\n"
                f"task={'yes' if stab.task_evoked else 'no'}\n"
                f"{flag}"
            )
            ax.text(
                0.02, 0.98, ann, transform=ax.transAxes, va="top", ha="left", fontsize=6,
                bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.75, edgecolor="none"),
            )

        ax.set_title(title, fontsize=8)
        ax.tick_params(labelsize=6)
        ax.set_xlim(t_ms[0], t_ms[-1])

        ymin = np.nanmin(q25)
        ymax = np.nanmax(q75)
        if np.isfinite(ymin) and np.isfinite(ymax) and ymax > ymin:
            pad = 0.05 * (ymax - ymin)
            ax.set_ylim(ymin - pad, ymax + pad)

        if panel_idx % n_cols == 0:
            ax.set_ylabel("Δ MUA", fontsize=7)

    configure_array_time_axis(axes, t_ms)

    fig.suptitle(title, fontsize=10, y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    fig.savefig(out_path, format="pdf", dpi=150, bbox_inches="tight")
    plt.close(fig)


def stability_deep_dive_title(base_title: str, label: str, stab: ChannelStability) -> str:
    flag = "stable" if stab.stable else "unstable"
    return (
        f"{base_title} | {label} ch{stab.channel:03d} ({stab.array_name})\n"
        f"median r={stab.median_pairwise_r:.2f}  ICC={stab.icc:.2f}  "
        f"sign={stab.n_same_sign}/{stab.n_sessions}  task={'yes' if stab.task_evoked else 'no'}  {flag}"
    )


def _deep_dive_subplot_grid(n_sessions: int) -> tuple[int, int, tuple[float, float]]:
    """Return (nrows, ncols, figsize) for one panel per session (5 columns)."""
    ncols = 5
    nrows = max(1, (n_sessions + ncols - 1) // ncols)
    return nrows, ncols, (14.0, 2.5 * nrows)


def plot_deep_dive_channel(
    summaries_by_session: dict[str, dict[int, ChannelSummary]],
    session_ids: list[str],
    channel: int,
    win_idx: np.ndarray,
    title: str,
    out_path: Path,
) -> None:
    nrows, ncols, figsize = _deep_dive_subplot_grid(len(session_ids))
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, sharex=True, sharey=True)
    axes_flat = np.atleast_1d(axes).ravel()

    y_vals = []
    for sid in session_ids:
        s = summaries_by_session.get(sid, {}).get(channel)
        if s is None:
            continue
        for arr in (s.mean_left, s.mean_right):
            if np.any(np.isfinite(arr)):
                y_vals.extend(arr[np.isfinite(arr)])

    if y_vals:
        ymin, ymax = np.percentile(y_vals, [2, 98])
        pad = 0.05 * (ymax - ymin) if ymax > ymin else 0.1
        ylim = (ymin - pad, ymax + pad)
    else:
        ylim = None

    for idx, sid in enumerate(session_ids):
        ax = axes_flat[idx]
        s = summaries_by_session.get(sid, {}).get(channel)
        if s is None:
            ax.set_title(f"{sid.split('.')[0]}\n[missing]", fontsize=6)
            ax.text(0.5, 0.5, "missing", transform=ax.transAxes, ha="center", va="center", fontsize=8, color="0.45")
            ax.set_xticks([])
            ax.set_yticks([])
            continue

        if np.any(np.isfinite(s.mean_left)):
            ax.plot(s.t_ms, s.mean_left, color=LEFT_COLOR, linewidth=1.0)
        if np.any(np.isfinite(s.mean_right)):
            ax.plot(s.t_ms, s.mean_right, color=RIGHT_COLOR, linewidth=1.0)

        if win_idx.size:
            ax.axvspan(s.t_ms[win_idx[0]], s.t_ms[win_idx[-1]], color="0.9", alpha=0.3)
        ax.axvline(0, color="0.5", linewidth=0.5, linestyle="--")
        annotate_task_evoked_text(
            ax,
            task_evoked_annotation_text(
                task_evoked=s.task_evoked,
                p_left=s.evoked_p_left,
                p_right=s.evoked_p_right,
            ),
        )
        ax.set_title(f"{sid.split('.')[0]}\nL={s.n_left} R={s.n_right}", fontsize=6)
        ax.tick_params(labelsize=5)
        if ylim:
            ax.set_ylim(ylim)

    for ax in axes_flat[len(session_ids):]:
        ax.axis("off")

    if session_ids:
        first = summaries_by_session.get(session_ids[0], {}).get(channel)
        if first is not None:
            configure_array_time_axis(axes, first.t_ms)

    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(out_path, format="pdf", dpi=150, bbox_inches="tight")
    plt.close(fig)


def tuned_stable_deep_dive_title(base_title: str, stab: ChannelStability) -> str:
    return (
        f"{base_title} | best10_tuned_stable ch{stab.channel:03d} ({stab.array_name})\n"
        f"median |SI|={stab.si_median_abs:.2f}  si_std={stab.si_std:.2f}  "
        f"sign={stab.n_same_sign}/{stab.n_sessions}  "
        f"task={'yes' if stab.task_evoked else 'no'}  "
        f"median r={stab.median_pairwise_r:.2f}"
    )


def write_stability_csv(
    stabilities: list[ChannelStability],
    out_path: Path,
    tuned_stable: list[ChannelStability] | None = None,
) -> None:
    if not stabilities:
        return
    rank_map = {}
    if tuned_stable:
        rank_map = {s.channel: i + 1 for i, s in enumerate(tuned_stable)}
    fieldnames = list(asdict(stabilities[0]).keys())
    if rank_map:
        fieldnames = fieldnames + ["tuned_stable_rank"]
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in stabilities:
            d = asdict(row)
            if rank_map:
                d["tuned_stable_rank"] = rank_map.get(row.channel, "")
            writer.writerow(d)
