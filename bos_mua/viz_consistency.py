from __future__ import annotations

import csv
from dataclasses import asdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize

from bos_mua.features import ChannelSummary
from bos_mua.io import ARRAY_NAMES, CHANNELS_PER_ARRAY, channel_label
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
    ch_mask = [
        i for i, ch in enumerate(channels)
        if (ch - 1) // CHANNELS_PER_ARRAY == ARRAY_NAMES.index(array_name)
    ]
    if not ch_mask:
        return

    session_labels = _session_short_labels(session_ids)
    session_colors = _session_colors(len(session_ids), session_colormap)
    ref_label = session_labels[reference_idx]

    fig, axes = plt.subplots(2, 1, figsize=(11, 9))

    ax = axes[0]
    ref_si = si_matrix[reference_idx, ch_mask]
    ch_nums = [channels[i] for i in ch_mask]
    for s_idx in range(si_matrix.shape[0]):
        y = si_matrix[s_idx, ch_mask]
        ax.scatter(
            ref_si, y,
            s=18, alpha=0.65, color=session_colors[s_idx],
            label=session_labels[s_idx], edgecolors="none",
        )
    lims = [np.nanmin(si_matrix[:, ch_mask]), np.nanmax(si_matrix[:, ch_mask])]
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
    data = [si_matrix[:, i] for i in ch_mask]
    parts = ax.violinplot(data, showmedians=True, showextrema=False)
    for body in parts["bodies"]:
        body.set_facecolor("#aec7e8")
        body.set_alpha(0.7)
    if "cmedians" in parts:
        parts["cmedians"].set_color("k")
    ax.set_xticks(range(1, len(ch_nums) + 1))
    ax.set_xticklabels([f"{c}" for c in ch_nums], fontsize=6, rotation=90)
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

    ch_start = array_index * CHANNELS_PER_ARRAY
    ch_end = ch_start + CHANNELS_PER_ARRAY

    for panel_idx, ax in enumerate(axes_flat):
        ch_num = ch_start + panel_idx + 1
        if ch_num > ch_end or ch_num not in channels:
            ax.axis("off")
            continue

        ch_idx = channels.index(ch_num)
        traces = diff_tensor[:, ch_idx, :]

        if show_session_traces:
            for s_idx, trace in enumerate(traces):
                if np.all(np.isnan(trace)):
                    continue
                ax.plot(t_ms, trace, color=colors[s_idx], linewidth=0.6, alpha=0.55)

        median = np.nanmedian(traces, axis=0)
        q25 = np.nanpercentile(traces, 25, axis=0)
        q75 = np.nanpercentile(traces, 75, axis=0)
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
                f"ICC={stab.icc:.2f}\n{flag}"
            )
            ax.text(
                0.02, 0.98, ann, transform=ax.transAxes, va="top", ha="left", fontsize=6,
                bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.75, edgecolor="none"),
            )

        _, idx_in_array = divmod(ch_num - 1, CHANNELS_PER_ARRAY)
        ax.set_title(channel_label(ch_num, array_name, idx_in_array + 1), fontsize=8)
        ax.tick_params(labelsize=6)
        ax.set_xlim(t_ms[0], t_ms[-1])

        ymin = np.nanmin(q25)
        ymax = np.nanmax(q75)
        if np.isfinite(ymin) and np.isfinite(ymax) and ymax > ymin:
            pad = 0.05 * (ymax - ymin)
            ax.set_ylim(ymin - pad, ymax + pad)

        if panel_idx % n_cols == 0:
            ax.set_ylabel("Δ MUA", fontsize=7)
        if panel_idx >= (n_rows - 1) * n_cols:
            ax.set_xlabel("Time (ms)", fontsize=7)

    fig.suptitle(title, fontsize=10, y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    fig.savefig(out_path, format="pdf", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_deep_dive_channel(
    summaries_by_session: dict[str, dict[int, ChannelSummary]],
    session_ids: list[str],
    channel: int,
    win_idx: np.ndarray,
    title: str,
    out_path: Path,
) -> None:
    fig, axes = plt.subplots(2, 5, figsize=(14, 5), sharex=True, sharey=True)
    axes_flat = axes.ravel()

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
            ax.set_title(sid.split(".")[0], fontsize=7)
            ax.axis("off")
            continue

        if np.any(np.isfinite(s.mean_left)):
            ax.plot(s.t_ms, s.mean_left, color=LEFT_COLOR, linewidth=1.0)
        if np.any(np.isfinite(s.mean_right)):
            ax.plot(s.t_ms, s.mean_right, color=RIGHT_COLOR, linewidth=1.0)

        if win_idx.size:
            ax.axvspan(s.t_ms[win_idx[0]], s.t_ms[win_idx[-1]], color="0.9", alpha=0.3)
        ax.axvline(0, color="0.5", linewidth=0.5, linestyle="--")
        ax.set_title(f"{sid.split('.')[0]}\nL={s.n_left} R={s.n_right}", fontsize=6)
        ax.tick_params(labelsize=5)
        if ylim:
            ax.set_ylim(ylim)

    for ax in axes_flat[len(session_ids):]:
        ax.axis("off")

    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(out_path, format="pdf", dpi=150, bbox_inches="tight")
    plt.close(fig)


def write_stability_csv(stabilities: list[ChannelStability], out_path: Path) -> None:
    if not stabilities:
        return
    fieldnames = list(asdict(stabilities[0]).keys())
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in stabilities:
            writer.writerow(asdict(row))
