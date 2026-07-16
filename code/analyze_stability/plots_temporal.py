"""Plots for temporal stability / inter-session gap analysis."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from analyze_stability.temporal_core import ChannelPairRow, GapBinSummary

HEATMAP_DPI = 300


def plot_r_vs_gap_scatter(
    rows: list[ChannelPairRow],
    stable_channels: set[int],
    title: str,
    out_path: Path,
) -> None:
    if not rows:
        return
    gaps = np.array([r.gap_days for r in rows], dtype=float)
    rs = np.array([r.waveform_r for r in rows], dtype=float)
    stable_mask = np.array([r.channel in stable_channels for r in rows], dtype=bool)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(
        gaps[~stable_mask], rs[~stable_mask],
        s=6, alpha=0.15, c="#aaaaaa", label="other channels", rasterized=True,
    )
    if np.any(stable_mask):
        ax.scatter(
            gaps[stable_mask], rs[stable_mask],
            s=10, alpha=0.5, c="#1f77b4", label="stable channels", rasterized=True,
        )
    ax.axhline(0.5, color="k", linestyle="--", linewidth=0.8, alpha=0.6, label="r=0.5")
    ax.set_xlabel("Inter-session gap (days)")
    ax.set_ylabel("Waveform r (Δ L−R)")
    ax.set_title(title, fontsize=10)
    ax.set_ylim(-1.05, 1.05)
    ax.legend(loc="lower left", fontsize=8)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, format="pdf", dpi=HEATMAP_DPI, bbox_inches="tight")
    plt.close(fig)


def plot_gap_bin_summary(
    summaries: list[GapBinSummary],
    title: str,
    out_path: Path,
) -> None:
    if not summaries:
        return
    labels = [s.bin_label for s in summaries]
    med = [s.median_waveform_r for s in summaries]
    p25 = [s.p25_waveform_r for s in summaries]
    x = np.arange(len(labels))

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(x, med, "o-", label="median r", color="#1f77b4")
    ax.plot(x, p25, "s--", label="p25 r", color="#ff7f0e")
    ax.axhline(0.5, color="k", linestyle=":", linewidth=0.8, alpha=0.6)
    ax.axhline(0.3, color="k", linestyle=":", linewidth=0.8, alpha=0.4)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_ylabel("Waveform r (stable-channel pairs)")
    ax.set_xlabel("Gap bin")
    ax.set_title(title, fontsize=10)
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=8)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, format="pdf", dpi=HEATMAP_DPI, bbox_inches="tight")
    plt.close(fig)


def plot_session_gap_heatmap(
    session_ids: list[str],
    sim_matrix: np.ndarray,
    gap_matrix: np.ndarray,
    title: str,
    out_path: Path,
) -> None:
    """Session×session similarity with gap days annotated on off-diagonal cells."""
    labels = [s.split(".")[0] for s in session_ids]
    n = len(session_ids)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))

    vmax = np.nanmax(np.abs(sim_matrix))
    if not np.isfinite(vmax) or vmax == 0:
        vmax = 1.0
    im0 = axes[0].imshow(sim_matrix, vmin=-vmax, vmax=vmax, cmap="RdBu_r", aspect="auto")
    axes[0].set_title("Median channel waveform r")
    fig.colorbar(im0, ax=axes[0], shrink=0.8)

    im1 = axes[1].imshow(gap_matrix, cmap="viridis", aspect="auto")
    axes[1].set_title("Gap (days)")
    fig.colorbar(im1, ax=axes[1], shrink=0.8)

    for ax in axes:
        ax.set_xticks(range(n))
        ax.set_yticks(range(n))
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=7)
        ax.set_yticklabels(labels, fontsize=7)

    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, format="pdf", dpi=HEATMAP_DPI, bbox_inches="tight")
    plt.close(fig)
