#!/usr/bin/env python3
"""Debug: trial-averaged z-scored trace using ALL trials in the event epoch."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat

from load_data.io import load_time_vector
from process_channels.preprocess import zscore_channel_trials
from load_data.sessions import load_dual_nhp_configs

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = Path(r"S:/taskcontroller/SCP_DATA/SCP-CTRL-01/MUA_export_per_session")
ALIGN = "A_InitialFixationReleaseTime_ms"
PREPOST = "pre1000ms.post1000ms"
CHANNEL = 69
OUT = REPO_ROOT / "figures/debug/dual_nhp_elmo_ch069_zscore_all_trials_mean.pdf"


def main() -> None:
    _, split = load_dual_nhp_configs(REPO_ROOT / "session_lists.m")
    sessions = split["Elmo"]

    fig, axes = plt.subplots(2, 2, figsize=(10, 7), sharex=True, sharey=True)
    axes_flat = axes.ravel()

    for ax, sid in zip(axes_flat, sessions):
        sdir = DATA_ROOT / sid
        ev = sdir / ALIGN
        ch_files = list(ev.glob(f"*.ch{CHANNEL:03d}.*.mat"))
        if not ch_files:
            ax.set_title(f"{sid.split('.')[0]}\n(no ch{CHANNEL} file)")
            continue

        mua = loadmat(ch_files[0])["cur_output_data"].astype(float)
        z = zscore_channel_trials(mua)
        t_ms = load_time_vector(ev, sid, ALIGN, PREPOST)

        row_ok = ~np.all(np.isnan(z), axis=1)
        z_ok = z[row_ok]
        mean_trace = np.nanmean(z_ok, axis=0)
        grand_mean = float(np.nanmean(z_ok))
        trace_time_mean = float(np.nanmean(mean_trace))

        ax.plot(t_ms, mean_trace, color="C0", lw=1.2)
        ax.axhline(0, color="k", ls="--", lw=0.8, alpha=0.6)
        ax.axvspan(-500, 500, color="0.9", zorder=0)
        ax.set_title(
            f"{sid.split('.')[0]}\n"
            f"n_trials={row_ok.sum()}  grand μ={grand_mean:.2e}  "
            f"mean(trace)={trace_time_mean:.3f}",
            fontsize=8,
        )
        ax.set_ylabel("z (trial mean)")

    for ax in axes_flat[len(sessions) :]:
        ax.set_visible(False)

    for ax in axes[-1, :]:
        ax.set_xlabel("Time (ms)")
    fig.suptitle(
        f"DUAL_NHP Elmo ch{CHANNEL} | {ALIGN} | {PREPOST}\n"
        "Mean across ALL trials after zscore_channel_trials (full trial×time μ/σ)",
        fontsize=10,
    )
    fig.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
