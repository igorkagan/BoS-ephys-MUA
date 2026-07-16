#!/usr/bin/env python3
"""Debug PDF: DUAL_NHP Elmo session using the same LR pipeline as plot_session_lr_mua."""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import matplotlib.pyplot as plt

from process_channels.evoked import TASK_EVOKED_ALPHA, TASK_EVOKED_BIN_MS, session_task_evoked, task_evoked_anova_pvalues
from load_data.io import (
    ARRAY_NAMES,
    build_base_mask,
    channel_files_by_number,
    choice_mask,
    filter_summary,
    load_time_vector,
    load_trial_labels,
    window_indices,
)
from process_channels.preprocess import trial_filters_for_dual_nhp
from load_data.sessions import load_dual_nhp_configs
from analyze_stability.plots_lr import (
    build_session_lr_suptitle,
    format_task_evoked_p,
    load_smoothed_lr_trials,
    make_session_array_figure,
)

DATA_ROOT = Path(r"S:/taskcontroller/SCP_DATA/SCP-CTRL-01/MUA_export_per_session")
ALIGN = "A_InitialFixationReleaseTime_ms"
PREPOST = "pre1000ms.post1000ms"
OUT_DIR = REPO_ROOT / "figures/debug"
DEBUG_CHANNEL = 69

LEFT_CHOICE = ["Al"]
RIGHT_CHOICE = ["Ar"]
INVALID_LABELS = frozenset({"NONE", "None", "none", ""})
ANALYSIS_WINDOW_MS = (-500, 500)
GAUSSIAN_SMOOTH_MS = 50
ZSCORE_MUA = True
MIN_TRIALS = 3
MWU_ALPHA = 0.05
SUBPLOT_GRID = (4, 8)
FIG_SIZE_IN = (11, 8.5)
DPI = 150


def main() -> None:
    _, split = load_dual_nhp_configs(REPO_ROOT / "session_lists.m")
    session_id = split["Elmo"][0]
    session_dir = DATA_ROOT / session_id
    event_dir = session_dir / ALIGN

    labels = load_trial_labels(session_dir, session_id)
    filters = trial_filters_for_dual_nhp()
    base_mask = build_base_mask(labels, filters, invalid_labels=INVALID_LABELS)
    left_mask = choice_mask(labels, base_mask, LEFT_CHOICE)
    right_mask = choice_mask(labels, base_mask, RIGHT_CHOICE)

    t_ms = load_time_vector(event_dir, session_id, ALIGN, PREPOST)
    win_idx = window_indices(t_ms, ANALYSIS_WINDOW_MS)
    channel_paths = channel_files_by_number(event_dir)

    n_left = int(left_mask.sum())
    n_right = int(right_mask.sum())
    suptitle_base = build_session_lr_suptitle(
        session_id,
        "DUAL_NHP Elmo",
        ALIGN,
        filter_summary(filters),
        zscore_mua=ZSCORE_MUA,
        gaussian_smooth_ms=GAUSSIAN_SMOOTH_MS,
        n_left=n_left,
        n_right=n_right,
        task_evoked_alpha=TASK_EVOKED_ALPHA,
    )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    stamp = session_id.split(".")[0]
    print(f"Session: {session_id}")
    print(f"trials L={n_left}, R={n_right}")

    for array_index, array_name in enumerate(ARRAY_NAMES):
        fig = make_session_array_figure(
            array_index,
            channel_paths,
            t_ms,
            win_idx,
            left_mask,
            right_mask,
            f"{suptitle_base} | {array_name}",
            gaussian_smooth_ms=GAUSSIAN_SMOOTH_MS,
            zscore_mua=ZSCORE_MUA,
            subplot_grid=SUBPLOT_GRID,
            fig_size=FIG_SIZE_IN,
            mwu_alpha=MWU_ALPHA,
            min_trials=MIN_TRIALS,
            task_evoked_alpha=TASK_EVOKED_ALPHA,
        )
        out_path = OUT_DIR / f"dual_nhp_elmo_task_evoked_{stamp}_{array_name}.pdf"
        fig.savefig(out_path, format="pdf", dpi=DPI, bbox_inches="tight")
        plt.close(fig)
        print(f"Wrote {out_path}")

    ch_path = channel_paths.get(DEBUG_CHANNEL)
    if ch_path is not None:
        left_trials, right_trials = load_smoothed_lr_trials(
            ch_path,
            left_mask,
            right_mask,
            t_ms,
            gaussian_smooth_ms=GAUSSIAN_SMOOTH_MS,
            zscore_mua=ZSCORE_MUA,
        )
        p_l, p_r = task_evoked_anova_pvalues(
            left_trials,
            right_trials,
            t_ms,
            bin_ms=TASK_EVOKED_BIN_MS,
            min_trials=MIN_TRIALS,
        )
        evoked = session_task_evoked(p_l, p_r, alpha=TASK_EVOKED_ALPHA)
        print(
            f"ch{DEBUG_CHANNEL:03d}: task={'yes' if evoked else 'no'} "
            f"pL={format_task_evoked_p(p_l)} pR={format_task_evoked_p(p_r)}"
        )


if __name__ == "__main__":
    main()
