#!/usr/bin/env python3
"""Plot left vs right MUA timecourses for one session, all channels.

Usage:
    pip install -r requirements.txt
    python plot_session_lr_mua.py
"""

from __future__ import annotations

import warnings
from pathlib import Path

import matplotlib.pyplot as plt

from bos_mua.evoked import TASK_EVOKED_ALPHA
from bos_mua.io import (
    ARRAY_NAMES,
    build_base_mask,
    channel_files_by_number,
    choice_mask,
    filter_summary,
    load_time_vector,
    load_trial_labels,
    window_indices,
)
from bos_mua.preprocess import (
    MONKEY_CONDITIONS,
    recording_monkey,
    resolve_condition_output_dir,
    trial_filters_for_condition,
    zscore_reference_mask,
)
from bos_mua.run_context import resolve_session_dir, session_ids_for_run
from bos_mua.viz.lr import build_session_lr_suptitle, make_session_array_figure

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DATA_ROOT = r"S:\taskcontroller\SCP_DATA\SCP-CTRL-01\MUA_curated_sessions"
CONDITION_FOLDER = None  # None = all MONKEY_CONDITIONS; or set one e.g. "Curius_BLOCKED"
SESSION_ID = None  # None = all sessions in condition; or set a single session ID
ALIGNMENT_EVENT = "A_InitialFixationReleaseTime_ms"
PRE_POST_TAG = "pre1000ms.post1000ms"

TRIAL_FILTERS = None  # None = per-condition defaults via trial_filters_for_condition
CHOICE_FIELD = "A_LR_pos_list"
LEFT_CHOICE = ["Al"]
RIGHT_CHOICE = ["Ar"]
INVALID_LABELS = frozenset({"NONE", "None", "none", ""})

ANALYSIS_WINDOW_MS = (-500, 500)
GAUSSIAN_SMOOTH_MS = 50
ZSCORE_MUA = False  # ignored when RUN_BOTH_PROCESSING is True
RUN_BOTH_PROCESSING = True
MWU_ALPHA = 0.05
MIN_TRIALS_PER_GROUP = 3

SUBPLOT_GRID = (4, 8)
OUTPUT_DIR = r"./figures"
FIG_SIZE_IN = (11, 8.5)
DPI = 150

# ---------------------------------------------------------------------------


def plot_session(
    session_id: str,
    condition_folder: str,
    trial_filters: dict,
    output_dir: Path,
    zscore_mua: bool,
) -> None:
    session_dir = resolve_session_dir(
        session_id, data_root=DATA_ROOT, condition_folder=condition_folder,
    )
    event_dir = session_dir / ALIGNMENT_EVENT

    if not session_dir.exists():
        raise FileNotFoundError(f"Session directory not found: {session_dir}")
    if not event_dir.exists():
        raise FileNotFoundError(f"Event directory not found: {event_dir}")

    labels = load_trial_labels(session_dir, session_id)
    base_mask = build_base_mask(labels, trial_filters, invalid_labels=INVALID_LABELS)
    left_mask = choice_mask(labels, base_mask, LEFT_CHOICE, field=CHOICE_FIELD)
    right_mask = choice_mask(labels, base_mask, RIGHT_CHOICE, field=CHOICE_FIELD)

    n_left = int(left_mask.sum())
    n_right = int(right_mask.sum())
    print(f"{session_id}: trials L={n_left}, R={n_right}")

    zscore_ref = None
    if zscore_mua:
        monkey = recording_monkey(session_id=session_id, condition_label=condition_folder)
        zscore_ref = zscore_reference_mask(labels, monkey)

    t_ms = load_time_vector(event_dir, session_id, ALIGNMENT_EVENT, PRE_POST_TAG)
    win_idx = window_indices(t_ms, ANALYSIS_WINDOW_MS)
    if win_idx.size == 0:
        raise ValueError(f"No time points in analysis window {ANALYSIS_WINDOW_MS}")

    channel_paths = channel_files_by_number(event_dir)
    suptitle_base = build_session_lr_suptitle(
        session_id,
        condition_folder,
        ALIGNMENT_EVENT,
        filter_summary(trial_filters),
        zscore_mua=zscore_mua,
        gaussian_smooth_ms=GAUSSIAN_SMOOTH_MS,
        n_left=n_left,
        n_right=n_right,
        task_evoked_alpha=TASK_EVOKED_ALPHA,
    )

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
            zscore_mua=zscore_mua,
            zscore_reference=zscore_ref,
            subplot_grid=SUBPLOT_GRID,
            fig_size=FIG_SIZE_IN,
            mwu_alpha=MWU_ALPHA,
            min_trials=MIN_TRIALS_PER_GROUP,
            task_evoked_alpha=TASK_EVOKED_ALPHA,
        )

        out_path = output_dir / f"{session_id}_{ALIGNMENT_EVENT}_{array_name}_LR.pdf"
        fig.savefig(out_path, format="pdf", dpi=DPI, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved {out_path.name}")


def run_condition(condition_folder: str, zscore_mua: bool) -> None:
    session_ids = session_ids_for_run(
        DATA_ROOT, condition_folder, session_id=SESSION_ID,
    )
    if not session_ids:
        warnings.warn(f"No sessions for {condition_folder}")
        return

    trial_filters = TRIAL_FILTERS or trial_filters_for_condition(condition_folder)
    output_dir = resolve_condition_output_dir(OUTPUT_DIR, zscore_mua, condition_folder)
    output_dir.mkdir(parents=True, exist_ok=True)
    label = "z-scored" if zscore_mua else "original"
    print(f"\n=== {condition_folder} | {label} | {len(session_ids)} session(s) -> {output_dir} ===")
    for session_id in session_ids:
        try:
            plot_session(session_id, condition_folder, trial_filters, output_dir, zscore_mua)
        except Exception as exc:
            warnings.warn(f"Skipping {session_id} ({label}): {exc}")


def main() -> None:
    conditions = [CONDITION_FOLDER] if CONDITION_FOLDER else MONKEY_CONDITIONS
    modes = (False, True) if RUN_BOTH_PROCESSING else (ZSCORE_MUA,)
    for zscore_mua in modes:
        for condition_folder in conditions:
            run_condition(condition_folder, zscore_mua)


if __name__ == "__main__":
    main()
