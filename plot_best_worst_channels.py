#!/usr/bin/env python3
"""Generate best10/worst10 and best10_tuned_stable channel deep-dive PDFs for both monkeys."""

from __future__ import annotations

import warnings
from pathlib import Path

from assess_cross_session_consistency import (
    ALIGNMENT_EVENT,
    ANALYSIS_WINDOW_MS,
    BEST_WORST_N,
    DATA_ROOT,
    GAUSSIAN_SMOOTH_MS,
    LEFT_CHOICE,
    MIN_TRIALS_PER_GROUP,
    OUTPUT_DIR,
    PRE_POST_TAG,
    RIGHT_CHOICE,
    RUN_BOTH_PROCESSING,
    SESSION_IDS,
    TRIAL_FILTERS,
    TUNED_STABLE_N,
    ZSCORE_MUA,
    compute_stabilities,
    plot_best_worst_channels,
    plot_tuned_stable_channels,
    summaries_lookup,
)
from bos_mua.features import extract_session_summaries
from bos_mua.io import discover_sessions, filter_summary, window_indices
from bos_mua.preprocess import processing_label, resolve_figures_dir
from bos_mua.tensors import build_tensors

MONKEY_CONDITIONS = ["Elmo_BLOCKED", "Curius_BLOCKED"]


def run_channel_rank_plots(condition_folder: str, zscore_mua: bool) -> None:
    condition_dir = Path(DATA_ROOT) / condition_folder
    session_ids = SESSION_IDS or discover_sessions(condition_dir)
    output_dir = resolve_figures_dir(OUTPUT_DIR, zscore_mua) / condition_folder
    output_dir.mkdir(parents=True, exist_ok=True)
    label = "z-scored" if zscore_mua else "original"

    print(f"\n=== {condition_folder} | {label} | channel rank plots ===")
    all_summaries = []
    for sid in session_ids:
        try:
            summaries = extract_session_summaries(
                condition_dir / sid, sid,
                ALIGNMENT_EVENT, PRE_POST_TAG,
                TRIAL_FILTERS, LEFT_CHOICE, RIGHT_CHOICE,
                ANALYSIS_WINDOW_MS, GAUSSIAN_SMOOTH_MS,
                min_trials=MIN_TRIALS_PER_GROUP,
                zscore_mua=zscore_mua,
            )
            all_summaries.extend(summaries)
            print(f"  {sid}: {len(summaries)} channels with data")
        except Exception as exc:
            warnings.warn(f"Skipping {sid}: {exc}")

    if not all_summaries:
        print(f"No summaries for {condition_folder} ({label}), skipping.")
        return

    channels, si_matrix, _, diff_tensor, t_ms = build_tensors(all_summaries, session_ids)
    win_idx = window_indices(t_ms, ANALYSIS_WINDOW_MS)
    lookup = summaries_lookup(all_summaries)
    stabilities, _ = compute_stabilities(channels, si_matrix, diff_tensor)
    base_title = (
        f"{condition_folder} | {ALIGNMENT_EVENT} | {filter_summary(TRIAL_FILTERS)}"
        f" | {processing_label(GAUSSIAN_SMOOTH_MS, zscore_mua)}"
    )
    plot_best_worst_channels(
        lookup, session_ids, stabilities, win_idx, base_title, output_dir, n=BEST_WORST_N,
    )
    plot_tuned_stable_channels(
        lookup, session_ids, stabilities, win_idx, base_title, output_dir, n=TUNED_STABLE_N,
    )


def main() -> None:
    modes = (False, True) if RUN_BOTH_PROCESSING else (ZSCORE_MUA,)
    for condition in MONKEY_CONDITIONS:
        if not (Path(DATA_ROOT) / condition).exists():
            warnings.warn(f"Condition folder not found: {condition}")
            continue
        for zscore_mua in modes:
            run_channel_rank_plots(condition, zscore_mua)
    print("\nAll done.")


if __name__ == "__main__":
    main()
