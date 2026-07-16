#!/usr/bin/env python3
"""Replot SI heatmaps only (portrait full-page, original + z-scored)."""

import _bootstrap  # noqa: F401
from pathlib import Path

from analyze_stability.consistency import (
    ALPHA,
    ANALYSIS_WINDOW_MS,
    CHOICE_FIELD,
    CONDITION_FOLDER,
    DATA_ROOT,
    GAUSSIAN_SMOOTH_MS,
    LEFT_CHOICE,
    MIN_TRIALS_PER_GROUP,
    OUTPUT_DIR,
    PRE_POST_TAG,
    RIGHT_CHOICE,
    SESSION_IDS,
    TRIAL_FILTERS,
    alignment_event_for_session,
    alignment_event_label_for_sessions,
    filter_summary,
)
from run_pipeline.config import zscore_modes
from process_channels.features import extract_session_summaries
from load_data.io import discover_sessions, filter_summary
from process_channels.preprocess import processing_label, resolve_consistency_dir, trial_filters_for_condition
from analyze_stability.tensors import build_tensors
from analyze_stability.plots_consistency import plot_si_heatmap, plot_signed_sig_heatmap


def replot_heatmaps(zscore_mua: bool) -> None:
    condition_dir = Path(DATA_ROOT) / CONDITION_FOLDER
    session_ids = SESSION_IDS or discover_sessions(condition_dir)
    label = "z-scored" if zscore_mua else "original"
    output_dir = resolve_consistency_dir(OUTPUT_DIR, zscore_mua, CONDITION_FOLDER)
    output_dir.mkdir(parents=True, exist_ok=True)

    trial_filters = TRIAL_FILTERS or trial_filters_for_condition(CONDITION_FOLDER)
    print(f"\n=== {label} | Extracting {len(session_ids)} sessions ===")
    all_summaries = []
    for sid in session_ids:
        summaries = extract_session_summaries(
            condition_dir / sid, sid,
            alignment_event_for_session(sid), PRE_POST_TAG,
            trial_filters, CHOICE_FIELD, LEFT_CHOICE, RIGHT_CHOICE,
            ANALYSIS_WINDOW_MS, GAUSSIAN_SMOOTH_MS,
            min_trials=MIN_TRIALS_PER_GROUP,
            zscore_mua=zscore_mua,
            condition_label=CONDITION_FOLDER,
        )
        all_summaries.extend(summaries)
        print(f"  {sid}: {len(summaries)} channels")

    channels, si_matrix, p_matrix, _, _ = build_tensors(all_summaries, session_ids)
    base_title = (
        f"{CONDITION_FOLDER} | {alignment_event_label_for_sessions(session_ids)}"
        f" | {filter_summary(trial_filters)}"
        f" | {processing_label(GAUSSIAN_SMOOTH_MS, zscore_mua)}"
    )

    plot_si_heatmap(
        si_matrix, session_ids, channels,
        f"{base_title}\nSelectivity index",
        output_dir / "si_heatmap.pdf",
    )
    plot_signed_sig_heatmap(
        si_matrix, p_matrix, session_ids, channels, ALPHA,
        f"{base_title}\nSigned significance",
        output_dir / "signed_sig_heatmap.pdf",
    )
    print(f"Done ({label}): {output_dir}")


def main() -> None:
    for zscore_mua in zscore_modes():
        replot_heatmaps(zscore_mua)


if __name__ == "__main__":
    main()
