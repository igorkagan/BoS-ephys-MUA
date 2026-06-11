#!/usr/bin/env python3
"""Replot SI heatmaps only (portrait full-page, original + z-scored)."""

from pathlib import Path

from assess_cross_session_consistency import (
    ALIGNMENT_EVENT,
    ALPHA,
    ANALYSIS_WINDOW_MS,
    CONDITION_FOLDER,
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
    ZSCORE_MUA,
    build_tensors,
    filter_summary,
)
from bos_mua.features import extract_session_summaries
from bos_mua.io import discover_sessions
from bos_mua.preprocess import processing_label, resolve_figures_dir
from bos_mua.viz_consistency import plot_si_heatmap, plot_signed_sig_heatmap


def replot_heatmaps(zscore_mua: bool) -> None:
    condition_dir = Path(DATA_ROOT) / CONDITION_FOLDER
    session_ids = SESSION_IDS or discover_sessions(condition_dir)
    label = "z-scored" if zscore_mua else "original"
    output_dir = resolve_figures_dir(OUTPUT_DIR, zscore_mua) / CONDITION_FOLDER
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n=== {label} | Extracting {len(session_ids)} sessions ===")
    all_summaries = []
    for sid in session_ids:
        summaries = extract_session_summaries(
            condition_dir / sid, sid,
            ALIGNMENT_EVENT, PRE_POST_TAG,
            TRIAL_FILTERS, LEFT_CHOICE, RIGHT_CHOICE,
            ANALYSIS_WINDOW_MS, GAUSSIAN_SMOOTH_MS,
            min_trials=MIN_TRIALS_PER_GROUP,
            zscore_mua=zscore_mua,
        )
        all_summaries.extend(summaries)
        print(f"  {sid}: {len(summaries)} channels")

    channels, si_matrix, p_matrix, _, _ = build_tensors(all_summaries, session_ids)
    base_title = (
        f"{CONDITION_FOLDER} | {ALIGNMENT_EVENT} | {filter_summary(TRIAL_FILTERS)}"
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
    modes = (False, True) if RUN_BOTH_PROCESSING else (ZSCORE_MUA,)
    for zscore_mua in modes:
        replot_heatmaps(zscore_mua)


if __name__ == "__main__":
    main()
