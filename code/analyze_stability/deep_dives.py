#!/usr/bin/env python3
"""Generate best10/worst10 and best10_tuned_stable channel deep-dive PDFs for both monkeys."""

from __future__ import annotations

import warnings
from pathlib import Path

from analyze_stability.consistency import (
    ANALYSIS_WINDOW_MS,
    BEST_WORST_N,
    CHOICE_FIELD,
    DATA_ROOT,
    GAUSSIAN_SMOOTH_MS,
    LEFT_CHOICE,
    MIN_TRIALS_PER_GROUP,
    OUTPUT_DIR,
    PRE_POST_TAG,
    RIGHT_CHOICE,
    SESSION_IDS,
    TRIAL_FILTERS,
    TUNED_STABLE_N,
    alignment_event_for_session,
    alignment_event_label_for_sessions,
    compute_stabilities,
    plot_best_worst_channels,
    plot_tuned_stable_channels,
    summaries_lookup,
)
from run_pipeline.config import zscore_modes
from process_channels.features import extract_session_summaries
from load_data.io import filter_summary, window_indices
from process_channels.preprocess import (
    MONKEY_CONDITIONS,
    processing_label,
    resolve_consistency_dir,
    trial_filters_for_condition,
)
from run_pipeline.context import get_active_context, resolve_session_dir, session_ids_for_run
from analyze_stability.tensors import build_tensors


def run_channel_rank_plots_from_summaries(
    condition_folder: str,
    zscore_mua: bool,
    session_ids: list[str],
    all_summaries: list,
) -> None:
    ctx = get_active_context()
    if ctx is not None:
        trial_filters = ctx.trial_filters
    elif TRIAL_FILTERS:
        trial_filters = TRIAL_FILTERS
    else:
        from process_channels.preprocess import (
            recording_actor_side,
            recording_monkey_from_condition_label,
        )
        monkey = recording_monkey_from_condition_label(condition_folder)
        actor_side = recording_actor_side(session_ids[0], monkey)
        trial_filters = trial_filters_for_condition(condition_folder, actor_side)
    output_dir = resolve_consistency_dir(OUTPUT_DIR, zscore_mua, condition_folder)
    output_dir.mkdir(parents=True, exist_ok=True)
    label = "z-scored" if zscore_mua else "original"

    print(f"\n=== {condition_folder} | {label} | channel rank plots ===")
    if not all_summaries:
        print(f"No summaries for {condition_folder} ({label}), skipping.")
        return

    channels, si_matrix, _, diff_tensor, t_ms = build_tensors(all_summaries, session_ids)
    win_idx = window_indices(t_ms, ANALYSIS_WINDOW_MS)
    lookup = summaries_lookup(all_summaries)
    stabilities, _ = compute_stabilities(channels, si_matrix, diff_tensor, lookup, session_ids)
    base_title = (
        f"{condition_folder} | {alignment_event_label_for_sessions(session_ids)}"
        f" | {filter_summary(trial_filters)}"
        f" | {processing_label(GAUSSIAN_SMOOTH_MS, zscore_mua)}"
    )
    plot_best_worst_channels(
        lookup, session_ids, stabilities, win_idx, base_title, output_dir, n=BEST_WORST_N,
    )
    plot_tuned_stable_channels(
        lookup, session_ids, stabilities, win_idx, base_title, output_dir, n=TUNED_STABLE_N,
    )


def run_from_cache(cache, condition_folder: str, zscore_mua: bool) -> None:
    """Generate rank plots from cached summaries."""
    run_channel_rank_plots_from_summaries(
        condition_folder,
        zscore_mua,
        cache.session_ids,
        cache.summaries.get(zscore_mua, []),
    )


def run_channel_rank_plots(condition_folder: str, zscore_mua: bool) -> None:
    session_ids = session_ids_for_run(
        DATA_ROOT, condition_folder, explicit_ids=SESSION_IDS,
    )
    ctx = get_active_context()
    if ctx is not None:
        trial_filters = ctx.trial_filters
    elif TRIAL_FILTERS:
        trial_filters = TRIAL_FILTERS
    else:
        from process_channels.preprocess import (
            recording_actor_side,
            recording_monkey_from_condition_label,
        )
        monkey = recording_monkey_from_condition_label(condition_folder)
        actor_side = recording_actor_side(session_ids[0], monkey)
        trial_filters = trial_filters_for_condition(condition_folder, actor_side)
    output_dir = resolve_consistency_dir(OUTPUT_DIR, zscore_mua, condition_folder)
    output_dir.mkdir(parents=True, exist_ok=True)
    label = "z-scored" if zscore_mua else "original"

    print(f"\n=== {condition_folder} | {label} | channel rank plots ===")
    all_summaries = []
    for sid in session_ids:
        session_dir = resolve_session_dir(
            sid, data_root=DATA_ROOT, condition_folder=condition_folder,
        )
        try:
            summaries = extract_session_summaries(
                session_dir, sid,
                alignment_event_for_session(sid), PRE_POST_TAG,
                trial_filters, CHOICE_FIELD, LEFT_CHOICE, RIGHT_CHOICE,
                ANALYSIS_WINDOW_MS, GAUSSIAN_SMOOTH_MS,
                min_trials=MIN_TRIALS_PER_GROUP,
                zscore_mua=zscore_mua,
                condition_label=condition_folder,
            )
            all_summaries.extend(summaries)
            print(f"  {sid}: {len(summaries)} channels with data")
        except Exception as exc:
            warnings.warn(f"Skipping {sid}: {exc}")

    if not all_summaries:
        print(f"No summaries for {condition_folder} ({label}), skipping.")
        return

    run_channel_rank_plots_from_summaries(
        condition_folder, zscore_mua, session_ids, all_summaries,
    )


def main() -> None:
    for condition in MONKEY_CONDITIONS:
        if not (Path(DATA_ROOT) / condition).exists():
            warnings.warn(f"Condition folder not found: {condition}")
            continue
        for zscore_mua in zscore_modes():
            run_channel_rank_plots(condition, zscore_mua)
    print("\nAll done.")


if __name__ == "__main__":
    main()
