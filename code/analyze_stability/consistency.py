#!/usr/bin/env python3
"""Assess cross-session consistency of nominal channel tuning and waveforms.

Usage:
    pip install -r requirements.txt
    python assess_cross_session_consistency.py
"""

from __future__ import annotations

import warnings
from collections import defaultdict
from pathlib import Path

import numpy as np

from process_channels.features import ChannelSummary
from load_data.io import (
    ARRAY_NAMES,
    CHANNELS_PER_ARRAY,
    filter_summary,
    session_sort_key,
    window_indices,
)
from process_channels.preprocess import (
    alignment_event_for_recording,
    processing_label,
    recording_actor_side,
    recording_monkey_from_condition_label,
    resolve_consistency_dir,
    trial_filters_for_condition,
)
from run_pipeline.context import get_active_context, resolve_session_dir, session_ids_for_run
from analyze_stability.metrics import (
    assess_channel_stability,
    channel_task_evoked_all_sessions,
    rank_best_worst_channels,
    rank_tuned_stable_channels,
    session_similarity_matrix,
)
from analyze_stability.tensors import (
    build_tensors,
    channel_presence_matrix,
    write_channel_presence_csv,
)
from analyze_stability.plots_consistency import (
    plot_deep_dive_channel,
    plot_delta_consensus_array,
    plot_session_similarity,
    plot_si_heatmap,
    plot_si_stability_by_array,
    plot_signed_sig_heatmap,
    stability_deep_dive_title,
    tuned_stable_deep_dive_title,
    write_stability_csv,
)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DATA_ROOT = r"S:\taskcontroller\SCP_DATA\SCP-CTRL-01\MUA_curated_sessions"
CONDITION_FOLDER = "Curius_BLOCKED"
SESSION_IDS = None  # None = auto-discover all sessions in folder
REFERENCE_SESSION = None  # None = earliest by date in session ID

ALIGNMENT_EVENT = "A_InitialFixationReleaseTime_ms"
PRE_POST_TAG = "pre1000ms.post1000ms"

TRIAL_FILTERS: dict[str, list[str]] | None = None  # None = auto from CONDITION_FOLDER
CHOICE_FIELD = "A_LR_pos_list"
LEFT_CHOICE = ["Al"]
RIGHT_CHOICE = ["Ar"]

ANALYSIS_WINDOW_MS = (-500, 500)
GAUSSIAN_SMOOTH_MS = 50
from run_pipeline.config import RUN_BOTH_PROCESSING, ZSCORE_MUA, zscore_modes
ALPHA = 0.05
MIN_TRIALS_PER_GROUP = 3
MIN_SESSIONS = 3

R_STABLE_THRESH = 0.5
ICC_STABLE_THRESH = 0.4
SIGN_CONCORDANCE_THRESH = 0.7

SHOW_SESSION_TRACES = True
SESSION_COLORMAP = "cool"
SUBPLOT_GRID = (4, 8)
FIG_SIZE_IN = (11, 8.5)

DEEP_DIVE_UNSTABLE_ONLY = True
MAX_DEEP_DIVE_CHANNELS = 20
BEST_WORST_N = 10
TUNED_STABLE_N = 10
TUNED_STABLE_SI_STD_MAX = 0.30
TUNED_STABLE_SI_ABS_MIN = 0.10

OUTPUT_DIR = r"./figures"

# ---------------------------------------------------------------------------


def effective_trial_filters(session_ids: list[str] | None = None) -> dict[str, list[str]]:
    if TRIAL_FILTERS is not None:
        return TRIAL_FILTERS
    from run_pipeline.context import get_active_context
    from process_channels.preprocess import (
        recording_actor_side,
        recording_monkey_from_condition_label,
    )

    ctx = get_active_context()
    if ctx is not None:
        return dict(ctx.trial_filters)
    if not session_ids:
        raise ValueError(
            "effective_trial_filters needs session_ids when no pipeline context is active"
        )
    monkey = recording_monkey_from_condition_label(CONDITION_FOLDER)
    actor_side = recording_actor_side(session_ids[0], monkey)
    return trial_filters_for_condition(CONDITION_FOLDER, actor_side)


def alignment_event_for_session(session_id: str) -> str:
    ctx = get_active_context()
    if ctx is not None:
        return ctx.alignment_event(session_id)
    monkey = recording_monkey_from_condition_label(CONDITION_FOLDER)
    return alignment_event_for_recording(session_id, monkey)


def alignment_event_label_for_sessions(session_ids: list[str]) -> str:
    ctx = get_active_context()
    if ctx is not None:
        return ctx.alignment_event_label(session_ids)
    monkey = recording_monkey_from_condition_label(CONDITION_FOLDER)
    events = {alignment_event_for_recording(sid, monkey) for sid in session_ids}
    if len(events) == 1:
        return events.pop()
    return "actor-side fixation release"


def summaries_lookup(all_summaries: list[ChannelSummary]) -> dict[str, dict[int, ChannelSummary]]:
    out: dict[str, dict[int, ChannelSummary]] = defaultdict(dict)
    for s in all_summaries:
        out[s.session_id][s.channel] = s
    return dict(out)


def compute_stabilities(
    channels: list[int],
    si_matrix: np.ndarray,
    diff_tensor: np.ndarray,
    lookup: dict[str, dict[int, ChannelSummary]],
    session_ids: list[str],
) -> tuple[list, dict[int, object]]:
    stabilities = []
    stab_by_ch: dict[int, object] = {}
    for ch in channels:
        ch_idx = ch - 1
        traces = diff_tensor[:, ch_idx, :]
        valid_sess = np.any(np.isfinite(traces), axis=1)
        if int(valid_sess.sum()) < MIN_SESSIONS:
            continue
        traces_present = traces[valid_sess]
        si_vals = si_matrix[valid_sess, ch_idx]
        array_name = ARRAY_NAMES[(ch - 1) // CHANNELS_PER_ARRAY]
        task_evoked, n_evoked, _ = channel_task_evoked_all_sessions(lookup, session_ids, ch)
        stab = assess_channel_stability(
            ch,
            array_name,
            traces_present,
            si_vals,
            R_STABLE_THRESH,
            ICC_STABLE_THRESH,
            SIGN_CONCORDANCE_THRESH,
            task_evoked=task_evoked,
            n_sessions_task_evoked=n_evoked,
        )
        stabilities.append(stab)
        stab_by_ch[ch] = stab
    return stabilities, stab_by_ch


def purge_stale_pdfs(output_dir: Path, patterns: list[str]) -> int:
    """Remove leftover channel PDFs from prior runs."""
    removed = 0
    for pattern in patterns:
        for path in output_dir.glob(pattern):
            path.unlink(missing_ok=True)
            removed += 1
    if removed:
        print(f"Removed {removed} stale PDF(s) from {output_dir}")
    return removed


def plot_best_worst_channels(
    lookup: dict[str, dict[int, ChannelSummary]],
    session_ids: list[str],
    stabilities: list,
    win_idx: np.ndarray,
    base_title: str,
    output_dir: Path,
    n: int = BEST_WORST_N,
) -> None:
    best, worst = rank_best_worst_channels(stabilities, n=n)
    for stab in best:
        plot_deep_dive_channel(
            lookup, session_ids, stab.channel, win_idx,
            stability_deep_dive_title(base_title, "best10", stab),
            output_dir / f"best10_ch{stab.channel:03d}.pdf",
        )
    for stab in worst:
        plot_deep_dive_channel(
            lookup, session_ids, stab.channel, win_idx,
            stability_deep_dive_title(base_title, "worst10", stab),
            output_dir / f"worst10_ch{stab.channel:03d}.pdf",
        )
    if best or worst:
        print(
            f"Saved {len(best)} best10 + {len(worst)} worst10 deep-dive PDFs "
            f"(best10: stable + task-evoked; ranked by median pairwise r)"
        )


def plot_tuned_stable_channels(
    lookup: dict[str, dict[int, ChannelSummary]],
    session_ids: list[str],
    stabilities: list,
    win_idx: np.ndarray,
    base_title: str,
    output_dir: Path,
    n: int = TUNED_STABLE_N,
    sign_thresh: float = SIGN_CONCORDANCE_THRESH,
    si_std_max: float = TUNED_STABLE_SI_STD_MAX,
    si_abs_min: float = TUNED_STABLE_SI_ABS_MIN,
) -> list:
    tuned = rank_tuned_stable_channels(
        stabilities, n=n,
        sign_thresh=sign_thresh,
        si_std_max=si_std_max,
        si_abs_min=si_abs_min,
    )
    if len(tuned) < n:
        warnings.warn(
            f"Only {len(tuned)} channels pass tuned-stable gate "
            f"(sign>={sign_thresh}, si_std<={si_std_max}, median|SI|>={si_abs_min}, task-evoked); "
            f"requested {n}"
        )
    for stab in tuned:
        plot_deep_dive_channel(
            lookup, session_ids, stab.channel, win_idx,
            tuned_stable_deep_dive_title(base_title, stab),
            output_dir / f"best10_tuned_stable_ch{stab.channel:03d}.pdf",
        )
    if tuned:
        print(
            f"Saved {len(tuned)} best10_tuned_stable deep-dive PDFs "
            f"(ranked by median |SI|, SI-stable + task-evoked gate)"
        )
    return tuned


def run_consistency_plots(
    session_ids: list[str],
    ref_session: str,
    ref_idx: int,
    zscore_mua: bool,
    all_summaries: list[ChannelSummary],
) -> None:
    output_dir = resolve_consistency_dir(OUTPUT_DIR, zscore_mua, CONDITION_FOLDER)
    output_dir.mkdir(parents=True, exist_ok=True)
    label = "z-scored" if zscore_mua else "original"
    trial_filters = effective_trial_filters(session_ids)
    print(f"\n=== {label} | {len(session_ids)} sessions -> {output_dir} ===")

    if not all_summaries:
        raise RuntimeError(f"No channel summaries extracted ({label}).")

    channels, si_matrix, p_matrix, diff_tensor, t_ms = build_tensors(all_summaries, session_ids)
    presence = channel_presence_matrix(all_summaries, session_ids)
    write_channel_presence_csv(session_ids, presence, output_dir / "channel_presence.csv")
    win_idx = window_indices(t_ms, ANALYSIS_WINDOW_MS)
    lookup = summaries_lookup(all_summaries)

    stabilities, stab_by_ch = compute_stabilities(channels, si_matrix, diff_tensor, lookup, session_ids)

    base_title = (
        f"{CONDITION_FOLDER} | {alignment_event_label_for_sessions(session_ids)}"
        f" | {filter_summary(trial_filters)} | {processing_label(GAUSSIAN_SMOOTH_MS, zscore_mua)}"
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

    sim = session_similarity_matrix(diff_tensor)
    plot_session_similarity(
        sim, session_ids,
        f"{base_title}\nSession similarity (median r of Δ)",
        output_dir / "session_similarity.pdf",
    )

    for array_name in ARRAY_NAMES:
        plot_si_stability_by_array(
            si_matrix, session_ids, channels, ref_idx, array_name,
            f"{base_title} | ref {ref_session.split('.')[0]}",
            output_dir / f"si_stability_{array_name}.pdf",
            session_colormap=SESSION_COLORMAP,
        )

    for array_index, array_name in enumerate(ARRAY_NAMES):
        plot_delta_consensus_array(
            diff_tensor, session_ids, channels, stab_by_ch, t_ms, win_idx,
            array_name, array_index, SHOW_SESSION_TRACES, SESSION_COLORMAP,
            SUBPLOT_GRID, FIG_SIZE_IN,
            f"{base_title} | {array_name} | Δ=L-R",
            output_dir / f"delta_consensus_{array_name}.pdf",
        )

    csv_path = output_dir / "channel_stability.csv"
    purge_stale_pdfs(
        output_dir,
        [
            "best10_tuned_stable_ch*.pdf",
            "best10_ch*.pdf",
            "worst10_ch*.pdf",
            "deep_dive_ch*.pdf",
        ],
    )
    tuned_stable = plot_tuned_stable_channels(
        lookup, session_ids, stabilities, win_idx, base_title, output_dir, n=TUNED_STABLE_N,
    )
    write_stability_csv(stabilities, csv_path, tuned_stable=tuned_stable)
    print(f"Saved {csv_path} ({sum(s.stable for s in stabilities)} stable / {len(stabilities)} channels with >={MIN_SESSIONS} sessions)")

    unstable = [s for s in stabilities if not s.stable]
    unstable.sort(key=lambda s: (s.median_pairwise_r if np.isfinite(s.median_pairwise_r) else 999))
    if DEEP_DIVE_UNSTABLE_ONLY:
        dive_channels = [s.channel for s in unstable[:MAX_DEEP_DIVE_CHANNELS]]
    else:
        dive_channels = channels[:MAX_DEEP_DIVE_CHANNELS]

    for ch in dive_channels:
        plot_deep_dive_channel(
            lookup, session_ids, ch, win_idx,
            f"{base_title} | deep dive ch{ch:03d}",
            output_dir / f"deep_dive_ch{ch:03d}.pdf",
        )
    if dive_channels:
        print(f"Saved {len(dive_channels)} deep-dive PDFs")

    plot_best_worst_channels(
        lookup, session_ids, stabilities, win_idx, base_title, output_dir, n=BEST_WORST_N,
    )

    print(f"Done ({label}). Outputs in {output_dir}")


def run_consistency(
    session_ids: list[str],
    ref_session: str,
    ref_idx: int,
    zscore_mua: bool,
) -> None:
    from process_channels.features import extract_session_summaries

    label = "z-scored" if zscore_mua else "original"
    trial_filters = effective_trial_filters(session_ids)
    print(f"\n=== {label} | Processing {len(session_ids)} sessions ===")
    all_summaries: list[ChannelSummary] = []
    for sid in session_ids:
        session_dir = resolve_session_dir(
            sid, data_root=DATA_ROOT, condition_folder=CONDITION_FOLDER,
        )
        try:
            summaries = extract_session_summaries(
                session_dir,
                sid,
                alignment_event_for_session(sid),
                PRE_POST_TAG,
                trial_filters,
                CHOICE_FIELD,
                LEFT_CHOICE,
                RIGHT_CHOICE,
                ANALYSIS_WINDOW_MS,
                GAUSSIAN_SMOOTH_MS,
                min_trials=MIN_TRIALS_PER_GROUP,
                zscore_mua=zscore_mua,
                condition_label=CONDITION_FOLDER,
            )
            all_summaries.extend(summaries)
            print(f"  {sid}: {len(summaries)} channels with data")
        except Exception as exc:
            warnings.warn(f"Skipping {sid} ({label}): {exc}")
    run_consistency_plots(session_ids, ref_session, ref_idx, zscore_mua, all_summaries)


def run_from_cache(cache, session_ids: list[str], ref_session: str, ref_idx: int) -> None:
    """Run consistency plots from a populated :class:`RunDataCache`."""
    for zscore_mua in zscore_modes():
        all_summaries = cache.summaries.get(zscore_mua, [])
        run_consistency_plots(session_ids, ref_session, ref_idx, zscore_mua, all_summaries)


def main() -> None:
    ctx = get_active_context()
    if ctx is None:
        condition_dir = Path(DATA_ROOT) / CONDITION_FOLDER
        if not condition_dir.exists():
            raise FileNotFoundError(f"Condition folder not found: {condition_dir}")

    session_ids = session_ids_for_run(
        DATA_ROOT, CONDITION_FOLDER, explicit_ids=SESSION_IDS,
    )
    if len(session_ids) < MIN_SESSIONS:
        raise ValueError(f"Need at least {MIN_SESSIONS} sessions, found {len(session_ids)}")

    if REFERENCE_SESSION and REFERENCE_SESSION in session_ids:
        ref_session = REFERENCE_SESSION
    else:
        ref_session = min(session_ids, key=session_sort_key)
    ref_idx = session_ids.index(ref_session)

    modes = zscore_modes()
    for zscore_mua in modes:
        run_consistency(session_ids, ref_session, ref_idx, zscore_mua)


if __name__ == "__main__":
    main()
