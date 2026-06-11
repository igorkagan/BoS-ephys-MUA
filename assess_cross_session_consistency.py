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

from bos_mua.features import ChannelSummary, extract_session_summaries
from bos_mua.io import (
    ARRAY_NAMES,
    CHANNELS_PER_ARRAY,
    discover_sessions,
    filter_summary,
    session_sort_key,
    window_indices,
)
from bos_mua.preprocess import processing_label, resolve_figures_dir
from bos_mua.stability import assess_channel_stability, session_similarity_matrix
from bos_mua.viz_consistency import (
    plot_deep_dive_channel,
    plot_delta_consensus_array,
    plot_session_similarity,
    plot_si_heatmap,
    plot_si_stability_by_array,
    plot_signed_sig_heatmap,
    write_stability_csv,
)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DATA_ROOT = r"S:\taskcontroller\SCP_DATA\SCP-CTRL-01\MUA_curated_sessions"
CONDITION_FOLDER = "Elmo_BLOCKED"
SESSION_IDS = None  # None = auto-discover all sessions in folder
REFERENCE_SESSION = None  # None = earliest by date in session ID

ALIGNMENT_EVENT = "A_InitialFixationReleaseTime_ms"
PRE_POST_TAG = "pre1000ms.post1000ms"

TRIAL_FILTERS = {
    "TrialSubType_list": ["Dyadic"],
    "conf_predictability_list": ["Blocked"],
    "go_seq_500_list": ["AgoB"],
    "A_Reward_list": ["RA1", "RA2", "RA3", "RA4"],
}
LEFT_CHOICE = ["Al"]
RIGHT_CHOICE = ["Ar"]

ANALYSIS_WINDOW_MS = (-500, 500)
GAUSSIAN_SMOOTH_MS = 50
ZSCORE_MUA = False  # ignored when RUN_BOTH_PROCESSING is True
RUN_BOTH_PROCESSING = True
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

OUTPUT_DIR = r"./figures/consistency"

# ---------------------------------------------------------------------------


def build_tensors(
    all_summaries: list[ChannelSummary],
    session_ids: list[str],
) -> tuple[list[int], np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    channels = sorted({s.channel for s in all_summaries})
    ch_to_idx = {ch: i for i, ch in enumerate(channels)}
    sess_to_idx = {sid: i for i, sid in enumerate(session_ids)}

    n_sess = len(session_ids)
    n_ch = len(channels)
    t_ms = all_summaries[0].t_ms
    n_time = len(t_ms)

    si_matrix = np.full((n_sess, n_ch), np.nan)
    p_matrix = np.full((n_sess, n_ch), np.nan)
    diff_tensor = np.full((n_sess, n_ch, n_time), np.nan)

    for s in all_summaries:
        si = sess_to_idx.get(s.session_id)
        ci = ch_to_idx.get(s.channel)
        if si is None or ci is None:
            continue
        si_matrix[si, ci] = s.si
        if s.mwu_p is not None:
            p_matrix[si, ci] = s.mwu_p
        diff_tensor[si, ci] = s.diff

    return channels, si_matrix, p_matrix, diff_tensor, t_ms


def summaries_lookup(all_summaries: list[ChannelSummary]) -> dict[str, dict[int, ChannelSummary]]:
    out: dict[str, dict[int, ChannelSummary]] = defaultdict(dict)
    for s in all_summaries:
        out[s.session_id][s.channel] = s
    return dict(out)


def run_consistency(
    condition_dir: Path,
    session_ids: list[str],
    ref_session: str,
    ref_idx: int,
    zscore_mua: bool,
) -> None:
    output_dir = resolve_figures_dir(OUTPUT_DIR, zscore_mua) / CONDITION_FOLDER
    output_dir.mkdir(parents=True, exist_ok=True)
    label = "z-scored" if zscore_mua else "original"

    print(f"\n=== {label} | Processing {len(session_ids)} sessions -> {output_dir} ===")
    all_summaries: list[ChannelSummary] = []
    for sid in session_ids:
        session_dir = condition_dir / sid
        try:
            summaries = extract_session_summaries(
                session_dir,
                sid,
                ALIGNMENT_EVENT,
                PRE_POST_TAG,
                TRIAL_FILTERS,
                LEFT_CHOICE,
                RIGHT_CHOICE,
                ANALYSIS_WINDOW_MS,
                GAUSSIAN_SMOOTH_MS,
                min_trials=MIN_TRIALS_PER_GROUP,
                zscore_mua=zscore_mua,
            )
            all_summaries.extend(summaries)
            print(f"  {sid}: {len(summaries)} channels")
        except Exception as exc:
            warnings.warn(f"Skipping {sid} ({label}): {exc}")

    if not all_summaries:
        raise RuntimeError(f"No channel summaries extracted ({label}).")

    channels, si_matrix, p_matrix, diff_tensor, t_ms = build_tensors(all_summaries, session_ids)
    win_idx = window_indices(t_ms, ANALYSIS_WINDOW_MS)
    lookup = summaries_lookup(all_summaries)

    stabilities = []
    stab_by_ch: dict[int, object] = {}
    for ch in channels:
        ch_idx = channels.index(ch)
        traces = diff_tensor[:, ch_idx, :]
        valid_sess = np.any(np.isfinite(traces), axis=1)
        if valid_sess.sum() < MIN_SESSIONS:
            continue
        traces = traces[valid_sess]
        si_vals = si_matrix[valid_sess, ch_idx]
        array_name = ARRAY_NAMES[(ch - 1) // CHANNELS_PER_ARRAY]
        stab = assess_channel_stability(
            ch, array_name, traces, si_vals,
            R_STABLE_THRESH, ICC_STABLE_THRESH, SIGN_CONCORDANCE_THRESH,
        )
        stabilities.append(stab)
        stab_by_ch[ch] = stab

    base_title = f"{CONDITION_FOLDER} | {ALIGNMENT_EVENT} | {filter_summary(TRIAL_FILTERS)} | {processing_label(GAUSSIAN_SMOOTH_MS, zscore_mua)}"

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
    write_stability_csv(stabilities, csv_path)
    print(f"Saved {csv_path} ({sum(s.stable for s in stabilities)} stable / {len(stabilities)} channels)")

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

    print(f"Done ({label}). Outputs in {output_dir}")


def main() -> None:
    condition_dir = Path(DATA_ROOT) / CONDITION_FOLDER
    if not condition_dir.exists():
        raise FileNotFoundError(f"Condition folder not found: {condition_dir}")

    session_ids = SESSION_IDS or discover_sessions(condition_dir)
    if len(session_ids) < MIN_SESSIONS:
        raise ValueError(f"Need at least {MIN_SESSIONS} sessions, found {len(session_ids)}")

    if REFERENCE_SESSION and REFERENCE_SESSION in session_ids:
        ref_session = REFERENCE_SESSION
    else:
        ref_session = min(session_ids, key=session_sort_key)
    ref_idx = session_ids.index(ref_session)

    modes = (False, True) if RUN_BOTH_PROCESSING else (ZSCORE_MUA,)
    for zscore_mua in modes:
        run_consistency(condition_dir, session_ids, ref_session, ref_idx, zscore_mua)


if __name__ == "__main__":
    main()
