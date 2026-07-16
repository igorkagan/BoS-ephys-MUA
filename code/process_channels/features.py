from __future__ import annotations

import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.io import loadmat
from scipy.stats import mannwhitneyu

from load_data.io import (
    build_base_mask,
    channel_number,
    channel_to_array,
    choice_mask,
    discover_channel_files,
    gaussian_smooth_trials,
    load_time_vector,
    load_trial_labels,
    trial_window_means,
    window_indices,
)
from process_channels.preprocess import (
    recording_monkey,
    zscore_channel_trials,
    zscore_reference_mask,
)
from process_channels.evoked import TASK_EVOKED_ALPHA, task_evoked_anova_pvalues, session_task_evoked


@dataclass
class ChannelExtractResult:
    summary: ChannelSummary
    left_trials: np.ndarray
    right_trials: np.ndarray


@dataclass
class ChannelSummary:
    session_id: str
    channel: int
    array_name: str
    index_in_array: int
    n_left: int
    n_right: int
    t_ms: np.ndarray
    mean_left: np.ndarray
    mean_right: np.ndarray
    diff: np.ndarray
    si: float
    mwu_p: float | None
    pref_side: str
    evoked_p_left: float | None
    evoked_p_right: float | None
    task_evoked: bool


def selectivity_index(left_mean: float, right_mean: float) -> float:
    denom = abs(left_mean) + abs(right_mean)
    if not np.isfinite(denom) or denom == 0:
        return np.nan
    return (left_mean - right_mean) / denom


def run_mann_whitney(
    left_rates: np.ndarray,
    right_rates: np.ndarray,
    min_trials: int = 3,
) -> tuple[float | None, str]:
    left = left_rates[~np.isnan(left_rates)]
    right = right_rates[~np.isnan(right_rates)]

    if len(left) < min_trials or len(right) < min_trials:
        return None, ""

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            _, p = mannwhitneyu(left, right, alternative="two-sided")
        except ValueError:
            return None, ""

    side = ""
    if np.nanmean(left) > np.nanmean(right):
        side = "L"
    elif np.nanmean(right) > np.nanmean(left):
        side = "R"
    return float(p), side


def process_channel_mua(
    mua: np.ndarray,
    *,
    ch_num: int,
    session_id: str,
    t_ms: np.ndarray,
    win_idx: np.ndarray,
    left_mask: np.ndarray,
    right_mask: np.ndarray,
    gaussian_smooth_ms: float,
    min_trials: int = 3,
) -> ChannelExtractResult | None:
    """Process pre-loaded MUA for one channel (no file I/O)."""
    array_name, index_in_array = channel_to_array(ch_num)
    row_ok = ~np.all(np.isnan(mua), axis=1)

    left_trials = gaussian_smooth_trials(mua[left_mask & row_ok], t_ms, gaussian_smooth_ms)
    right_trials = gaussian_smooth_trials(mua[right_mask & row_ok], t_ms, gaussian_smooth_ms)

    n_left = len(left_trials)
    n_right = len(right_trials)
    if n_left < min_trials and n_right < min_trials:
        return None

    mean_left = np.nanmean(left_trials, axis=0) if n_left else np.full_like(t_ms, np.nan, dtype=float)
    mean_right = np.nanmean(right_trials, axis=0) if n_right else np.full_like(t_ms, np.nan, dtype=float)
    diff = mean_left - mean_right

    left_rates = trial_window_means(left_trials, win_idx)
    right_rates = trial_window_means(right_trials, win_idx)
    left_wm = float(np.nanmean(left_rates)) if left_rates.size else np.nan
    right_wm = float(np.nanmean(right_rates)) if right_rates.size else np.nan
    si = selectivity_index(left_wm, right_wm)
    mwu_p, pref_side = run_mann_whitney(left_rates, right_rates, min_trials=min_trials)
    evoked_p_left, evoked_p_right = task_evoked_anova_pvalues(
        left_trials, right_trials, t_ms, min_trials=min_trials,
    )
    task_evoked = session_task_evoked(evoked_p_left, evoked_p_right, alpha=TASK_EVOKED_ALPHA)

    summary = ChannelSummary(
        session_id=session_id,
        channel=ch_num,
        array_name=array_name,
        index_in_array=index_in_array,
        n_left=n_left,
        n_right=n_right,
        t_ms=t_ms,
        mean_left=mean_left,
        mean_right=mean_right,
        diff=diff,
        si=si,
        mwu_p=mwu_p,
        pref_side=pref_side,
        evoked_p_left=evoked_p_left,
        evoked_p_right=evoked_p_right,
        task_evoked=task_evoked,
    )
    return ChannelExtractResult(summary=summary, left_trials=left_trials, right_trials=right_trials)


def extract_channel_data(
    ch_path: Path,
    session_id: str,
    t_ms: np.ndarray,
    win_idx: np.ndarray,
    left_mask: np.ndarray,
    right_mask: np.ndarray,
    gaussian_smooth_ms: float,
    min_trials: int = 3,
    zscore_mua: bool = False,
    zscore_reference: np.ndarray | None = None,
    *,
    loadmat_hook=loadmat,
) -> ChannelExtractResult | None:
    ch_num = channel_number(ch_path)
    mua = loadmat_hook(ch_path)["cur_output_data"]
    if zscore_mua:
        mua = zscore_channel_trials(mua, reference_mask=zscore_reference)
    return process_channel_mua(
        mua,
        ch_num=ch_num,
        session_id=session_id,
        t_ms=t_ms,
        win_idx=win_idx,
        left_mask=left_mask,
        right_mask=right_mask,
        gaussian_smooth_ms=gaussian_smooth_ms,
        min_trials=min_trials,
    )


def extract_channel_summary(
    ch_path: Path,
    session_id: str,
    t_ms: np.ndarray,
    win_idx: np.ndarray,
    left_mask: np.ndarray,
    right_mask: np.ndarray,
    gaussian_smooth_ms: float,
    min_trials: int = 3,
    zscore_mua: bool = False,
    zscore_reference: np.ndarray | None = None,
) -> ChannelSummary | None:
    result = extract_channel_data(
        ch_path,
        session_id,
        t_ms,
        win_idx,
        left_mask,
        right_mask,
        gaussian_smooth_ms,
        min_trials=min_trials,
        zscore_mua=zscore_mua,
        zscore_reference=zscore_reference,
    )
    return result.summary if result is not None else None


def extract_session_summaries(
    session_dir: Path,
    session_id: str,
    alignment_event: str,
    pre_post_tag: str,
    trial_filters: dict[str, list[str]],
    choice_field: str,
    left_choice: list[str],
    right_choice: list[str],
    analysis_window_ms: tuple[float, float],
    gaussian_smooth_ms: float,
    min_trials: int = 3,
    zscore_mua: bool = False,
    condition_label: str = "",
) -> list[ChannelSummary]:
    event_dir = session_dir / alignment_event
    labels = load_trial_labels(session_dir, session_id)
    base_mask = build_base_mask(labels, trial_filters)
    left_mask = choice_mask(labels, base_mask, left_choice, field=choice_field)
    right_mask = choice_mask(labels, base_mask, right_choice, field=choice_field)
    zscore_ref = None
    if zscore_mua:
        from run_pipeline.context import get_active_context

        ctx = get_active_context()
        if ctx is not None:
            monkey = ctx.resolved_recording_monkey(session_id)
        else:
            monkey = recording_monkey(session_id=session_id, condition_label=condition_label)
        zscore_ref = zscore_reference_mask(labels, monkey, session_id=session_id)

    t_ms = load_time_vector(event_dir, session_id, alignment_event, pre_post_tag)
    win_idx = window_indices(t_ms, analysis_window_ms)
    if win_idx.size == 0:
        raise ValueError(f"No time points in analysis window {analysis_window_ms}")

    summaries: list[ChannelSummary] = []
    for ch_path in discover_channel_files(event_dir):
        result = extract_channel_data(
            ch_path,
            session_id,
            t_ms,
            win_idx,
            left_mask,
            right_mask,
            gaussian_smooth_ms,
            min_trials=min_trials,
            zscore_mua=zscore_mua,
            zscore_reference=zscore_ref,
        )
        if result is not None:
            summaries.append(result.summary)
    return summaries
