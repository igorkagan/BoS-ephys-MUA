"""Build Decodanda-ready session dicts from curated MUA."""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
from scipy.io import loadmat

from analyze_decoding.config import (
    BIN_STEP_MS,
    BIN_WIDTH_MS,
    DECODE_WINDOW_MS,
    GAUSSIAN_SMOOTH_MODE,
    GAUSSIAN_SMOOTH_MS,
    PRE_POST_TAG,
    ZSCORE_MUA,
)
from analyze_decoding.targets import DecodeTarget, all_grid_labels
from load_data.io import (
    INVALID_LABELS,
    build_base_mask,
    channel_files_by_number,
    choice_mask,
    gaussian_smooth_trials,
    load_time_vector,
    load_trial_labels,
    window_indices,
)
from process_channels.preprocess import (
    alignment_event_for_actor_side,
    choice_config_for_actor_side,
    recording_actor_side,
    recording_monkey_from_condition_label,
    trial_filters_for_go_seq,
    trial_filters_for_solo_from_dyadic,
    zscore_channel_trials,
    zscore_reference_mask,
)


@dataclass(frozen=True)
class SessionDecodeData:
    """One session ready for time-resolved decoding."""

    session_id: str
    go_seq: str
    trial_type: str
    actor_side: str
    alignment_event: str
    data: dict[str, np.ndarray]
    conditions: dict[str, list[str]]
    bin_centers_ms: np.ndarray
    n_trials: int
    n_channels: int
    n_left: int
    n_right: int
    channel_numbers: list[int]


def trial_filters_for_branch(
    condition: str,
    go_seq: str,
    trial_type: str,
    actor_side: str,
) -> dict[str, list[str]]:
    dyadic = trial_filters_for_go_seq(condition, go_seq, actor_side)
    if trial_type == "Dyadic":
        return dyadic
    if trial_type in ("SoloA", "SoloB"):
        return trial_filters_for_solo_from_dyadic(dyadic, actor_side)
    raise ValueError(f"Unknown trial_type {trial_type!r}")


def overlapping_bin_starts(
    t_ms: np.ndarray,
    width_ms: float,
    step_ms: float,
) -> np.ndarray:
    """Start times for full-width windows that fit in ``t_ms`` span."""
    start = float(t_ms[0])
    end = float(t_ms[-1])
    last_start = end - width_ms
    if last_start < start - 1e-9:
        return np.array([], dtype=float)
    # include last_start (arange stop is exclusive)
    return np.arange(start, last_start + step_ms * 0.5, step_ms)


def bin_trials_overlapping(
    t_ms: np.ndarray,
    trials: np.ndarray,
    width_ms: float,
    step_ms: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Mean within overlapping windows; return (n_trials, n_bins), centers_ms."""
    if trials.size == 0:
        return np.empty((0, 0)), np.array([], dtype=float)

    starts = overlapping_bin_starts(t_ms, width_ms, step_ms)
    if starts.size == 0:
        return np.empty((trials.shape[0], 0)), np.array([], dtype=float)

    cols: list[np.ndarray] = []
    centers: list[float] = []
    t_end = float(t_ms[-1])
    for i, lo in enumerate(starts):
        hi = float(lo) + width_ms
        # last window: include endpoint sample
        if i == starts.size - 1 or hi >= t_end - 1e-9:
            idx = (t_ms >= lo) & (t_ms <= hi + 1e-9)
        else:
            idx = (t_ms >= lo) & (t_ms < hi)
        if not np.any(idx):
            continue
        cols.append(np.nanmean(trials[:, idx], axis=1))
        centers.append(float(np.mean(t_ms[idx])))

    if not cols:
        return np.empty((trials.shape[0], 0)), np.array([], dtype=float)
    return np.column_stack(cols), np.asarray(centers, dtype=float)


def _load_channel_mua(path: Path) -> np.ndarray:
    return np.asarray(loadmat(path)["cur_output_data"], dtype=float)


def _stack_binned_channels(
    *,
    labels: dict[str, np.ndarray],
    trial_idx: np.ndarray,
    event_dir: Path,
    t_ms_full: np.ndarray,
    win_idx: np.ndarray,
    t_win: np.ndarray,
    bin_width_ms: float,
    bin_step_ms: float,
    smooth_ms: float,
    zscore_mua: bool,
    zscore_ref: np.ndarray | None,
    smooth_mode: str = GAUSSIAN_SMOOTH_MODE,
) -> tuple[np.ndarray, list[int], np.ndarray]:
    """Return X (n_trials, n_ch, n_bins), channel_numbers, bin_centers."""
    ch_paths = channel_files_by_number(event_dir)
    channel_numbers: list[int] = []
    binned_cols: list[np.ndarray] = []
    bin_centers: np.ndarray | None = None
    n_label_trials = len(next(iter(labels.values())))

    for ch_num in sorted(ch_paths):
        mua = _load_channel_mua(ch_paths[ch_num])
        if mua.shape[0] != n_label_trials:
            continue
        if zscore_mua:
            mua = zscore_channel_trials(mua, reference_mask=zscore_ref)
        mua = gaussian_smooth_trials(mua, t_ms_full, smooth_ms, mode=smooth_mode)
        trials = mua[trial_idx][:, win_idx]
        if not np.any(np.isfinite(trials)):
            continue
        binned, centers = bin_trials_overlapping(
            t_win, trials, bin_width_ms, bin_step_ms,
        )
        if binned.size == 0 or binned.shape[1] == 0:
            continue
        if bin_centers is None:
            bin_centers = centers
        elif centers.size != bin_centers.size or not np.allclose(centers, bin_centers):
            raise RuntimeError("Inconsistent bin centers across channels")
        channel_numbers.append(ch_num)
        binned_cols.append(binned)

    if not binned_cols or bin_centers is None:
        raise ValueError("no usable channels")

    n_bins = binned_cols[0].shape[1]
    for b in binned_cols:
        if b.shape[1] != n_bins:
            raise RuntimeError("Inconsistent bin counts across channels")
    X = np.stack(binned_cols, axis=1)
    return X, channel_numbers, bin_centers


def _flatten_decode_dict(
    X: np.ndarray,
    bin_centers: np.ndarray,
    label_cols: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    n_trials, n_channels, n_bins = X.shape
    raster = np.reshape(np.transpose(X, (0, 2, 1)), (n_trials * n_bins, n_channels))
    data: dict[str, np.ndarray] = {
        "raster": raster,
        "trial": np.repeat(np.arange(n_trials, dtype=int), n_bins),
        "time_from_onset": np.tile(bin_centers, n_trials),
    }
    for key, arr in label_cols.items():
        data[key] = np.repeat(arr, n_bins)
    return data


def _class_counts(labels: np.ndarray, levels: list[str]) -> tuple[int, int]:
    return int(np.sum(labels == levels[0])), int(np.sum(labels == levels[1]))


def build_session_decode_data(
    session_id: str,
    *,
    condition: str,
    go_seq: str,
    trial_type: str,
    target: DecodeTarget,
    data_root: Path,
    alignment_event: str | None = None,
    window_ms: tuple[float, float] = DECODE_WINDOW_MS,
    bin_width_ms: float = BIN_WIDTH_MS,
    bin_step_ms: float = BIN_STEP_MS,
    smooth_ms: float = GAUSSIAN_SMOOTH_MS,
    zscore_mua: bool = ZSCORE_MUA,
    pre_post_tag: str = PRE_POST_TAG,
) -> SessionDecodeData:
    """Stack channels, filter trials for one target, bin, flatten."""
    monkey = recording_monkey_from_condition_label(condition)
    actor_side = recording_actor_side(session_id, monkey)
    event = alignment_event or alignment_event_for_actor_side(actor_side)
    session_dir = data_root / condition / session_id
    event_dir = session_dir / event
    if not event_dir.is_dir():
        raise FileNotFoundError(f"Missing event dir: {event_dir}")

    labels = load_trial_labels(session_dir, session_id)
    filters = trial_filters_for_branch(condition, go_seq, trial_type, actor_side)
    choice = choice_config_for_actor_side(actor_side)
    base_mask = build_base_mask(labels, filters, invalid_labels=INVALID_LABELS)
    left_mask = choice_mask(labels, base_mask, choice.left, field=choice.field)
    right_mask = choice_mask(labels, base_mask, choice.right, field=choice.field)
    keep = left_mask | right_mask
    trial_idx = np.flatnonzero(keep)
    if trial_idx.size == 0:
        raise ValueError(f"{session_id}: no trials after filters ({trial_type}/{go_seq})")

    t_ms_full = load_time_vector(event_dir, session_id, event, pre_post_tag)
    win_idx = window_indices(t_ms_full, window_ms)
    if win_idx.size == 0:
        raise ValueError(f"No samples in window {window_ms}")
    t_win = t_ms_full[win_idx]
    zscore_ref = zscore_reference_mask(labels, monkey, session_id=session_id) if zscore_mua else None

    X, channel_numbers, bin_centers = _stack_binned_channels(
        labels=labels,
        trial_idx=trial_idx,
        event_dir=event_dir,
        t_ms_full=t_ms_full,
        win_idx=win_idx,
        t_win=t_win,
        bin_width_ms=bin_width_ms,
        bin_step_ms=bin_step_ms,
        smooth_ms=smooth_ms,
        zscore_mua=zscore_mua,
        zscore_ref=zscore_ref,
    )
    label_cols = target.labels_for_trials(labels, trial_idx, actor_side=actor_side)
    for key, arr in label_cols.items():
        if np.any(arr == ""):
            raise ValueError(f"{session_id}: empty labels for {key}")

    primary = next(iter(label_cols))
    levels = list(target.conditions[primary])
    n_left, n_right = _class_counts(label_cols[primary], levels)
    data = _flatten_decode_dict(X, bin_centers, label_cols)
    return SessionDecodeData(
        session_id=session_id,
        go_seq=go_seq,
        trial_type=trial_type,
        actor_side=actor_side,
        alignment_event=event,
        data=data,
        conditions=dict(target.conditions),
        bin_centers_ms=bin_centers,
        n_trials=X.shape[0],
        n_channels=X.shape[1],
        n_left=n_left,
        n_right=n_right,
        channel_numbers=channel_numbers,
    )


def build_aligned_multilabel_bundle(
    session_id: str,
    *,
    condition: str,
    go_seq: str,
    trial_type: str,
    data_root: Path,
    alignment_event: str,
    window_ms: tuple[float, float] = DECODE_WINDOW_MS,
    bin_width_ms: float = BIN_WIDTH_MS,
    bin_step_ms: float = BIN_STEP_MS,
    smooth_ms: float = GAUSSIAN_SMOOTH_MS,
    zscore_mua: bool = ZSCORE_MUA,
    pre_post_tag: str = PRE_POST_TAG,
) -> SessionDecodeData:
    """Dyadic/grid: neural @ alignment with A_choice, B_choice, same_diff attached.

    Trials = filter base_mask only (not restricted to one side's L/R). Labels may
    contain ``\"\"`` for missing choices; use ``subset_for_conditions`` before decode.
    """
    monkey = recording_monkey_from_condition_label(condition)
    actor_side = recording_actor_side(session_id, monkey)
    session_dir = data_root / condition / session_id
    event_dir = session_dir / alignment_event
    if not event_dir.is_dir():
        raise FileNotFoundError(f"Missing event dir: {event_dir}")

    labels = load_trial_labels(session_dir, session_id)
    filters = trial_filters_for_branch(condition, go_seq, trial_type, actor_side)
    base_mask = build_base_mask(labels, filters, invalid_labels=INVALID_LABELS)
    trial_idx = np.flatnonzero(base_mask)
    if trial_idx.size == 0:
        raise ValueError(f"{session_id}: no trials after filters ({trial_type}/{go_seq})")

    t_ms_full = load_time_vector(event_dir, session_id, alignment_event, pre_post_tag)
    win_idx = window_indices(t_ms_full, window_ms)
    if win_idx.size == 0:
        raise ValueError(f"No samples in window {window_ms}")
    t_win = t_ms_full[win_idx]
    zscore_ref = zscore_reference_mask(labels, monkey, session_id=session_id) if zscore_mua else None

    X, channel_numbers, bin_centers = _stack_binned_channels(
        labels=labels,
        trial_idx=trial_idx,
        event_dir=event_dir,
        t_ms_full=t_ms_full,
        win_idx=win_idx,
        t_win=t_win,
        bin_width_ms=bin_width_ms,
        bin_step_ms=bin_step_ms,
        smooth_ms=smooth_ms,
        zscore_mua=zscore_mua,
        zscore_ref=zscore_ref,
    )
    label_cols = all_grid_labels(labels, trial_idx)
    data = _flatten_decode_dict(X, bin_centers, label_cols)
    return SessionDecodeData(
        session_id=session_id,
        go_seq=go_seq,
        trial_type=trial_type,
        actor_side=actor_side,
        alignment_event=alignment_event,
        data=data,
        conditions={},  # set by subset_for_conditions
        bin_centers_ms=bin_centers,
        n_trials=X.shape[0],
        n_channels=X.shape[1],
        n_left=0,
        n_right=0,
        channel_numbers=channel_numbers,
    )


def subset_for_conditions(
    bundle: SessionDecodeData,
    conditions: dict[str, list[str]],
    *,
    count_key: str | None = None,
) -> SessionDecodeData:
    """Keep trials where every condition key has a non-empty label; set conditions."""
    n_bins = bundle.bin_centers_ms.size
    trial_ids = np.asarray(bundle.data["trial"], dtype=int)
    n_trials = int(trial_ids.max()) + 1 if trial_ids.size else 0

    # per-trial validity from first sample of each trial
    keep_trial = np.ones(n_trials, dtype=bool)
    for key in conditions:
        lab = np.asarray(bundle.data[key])
        # label of trial t = lab at first bin of that trial
        for t in range(n_trials):
            idx = np.flatnonzero(trial_ids == t)
            if idx.size == 0 or lab[idx[0]] == "":
                keep_trial[t] = False

    kept = np.flatnonzero(keep_trial)
    if kept.size == 0:
        raise ValueError(
            f"{bundle.session_id}: no trials with valid labels for {list(conditions)}"
        )

    # remap trial ids to 0..n_kept-1
    old_to_new = {int(old): new for new, old in enumerate(kept)}
    sample_keep = np.isin(trial_ids, kept)
    new_data: dict[str, np.ndarray] = {}
    for key, val in bundle.data.items():
        arr = np.asarray(val)[sample_keep]
        if key == "trial":
            arr = np.array([old_to_new[int(t)] for t in arr], dtype=int)
        new_data[key] = arr

    primary = count_key or next(iter(conditions))
    levels = list(conditions[primary])
    # per-trial labels for counts
    primary_trial = np.asarray(bundle.data[primary])[np.isin(trial_ids, kept)]
    # one per trial
    pt = primary_trial[::n_bins] if n_bins else primary_trial
    n0, n1 = _class_counts(pt, levels)

    return replace(
        bundle,
        data=new_data,
        conditions=dict(conditions),
        n_trials=kept.size,
        n_left=n0,
        n_right=n1,
    )
