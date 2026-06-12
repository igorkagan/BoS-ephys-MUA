#!/usr/bin/env python3
"""Run Yorgos's epoch code verbatim, write his attrs pickle, compare trial sets.

Trial identity = row index in cur_output_data / trialinfo (0-based). There is no
separate trial-ID field in the curated files; row index is the canonical key.

Yorgos's pickle stores only Trial Type, Go Sequence, Condition (post-RA0 rows).
Reward is not in the pickle; we keep it from the same load step as additional
data needed to apply RA1–RA4 filtering.

Usage:
    python yorgos_generatte_epochs_test.py
"""

from __future__ import annotations

import os
import pickle
import re
from pathlib import Path

import numpy as np
from scipy.io import loadmat

from bos_mua.io import build_base_mask, discover_sessions, load_trial_labels

# ---------------------------------------------------------------------------
DATA_ROOT = r"S:\taskcontroller\SCP_DATA\SCP-CTRL-01\MUA_curated_sessions"
CONDITION_FOLDER = "Curius_BLOCKED"
SESSION_ID = None  # None = earliest session in folder
ALIGN = "A_InitialFixationReleaseTime_ms"
OUTPUT_DIR = Path("./figures/yorgos_test")

TRIAL_FILTERS = {
    "TrialSubType_list": ["Dyadic"],
    "conf_predictability_list": ["Blocked"],
    "go_seq_500_list": ["AgoB"],
    "A_Reward_list": ["RA1", "RA2", "RA3", "RA4"],
}

# Fields used to build a human-readable trial fingerprint for mismatch reports
FINGERPRINT_FIELDS = (
    "TrialSubType_list",
    "conf_predictability_list",
    "go_seq_500_list",
    "A_Reward_list",
    "A_LR_pos_list",
)
# ---------------------------------------------------------------------------


def run_yorgos_session(session: str, data_dir: Path, align: str, epochs_path: Path) -> dict:
    """Body copied verbatim from yorgos_generate_epochs.py (session loop).

    Only deviations: .mat extensions on loadmat paths (required on curated layout),
    sorted channel list (his listdir order is undefined), and we record
    surviving row indices for downstream comparison.
    """
    session_dir = data_dir / session

    # --- yorgos_generate_epochs.py lines 18-31 ---
    trials_file = f"{session}.trialinfo.4python.mat"
    trial_info = loadmat(session_dir / trials_file)["cur_raster_labels"]
    go_seq = trial_info["go_seq_500_list"][0][0].squeeze()
    go_seq = np.array([item[0] for item in go_seq])
    direction_A_lr = trial_info["A_LR_pos_list"][0][0].squeeze()
    direction_A_lr = np.array([item[0][1] for item in direction_A_lr])
    trial_type = trial_info["TrialSubType_list"][0][0].squeeze()
    trial_type = np.array([item[0] for item in trial_type])
    rewards = trial_info["A_Reward_list"][0][0].squeeze()
    rewards = np.array([item[0] for item in rewards])
    aborted_trials_ind = np.where(rewards == "RA0")[0]
    condition = trial_info["conf_predictability_list"][0][0].squeeze()
    condition = np.array([item[0] for item in condition])

    # --- lines 33-36 ---
    times_file = f"{session}.{align}.MUA.pre1000ms.post1000ms.x_vector_ms.mat"
    time_vec = loadmat(session_dir / align / times_file)
    times = time_vec["x_vector_ms"].squeeze()

    # --- lines 38-43 (channels sorted so column order is reproducible) ---
    channel_dir_files = sorted(os.listdir(session_dir / align))
    chInds = []
    for f, file in enumerate(channel_dir_files):
        if re.search("data", file):
            chInds.append(f)
    channel_filenames = np.array(channel_dir_files)[chInds]

    # --- lines 45-62 ---
    ntrials, nchannels, ntimes = go_seq.size, len(channel_filenames), times.size
    mua = np.zeros((ntrials, nchannels, ntimes))
    channels = np.zeros(nchannels, dtype=object)
    index = session.find("A_")
    split_ch_id = 64 if session[index + 2] == "C" else 32

    for ii, ch_file in enumerate(channel_filenames):
        data = loadmat(session_dir / align / ch_file)
        mua[:, ii, :] = data["cur_output_data"]

        ind = ch_file.find("ch")
        ch_id = ch_file[ind + 2 : ind + 5]
        if int(ch_id) <= split_ch_id:
            channels[ii] = "PMv" + ch_id
        else:
            channels[ii] = "PMd" + ch_id

    # --- lines 64-70 ---
    ntrials = go_seq.size
    original_row_index = np.arange(ntrials, dtype=int)
    mask = np.ones(ntrials, dtype=bool)
    mask[aborted_trials_ind] = False
    mua = mua[mask]
    go_seq, condition = go_seq[mask], condition[mask]
    direction_A_lr, trial_type = direction_A_lr[mask], trial_type[mask]
    rewards_post_ra0 = rewards[mask]
    surviving_row_index = original_row_index[mask]

    # --- lines 72-78 ---
    zmean = mua.mean(axis=(0, 2), keepdims=True)
    zstd = mua.std(axis=(0, 2), keepdims=True)
    muaNormalized = (mua - zmean) / zstd
    # xr_mua = xr.DataArray(...)  # skipped: not needed for pickle comparison

    attrs = {"Trial Type": trial_type, "Go Sequence": go_seq, "Condition": condition}

    # --- lines 80-91 ---
    epochs_path.mkdir(parents=True, exist_ok=True)
    attrs_filename = f"{session}.{align}_attrs.pkl"
    attrs_path = epochs_path / attrs_filename
    with open(attrs_path, "wb") as handle:
        pickle.dump(attrs, handle)

    return {
        "attrs_path": attrs_path,
        "attrs": attrs,
        "surviving_row_index": surviving_row_index,
        "rewards_post_ra0": rewards_post_ra0,
        "direction_post_ra0": direction_A_lr,
        "n_total": ntrials,
        "n_after_ra0": int(mask.sum()),
        "n_aborted": int(len(aborted_trials_ind)),
        "n_channels": nchannels,
        "mua_shape": muaNormalized.shape,
    }


def load_yorgos_pickle(attrs_path: Path) -> dict:
    with open(attrs_path, "rb") as handle:
        return pickle.load(handle)


def analysis_mask_from_pickle(
    attrs: dict,
    rewards_post_ra0: np.ndarray,
) -> np.ndarray:
    """Dyadic / Blocked / AgoB / RA1–RA4 on post-RA0 rows (pickle + rewards)."""
    mask = (
        (attrs["Trial Type"] == "Dyadic")
        & (attrs["Condition"] == "Blocked")
        & (attrs["Go Sequence"] == "AgoB")
        & np.isin(rewards_post_ra0, ["RA1", "RA2", "RA3", "RA4"])
    )
    return mask


def our_analysis_row_indices(session_dir: Path, session_id: str) -> np.ndarray:
    labels = load_trial_labels(session_dir, session_id)
    return np.flatnonzero(build_base_mask(labels, TRIAL_FILTERS))


def trial_fingerprint(labels: dict[str, np.ndarray], row: int) -> tuple:
    return tuple(str(labels[field][row]) for field in FINGERPRINT_FIELDS)


def format_trial(row: int, labels: dict[str, np.ndarray]) -> str:
    fp = trial_fingerprint(labels, row)
    parts = [f"row={row}"]
    for field, val in zip(FINGERPRINT_FIELDS, fp):
        short = field.replace("_list", "")
        parts.append(f"{short}={val!r}")
    return "  ".join(parts)


def compare_session(session_id: str) -> None:
    data_dir = Path(DATA_ROOT) / CONDITION_FOLDER
    session_dir = data_dir / session_id
    epochs_path = OUTPUT_DIR / CONDITION_FOLDER / "MUA_Epochs"

    print(f"Session: {session_id}")
    print(f"Data:    {session_dir}")
    print(f"Pickle:  {epochs_path}\n")

    run = run_yorgos_session(session_id, data_dir, ALIGN, epochs_path)
    attrs_disk = load_yorgos_pickle(run["attrs_path"])
    assert attrs_disk.keys() == {"Trial Type", "Go Sequence", "Condition"}, attrs_disk.keys()

    rewards = run["rewards_post_ra0"]
    surviving = run["surviving_row_index"]
    post_ra0_mask = analysis_mask_from_pickle(attrs_disk, rewards)
    yorgos_rows = surviving[post_ra0_mask]

    our_rows = our_analysis_row_indices(session_dir, session_id)
    labels = load_trial_labels(session_dir, session_id)

    only_yorgos = np.setdiff1d(yorgos_rows, our_rows)
    only_ours = np.setdiff1d(our_rows, yorgos_rows)
    counts_match = len(yorgos_rows) == len(our_rows)
    identity_match = np.array_equal(np.sort(yorgos_rows), np.sort(our_rows))

    print("Yorgos (exact code path -> pickle on disk -> reload)")
    print(f"  pickle keys:               {list(attrs_disk.keys())}")
    print(f"  total trial rows:          {run['n_total']}")
    print(f"  after RA0:                 {run['n_after_ra0']} ({run['n_aborted']} aborted)")
    print(f"  pickle rows (post-RA0):    {len(attrs_disk['Trial Type'])}")
    print(f"  channels loaded:           {run['n_channels']}")
    print(f"  mua tensor shape:          {run['mua_shape']}")
    print(f"  analysis subset rows:      {len(yorgos_rows)}")

    print("\nOur pipeline (build_base_mask)")
    print(f"  analysis subset rows:      {len(our_rows)}")

    print("\nTrial identity = cur_output_data row index (0-based)")
    print(f"  counts match:              {counts_match}")
    print(f"  set match:                 {identity_match}")

    if identity_match:
        print("\n  All analysis trial row indices identical.")
        print(f"  Example rows (first 10):   {list(yorgos_rows[:10])}")
        print("\n  Fingerprint check (first 3 shared rows):")
        for row in yorgos_rows[:3]:
            print(f"    {format_trial(int(row), labels)}")
        return

    print(f"\n  only in Yorgos path ({len(only_yorgos)}):")
    for row in only_yorgos[:10]:
        print(f"    {format_trial(int(row), labels)}")

    print(f"\n  only in our pipeline ({len(only_ours)}):")
    for row in only_ours[:10]:
        print(f"    {format_trial(int(row), labels)}")

    # Cross-check: same row -> same fingerprint reachable from pickle slice?
    print("\n  Pickle slice vs full labels (first mismatch if any):")
    for row in only_yorgos[:3]:
        pos = int(np.where(surviving == row)[0][0])
        print(
            f"    row={row}  pickle: type={attrs_disk['Trial Type'][pos]!r} "
            f"go={attrs_disk['Go Sequence'][pos]!r} "
            f"cond={attrs_disk['Condition'][pos]!r} "
            f"reward={rewards[pos]!r}  dir={run['direction_post_ra0'][pos]!r}"
        )


def main() -> None:
    condition_dir = Path(DATA_ROOT) / CONDITION_FOLDER
    session_id = SESSION_ID or discover_sessions(condition_dir)[0]
    compare_session(session_id)


if __name__ == "__main__":
    main()
