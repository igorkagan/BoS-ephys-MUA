from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from process_channels.features import ChannelSummary
from load_data.io import NOMINAL_CHANNELS, nominal_channel_list


def build_tensors(
    all_summaries: list[ChannelSummary],
    session_ids: list[str],
) -> tuple[list[int], np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build session × nominal-channel tensors; missing session/channel → NaN."""
    channels = nominal_channel_list()
    sess_to_idx = {sid: i for i, sid in enumerate(session_ids)}

    n_sess = len(session_ids)
    n_ch = len(channels)
    t_ms = next(s.t_ms for s in all_summaries)

    si_matrix = np.full((n_sess, n_ch), np.nan)
    p_matrix = np.full((n_sess, n_ch), np.nan)
    diff_tensor = np.full((n_sess, n_ch, len(t_ms)), np.nan)

    for s in all_summaries:
        si = sess_to_idx.get(s.session_id)
        if si is None or not (1 <= s.channel <= NOMINAL_CHANNELS):
            continue
        ci = s.channel - 1
        si_matrix[si, ci] = s.si
        if s.mwu_p is not None:
            p_matrix[si, ci] = s.mwu_p
        diff_tensor[si, ci] = s.diff

    return channels, si_matrix, p_matrix, diff_tensor, t_ms


def channel_presence_matrix(
    all_summaries: list[ChannelSummary],
    session_ids: list[str],
) -> np.ndarray:
    """Boolean (n_sessions, n_channels): True where channel had usable data."""
    present = np.zeros((len(session_ids), NOMINAL_CHANNELS), dtype=bool)
    sess_to_idx = {sid: i for i, sid in enumerate(session_ids)}
    for s in all_summaries:
        si = sess_to_idx.get(s.session_id)
        if si is None or not (1 <= s.channel <= NOMINAL_CHANNELS):
            continue
        present[si, s.channel - 1] = True
    return present


def write_channel_presence_csv(
    session_ids: list[str],
    presence: np.ndarray,
    out_path: Path,
) -> None:
    """CSV: one row per session × array with missing nominal channel IDs."""
    from load_data.io import ARRAY_NAMES, CHANNELS_PER_ARRAY

    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "session_id",
                "array",
                "n_present",
                "n_missing",
                "missing_channels",
            ],
        )
        writer.writeheader()
        for s_idx, sid in enumerate(session_ids):
            for a_idx, array_name in enumerate(ARRAY_NAMES):
                ch_start = a_idx * CHANNELS_PER_ARRAY
                ch_nums = list(range(ch_start + 1, ch_start + CHANNELS_PER_ARRAY + 1))
                missing = [ch for ch in ch_nums if not presence[s_idx, ch - 1]]
                writer.writerow(
                    {
                        "session_id": sid,
                        "array": array_name,
                        "n_present": CHANNELS_PER_ARRAY - len(missing),
                        "n_missing": len(missing),
                        "missing_channels": ",".join(f"{c:03d}" for c in missing),
                    }
                )
