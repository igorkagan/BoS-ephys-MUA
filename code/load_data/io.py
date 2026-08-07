from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path

import numpy as np
from scipy.io import loadmat
from scipy.ndimage import gaussian_filter1d

CHANNEL_RE = re.compile(r"\.ch(\d+)\.", re.IGNORECASE)
SESSION_DATETIME_RE = re.compile(r"^(\d{8}T\d{6})")
INVALID_LABELS = frozenset({"NONE", "None", "none", ""})

CHANNELS_PER_ARRAY = 32
ARRAY_NAMES = ["A1", "A2", "A3", "A4", "A5"]
NOMINAL_CHANNELS = len(ARRAY_NAMES) * CHANNELS_PER_ARRAY  # 160


def nominal_channel_list() -> list[int]:
    return list(range(1, NOMINAL_CHANNELS + 1))


def array_nominal_channels(array_index: int) -> list[int]:
    start = array_index * CHANNELS_PER_ARRAY
    return list(range(start + 1, start + CHANNELS_PER_ARRAY + 1))


def as_str_array(field) -> np.ndarray:
    flat = np.atleast_1d(field).ravel()
    out = np.empty(flat.size, dtype=object)
    for i, item in enumerate(flat):
        if isinstance(item, str):
            out[i] = item.strip()
        elif isinstance(item, bytes):
            out[i] = item.decode("utf-8").strip()
        elif item is None or (isinstance(item, float) and np.isnan(item)):
            out[i] = ""
        else:
            out[i] = str(item).strip()
    return out


def load_trial_labels(session_dir: Path, session_id: str) -> dict[str, np.ndarray]:
    trialinfo_path = session_dir / f"{session_id}.trialinfo.4python.mat"
    if not trialinfo_path.exists():
        raise FileNotFoundError(f"Trial info not found: {trialinfo_path}")

    trialinfo = loadmat(trialinfo_path, struct_as_record=False, squeeze_me=True)
    labels_struct = trialinfo["cur_raster_labels"]
    field_names = [name for name in dir(labels_struct) if name.endswith("_list")]
    return {name: as_str_array(getattr(labels_struct, name)) for name in field_names}


def build_base_mask(
    labels: dict[str, np.ndarray],
    filters: dict[str, list[str]],
    invalid_labels: frozenset[str] = INVALID_LABELS,
) -> np.ndarray:
    n_trials = len(next(iter(labels.values())))
    mask = np.ones(n_trials, dtype=bool)

    for field, allowed in filters.items():
        if field not in labels:
            raise KeyError(f"Missing label field: {field}")
        values = labels[field]
        mask &= np.isin(values, allowed)
        mask &= ~np.isin(values, list(invalid_labels))

    if "A_LR_pos_list" in labels:
        mask &= ~np.isin(labels["A_LR_pos_list"], list(invalid_labels))
    if "B_LR_pos_list" in labels:
        mask &= ~np.isin(labels["B_LR_pos_list"], list(invalid_labels))

    return mask


def choice_mask(
    labels: dict[str, np.ndarray],
    base_mask: np.ndarray,
    choices: list[str],
    *,
    field: str = "A_LR_pos_list",
) -> np.ndarray:
    if field not in labels:
        raise KeyError(f"Missing choice label field: {field}")
    return base_mask & np.isin(labels[field], choices)


def load_time_vector(event_dir: Path, session_id: str, event: str, pre_post_tag: str) -> np.ndarray:
    pattern = f"{session_id}.{event}.MUA.{pre_post_tag}.x_vector_ms.mat"
    time_path = event_dir / pattern
    if not time_path.exists():
        matches = sorted(event_dir.glob("*.x_vector_ms.mat"))
        if not matches:
            raise FileNotFoundError(f"Time vector not found in {event_dir}")
        time_path = matches[0]
    return loadmat(time_path)["x_vector_ms"].ravel()


def discover_channel_files(event_dir: Path) -> list[Path]:
    files = sorted(event_dir.glob("*.event_aligned_data.mat"))
    if not files:
        raise FileNotFoundError(f"No channel data in {event_dir}")
    return files


def channel_files_by_number(event_dir: Path) -> dict[int, Path]:
    """Map nominal channel number (1–160) to its event-aligned file, if present."""
    out: dict[int, Path] = {}
    for path in discover_channel_files(event_dir):
        try:
            out[channel_number(path)] = path
        except ValueError:
            continue
    return out


def discover_sessions(condition_dir: Path) -> list[str]:
    sessions = [p.name for p in condition_dir.iterdir() if p.is_dir()]
    return sorted(sessions, key=session_sort_key)


def session_sort_key(session_id: str) -> str:
    return session_id.split(".")[0]


def parse_session_datetime(session_id: str) -> datetime:
    """Parse recording datetime from session id prefix (``YYYYMMDDTHHMMSS[U|B]``)."""
    prefix = session_id.split(".", 1)[0]
    match = SESSION_DATETIME_RE.match(prefix)
    if not match:
        raise ValueError(f"Cannot parse datetime from session id {session_id!r}")
    return datetime.strptime(match.group(1), "%Y%m%dT%H%M%S")


def session_gap_days(session_a: str, session_b: str) -> float:
    """Absolute calendar gap between two sessions in days (fractional)."""
    dt_a = parse_session_datetime(session_a)
    dt_b = parse_session_datetime(session_b)
    return abs((dt_b - dt_a).total_seconds()) / 86400.0


def channel_number(path: Path) -> int:
    match = CHANNEL_RE.search(path.name)
    if not match:
        raise ValueError(f"Cannot parse channel number from {path.name}")
    return int(match.group(1))


def channel_to_array(channel: int) -> tuple[str, int]:
    idx = (channel - 1) // CHANNELS_PER_ARRAY
    array_name = ARRAY_NAMES[idx] if idx < len(ARRAY_NAMES) else f"A{idx + 1}"
    index_in_array = (channel - 1) % CHANNELS_PER_ARRAY + 1
    return array_name, index_in_array


def channel_label(ch_num: int, array_name: str, index_in_array: int) -> str:
    return f"ch{ch_num:03d} ({array_name}-{index_in_array})"


def array_channel_files(all_files: list[Path], array_index: int) -> list[Path]:
    """Legacy helper: slice sorted file list by array index (assumes contiguous ch001–ch160 files)."""
    start = array_index * CHANNELS_PER_ARRAY
    return all_files[start : start + CHANNELS_PER_ARRAY]


def window_indices(t_ms: np.ndarray, window_ms: tuple[float, float]) -> np.ndarray:
    lo, hi = window_ms
    return np.flatnonzero((t_ms >= lo) & (t_ms <= hi))


def gaussian_smooth_trials(
    trials: np.ndarray,
    t_ms: np.ndarray,
    smooth_ms: float,
    *,
    mode: str = "reflect",
) -> np.ndarray:
    if trials.size == 0 or smooth_ms <= 0:
        return trials

    dt = float(np.median(np.diff(t_ms)))
    sigma_samples = (smooth_ms / 2.354820045) / dt
    out = trials.copy()
    valid = ~np.all(np.isnan(trials), axis=1)
    if np.any(valid):
        out[valid] = gaussian_filter1d(
            trials[valid], sigma=sigma_samples, axis=1, mode=mode,
        )
    return out


def trial_window_means(trials: np.ndarray, win_idx: np.ndarray) -> np.ndarray:
    if trials.size == 0:
        return np.array([], dtype=float)
    return np.nanmean(trials[:, win_idx], axis=1)


def filter_summary(filters: dict[str, list[str]]) -> str:
    parts = []
    for key, values in filters.items():
        short_key = key.replace("_list", "")
        parts.append(f"{short_key}={','.join(values)}")
    return "; ".join(parts)
