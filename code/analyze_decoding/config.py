"""Defaults for session decoding (v1: actor L/R)."""

from __future__ import annotations

from pathlib import Path

PRE_POST_TAG = "pre1000ms.post1000ms"
DECODE_WINDOW_MS = (-1000.0, 1000.0)
GAUSSIAN_SMOOTH_MS = 50.0
# scipy.ndimage.gaussian_filter1d pad mode (decode + evoked MUA via io.gaussian_smooth_trials).
GAUSSIAN_SMOOTH_MODE = "reflect"
# Overlapping time bins for decode (width × step).
BIN_WIDTH_MS = 100.0
BIN_STEP_MS = 50.0
# Backward-compat alias (= step); prefer BIN_WIDTH_MS / BIN_STEP_MS.
BIN_MS = BIN_STEP_MS
ZSCORE_MUA = True

MIN_TRIALS_PER_CONDITION = 5
TRAINING_FRACTION = 0.8
CROSS_VALIDATIONS = 20
NSHUFFLES = 25

REPO_ROOT = Path(__file__).resolve().parents[2]
FIGURES_ROOT = REPO_ROOT / "figures"

# Linux mount used on this machine; Windows path is the repo default elsewhere.
_LINUX_CURATED = Path(
    "/home/igor/snd/taskcontroller/SCP_DATA/SCP-CTRL-01/MUA_curated_sessions"
)
_WINDOWS_CURATED = Path(
    r"S:\taskcontroller\SCP_DATA\SCP-CTRL-01\MUA_curated_sessions"
)


def default_curated_data_root() -> Path:
    if _LINUX_CURATED.is_dir():
        return _LINUX_CURATED
    return _WINDOWS_CURATED


def bin_settings_label(
    width_ms: float = BIN_WIDTH_MS,
    step_ms: float = BIN_STEP_MS,
) -> str:
    return f"bin={width_ms:g}/{step_ms:g} ms (w/step)"
