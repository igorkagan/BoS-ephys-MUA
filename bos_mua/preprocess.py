from __future__ import annotations

from pathlib import Path

import numpy as np

MONKEY_CONDITIONS = [
    "Curius_BLOCKED",
    "Curius_SHUFFLED",
    "Elmo_BLOCKED",
    "Elmo_SHUFFLED",
]

_BASE_TRIAL_FILTERS: dict[str, list[str]] = {
    "TrialSubType_list": ["Dyadic"],
    "go_seq_500_list": ["AgoB"],
    "A_Reward_list": ["RA1", "RA2", "RA3", "RA4"],
}


def zscore_channel_trials(mua: np.ndarray) -> np.ndarray:
    """Z-score all trial×time samples for one channel within a session."""
    out = np.array(mua, dtype=float, copy=True)
    finite = np.isfinite(out)
    if not np.any(finite):
        return out

    vals = out[finite]
    mu = float(np.mean(vals))
    sd = float(np.std(vals, ddof=0))
    if sd == 0 or not np.isfinite(sd):
        out[finite] = 0.0
    else:
        out[finite] = (out[finite] - mu) / sd
    return out


def trial_filters_for_condition(condition_folder: str) -> dict[str, list[str]]:
    """Return trial filters with Blocked vs Shuffled predictability."""
    filters = dict(_BASE_TRIAL_FILTERS)
    if condition_folder.endswith("_SHUFFLED"):
        filters["conf_predictability_list"] = ["Shuffled"]
    elif condition_folder.endswith("_BLOCKED"):
        filters["conf_predictability_list"] = ["Blocked"]
    else:
        raise ValueError(f"Unknown condition folder: {condition_folder}")
    return filters


def resolve_figures_dir(base: str | Path, zscore_mua: bool) -> Path:
    """Nest outputs under figures/original/ or figures/zscored/."""
    path = Path(base)
    subfolder = "zscored" if zscore_mua else "original"
    if path.name == "figures":
        return path / subfolder
    if path.parent.name == "figures":
        return path.parent / subfolder / path.name
    return path.parent / subfolder / path.name


def resolve_condition_output_dir(
    base: str | Path,
    zscore_mua: bool,
    condition_folder: str,
    subfolder: str = "",
) -> Path:
    """figures/{mode}/{condition}/[{subfolder}/]."""
    out = resolve_figures_dir(base, zscore_mua) / condition_folder
    return out / subfolder if subfolder else out


def resolve_consistency_dir(
    base: str | Path,
    zscore_mua: bool,
    condition_folder: str,
) -> Path:
    """figures/{mode}/consistency/{condition}/."""
    path = Path(base)
    if path.name == "consistency":
        root = resolve_figures_dir(path.parent, zscore_mua) / "consistency"
    else:
        root = resolve_figures_dir(base, zscore_mua) / "consistency"
    return root / condition_folder


def processing_label(gaussian_smooth_ms: float, zscore_mua: bool) -> str:
    parts: list[str] = []
    if zscore_mua:
        parts.append("z-scored per channel/session")
    if gaussian_smooth_ms > 0:
        parts.append(f"smooth {gaussian_smooth_ms} ms")
    return " | ".join(parts) if parts else "raw"
