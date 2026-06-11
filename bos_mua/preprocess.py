from __future__ import annotations

from pathlib import Path

import numpy as np


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


def resolve_figures_dir(base: str | Path, zscore_mua: bool) -> Path:
    """Nest outputs under figures/original/ or figures/zscored/."""
    path = Path(base)
    subfolder = "zscored" if zscore_mua else "original"
    if path.name == "figures":
        return path / subfolder
    if path.parent.name == "figures":
        return path.parent / subfolder / path.name
    return path.parent / subfolder / path.name


def processing_label(gaussian_smooth_ms: float, zscore_mua: bool) -> str:
    parts: list[str] = []
    if zscore_mua:
        parts.append("z-scored per channel/session")
    if gaussian_smooth_ms > 0:
        parts.append(f"smooth {gaussian_smooth_ms} ms")
    return " | ".join(parts) if parts else "raw"
