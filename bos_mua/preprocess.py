from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

MONKEY_CONDITIONS = [
    "Curius_BLOCKED",
    "Curius_SHUFFLED",
    "Elmo_BLOCKED",
    "Elmo_SHUFFLED",
]

DUAL_NHP_GO_SEQS = ("AgoB", "BgoA")

# TrialSubType_list values where the recorded monkey acts (see session_lists.m comments).
CURIUS_ACTIVE_SOLO_SUBTYPES = ("SoloARewardAB", "SoloA")
ELMO_ACTIVE_SOLO_SUBTYPES = ("SoloBRewardAB", "SoloB")

_BASE_TRIAL_FILTERS: dict[str, list[str]] = {
    "TrialSubType_list": ["Dyadic"],
    "go_seq_500_list": ["AgoB"],
    "A_Reward_list": ["RA1", "RA2", "RA3", "RA4"],
}


@dataclass(frozen=True)
class ChoiceConfig:
    field: str
    left: list[str]
    right: list[str]


def default_choice_config() -> ChoiceConfig:
    return ChoiceConfig(field="A_LR_pos_list", left=["Al"], right=["Ar"])


def choice_config_for_monkey(monkey: str) -> ChoiceConfig:
    """L/R choice field and labels for the recorded monkey."""
    if monkey == "Elmo":
        return ChoiceConfig(field="B_LR_pos_list", left=["Bl"], right=["Br"])
    if monkey == "Curius":
        return default_choice_config()
    raise ValueError(f"Unknown monkey: {monkey}")


def dual_nhp_choice_config(monkey: str) -> ChoiceConfig:
    """Alias for choice_config_for_monkey (DUAL_NHP and confederate runs)."""
    return choice_config_for_monkey(monkey)


def trial_filters_for_go_seq(condition_key: str, go_seq: str) -> dict[str, list[str]]:
    """Dyadic filters for one go sequence, optionally with Blocked/Shuffled predictability."""
    if go_seq not in DUAL_NHP_GO_SEQS:
        raise ValueError(
            f"Unknown go sequence {go_seq!r}; expected one of {DUAL_NHP_GO_SEQS}"
        )
    filters = dict(_BASE_TRIAL_FILTERS)
    filters["go_seq_500_list"] = [go_seq]
    if condition_key.endswith("_SHUFFLED"):
        filters["conf_predictability_list"] = ["Shuffled"]
    elif condition_key.endswith("_BLOCKED"):
        filters["conf_predictability_list"] = ["Blocked"]
    return filters


def trial_filters_for_dual_nhp_go_seq(go_seq: str) -> dict[str, list[str]]:
    """Dyadic DUAL_NHP filters for one timing condition (AgoB or BgoA)."""
    return trial_filters_for_go_seq("DUAL_NHP", go_seq)


def trial_filters_for_dual_nhp_monkey(
    monkey: str,
    go_seq: str = "AgoB",
) -> dict[str, list[str]]:
    """Return DUAL_NHP trial filters for one monkey perspective and go sequence."""
    if monkey not in ("Elmo", "Curius"):
        raise ValueError(f"Unknown DUAL_NHP monkey: {monkey}")
    return trial_filters_for_dual_nhp_go_seq(go_seq)


def recording_monkey(*, session_id: str, condition_label: str = "") -> str:
    """Infer which monkey's electrodes a session export represents."""
    from bos_mua.run_context import get_active_context

    ctx = get_active_context()
    if ctx is not None and ctx.recording_monkey:
        return ctx.recording_monkey

    return recording_monkey_from_session_id(session_id, condition_label=condition_label)


def recording_monkey_from_condition_label(condition_label: str) -> str:
    """Infer recorded monkey from a condition or run label (Elmo_BLOCKED, Curius_AgoB, …)."""
    if condition_label.startswith("Elmo"):
        return "Elmo"
    if condition_label.startswith("Curius"):
        return "Curius"
    raise ValueError(f"Cannot infer recording monkey from condition label {condition_label!r}")


def recording_monkey_from_session_id(session_id: str, *, condition_label: str = "") -> str:
    """Infer recorded monkey from session id / condition label only."""
    datetime_part = session_id.split(".", 1)[0]
    if datetime_part.endswith("B"):
        return "Elmo"
    if datetime_part.endswith("U"):
        return "Curius"
    if condition_label.startswith("Elmo"):
        return "Elmo"
    if condition_label.startswith("Curius"):
        return "Curius"
    if ".A_Elmo." in session_id:
        return "Elmo"
    if ".A_Curius." in session_id:
        return "Curius"
    raise ValueError(
        f"Cannot infer recording monkey for session {session_id!r} "
        f"(condition_label={condition_label!r})"
    )


def zscore_reference_mask(labels: dict[str, np.ndarray], monkey: str) -> np.ndarray:
    """Trials where the recorded monkey acts: Dyadic + that monkey's active solos."""
    sub = labels["TrialSubType_list"].astype(str)
    mask = sub == "Dyadic"
    if monkey == "Curius":
        mask |= np.isin(sub, CURIUS_ACTIVE_SOLO_SUBTYPES)
    elif monkey == "Elmo":
        mask |= np.isin(sub, ELMO_ACTIVE_SOLO_SUBTYPES)
    else:
        raise ValueError(f"Unknown monkey: {monkey!r}")
    return mask


def zscore_channel_trials(
    mua: np.ndarray,
    *,
    reference_mask: np.ndarray | None = None,
) -> np.ndarray:
    """Z-score one channel's trial×time MUA within a session.

    μ and σ are computed from finite samples in reference rows only (when
    reference_mask is set); the same transform is applied to all finite values.
    """
    out = np.array(mua, dtype=float, copy=True)
    finite = np.isfinite(out)
    if not np.any(finite):
        return out

    row_has_data = np.any(finite, axis=1)
    if reference_mask is not None:
        ref = np.asarray(reference_mask, dtype=bool)
        if ref.shape[0] != out.shape[0]:
            raise ValueError(
                f"reference_mask length {ref.shape[0]} != n_trials {out.shape[0]}"
            )
        ref_finite = finite & (ref & row_has_data)[:, np.newaxis]
        if not np.any(ref_finite):
            ref_finite = finite
    else:
        ref_finite = finite

    vals = out[ref_finite]
    mu = float(np.mean(vals))
    sd = float(np.std(vals, ddof=0))
    if sd == 0 or not np.isfinite(sd):
        out[finite] = 0.0
    else:
        out[finite] = (out[finite] - mu) / sd
    return out


def trial_filters_for_condition(condition_folder: str) -> dict[str, list[str]]:
    """Return trial filters with Blocked vs Shuffled predictability (AgoB only)."""
    return trial_filters_for_go_seq(condition_folder, "AgoB")


def trial_filters_for_dual_nhp() -> dict[str, list[str]]:
    """Return base dyadic trial filters without conf_predictability."""
    return dict(_BASE_TRIAL_FILTERS)


def _nest_condition_in_figure_paths(condition_folder: str) -> bool:
    """Curated runs nest under figures/{mode}/{condition}/; flat export runs do not."""
    if not condition_folder:
        return False
    from bos_mua.run_context import get_active_context

    ctx = get_active_context()
    if ctx is not None and ctx.layout == "flat":
        return False
    return True


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
    """figures/{mode}/[{condition}/][{subfolder}/]."""
    out = resolve_figures_dir(base, zscore_mua)
    if _nest_condition_in_figure_paths(condition_folder):
        out = out / condition_folder
    return out / subfolder if subfolder else out


def resolve_consistency_dir(
    base: str | Path,
    zscore_mua: bool,
    condition_folder: str,
) -> Path:
    """figures/{mode}/consistency/[{condition}/]."""
    path = Path(base)
    if path.name == "consistency":
        root = resolve_figures_dir(path.parent, zscore_mua) / "consistency"
    else:
        root = resolve_figures_dir(base, zscore_mua) / "consistency"
    if _nest_condition_in_figure_paths(condition_folder):
        return root / condition_folder
    return root


def processing_label(gaussian_smooth_ms: float, zscore_mua: bool) -> str:
    parts: list[str] = []
    if zscore_mua:
        parts.append("z-scored per channel/session (actor trials)")
    if gaussian_smooth_ms > 0:
        parts.append(f"smooth {gaussian_smooth_ms} ms")
    return " | ".join(parts) if parts else "raw"
