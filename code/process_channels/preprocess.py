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

# TrialSubType_list values where A or B acts solo (see session_lists.m comments).
A_SIDE_SOLO_SUBTYPES = ("SoloARewardAB", "SoloA")
B_SIDE_SOLO_SUBTYPES = ("SoloBRewardAB", "SoloB")

REWARDED_A_TIERS = ["RA1", "RA2", "RA3", "RA4"]
REWARDED_B_TIERS = ["RB1", "RB2", "RB3", "RB4"]


@dataclass(frozen=True)
class ChoiceConfig:
    field: str
    left: list[str]
    right: list[str]


def default_choice_config() -> ChoiceConfig:
    return ChoiceConfig(field="A_LR_pos_list", left=["Al"], right=["Ar"])


def choice_config_for_actor_side(actor_side: str) -> ChoiceConfig:
    """L/R choice field for the actor side derived from session ID."""
    if actor_side == "A":
        return default_choice_config()
    if actor_side == "B":
        return ChoiceConfig(field="B_LR_pos_list", left=["Bl"], right=["Br"])
    raise ValueError(f"Unknown actor side: {actor_side!r}")


def choice_config_for_recording(session_id: str, recording_monkey: str) -> ChoiceConfig:
    """L/R choice config from session ID actor side."""
    return choice_config_for_actor_side(
        recording_actor_side(session_id, recording_monkey),
    )


def recording_actor_side(session_id: str, recording_monkey: str) -> str:
    """Return ``A`` or ``B``: which side the recorded monkey plays in this session."""
    parts = session_id.split(".")
    if len(parts) >= 2 and parts[1].startswith("A_"):
        if parts[1][2:] == recording_monkey:
            return "A"
    if len(parts) >= 3 and parts[2].startswith("B_"):
        if parts[2][2:] == recording_monkey:
            return "B"
    datetime_part = parts[0]
    if datetime_part.endswith("U") and recording_monkey == "Curius":
        return "A"
    if datetime_part.endswith("B") and recording_monkey == "Elmo":
        return "B"
    raise ValueError(
        f"Cannot infer actor side for {recording_monkey!r} in session {session_id!r}"
    )


def active_solo_subtypes_for_actor_side(actor_side: str) -> tuple[str, ...]:
    """Solo trial subtypes where the given actor side (A or B) acts."""
    if actor_side == "A":
        return A_SIDE_SOLO_SUBTYPES
    if actor_side == "B":
        return B_SIDE_SOLO_SUBTYPES
    raise ValueError(f"Unknown actor side: {actor_side!r}")


def reward_tiers_for_actor_side(actor_side: str) -> tuple[str, list[str]]:
    """Reward list field and rewarded-tier labels for solo on actor side A or B."""
    if actor_side == "A":
        return "A_Reward_list", list(REWARDED_A_TIERS)
    if actor_side == "B":
        return "B_Reward_list", list(REWARDED_B_TIERS)
    raise ValueError(f"Unknown actor side: {actor_side!r}")


def solo_output_subdir_for_actor_side(actor_side: str) -> str:
    """Output subdirectory for solo trials (SoloA = A acts, SoloB = B acts)."""
    if actor_side == "A":
        return "SoloA"
    if actor_side == "B":
        return "SoloB"
    raise ValueError(f"Unknown actor side: {actor_side!r}")


def alignment_event_for_actor_side(actor_side: str) -> str:
    """MUA alignment folder for the recorded monkey's fixation release."""
    if actor_side == "A":
        return "A_InitialFixationReleaseTime_ms"
    if actor_side == "B":
        return "B_InitialFixationReleaseTime_ms"
    raise ValueError(f"Unknown actor side: {actor_side!r}")


def alignment_event_for_recording(session_id: str, recording_monkey: str) -> str:
    """Alignment event folder for one session's recorded monkey (actor side)."""
    return alignment_event_for_actor_side(
        recording_actor_side(session_id, recording_monkey),
    )


def trial_filters_for_solo_go_seq(
    condition_key: str,
    go_seq: str,
    actor_side: str,
) -> dict[str, list[str]]:
    """Solo trial filters for one go sequence and actor side (A or B)."""
    filters = trial_filters_for_go_seq(condition_key, go_seq, actor_side)
    return trial_filters_for_solo_from_dyadic(filters, actor_side)


def _dyadic_base_filters(actor_side: str, go_seq: str) -> dict[str, list[str]]:
    reward_field, tiers = reward_tiers_for_actor_side(actor_side)
    return {
        "TrialSubType_list": ["Dyadic"],
        "go_seq_500_list": [go_seq],
        reward_field: tiers,
    }


def trial_filters_for_solo_from_dyadic(
    dyadic_filters: dict[str, list[str]],
    actor_side: str,
) -> dict[str, list[str]]:
    """Derive solo filters: actor solo subtypes, rewarded tiers, same go_seq as dyadic.

    Solo trials are not tagged Blocked/Shuffled (typically ``Free``); drop
    ``conf_predictability_list`` inherited from dyadic filters.
    """
    filters = dict(dyadic_filters)
    filters.pop("A_Reward_list", None)
    filters.pop("B_Reward_list", None)
    filters.pop("conf_predictability_list", None)
    reward_field, tiers = reward_tiers_for_actor_side(actor_side)
    filters["TrialSubType_list"] = list(active_solo_subtypes_for_actor_side(actor_side))
    filters[reward_field] = tiers
    return filters


@dataclass(frozen=True)
class TrialBranchSpec:
    """One analysis branch (go-seq × social) within a condition load."""

    name: str
    output_subdir: str
    trial_filters: dict[str, list[str]]
    condition_label: str | None = None


def branch_specs_for_run(
    base_ctx: "PipelineContext",
    *,
    include_solo: bool = True,
) -> list[TrialBranchSpec]:
    """Dyadic branch plus optional solo branch for the recorded monkey."""
    from run_pipeline.context import PipelineContext

    if not isinstance(base_ctx, PipelineContext):
        raise TypeError(f"Expected PipelineContext, got {type(base_ctx)!r}")

    monkey = base_ctx.recording_monkey or recording_monkey_from_condition_label(
        base_ctx.condition_label,
    )
    specs = [
        TrialBranchSpec("dyadic", "Dyadic", dict(base_ctx.trial_filters)),
    ]
    if include_solo:
        if not base_ctx.session_ids:
            raise ValueError("Cannot build solo branch without session_ids")
        actor_side = recording_actor_side(base_ctx.session_ids[0], monkey)
        solo_filters = trial_filters_for_solo_from_dyadic(base_ctx.trial_filters, actor_side)
        specs.append(
            TrialBranchSpec(
                f"solo_{actor_side.lower()}",
                solo_output_subdir_for_actor_side(actor_side),
                solo_filters,
            ),
        )
    return specs


def trial_filters_for_go_seq(
    condition_key: str,
    go_seq: str,
    actor_side: str,
) -> dict[str, list[str]]:
    """Dyadic filters for one go sequence, actor side, and optional predictability."""
    if go_seq not in DUAL_NHP_GO_SEQS:
        raise ValueError(
            f"Unknown go sequence {go_seq!r}; expected one of {DUAL_NHP_GO_SEQS}"
        )
    filters = _dyadic_base_filters(actor_side, go_seq)
    if condition_key.endswith("_SHUFFLED"):
        filters["conf_predictability_list"] = ["Shuffled"]
    elif condition_key.endswith("_BLOCKED"):
        filters["conf_predictability_list"] = ["Blocked"]
    return filters


def trial_filters_for_dual_nhp_go_seq(go_seq: str, actor_side: str) -> dict[str, list[str]]:
    """Dyadic DUAL_NHP filters for one timing condition (AgoB or BgoA)."""
    return trial_filters_for_go_seq("DUAL_NHP", go_seq, actor_side)


def trial_filters_for_dual_nhp_monkey(
    monkey: str,
    go_seq: str = "AgoB",
    *,
    session_id: str,
) -> dict[str, list[str]]:
    """Return DUAL_NHP trial filters for one session's actor side and go sequence."""
    if monkey not in ("Elmo", "Curius"):
        raise ValueError(f"Unknown DUAL_NHP monkey: {monkey}")
    actor_side = recording_actor_side(session_id, monkey)
    return trial_filters_for_dual_nhp_go_seq(go_seq, actor_side)


def recording_monkey(*, session_id: str, condition_label: str = "") -> str:
    """Infer which monkey's electrodes a session export represents."""
    from run_pipeline.context import get_active_context

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


def recording_monkey_for_decode(
    condition: str,
    *,
    session_id: str = "",
    recording_monkey: str | None = None,
) -> str:
    """Recorded monkey for decode paths and trial labels.

    Confederate / curated labels start with Elmo_/Curius_. ``DUAL_NHP`` needs
    ``recording_monkey`` or a ``session_id`` (U vs B export).
    """
    if recording_monkey:
        return recording_monkey
    if condition == "DUAL_NHP":
        if not session_id:
            raise ValueError("DUAL_NHP decode needs recording_monkey or session_id")
        return recording_monkey_from_session_id(session_id, condition_label=condition)
    return recording_monkey_from_condition_label(condition)


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


def zscore_reference_mask(
    labels: dict[str, np.ndarray],
    recording_monkey: str,
    *,
    session_id: str = "",
    actor_side: str | None = None,
) -> np.ndarray:
    """Trials where the recorded monkey acts: Dyadic + that side's active solos."""
    sub = labels["TrialSubType_list"].astype(str)
    mask = sub == "Dyadic"
    if actor_side is None:
        if not session_id:
            raise ValueError("zscore_reference_mask needs session_id or actor_side")
        actor_side = recording_actor_side(session_id, recording_monkey)
    mask |= np.isin(sub, active_solo_subtypes_for_actor_side(actor_side))
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


def trial_filters_for_condition(
    condition_folder: str,
    actor_side: str,
) -> dict[str, list[str]]:
    """Return trial filters with Blocked vs Shuffled predictability (AgoB only)."""
    return trial_filters_for_go_seq(condition_folder, "AgoB", actor_side)


def trial_filters_for_dual_nhp(actor_side: str) -> dict[str, list[str]]:
    """Return base dyadic DUAL_NHP trial filters (AgoB, no conf_predictability)."""
    return trial_filters_for_dual_nhp_go_seq("AgoB", actor_side)


def _nest_condition_in_figure_paths(condition_folder: str) -> bool:
    """Curated runs nest under figures/{mode}/{condition}/; flat export runs do not."""
    if not condition_folder:
        return False
    from run_pipeline.context import get_active_context

    ctx = get_active_context()
    # Active pipeline contexts already carry a fully scoped branch output_base.
    if ctx is not None:
        return False
    return True


def resolve_figures_dir(base: str | Path, zscore_mua: bool) -> Path:
    """Branch output dir: flat when z-scored only; else original/ or zscored/."""
    from run_pipeline.config import dual_processing_modes

    path = Path(base)
    if not dual_processing_modes():
        return path
    subfolder = "zscored" if zscore_mua else "original"
    return path / subfolder


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
