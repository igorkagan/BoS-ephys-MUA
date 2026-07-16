"""Pipeline trial counts from trial label dicts (actor-side selection rules)."""

from __future__ import annotations

import numpy as np

from load_data.io import build_base_mask, choice_mask
from process_channels.preprocess import (
    DUAL_NHP_GO_SEQS,
    choice_config_for_actor_side,
    recording_actor_side,
    trial_filters_for_go_seq,
    trial_filters_for_solo_from_dyadic,
)


def filter_choice_counts(
    labels: dict[str, np.ndarray],
    trial_filters: dict[str, list[str]],
    *,
    choice_field: str,
    left: list[str],
    right: list[str],
) -> dict[str, int]:
    base = build_base_mask(labels, trial_filters)
    left_n = int(choice_mask(labels, base, left, field=choice_field).sum())
    right_n = int(choice_mask(labels, base, right, field=choice_field).sum())
    return {"total": int(base.sum()), "left": left_n, "right": right_n}


def pipeline_trial_counts(
    labels: dict[str, np.ndarray],
    *,
    condition_key: str,
    session_id: str,
    recording_monkey: str,
) -> dict[str, int]:
    """Counts used by curated/confederate pipelines (dyadic + solo, both go_seqs)."""
    actor_side = recording_actor_side(session_id, recording_monkey)
    choice = choice_config_for_actor_side(actor_side)
    out: dict[str, int] = {"n_trials": len(next(iter(labels.values())))}

    for go_seq in DUAL_NHP_GO_SEQS:
        dyadic_filters = trial_filters_for_go_seq(condition_key, go_seq, actor_side)
        solo_filters = trial_filters_for_solo_from_dyadic(dyadic_filters, actor_side)

        dy = filter_choice_counts(
            labels,
            dyadic_filters,
            choice_field=choice.field,
            left=choice.left,
            right=choice.right,
        )
        so = filter_choice_counts(
            labels,
            solo_filters,
            choice_field=choice.field,
            left=choice.left,
            right=choice.right,
        )
        prefix_dy = f"dyadic_{go_seq}"
        prefix_so = f"solo_{go_seq}"
        out[f"{prefix_dy}_total"] = dy["total"]
        out[f"{prefix_dy}_left"] = dy["left"]
        out[f"{prefix_dy}_right"] = dy["right"]
        out[f"{prefix_so}_total"] = so["total"]
        out[f"{prefix_so}_left"] = so["left"]
        out[f"{prefix_so}_right"] = so["right"]

    return out


def conf_list_for_curated_condition(condition: str) -> str:
    """Map curated folder name to confederate list in session_lists.m."""
    return f"{condition}_CONF"


def curated_sessions_with_export_counterpart(
    curated_session_ids: list[str],
    conf_session_ids: list[str],
    *,
    n: int = 10,
    conf_list_name: str = "",
) -> list[str]:
    """First ``n`` curated IDs that also appear in the confederate export list."""
    conf_set = set(conf_session_ids)
    picked: list[str] = []
    for sid in curated_session_ids:
        if sid in conf_set:
            picked.append(sid)
            if len(picked) == n:
                break
    if len(picked) < n:
        only_in_curated = [s for s in curated_session_ids if s not in conf_set][:5]
        raise ValueError(
            f"Only {len(picked)}/{n} curated sessions found in {conf_list_name!r}; "
            f"examples without export list entry: {only_in_curated}"
        )
    return picked
