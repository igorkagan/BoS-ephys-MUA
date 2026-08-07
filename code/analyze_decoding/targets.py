"""Decode-target registry and alignment-order helpers."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from process_channels.preprocess import ChoiceConfig, choice_config_for_actor_side

A_EVENT = "A_InitialFixationReleaseTime_ms"
B_EVENT = "B_InitialFixationReleaseTime_ms"

BALANCED_AB_CONDITIONS: dict[str, list[str]] = {
    "A_choice": ["left", "right"],
    "B_choice": ["left", "right"],
}


def align_events_in_order(go_seq: str) -> tuple[str, str]:
    """Return (1st action event, 2nd action event) for go-seq order."""
    if go_seq == "AgoB":
        return A_EVENT, B_EVENT
    if go_seq == "BgoA":
        return B_EVENT, A_EVENT
    raise ValueError(f"Unknown go_seq {go_seq!r}")


def align_short_label(event: str) -> str:
    if event.startswith("A_"):
        return "A release"
    if event.startswith("B_"):
        return "B release"
    return event


def _map_lr(raw: np.ndarray, left: set[str], right: set[str]) -> np.ndarray:
    out = np.empty(raw.shape[0], dtype=object)
    for i, v in enumerate(raw):
        if v in left:
            out[i] = "left"
        elif v in right:
            out[i] = "right"
        else:
            out[i] = ""
    return out


def side_choice_labels(
    trial_labels: dict[str, np.ndarray],
    trial_idx: np.ndarray,
    side: str,
) -> np.ndarray:
    cfg: ChoiceConfig = choice_config_for_actor_side(side)
    raw = trial_labels[cfg.field][trial_idx]
    return _map_lr(raw, set(cfg.left), set(cfg.right))


def same_diff_labels(
    trial_labels: dict[str, np.ndarray],
    trial_idx: np.ndarray,
) -> np.ndarray:
    a = side_choice_labels(trial_labels, trial_idx, "A")
    b = side_choice_labels(trial_labels, trial_idx, "B")
    out = np.empty(a.shape[0], dtype=object)
    for i in range(a.shape[0]):
        if a[i] == "" or b[i] == "":
            out[i] = ""
        elif a[i] == b[i]:
            out[i] = "same"
        else:
            out[i] = "diff"
    return out


def all_grid_labels(
    trial_labels: dict[str, np.ndarray],
    trial_idx: np.ndarray,
) -> dict[str, np.ndarray]:
    return {
        "A_choice": side_choice_labels(trial_labels, trial_idx, "A"),
        "B_choice": side_choice_labels(trial_labels, trial_idx, "B"),
        "same_diff": same_diff_labels(trial_labels, trial_idx),
    }


@dataclass(frozen=True)
class DecodeTarget:
    """One decode target: name + Decodanda conditions + label extractor."""

    name: str
    conditions: dict[str, list[str]] = field(default_factory=dict)

    def labels_for_trials(
        self,
        trial_labels: dict[str, np.ndarray],
        trial_idx: np.ndarray,
        *,
        actor_side: str,
    ) -> dict[str, np.ndarray]:
        raise NotImplementedError


@dataclass(frozen=True)
class ActorChoiceTarget(DecodeTarget):
    name: str = "actor_choice"
    conditions: dict[str, list[str]] = field(
        default_factory=lambda: {"actor_choice": ["left", "right"]},
    )

    def labels_for_trials(
        self,
        trial_labels: dict[str, np.ndarray],
        trial_idx: np.ndarray,
        *,
        actor_side: str,
    ) -> dict[str, np.ndarray]:
        cfg = choice_config_for_actor_side(actor_side)
        raw = trial_labels[cfg.field][trial_idx]
        return {"actor_choice": _map_lr(raw, set(cfg.left), set(cfg.right))}


@dataclass(frozen=True)
class AChoiceTarget(DecodeTarget):
    name: str = "A_choice"
    conditions: dict[str, list[str]] = field(
        default_factory=lambda: {"A_choice": ["left", "right"]},
    )

    def labels_for_trials(
        self,
        trial_labels: dict[str, np.ndarray],
        trial_idx: np.ndarray,
        *,
        actor_side: str,
    ) -> dict[str, np.ndarray]:
        return {"A_choice": side_choice_labels(trial_labels, trial_idx, "A")}


@dataclass(frozen=True)
class BChoiceTarget(DecodeTarget):
    name: str = "B_choice"
    conditions: dict[str, list[str]] = field(
        default_factory=lambda: {"B_choice": ["left", "right"]},
    )

    def labels_for_trials(
        self,
        trial_labels: dict[str, np.ndarray],
        trial_idx: np.ndarray,
        *,
        actor_side: str,
    ) -> dict[str, np.ndarray]:
        return {"B_choice": side_choice_labels(trial_labels, trial_idx, "B")}


@dataclass(frozen=True)
class SameDiffTarget(DecodeTarget):
    name: str = "same_diff"
    conditions: dict[str, list[str]] = field(
        default_factory=lambda: {"same_diff": ["same", "diff"]},
    )

    def labels_for_trials(
        self,
        trial_labels: dict[str, np.ndarray],
        trial_idx: np.ndarray,
        *,
        actor_side: str,
    ) -> dict[str, np.ndarray]:
        return {"same_diff": same_diff_labels(trial_labels, trial_idx)}


TARGETS: dict[str, DecodeTarget] = {
    "actor_choice": ActorChoiceTarget(),
    "A_choice": AChoiceTarget(),
    "B_choice": BChoiceTarget(),
    "same_diff": SameDiffTarget(),
}


def get_target(name: str) -> DecodeTarget:
    if name not in TARGETS:
        raise ValueError(
            f"Unknown decode target {name!r}; expected one of {sorted(TARGETS)}"
        )
    return TARGETS[name]
