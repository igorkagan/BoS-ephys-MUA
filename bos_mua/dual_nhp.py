"""DUAL_NHP (and future single-monkey confederate) pipeline run definitions."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

from bos_mua.preprocess import (
    DUAL_NHP_GO_SEQS,
    choice_config_for_monkey,
    trial_filters_for_dual_nhp_go_seq,
)
from bos_mua.run_context import PipelineContext
from bos_mua.session_lists import (
    DUAL_NHP_MONKEYS,
    SessionListConfig,
    load_dual_nhp_configs,
    split_sessions_by_recording_monkey,
)


@dataclass(frozen=True)
class DualNhpRunKey:
    """One exported analysis condition: recorded monkey × go sequence."""

    monkey: str
    go_seq: str

    @property
    def label(self) -> str:
        return dual_nhp_run_label(self.monkey, self.go_seq)


ALL_DUAL_NHP_RUNS: tuple[DualNhpRunKey, ...] = tuple(
    DualNhpRunKey(monkey, go_seq)
    for monkey in DUAL_NHP_MONKEYS
    for go_seq in DUAL_NHP_GO_SEQS
)


def dual_nhp_run_label(monkey: str, go_seq: str) -> str:
    """Output folder / condition label, e.g. Curius_AgoB."""
    if monkey not in DUAL_NHP_MONKEYS:
        raise ValueError(f"Unknown monkey {monkey!r}; expected one of {DUAL_NHP_MONKEYS}")
    if go_seq not in DUAL_NHP_GO_SEQS:
        raise ValueError(f"Unknown go_seq {go_seq!r}; expected one of {DUAL_NHP_GO_SEQS}")
    return f"{monkey}_{go_seq}"


def resolve_go_seqs(go_seq: str) -> list[str]:
    if go_seq == "all":
        return list(DUAL_NHP_GO_SEQS)
    if go_seq not in DUAL_NHP_GO_SEQS:
        raise ValueError(f"Unknown go_seq {go_seq!r}; expected all, AgoB, or BgoA")
    return [go_seq]


def build_dual_nhp_context(
    cfg: SessionListConfig,
    monkey: str,
    go_seq: str,
    session_ids: list[str],
) -> PipelineContext:
    """One pipeline run: one recorded monkey × one timing condition."""
    choice = choice_config_for_monkey(monkey)
    label = dual_nhp_run_label(monkey, go_seq)
    return PipelineContext(
        data_root=cfg.root_folder,
        session_ids=list(session_ids),
        output_base=cfg.output_folder / label,
        condition_label=label,
        condition_key=label,
        layout="flat",
        trial_filters=trial_filters_for_dual_nhp_go_seq(go_seq),
        choice_field=choice.field,
        left_choice=list(choice.left),
        right_choice=list(choice.right),
        recording_monkey=monkey,
    )


def iter_dual_nhp_runs(
    session_lists_path: str | Path,
    *,
    monkey: str | None = None,
    go_seq: str = "all",
) -> Iterator[PipelineContext]:
    """Yield PipelineContext for each monkey × go_seq with at least one session."""
    cfg, split = load_dual_nhp_configs(session_lists_path)
    monkeys = [monkey] if monkey else list(DUAL_NHP_MONKEYS)
    go_seqs = resolve_go_seqs(go_seq)

    for name in monkeys:
        if name not in DUAL_NHP_MONKEYS:
            raise ValueError(f"Unknown monkey {name!r}; expected one of {DUAL_NHP_MONKEYS}")
        session_ids = split[name]
        if not session_ids:
            continue
        for timing in go_seqs:
            yield build_dual_nhp_context(cfg, name, timing, session_ids)


def split_for_list(session_ids: list[str]) -> dict[str, list[str]]:
    """Group flat-export sessions by recorded monkey (paired U/B or confederate)."""
    return split_sessions_by_recording_monkey(session_ids)
