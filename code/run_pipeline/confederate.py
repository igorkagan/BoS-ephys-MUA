"""Confederate session-list pipeline: one monkey × AgoB/BgoA runs."""

from __future__ import annotations

from collections.abc import Iterator

from run_pipeline.dual_nhp import dual_nhp_run_label, resolve_go_seqs
from process_channels.preprocess import (
    choice_config_for_actor_side,
    recording_actor_side,
    recording_monkey_from_condition_label,
    trial_filters_for_go_seq,
)
from run_pipeline.context import PipelineContext
from load_data.sessions import SessionListConfig, is_confederate_list


def build_confederate_context(
    cfg: SessionListConfig,
    go_seq: str,
) -> PipelineContext:
    """One pipeline run: confederate list × one timing condition."""
    monkey = recording_monkey_from_condition_label(cfg.condition_key)
    if not cfg.session_ids:
        raise ValueError(f"No sessions in confederate list {cfg.list_name!r}")
    actor_side = recording_actor_side(cfg.session_ids[0], monkey)
    choice = choice_config_for_actor_side(actor_side)
    label = dual_nhp_run_label(monkey, go_seq)
    return PipelineContext(
        data_root=cfg.root_folder,
        session_ids=list(cfg.session_ids),
        output_base=cfg.output_folder / label,
        condition_label=label,
        condition_key=cfg.condition_key,
        layout="flat",
        trial_filters=trial_filters_for_go_seq(cfg.condition_key, go_seq, actor_side),
        choice_field=choice.field,
        left_choice=list(choice.left),
        right_choice=list(choice.right),
        recording_monkey=monkey,
    )


def iter_confederate_runs(
    cfg: SessionListConfig,
    *,
    go_seq: str = "all",
) -> Iterator[PipelineContext]:
    """Yield PipelineContext for each go_seq (AgoB, BgoA) in a confederate list."""
    if not is_confederate_list(cfg.list_name):
        raise ValueError(f"Not a confederate list: {cfg.list_name!r}")
    for timing in resolve_go_seqs(go_seq):
        yield build_confederate_context(cfg, timing)
