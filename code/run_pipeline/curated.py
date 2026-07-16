"""Curated condition pipeline: discover sessions from MUA_curated_sessions/."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

from load_data.io import discover_sessions
from run_pipeline.context import PipelineContext, session_ids_for_run
from run_pipeline.dual_nhp import dual_nhp_run_label, resolve_go_seqs
from process_channels.preprocess import (
    MONKEY_CONDITIONS,
    choice_config_for_actor_side,
    recording_actor_side,
    recording_monkey_from_condition_label,
    trial_filters_for_go_seq,
)

DEFAULT_DATA_ROOT = Path(
    r"S:\taskcontroller\SCP_DATA\SCP-CTRL-01\MUA_curated_sessions",
)
REPO_ROOT = Path(__file__).resolve().parents[2]


def curated_condition_output(
    condition: str,
    *,
    figures_root: Path | None = None,
) -> Path:
    """Parent output folder for one curated condition, e.g. repo/figures/Elmo_BLOCKED."""
    root = figures_root if figures_root is not None else REPO_ROOT / "figures"
    return root / condition


def curated_data_root(data_root: Path | None = None) -> Path:
    return data_root if data_root is not None else DEFAULT_DATA_ROOT


def curated_output_base(
    condition: str,
    *,
    figures_root: Path | None = None,
) -> Path:
    """Condition-level output (comparisons, etc.)."""
    return curated_condition_output(condition, figures_root=figures_root)


def discover_curated_sessions(
    condition: str,
    *,
    data_root: Path | None = None,
) -> list[str]:
    root = curated_data_root(data_root)
    if condition not in MONKEY_CONDITIONS:
        raise ValueError(
            f"Unknown condition {condition!r}; expected one of {MONKEY_CONDITIONS}"
        )
    condition_dir = root / condition
    if not condition_dir.exists():
        raise FileNotFoundError(f"Missing condition folder: {condition_dir}")
    return discover_sessions(condition_dir)


def build_curated_context(
    condition: str,
    go_seq: str,
    session_ids: list[str],
    *,
    data_root: Path | None = None,
    output_base: Path | None = None,
) -> PipelineContext:
    """One pipeline run: curated condition x one timing (AgoB or BgoA)."""
    root = curated_data_root(data_root)
    monkey = recording_monkey_from_condition_label(condition)
    if not session_ids:
        raise ValueError(f"No sessions for curated condition {condition!r}")
    actor_side = recording_actor_side(session_ids[0], monkey)
    choice = choice_config_for_actor_side(actor_side)
    label = dual_nhp_run_label(monkey, go_seq)
    figures_root = output_base if output_base is not None else None
    run_out = curated_condition_output(condition, figures_root=figures_root) / label
    return PipelineContext(
        data_root=root,
        session_ids=list(session_ids),
        output_base=run_out,
        condition_label=label,
        condition_key=condition,
        layout="curated",
        trial_filters=trial_filters_for_go_seq(condition, go_seq, actor_side),
        source_kind="curated",
        session_parent=condition,
        choice_field=choice.field,
        left_choice=list(choice.left),
        right_choice=list(choice.right),
        recording_monkey=monkey,
    )


def iter_curated_runs(
    condition: str,
    *,
    go_seq: str = "all",
    data_root: Path | None = None,
    output_base: Path | None = None,
) -> Iterator[PipelineContext]:
    """Yield PipelineContext for each go_seq with sessions discovered from disk."""
    session_ids = discover_curated_sessions(condition, data_root=data_root)
    if not session_ids:
        raise FileNotFoundError(f"No sessions under {curated_data_root(data_root) / condition}")
    for timing in resolve_go_seqs(go_seq):
        yield build_curated_context(
            condition,
            timing,
            session_ids,
            data_root=data_root,
            output_base=output_base,
        )


def verify_curated_condition(condition: str, *, data_root: Path | None = None) -> None:
    root = curated_data_root(data_root)
    condition_dir = root / condition
    if not condition_dir.exists():
        raise FileNotFoundError(f"Missing condition folder: {condition_dir}")
