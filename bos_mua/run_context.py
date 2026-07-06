"""Shared pipeline context for curated and flat session-list layouts."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from bos_mua.preprocess import choice_config_for_monkey, recording_monkey as infer_recording_monkey

Layout = Literal["curated", "flat"]

_active: "PipelineContext | None" = None


@dataclass(frozen=True)
class PipelineContext:
    data_root: Path
    session_ids: list[str]
    output_base: Path
    condition_label: str
    condition_key: str
    layout: Layout
    trial_filters: dict
    choice_field: str = "A_LR_pos_list"
    left_choice: list[str] = field(default_factory=lambda: ["Al"])
    right_choice: list[str] = field(default_factory=lambda: ["Ar"])
    recording_monkey: str = ""

    def session_dir(self, session_id: str) -> Path:
        if self.layout == "flat":
            return self.data_root / session_id
        return self.data_root / self.condition_key / session_id

    def resolved_recording_monkey(self, session_id: str) -> str:
        """Recorded monkey for z-scoring; explicit on context or inferred from session."""
        if self.recording_monkey:
            return self.recording_monkey
        return infer_recording_monkey(
            session_id=session_id,
            condition_label=self.condition_label,
        )


def get_active_context() -> PipelineContext | None:
    return _active


def set_active_context(ctx: PipelineContext | None) -> None:
    global _active
    _active = ctx


def resolve_session_dir(
    session_id: str,
    *,
    data_root: str | Path,
    condition_folder: str,
    layout: Layout = "curated",
) -> Path:
    """Resolve session directory for curated or flat layout."""
    ctx = get_active_context()
    if ctx is not None:
        return ctx.session_dir(session_id)

    data_root = Path(data_root)
    if layout == "flat":
        return data_root / session_id
    return data_root / condition_folder / session_id


def session_ids_for_run(
    data_root: str | Path,
    condition_folder: str,
    *,
    session_id: str | None = None,
    explicit_ids: list[str] | None = None,
) -> list[str]:
    """Return session IDs from active context, explicit list, or filesystem discovery."""
    ctx = get_active_context()
    if ctx is not None:
        return list(ctx.session_ids)
    if session_id:
        return [session_id]
    if explicit_ids:
        return list(explicit_ids)

    from bos_mua.io import discover_sessions

    condition_dir = Path(data_root) / condition_folder
    if not condition_dir.exists():
        return []
    return discover_sessions(condition_dir)


def apply_pipeline_context(ctx: PipelineContext) -> None:
    """Patch imported analysis modules for the current pipeline run."""
    from bos_mua.steps import best_worst as pbw
    from bos_mua.steps import combine as cs
    from bos_mua.steps import consistency as acc
    from bos_mua.steps import session_lr as psl

    figures_dir = ctx.output_base / "figures"
    data_root = str(ctx.data_root)
    output_dir = str(figures_dir)

    set_active_context(ctx)

    acc.DATA_ROOT = data_root
    acc.CONDITION_FOLDER = ctx.condition_label
    acc.SESSION_IDS = list(ctx.session_ids)
    acc.TRIAL_FILTERS = ctx.trial_filters
    acc.CHOICE_FIELD = ctx.choice_field
    acc.LEFT_CHOICE = list(ctx.left_choice)
    acc.RIGHT_CHOICE = list(ctx.right_choice)
    acc.OUTPUT_DIR = output_dir

    psl.DATA_ROOT = data_root
    psl.CONDITION_FOLDER = ctx.condition_label
    psl.TRIAL_FILTERS = ctx.trial_filters
    psl.CHOICE_FIELD = ctx.choice_field
    psl.LEFT_CHOICE = list(ctx.left_choice)
    psl.RIGHT_CHOICE = list(ctx.right_choice)
    psl.OUTPUT_DIR = output_dir

    cs.DATA_ROOT = data_root
    cs.TRIAL_FILTERS = ctx.trial_filters
    cs.CHOICE_FIELD = ctx.choice_field
    cs.LEFT_CHOICE = list(ctx.left_choice)
    cs.RIGHT_CHOICE = list(ctx.right_choice)
    cs.OUTPUT_DIR = output_dir

    pbw.DATA_ROOT = data_root
    pbw.TRIAL_FILTERS = ctx.trial_filters
    pbw.CHOICE_FIELD = ctx.choice_field
    pbw.LEFT_CHOICE = list(ctx.left_choice)
    pbw.RIGHT_CHOICE = list(ctx.right_choice)
    pbw.OUTPUT_DIR = output_dir
