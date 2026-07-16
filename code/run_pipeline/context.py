"""Shared pipeline context for curated and flat session-list layouts."""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from load_data.io import session_sort_key
from process_channels.preprocess import recording_monkey as infer_recording_monkey
from process_channels.preprocess import alignment_event_for_recording

Layout = Literal["curated", "flat"]
SourceKind = Literal["curated", "session_list"]

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
    source_kind: SourceKind = "session_list"
    session_parent: str | None = None
    choice_field: str = "A_LR_pos_list"
    left_choice: list[str] = field(default_factory=lambda: ["Al"])
    right_choice: list[str] = field(default_factory=lambda: ["Ar"])
    recording_monkey: str = ""

    def __post_init__(self) -> None:
        if self.source_kind == "curated":
            if self.layout != "curated" or not self.session_parent:
                raise ValueError(
                    "Curated contexts require layout='curated' and an explicit session_parent"
                )
        elif self.layout != "flat" or self.session_parent is not None:
            raise ValueError(
                "Session-list contexts require layout='flat' and no session_parent"
            )

    def session_dir(self, session_id: str) -> Path:
        parent = self.data_root / self.session_parent if self.session_parent else self.data_root
        return parent / session_id

    def resolved_recording_monkey(self, session_id: str) -> str:
        """Recorded monkey for z-scoring; explicit on context or inferred from session."""
        if self.recording_monkey:
            return self.recording_monkey
        return infer_recording_monkey(
            session_id=session_id,
            condition_label=self.condition_label,
        )

    def alignment_event(self, session_id: str) -> str:
        """Per-session alignment folder (A or B fixation release)."""
        return alignment_event_for_recording(
            session_id, self.resolved_recording_monkey(session_id),
        )

    def alignment_event_label(self, session_ids: list[str]) -> str:
        """Title/filename label; single event name or generic if mixed."""
        events = {self.alignment_event(sid) for sid in session_ids}
        if len(events) == 1:
            return events.pop()
        return "actor-side fixation release"


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

    from load_data.io import discover_sessions

    condition_dir = Path(data_root) / condition_folder
    if not condition_dir.exists():
        return []
    return discover_sessions(condition_dir)


def apply_pipeline_context(ctx: PipelineContext) -> None:
    """Patch imported analysis modules for the current pipeline run."""
    from analyze_stability import deep_dives as pbw
    from analyze_stability import combine as cs
    from analyze_stability import consistency as acc
    from analyze_stability import session_lr as psl

    figures_dir = ctx.output_base
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


def verify_sessions(ctx: PipelineContext) -> list[str]:
    """Return session IDs with existing directories; warn on missing."""
    present: list[str] = []
    missing: list[str] = []
    for session_id in ctx.session_ids:
        if ctx.session_dir(session_id).exists():
            present.append(session_id)
        else:
            missing.append(session_id)
    for session_id in missing:
        warnings.warn(f"Missing session directory: {ctx.session_dir(session_id)}")
    if not present:
        raise FileNotFoundError(
            f"No session directories found under {ctx.data_root} for "
            f"{ctx.condition_label!r}"
        )
    return present
