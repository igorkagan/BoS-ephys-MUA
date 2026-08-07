"""Output paths for decoding under timing / trial-type branches."""

from __future__ import annotations

from pathlib import Path

from analyze_decoding.config import FIGURES_ROOT
from run_pipeline.dual_nhp import dual_nhp_run_label
from process_channels.preprocess import recording_monkey_from_condition_label


def timing_branch_dir(
    condition: str,
    go_seq: str,
    trial_type: str,
    *,
    figures_root: Path | None = None,
) -> Path:
    root = figures_root if figures_root is not None else FIGURES_ROOT
    monkey = recording_monkey_from_condition_label(condition)
    return root / condition / dual_nhp_run_label(monkey, go_seq) / trial_type


def decoding_dir(
    condition: str,
    go_seq: str,
    trial_type: str,
    target: str,
    *,
    figures_root: Path | None = None,
) -> Path:
    return (
        timing_branch_dir(condition, go_seq, trial_type, figures_root=figures_root)
        / "decoding"
        / target
    )


def session_decode_stem(session_id: str) -> str:
    return f"{session_id}_decode"


def combined_dir(decode_root: Path) -> Path:
    return decode_root / "combined"
