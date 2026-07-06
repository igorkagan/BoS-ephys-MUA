#!/usr/bin/env python3
"""Run full analysis pipeline for one or all monkey/condition folders.

Usage:
    python -u scripts/run_curated.py Elmo_SHUFFLED
    python -u scripts/run_curated.py Curius_BLOCKED
    python -u scripts/run_curated.py --all
    python -u scripts/run_curated.py Elmo_BLOCKED --steps session_lr,consistency
"""

from __future__ import annotations

import _bootstrap  # noqa: F401
import argparse
from pathlib import Path

from bos_mua.pipeline import (
    ALL_STEPS,
    build_curated_context,
    parse_steps,
    run_pipeline_steps,
)
from bos_mua.preprocess import MONKEY_CONDITIONS, trial_filters_for_condition
from bos_mua.steps import session_lr as psl


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run analysis pipeline for a monkey/condition across all sessions.",
    )
    parser.add_argument(
        "condition",
        nargs="?",
        help=f"Condition folder name, e.g. Elmo_SHUFFLED (one of {MONKEY_CONDITIONS})",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Run all conditions in MONKEY_CONDITIONS",
    )
    parser.add_argument(
        "--steps",
        default=",".join(ALL_STEPS),
        help=f"Comma-separated steps to run (default: all). Choices: {','.join(ALL_STEPS)}",
    )
    return parser.parse_args(argv)


def resolve_conditions(args: argparse.Namespace) -> list[str]:
    if args.all and args.condition:
        raise SystemExit("Pass either a condition name or --all, not both.")
    if args.all:
        return list(MONKEY_CONDITIONS)
    if not args.condition:
        argparse.ArgumentParser(
            description="Run analysis pipeline for a monkey/condition across all sessions.",
        ).print_help()
        raise SystemExit("Provide a condition name or --all.")
    if args.condition not in MONKEY_CONDITIONS:
        raise SystemExit(
            f"Unknown condition {args.condition!r}. "
            f"Expected one of: {', '.join(MONKEY_CONDITIONS)}"
        )
    return [args.condition]


def verify_condition_folder(condition: str) -> None:
    condition_dir = Path(psl.DATA_ROOT) / condition
    if not condition_dir.exists():
        raise FileNotFoundError(f"Missing condition folder: {condition_dir}")


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    conditions = resolve_conditions(args)
    steps = parse_steps(args.steps)
    for condition in conditions:
        verify_condition_folder(condition)
        ctx = build_curated_context(
            condition, trial_filters_for_condition(condition),
        )
        run_pipeline_steps(ctx, steps)


if __name__ == "__main__":
    main()
