#!/usr/bin/env python3
"""Run full analysis pipeline for one or all monkey/condition folders.

Usage:
    python -u code/run_scripts/run_curated.py Elmo_SHUFFLED
    python -u code/run_scripts/run_curated.py Curius_BLOCKED
    python -u code/run_scripts/run_curated.py --all
    python -u code/run_scripts/run_curated.py Elmo_BLOCKED --steps session_lr,consistency
    python -u code/run_scripts/run_curated.py Curius_BLOCKED --go-seq AgoB --dyadic-only
"""

from __future__ import annotations

import _bootstrap  # noqa: F401
import argparse

from run_pipeline import ALL_STEPS, parse_steps, run_curated_pipeline
from process_channels.preprocess import DUAL_NHP_GO_SEQS, MONKEY_CONDITIONS
from run_pipeline.curated import verify_curated_condition
from run_scripts._processing import add_processing_args, apply_processing_args


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
    parser.add_argument(
        "--go-seq",
        choices=[*DUAL_NHP_GO_SEQS, "all"],
        default="all",
        help="AgoB, BgoA, or all (default: all)",
    )
    parser.add_argument(
        "--dyadic-only",
        action="store_true",
        help="Skip solo trial branches (SoloA/SoloB) and solo comparisons",
    )
    add_processing_args(parser)
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


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    apply_processing_args(args)
    conditions = resolve_conditions(args)
    steps = parse_steps(args.steps)
    for condition in conditions:
        verify_curated_condition(condition)
        run_curated_pipeline(
            condition,
            steps,
            go_seq=args.go_seq,
            include_solo=not args.dyadic_only,
        )


if __name__ == "__main__":
    main()
