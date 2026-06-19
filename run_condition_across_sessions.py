#!/usr/bin/env python3
"""Run full analysis pipeline for one or all monkey/condition folders.

Steps (all sessions in the condition):
  1. plot_session_lr_mua — per-session L/R PDFs (original + z-scored)
  2. assess_cross_session_consistency — cross-session figures + CSV
  3. combine_sessions — z-scored pooled L/R PDFs
  4. plot_best_worst_channels — best/worst/tuned-stable deep dives

Usage:
    python -u run_condition_across_sessions.py Elmo_SHUFFLED
    python -u run_condition_across_sessions.py Curius_BLOCKED
    python -u run_condition_across_sessions.py --all
    python -u run_condition_across_sessions.py Elmo_BLOCKED --steps session_lr,consistency
"""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import assess_cross_session_consistency as acc
import combine_sessions as cs
import plot_best_worst_channels as pbw
import plot_session_lr_mua as psl
from bos_mua.io import discover_sessions
from bos_mua.preprocess import (
    MONKEY_CONDITIONS,
    resolve_condition_output_dir,
    trial_filters_for_condition,
)

ALL_STEPS = ("session_lr", "consistency", "combine", "best_worst")


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


def parse_steps(steps_arg: str) -> list[str]:
    steps = [s.strip() for s in steps_arg.split(",") if s.strip()]
    unknown = set(steps) - set(ALL_STEPS)
    if unknown:
        raise SystemExit(
            f"Unknown step(s): {', '.join(sorted(unknown))}. "
            f"Valid: {', '.join(ALL_STEPS)}"
        )
    return steps


def apply_config(condition: str) -> None:
    """Patch imported modules for this condition (does not change file defaults on disk)."""
    trial_filters = trial_filters_for_condition(condition)
    acc.CONDITION_FOLDER = condition
    acc.TRIAL_FILTERS = trial_filters
    psl.CONDITION_FOLDER = condition
    pbw.TRIAL_FILTERS = trial_filters


def verify_data(condition: str) -> list[str]:
    condition_dir = Path(psl.DATA_ROOT) / condition
    if not condition_dir.exists():
        raise FileNotFoundError(f"Missing condition folder: {condition_dir}")
    session_ids = discover_sessions(condition_dir)
    print(f"DATA_ROOT: {psl.DATA_ROOT}")
    print(f"Condition: {condition} | sessions: {len(session_ids)}")
    for sid in session_ids:
        print(f"  {sid}")
    print(f"TRIAL_FILTERS: {acc.TRIAL_FILTERS}")
    return session_ids


def run_pipeline(condition: str, steps: list[str]) -> None:
    apply_config(condition)
    session_ids = verify_data(condition)

    if "session_lr" in steps:
        print("\n" + "=" * 72)
        print("STEP: plot_session_lr_mua")
        print("=" * 72)
        psl.main()

    if "consistency" in steps:
        print("\n" + "=" * 72)
        print("STEP: assess_cross_session_consistency")
        print("=" * 72)
        acc.main()

    if "combine" in steps:
        print("\n" + "=" * 72)
        print("STEP: combine_sessions (z-scored combined)")
        print("=" * 72)
        if len(session_ids) < cs.MIN_SESSIONS:
            warnings.warn(
                f"Skipping combine for {condition}: need >={cs.MIN_SESSIONS} sessions, "
                f"found {len(session_ids)}"
            )
        else:
            output_dir = resolve_condition_output_dir(cs.OUTPUT_DIR, True, condition, "combined")
            output_dir.mkdir(parents=True, exist_ok=True)
            cs.plot_condition_combined(condition, output_dir)

    if "best_worst" in steps:
        print("\n" + "=" * 72)
        print("STEP: plot_best_worst_channels")
        print("=" * 72)
        modes = (False, True) if acc.RUN_BOTH_PROCESSING else (acc.ZSCORE_MUA,)
        for zscore_mua in modes:
            pbw.run_channel_rank_plots(condition, zscore_mua)

    print(f"\nPipeline finished for {condition}")


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    conditions = resolve_conditions(args)
    steps = parse_steps(args.steps)

    for condition in conditions:
        print("\n" + "#" * 72)
        print(f"# {condition}")
        print("#" * 72)
        run_pipeline(condition, steps)


if __name__ == "__main__":
    main()
