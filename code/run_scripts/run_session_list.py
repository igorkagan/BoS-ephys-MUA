#!/usr/bin/env python3
"""Run full analysis pipeline for an explicit session list from session_lists.m.

Data layout: flat export tree root_folder/{session_id}/ (see session_lists.m).

Outputs:
  DUAL_NHP: {root}/DUAL_NHP/{Monkey}_{AgoB|BgoA}/figures/... and SoloA/ or SoloB/
  Confederate: {root}/{list_name}/{Monkey}_{AgoB|BgoA}/figures/... and SoloA/ or SoloB/

Usage:
    python -u code/run_scripts/run_session_list.py --list
    python -u code/run_scripts/run_session_list.py DUAL_NHP
    python -u code/run_scripts/run_session_list.py Curius_SHUFFLED_CONF
"""

from __future__ import annotations

import _bootstrap  # noqa: F401
import argparse
from pathlib import Path

from run_pipeline import (
    ALL_STEPS,
    build_flat_session_list_context,
    parse_steps,
    run_confederate_pipeline,
    run_dual_nhp_pipeline,
    run_pipeline_steps,
)
from process_channels.preprocess import DUAL_NHP_GO_SEQS
from load_data.sessions import (
    DUAL_NHP_MONKEYS,
    is_confederate_list,
    is_dual_nhp_list,
    load_session_list,
    summarize_session_lists,
)
from run_scripts._processing import add_processing_args, apply_processing_args


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run analysis pipeline for a named session list from session_lists.m.",
    )
    parser.add_argument(
        "list_name",
        nargs="?",
        help="Session list name, e.g. DUAL_NHP or Curius_SHUFFLED_CONF",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="Show parseable session list names and exit",
    )
    parser.add_argument(
        "--session-lists",
        default="session_lists.m",
        help="Path to session_lists.m (default: session_lists.m)",
    )
    parser.add_argument(
        "--steps",
        default=",".join(ALL_STEPS),
        help=f"Comma-separated steps (default: all). Choices: {','.join(ALL_STEPS)}",
    )
    parser.add_argument(
        "--monkey",
        choices=list(DUAL_NHP_MONKEYS),
        help="DUAL_NHP only: Curius or Elmo (default: both)",
    )
    parser.add_argument(
        "--go-seq",
        choices=[*DUAL_NHP_GO_SEQS, "all"],
        default="all",
        help="DUAL_NHP or confederate: AgoB, BgoA, or all (default: all)",
    )
    parser.add_argument(
        "--dyadic-only",
        action="store_true",
        help="Skip solo trial branches (SoloA/SoloB) and solo comparisons",
    )
    parser.add_argument(
        "--allow-legacy-flat",
        action="store_true",
        help=(
            "Explicitly allow an unbranched legacy list. Such lists do not support "
            "the full AgoB/BgoA comparison matrix."
        ),
    )
    add_processing_args(parser)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    apply_processing_args(args)
    session_lists_path = Path(args.session_lists)

    if args.list:
        if not session_lists_path.exists():
            raise FileNotFoundError(f"Session lists file not found: {session_lists_path}")
        for name, count, output_folder, detail in summarize_session_lists(session_lists_path):
            suffix = f" ({detail})" if detail else ""
            print(f"{name}: {count} session(s) -> {output_folder}{suffix}")
        return

    if not args.list_name:
        parse_args(["--help"])
        raise SystemExit(1)

    if not session_lists_path.exists():
        raise FileNotFoundError(f"Session lists file not found: {session_lists_path}")

    steps = parse_steps(args.steps)

    if args.monkey and not is_dual_nhp_list(args.list_name):
        raise SystemExit("--monkey is only valid with DUAL_NHP.")

    if is_dual_nhp_list(args.list_name):
        run_dual_nhp_pipeline(
            session_lists_path,
            steps,
            monkey=args.monkey,
            go_seq=args.go_seq,
            include_solo=not args.dyadic_only,
        )
        return

    if is_confederate_list(args.list_name):
        run_confederate_pipeline(
            session_lists_path,
            args.list_name,
            steps,
            go_seq=args.go_seq,
            include_solo=not args.dyadic_only,
        )
        return

    if not args.allow_legacy_flat:
        raise SystemExit(
            f"Session list {args.list_name!r} is neither DUAL_NHP nor *_CONF. "
            "Refusing the reduced legacy flat pipeline; pass --allow-legacy-flat "
            "only if AgoB-only, unbranched behavior is intentional."
        )
    if "comparisons" in steps:
        raise SystemExit("Legacy flat lists do not support the comparisons step.")
    cfg = load_session_list(args.list_name, session_lists_path)
    ctx = build_flat_session_list_context(cfg)
    print("\n" + "#" * 72)
    print(f"# {ctx.condition_label}")
    print("#" * 72)
    run_pipeline_steps(ctx, steps)


if __name__ == "__main__":
    main()
