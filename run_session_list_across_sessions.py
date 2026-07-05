#!/usr/bin/env python3
"""Run full analysis pipeline for an explicit session list from session_lists.m.

Data layout: flat export tree root_folder/{session_id}/ (see session_lists.m).

Outputs:
  DUAL_NHP: {root}/DUAL_NHP/{Monkey}_{AgoB|BgoA}/figures/...
  Confederate: {root}/{list_name}/{Monkey}_{AgoB|BgoA}/figures/...

Usage:
    python -u run_session_list_across_sessions.py --list
    python -u run_session_list_across_sessions.py DUAL_NHP
    python -u run_session_list_across_sessions.py Curius_SHUFFLED_CONF
    python -u run_session_list_across_sessions.py DUAL_NHP --monkey Curius --go-seq BgoA
    python -u run_session_list_across_sessions.py Elmo_BLOCKED_CONF --steps session_lr,consistency
"""

from __future__ import annotations

import argparse
from pathlib import Path

from bos_mua.pipeline_runner import (
    ALL_STEPS,
    build_flat_session_list_context,
    parse_steps,
    run_confederate_pipeline,
    run_dual_nhp_pipeline,
    run_pipeline_steps,
)
from bos_mua.preprocess import DUAL_NHP_GO_SEQS
from bos_mua.session_lists import (
    DUAL_NHP_MONKEYS,
    is_confederate_list,
    is_dual_nhp_list,
    load_session_list,
    summarize_session_lists,
)


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
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
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
        )
        return

    if is_confederate_list(args.list_name):
        run_confederate_pipeline(
            session_lists_path,
            args.list_name,
            steps,
            go_seq=args.go_seq,
        )
        return

    cfg = load_session_list(args.list_name, session_lists_path)
    ctx = build_flat_session_list_context(cfg)
    print("\n" + "#" * 72)
    print(f"# {ctx.condition_label}")
    print("#" * 72)
    run_pipeline_steps(ctx, steps)


if __name__ == "__main__":
    main()
