"""Shared CLI for single-step pipeline reruns."""

from __future__ import annotations

import argparse
from pathlib import Path

from run_pipeline import (
    ALL_STEPS,
    build_flat_session_list_context,
    run_confederate_pipeline,
    run_curated_pipeline,
    run_dual_nhp_pipeline,
    run_pipeline_steps,
)
from process_channels.preprocess import DUAL_NHP_GO_SEQS, MONKEY_CONDITIONS
from load_data.sessions import (
    DUAL_NHP_MONKEYS,
    is_confederate_list,
    is_dual_nhp_list,
    load_session_list,
)
from run_pipeline.curated import verify_curated_condition
from run_scripts._processing import add_processing_args, apply_processing_args


def parse_partial_args(step: str, argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=f"Re-run pipeline step: {step}",
    )
    parser.add_argument(
        "target",
        help="Curated condition (e.g. Curius_BLOCKED) or session list name (e.g. DUAL_NHP)",
    )
    parser.add_argument(
        "--session-lists",
        default="session_lists.m",
        help="Path to session_lists.m (session-list targets only)",
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
        help="Skip solo trial branches",
    )
    parser.add_argument(
        "--monkey",
        choices=list(DUAL_NHP_MONKEYS),
        help="DUAL_NHP only: Curius or Elmo",
    )
    parser.add_argument(
        "--allow-legacy-flat",
        action="store_true",
        help="Explicitly allow a reduced, unbranched legacy session list",
    )
    if step == "comparisons":
        parser.add_argument(
            "--only-combined",
            action="store_true",
            help=(
                "Replot comparison combined PDFs only "
                "(skip session overlays, heatmaps, deep dives, global scatters)"
            ),
        )
    add_processing_args(parser)
    return parser.parse_args(argv)


def run_partial_step(step: str, args: argparse.Namespace) -> None:
    if step not in ALL_STEPS:
        raise ValueError(f"Unknown step {step!r}; valid: {', '.join(ALL_STEPS)}")
    steps = [step]
    include_solo = not args.dyadic_only
    only_combined = bool(getattr(args, "only_combined", False))
    target = args.target

    if target in MONKEY_CONDITIONS:
        verify_curated_condition(target)
        run_curated_pipeline(
            target,
            steps,
            go_seq=args.go_seq,
            include_solo=include_solo,
            only_combined=only_combined,
        )
        return

    session_lists_path = Path(args.session_lists)
    if not session_lists_path.exists():
        raise FileNotFoundError(f"Session lists file not found: {session_lists_path}")

    if args.monkey and not is_dual_nhp_list(target):
        raise SystemExit("--monkey is only valid with DUAL_NHP.")

    if is_dual_nhp_list(target):
        run_dual_nhp_pipeline(
            session_lists_path,
            steps,
            monkey=args.monkey,
            go_seq=args.go_seq,
            include_solo=include_solo,
            only_combined=only_combined,
        )
        return

    if is_confederate_list(target):
        run_confederate_pipeline(
            session_lists_path,
            target,
            steps,
            go_seq=args.go_seq,
            include_solo=include_solo,
            only_combined=only_combined,
        )
        return

    if not args.allow_legacy_flat:
        raise SystemExit(
            f"Session list {target!r} is neither DUAL_NHP nor *_CONF. "
            "Refusing the reduced legacy flat pipeline; pass --allow-legacy-flat "
            "only when that behavior is intentional."
        )
    if step == "comparisons":
        raise SystemExit("Legacy flat lists do not support the comparisons step.")
    cfg = load_session_list(target, session_lists_path)
    ctx = build_flat_session_list_context(cfg)
    print("\n" + "#" * 72)
    print(f"# {ctx.condition_label}")
    print("#" * 72)
    run_pipeline_steps(ctx, steps)


def main_for_step(step: str, argv: list[str] | None = None) -> None:
    args = parse_partial_args(step, argv)
    apply_processing_args(args)
    run_partial_step(step, args)
