#!/usr/bin/env python3
"""Run session-based decoding on curated conditions.

Usage:
    python -u code/run_scripts/run_decode_curated.py Elmo_BLOCKED --mode actor
    python -u code/run_scripts/run_decode_curated.py Elmo_BLOCKED --trial-type Dyadic --mode dyadic_all
    python -u code/run_scripts/run_decode_curated.py Elmo_BLOCKED --trial-type SoloA --mode actor
"""

from __future__ import annotations

import _bootstrap  # noqa: F401
import argparse
from pathlib import Path

from analyze_decoding.config import default_curated_data_root
from analyze_decoding.grid_run import (
    run_choice_ab_grid,
    run_dyadic_all_grids,
    run_same_diff_grid,
)
from analyze_decoding.run import run_decode_curated
from analyze_decoding.targets import TARGETS
from process_channels.preprocess import DUAL_NHP_GO_SEQS, MONKEY_CONDITIONS

MODES = (
    "actor",
    "choice_ab",
    "choice_ab_balanced",
    "same_diff",
    "dyadic_all",
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Curated session decoding (Decodanda).")
    p.add_argument(
        "condition",
        help=f"Curated condition folder, e.g. Elmo_BLOCKED (one of {MONKEY_CONDITIONS})",
    )
    p.add_argument(
        "--go-seq",
        choices=[*DUAL_NHP_GO_SEQS, "all"],
        default="all",
        help="AgoB, BgoA, or all (default: all)",
    )
    p.add_argument(
        "--trial-type",
        choices=["Dyadic", "SoloA", "both"],
        default="both",
        help="Trial branch (default: both). Grid modes require Dyadic.",
    )
    p.add_argument(
        "--mode",
        choices=MODES,
        default="actor",
        help=(
            "actor=legacy actor_choice @ actor align; "
            "choice_ab / choice_ab_balanced / same_diff / dyadic_all=Dyadic grids"
        ),
    )
    p.add_argument(
        "--target",
        default="actor_choice",
        choices=sorted(TARGETS),
        help="Decode target for --mode actor (default: actor_choice)",
    )
    p.add_argument("--data-root", type=Path, default=None)
    p.add_argument("--figures-root", type=Path, default=None)
    p.add_argument("--force", action="store_true")
    p.add_argument("--validate-only", action="store_true")
    p.add_argument("--replot-only", action="store_true")
    p.add_argument("--nshuffles", type=int, default=None)
    p.add_argument("--cv", type=int, default=None)
    p.add_argument("--max-sessions", type=int, default=None)
    p.add_argument("--max-bins", type=int, default=None)
    p.add_argument(
        "--smooth-ms",
        type=float,
        default=None,
        help="Gaussian FWHM ms (0=off). Default: config GAUSSIAN_SMOOTH_MS",
    )
    p.add_argument(
        "--decode-name",
        type=str,
        default=None,
        help="Output folder under decoding/ (default: --target name)",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if args.condition not in MONKEY_CONDITIONS:
        raise SystemExit(
            f"Unknown condition {args.condition!r}; expected one of {MONKEY_CONDITIONS}"
        )
    if args.validate_only and args.replot_only:
        raise SystemExit("Use only one of --validate-only / --replot-only")

    go_seqs = DUAL_NHP_GO_SEQS if args.go_seq == "all" else (args.go_seq,)
    data_root = args.data_root if args.data_root is not None else default_curated_data_root()
    print(f"DATA_ROOT: {data_root}")
    print(f"condition={args.condition} go_seqs={go_seqs} mode={args.mode}")

    grid_modes = {"choice_ab", "choice_ab_balanced", "same_diff", "dyadic_all"}
    if args.mode in grid_modes:
        if args.trial_type == "SoloA":
            raise SystemExit(
                f"--mode {args.mode} is Dyadic-only; use --mode actor for SoloA"
            )
        if args.trial_type == "both":
            print("[info] grid modes run Dyadic only (SoloA skipped)")
        if args.validate_only or args.replot_only:
            raise SystemExit("validate/replot not supported for grid modes yet")

        kw = dict(
            data_root=data_root,
            figures_root=args.figures_root,
            force=args.force,
            max_sessions=args.max_sessions,
            nshuffles=args.nshuffles,
            cross_validations=args.cv,
        )
        if args.mode == "dyadic_all":
            run_dyadic_all_grids(args.condition, go_seqs=go_seqs, **kw)
        elif args.mode == "choice_ab":
            for go in go_seqs:
                run_choice_ab_grid(args.condition, go, balanced=False, **kw)
        elif args.mode == "choice_ab_balanced":
            for go in go_seqs:
                run_choice_ab_grid(args.condition, go, balanced=True, **kw)
        else:
            for go in go_seqs:
                run_same_diff_grid(args.condition, go, **kw)
        return

    # actor mode (SoloA and/or Dyadic single-panel)
    if args.trial_type == "both":
        trial_types: tuple[str, ...] = ("Dyadic", "SoloA")
    else:
        trial_types = (args.trial_type,)

    decode_name = args.decode_name
    if decode_name is None and args.smooth_ms is not None and args.smooth_ms <= 0:
        decode_name = f"{args.target}_nosmooth"
    print(
        f"target={args.target} decode_name={decode_name or args.target} "
        f"smooth_ms={args.smooth_ms} trial_types={trial_types} "
        f"validate_only={args.validate_only} replot_only={args.replot_only} force={args.force}"
    )
    run_decode_curated(
        args.condition,
        go_seqs=go_seqs,
        trial_types=trial_types,
        target_name=args.target,
        decode_name=decode_name,
        data_root=data_root,
        figures_root=args.figures_root,
        force=args.force,
        validate_only=args.validate_only,
        replot_only=args.replot_only,
        nshuffles=args.nshuffles,
        cross_validations=args.cv,
        max_sessions=args.max_sessions,
        max_bins=args.max_bins,
        smooth_ms=args.smooth_ms,
    )


if __name__ == "__main__":
    main()
