#!/usr/bin/env python3
"""Run session decoding on a confederate list from session_lists.m.

Flat export: root_folder/{session_id}/
Outputs:     root_folder/{list_name}/{Monkey}_{AgoB|BgoA}/{Dyadic|SoloA}/decoding/

Usage:
    python -u code/run_scripts/run_decode_session_list.py Elmo_BLOCKED_CONF --mode dyadic_all
    python -u code/run_scripts/run_decode_session_list.py Elmo_BLOCKED_CONF --trial-type SoloA --mode actor
"""

from __future__ import annotations

import _bootstrap  # noqa: F401
import argparse
from pathlib import Path

from analyze_decoding.grid_run import (
    run_choice_ab_grid,
    run_dyadic_all_grids,
    run_same_diff_grid,
)
from analyze_decoding.run import run_decode_curated
from analyze_decoding.targets import TARGETS
from load_data.sessions import is_confederate_list, is_dual_nhp_list, load_session_list
from process_channels.preprocess import DUAL_NHP_GO_SEQS

MODES = (
    "actor",
    "choice_ab",
    "choice_ab_balanced",
    "same_diff",
    "dyadic_all",
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Session-list decoding (Decodanda) for *_CONF confederate lists.",
    )
    p.add_argument(
        "list_name",
        help="Session list name, e.g. Elmo_BLOCKED_CONF",
    )
    p.add_argument(
        "--session-lists",
        default="session_lists.m",
        help="Path to session_lists.m (default: session_lists.m)",
    )
    p.add_argument(
        "--go-seq",
        choices=[*DUAL_NHP_GO_SEQS, "all"],
        default="all",
    )
    p.add_argument(
        "--trial-type",
        choices=["Dyadic", "SoloA", "both"],
        default="both",
        help="Grid modes are Dyadic-only.",
    )
    p.add_argument("--mode", choices=MODES, default="actor")
    p.add_argument(
        "--target",
        default="actor_choice",
        choices=sorted(TARGETS),
        help="Decode target for --mode actor",
    )
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


def _list_decode_kwargs(cfg) -> dict:
    return dict(
        data_root=cfg.root_folder,
        figures_root=cfg.root_folder,
        session_ids=list(cfg.session_ids),
        session_parent="",
    )


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if is_dual_nhp_list(args.list_name):
        raise SystemExit(
            "DUAL_NHP decoding is not wired yet (two recorded monkeys). "
            "Use a confederate *_CONF list."
        )
    if not is_confederate_list(args.list_name):
        raise SystemExit(
            f"{args.list_name!r} is not a confederate list; expected e.g. Elmo_BLOCKED_CONF"
        )
    if args.validate_only and args.replot_only:
        raise SystemExit("Use only one of --validate-only / --replot-only")

    session_lists_path = Path(args.session_lists)
    cfg = load_session_list(args.list_name, session_lists_path)
    go_seqs = DUAL_NHP_GO_SEQS if args.go_seq == "all" else (args.go_seq,)
    src = _list_decode_kwargs(cfg)
    print(f"DATA_ROOT: {cfg.root_folder}")
    print(
        f"list={cfg.list_name} n_sessions={len(cfg.session_ids)} "
        f"go_seqs={go_seqs} mode={args.mode} layout=flat"
    )

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
            **src,
            force=args.force,
            max_sessions=args.max_sessions,
            nshuffles=args.nshuffles,
            cross_validations=args.cv,
        )
        if args.mode == "dyadic_all":
            run_dyadic_all_grids(cfg.list_name, go_seqs=go_seqs, **kw)
        elif args.mode == "choice_ab":
            for go in go_seqs:
                run_choice_ab_grid(cfg.list_name, go, balanced=False, **kw)
        elif args.mode == "choice_ab_balanced":
            for go in go_seqs:
                run_choice_ab_grid(cfg.list_name, go, balanced=True, **kw)
        else:
            for go in go_seqs:
                run_same_diff_grid(cfg.list_name, go, **kw)
        return

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
        cfg.list_name,
        go_seqs=go_seqs,
        trial_types=trial_types,
        target_name=args.target,
        decode_name=decode_name,
        force=args.force,
        validate_only=args.validate_only,
        replot_only=args.replot_only,
        nshuffles=args.nshuffles,
        cross_validations=args.cv,
        max_sessions=args.max_sessions,
        max_bins=args.max_bins,
        smooth_ms=args.smooth_ms,
        **src,
    )


if __name__ == "__main__":
    main()
