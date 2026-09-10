#!/usr/bin/env python3
"""Run session decoding on a confederate or DUAL_NHP list from session_lists.m.

Flat export: root_folder/{session_id}/
Outputs:
  CONF:     {root}/{list_name}/{Monkey}_{AgoB|BgoA}/{Dyadic|SoloA}/decoding/
  DUAL_NHP: {root}/DUAL_NHP/{Monkey}_{AgoB|BgoA}/{Dyadic|SoloA|SoloB}/decoding/

Usage:
    python -u code/run_scripts/run_decode_session_list.py Elmo_BLOCKED_CONF --mode dyadic_all
    python -u code/run_scripts/run_decode_session_list.py Elmo_BLOCKED_CONF --trial-type SoloA --mode actor
    python -u code/run_scripts/run_decode_session_list.py DUAL_NHP --mode dyadic_all --force
    python -u code/run_scripts/run_decode_session_list.py DUAL_NHP --mode actor --force
"""

from __future__ import annotations

import _bootstrap  # noqa: F401
import argparse
from collections.abc import Iterator
from pathlib import Path

from analyze_decoding.grid_run import (
    run_choice_ab_grid,
    run_dyadic_all_grids,
    run_same_diff_grid,
)
from analyze_decoding.run import run_decode_curated
from analyze_decoding.targets import TARGETS
from load_data.sessions import (
    DUAL_NHP_MONKEYS,
    SessionListConfig,
    is_confederate_list,
    is_dual_nhp_list,
    load_dual_nhp_configs,
    load_session_list,
)
from process_channels.preprocess import (
    DUAL_NHP_GO_SEQS,
    recording_actor_side,
    solo_output_subdir_for_actor_side,
)

MODES = (
    "actor",
    "choice_ab",
    "choice_ab_balanced",
    "same_diff",
    "dyadic_all",
)

GRID_MODES = frozenset({"choice_ab", "choice_ab_balanced", "same_diff", "dyadic_all"})
SOLO_TRIAL_TYPES = frozenset({"SoloA", "SoloB"})


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Session-list decoding (Decodanda) for *_CONF confederate lists "
            "or DUAL_NHP (split by recorded monkey)."
        ),
    )
    p.add_argument(
        "list_name",
        help="Session list name, e.g. Elmo_BLOCKED_CONF or DUAL_NHP",
    )
    p.add_argument(
        "--session-lists",
        default="session_lists.m",
        help="Path to session_lists.m (default: session_lists.m)",
    )
    p.add_argument(
        "--monkey",
        choices=list(DUAL_NHP_MONKEYS),
        help="DUAL_NHP only: Curius or Elmo (default: both)",
    )
    p.add_argument(
        "--go-seq",
        choices=[*DUAL_NHP_GO_SEQS, "all"],
        default="all",
    )
    p.add_argument(
        "--trial-type",
        choices=["Dyadic", "SoloA", "SoloB", "both"],
        default="both",
        help="Grid modes are Dyadic-only. both = Dyadic + that monkey's solo folder.",
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


def iter_decode_jobs(
    list_name: str,
    session_lists_path: Path,
    *,
    monkey: str | None = None,
) -> Iterator[tuple[SessionListConfig, str | None, list[str]]]:
    """Yield (cfg, recording_monkey, session_ids). monkey is None for CONF lists."""
    if is_dual_nhp_list(list_name):
        cfg, split = load_dual_nhp_configs(session_lists_path)
        monkeys = [monkey] if monkey else list(DUAL_NHP_MONKEYS)
        for name in monkeys:
            if name not in DUAL_NHP_MONKEYS:
                raise ValueError(f"Unknown monkey {name!r}")
            sids = split[name]
            if not sids:
                continue
            yield cfg, name, sids
        return
    if monkey:
        raise SystemExit("--monkey is only valid with DUAL_NHP.")
    if not is_confederate_list(list_name):
        raise SystemExit(
            f"{list_name!r} is not a confederate list or DUAL_NHP; "
            "expected e.g. Elmo_BLOCKED_CONF or DUAL_NHP"
        )
    cfg = load_session_list(list_name, session_lists_path)
    yield cfg, None, list(cfg.session_ids)


def actor_trial_types(
    trial_type: str,
    *,
    recording_monkey: str | None,
    session_ids: list[str],
) -> tuple[str, ...]:
    """``both`` → Dyadic + SoloA (CONF / Curius) or SoloB (Elmo DUAL_NHP)."""
    if trial_type != "both":
        return (trial_type,)
    if recording_monkey is None:
        return ("Dyadic", "SoloA")
    actor = recording_actor_side(session_ids[0], recording_monkey)
    return ("Dyadic", solo_output_subdir_for_actor_side(actor))


def _list_decode_kwargs(
    cfg: SessionListConfig,
    session_ids: list[str],
    recording_monkey: str | None,
) -> dict:
    return dict(
        data_root=cfg.root_folder,
        figures_root=cfg.root_folder,
        session_ids=list(session_ids),
        session_parent="",
        recording_monkey=recording_monkey,
    )


def _run_job(
    args: argparse.Namespace,
    *,
    list_name: str,
    go_seqs: tuple[str, ...],
    recording_monkey: str | None,
    session_ids: list[str],
    src: dict,
) -> None:
    monkey_tag = recording_monkey or "list"
    print(
        f"list={list_name} monkey={monkey_tag} n_sessions={len(session_ids)} "
        f"go_seqs={go_seqs} mode={args.mode} layout=flat"
    )

    if args.mode in GRID_MODES:
        if args.trial_type in SOLO_TRIAL_TYPES:
            raise SystemExit(
                f"--mode {args.mode} is Dyadic-only; use --mode actor for {args.trial_type}"
            )
        if args.trial_type == "both":
            print("[info] grid modes run Dyadic only (solo skipped)")
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
            run_dyadic_all_grids(list_name, go_seqs=go_seqs, **kw)
        elif args.mode == "choice_ab":
            for go in go_seqs:
                run_choice_ab_grid(list_name, go, balanced=False, **kw)
        elif args.mode == "choice_ab_balanced":
            for go in go_seqs:
                run_choice_ab_grid(list_name, go, balanced=True, **kw)
        else:
            for go in go_seqs:
                run_same_diff_grid(list_name, go, **kw)
        return

    trial_types = actor_trial_types(
        args.trial_type,
        recording_monkey=recording_monkey,
        session_ids=session_ids,
    )
    decode_name = args.decode_name
    if decode_name is None and args.smooth_ms is not None and args.smooth_ms <= 0:
        decode_name = f"{args.target}_nosmooth"
    print(
        f"target={args.target} decode_name={decode_name or args.target} "
        f"smooth_ms={args.smooth_ms} trial_types={trial_types} "
        f"validate_only={args.validate_only} replot_only={args.replot_only} force={args.force}"
    )
    run_decode_curated(
        list_name,
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


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if args.validate_only and args.replot_only:
        raise SystemExit("Use only one of --validate-only / --replot-only")
    if args.monkey and not is_dual_nhp_list(args.list_name):
        raise SystemExit("--monkey is only valid with DUAL_NHP.")

    session_lists_path = Path(args.session_lists)
    go_seqs = DUAL_NHP_GO_SEQS if args.go_seq == "all" else (args.go_seq,)
    jobs = list(
        iter_decode_jobs(args.list_name, session_lists_path, monkey=args.monkey)
    )
    if not jobs:
        raise SystemExit(f"No decode jobs for {args.list_name!r}")

    print(f"DATA_ROOT: {jobs[0][0].root_folder}")
    for cfg, recording_monkey, session_ids in jobs:
        src = _list_decode_kwargs(cfg, session_ids, recording_monkey)
        _run_job(
            args,
            list_name=cfg.list_name,
            go_seqs=go_seqs,
            recording_monkey=recording_monkey,
            session_ids=session_ids,
            src=src,
        )


if __name__ == "__main__":
    main()
