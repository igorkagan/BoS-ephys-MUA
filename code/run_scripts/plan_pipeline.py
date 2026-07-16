#!/usr/bin/env python3
"""Print the exact pipeline/comparison matrix without loading channel data."""

from __future__ import annotations

import _bootstrap  # noqa: F401

import argparse
import json
from pathlib import Path

from load_data.sessions import (
    DUAL_NHP_MONKEYS,
    is_confederate_list,
    is_dual_nhp_list,
    load_session_list,
)
from process_channels.preprocess import (
    DUAL_NHP_GO_SEQS,
    MONKEY_CONDITIONS,
    branch_specs_for_run,
)
from run_pipeline.confederate import iter_confederate_runs
from run_pipeline.contracts import comparison_requests
from run_pipeline.curated import iter_curated_runs
from run_pipeline.dual_nhp import iter_dual_nhp_runs


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("target")
    parser.add_argument("--session-lists", default="session_lists.m")
    parser.add_argument("--monkey", choices=list(DUAL_NHP_MONKEYS))
    parser.add_argument(
        "--go-seq", choices=[*DUAL_NHP_GO_SEQS, "all"], default="all",
    )
    parser.add_argument("--dyadic-only", action="store_true")
    parser.add_argument("--also-original", action="store_true")
    return parser.parse_args(argv)


def _context_payload(ctx, *, include_solo: bool) -> dict:
    branches = branch_specs_for_run(ctx, include_solo=include_solo)
    return {
        "condition_key": ctx.condition_key,
        "condition_label": ctx.condition_label,
        "data_root": str(ctx.data_root.resolve()),
        "filters": ctx.trial_filters,
        "layout": ctx.layout,
        "output_base": str(ctx.output_base.resolve()),
        "recording_monkey": ctx.recording_monkey,
        "session_count": len(ctx.session_ids),
        "session_ids": list(ctx.session_ids),
        "session_parent": ctx.session_parent,
        "source_kind": ctx.source_kind,
        "trial_branches": [
            {
                "filters": branch.trial_filters,
                "name": branch.name,
                "output_subdir": branch.output_subdir or "Dyadic",
            }
            for branch in branches
        ],
    }


def build_plan(args: argparse.Namespace) -> dict:
    include_solo = not args.dyadic_only
    if args.target in MONKEY_CONDITIONS:
        contexts = list(iter_curated_runs(args.target, go_seq=args.go_seq))
    else:
        path = Path(args.session_lists)
        if is_dual_nhp_list(args.target):
            contexts = list(
                iter_dual_nhp_runs(path, monkey=args.monkey, go_seq=args.go_seq)
            )
        elif is_confederate_list(args.target):
            contexts = list(
                iter_confederate_runs(
                    load_session_list(args.target, path),
                    go_seq=args.go_seq,
                )
            )
        else:
            raise ValueError(
                f"Unsupported reduced legacy target {args.target!r}; "
                "only curated, DUAL_NHP, and *_CONF use the full shared engine"
            )
    monkeys = sorted({ctx.recording_monkey for ctx in contexts})
    requests = [
        {
            "monkey": monkey,
            **request.as_dict(),
        }
        for monkey in monkeys
        for request in comparison_requests(
            include_solo=include_solo,
            go_seq=args.go_seq,
        )
    ]
    return {
        "comparisons": requests,
        "include_solo": include_solo,
        "processing_modes": (
            ["original", "zscored"] if args.also_original else ["zscored"]
        ),
        "runs": [
            _context_payload(ctx, include_solo=include_solo)
            for ctx in contexts
        ],
        "target": args.target,
    }


def main(argv: list[str] | None = None) -> None:
    print(json.dumps(build_plan(parse_args(argv)), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
