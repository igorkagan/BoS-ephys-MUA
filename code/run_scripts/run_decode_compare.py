#!/usr/bin/env python3
"""Compare decode combined caches across confederate lists (no re-decode).

Usage:
    python -u code/run_scripts/run_decode_compare.py
    python -u code/run_scripts/run_decode_compare.py --monkeys Curius --go-seq AgoB
    python -u code/run_scripts/run_decode_compare.py --analysis actor_own
"""

from __future__ import annotations

import _bootstrap  # noqa: F401
import argparse
from pathlib import Path

from analyze_decoding.compare_lists import ANALYSES, run_decode_compare
from process_channels.preprocess import DUAL_NHP_GO_SEQS


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="BLOCKED vs SHUFFLED decode comparison from combined caches.",
    )
    p.add_argument(
        "--monkeys",
        default="Elmo,Curius",
        help="Comma-separated monkeys (default: Elmo,Curius)",
    )
    p.add_argument(
        "--go-seq",
        choices=[*DUAL_NHP_GO_SEQS, "all"],
        default=None,
        help="AgoB, BgoA, or all. Default: AgoB for same_diff, all for actor_own.",
    )
    p.add_argument(
        "--analysis",
        choices=list(ANALYSES),
        default="same_diff",
        help="same_diff=BLOCKED vs SHUFFLED; actor_own=Dyadic vs SoloA own action",
    )
    p.add_argument(
        "--lists",
        default="BLOCKED,SHUFFLED",
        help="Comma-separated BLOCKED,SHUFFLED (actor_own only)",
    )
    p.add_argument(
        "--session-lists",
        default="session_lists.m",
        help="Path to session_lists.m (data root is read from it)",
    )
    p.add_argument("--n-perm", type=int, default=5000)
    p.add_argument("--cluster-forming-p", type=float, default=0.05)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    monkeys = tuple(m.strip() for m in args.monkeys.split(",") if m.strip())
    if not monkeys:
        raise SystemExit("empty --monkeys")
    go_seq = args.go_seq
    if go_seq is None:
        go_seq = "all" if args.analysis == "actor_own" else "AgoB"
    list_tags = tuple(t.strip().upper() for t in args.lists.split(",") if t.strip())
    written = run_decode_compare(
        monkeys=monkeys,
        go_seq=go_seq,
        analysis=args.analysis,
        list_tags=list_tags,
        session_lists=Path(args.session_lists),
        n_perm=args.n_perm,
        cluster_forming_p=args.cluster_forming_p,
        seed=args.seed,
    )
    for path in written:
        print(f"[out] {path}")


if __name__ == "__main__":
    main()
