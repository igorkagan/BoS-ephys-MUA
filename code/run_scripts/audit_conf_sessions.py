#!/usr/bin/env python3
"""Audit session lists: rewarded trial counts and label problems.

Usage:
    python -u scripts/audit_conf_sessions.py --list Curius_SHUFFLED_CONF
    python -u scripts/audit_conf_sessions.py --all-conf
    python -u scripts/audit_conf_sessions.py --dual-nhp
"""

from __future__ import annotations

import _bootstrap  # noqa: F401
import argparse
from pathlib import Path

from load_data.audit import (
    audit_all_conf_lists,
    audit_dual_nhp_list,
    audit_session_list,
    format_markdown,
    write_csv,
)
from load_data.sessions import is_confederate_list, list_available_session_lists


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit session lists for trial labels and rewarded counts.",
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--list",
        metavar="NAME",
        help="Audit one confederate list (e.g. Curius_SHUFFLED_CONF)",
    )
    group.add_argument(
        "--all-conf",
        action="store_true",
        help="Audit all confederate lists (4 lists, ~90 sessions)",
    )
    group.add_argument(
        "--dual-nhp",
        action="store_true",
        help="Audit DUAL_NHP list (8 sessions) → audit/DUAL_NHP_audit.csv",
    )
    parser.add_argument(
        "--session-lists",
        default="session_lists.m",
        help="Path to session_lists.m (default: session_lists.m)",
    )
    parser.add_argument(
        "--output-dir",
        default="audit",
        help="Directory for CSV output (default: audit)",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    output_dir = Path(args.output_dir)

    if args.all_conf:
        rows = audit_all_conf_lists(session_lists_path=args.session_lists)
        csv_path = write_csv(rows, output_dir / "all_conf_audit.csv")
        title = "All confederate session lists"
    elif args.dual_nhp:
        rows = audit_dual_nhp_list(session_lists_path=args.session_lists)
        csv_path = write_csv(rows, output_dir / "DUAL_NHP_audit.csv")
        title = "DUAL_NHP"
    else:
        list_name = args.list
        if not is_confederate_list(list_name):
            available = [
                name
                for name in list_available_session_lists(args.session_lists)
                if is_confederate_list(name)
            ]
            raise SystemExit(
                f"{list_name!r} is not a confederate list. "
                f"Available: {', '.join(available)}"
            )
        rows = audit_session_list(list_name, session_lists_path=args.session_lists)
        csv_path = write_csv(rows, output_dir / f"{list_name}_audit.csv")
        title = list_name

    print(format_markdown(rows, title=title))
    print(f"\nWrote {csv_path}")


if __name__ == "__main__":
    main()
