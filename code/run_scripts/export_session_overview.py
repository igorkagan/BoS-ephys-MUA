#!/usr/bin/env python3
"""Write a per-session inclusion Excel table for confederate lists.

Usage:
    PYTHONPATH=code python -u code/run_scripts/export_session_overview.py
    PYTHONPATH=code python -u code/run_scripts/export_session_overview.py --output audit/conf_session_overview.xlsx
"""

from __future__ import annotations

import _bootstrap  # noqa: F401
import argparse
from pathlib import Path

from load_data.session_overview import (
    CONF_LISTS,
    build_overview_rows,
    write_overview_excel,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Excel overview: one row per CONF session, inclusion and drop reasons.",
    )
    parser.add_argument("--session-lists", default="session_lists.m")
    parser.add_argument(
        "--output",
        default="audit/conf_session_overview.xlsx",
        help="Workbook path (default: audit/conf_session_overview.xlsx)",
    )
    parser.add_argument(
        "--lists",
        default=",".join(CONF_LISTS),
        help="Comma-separated list names",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    list_names = tuple(name.strip() for name in args.lists.split(",") if name.strip())
    sessions = build_overview_rows(
        session_lists_path=args.session_lists,
        list_names=list_names,
    )
    path = write_overview_excel(sessions, Path(args.output))
    print(f"Wrote {path} ({len(sessions)} sessions, {len(sessions.columns)} columns)")


if __name__ == "__main__":
    main()
