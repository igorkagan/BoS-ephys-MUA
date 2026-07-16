#!/usr/bin/env python3
"""Run the fast, data-independent quality gate."""

from __future__ import annotations

import _bootstrap  # noqa: F401

import compileall
import os
import sys
import unittest
from pathlib import Path


CODE_ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    os.environ.setdefault("MPLBACKEND", "Agg")
    os.environ.pop("BOS_RUN_DATA_TESTS", None)
    if not compileall.compile_dir(CODE_ROOT, quiet=1):
        raise SystemExit("Python compilation failed")
    suite = unittest.defaultTestLoader.discover(
        str(CODE_ROOT / "tests"),
        pattern="test_*.py",
        top_level_dir=str(CODE_ROOT),
    )
    result = unittest.TextTestRunner(verbosity=1).run(suite)
    if not result.wasSuccessful():
        raise SystemExit(1)
    print(f"Fast quality gate passed: {result.testsRun} test(s)")


if __name__ == "__main__":
    main()
