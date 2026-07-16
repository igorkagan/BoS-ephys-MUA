#!/usr/bin/env python3
"""Deprecated compatibility wrapper for the comparisons step.

Usage:
    python -u code/run_scripts/run_timing_compare.py Curius_BLOCKED
    python -u code/run_scripts/run_timing_compare.py DUAL_NHP --monkey Curius
"""

from __future__ import annotations

import _bootstrap  # noqa: F401

from run_scripts._partial import main_for_step

if __name__ == "__main__":
    import warnings

    warnings.warn(
        "run_timing_compare.py is deprecated; use run_comparisons.py",
        DeprecationWarning,
        stacklevel=1,
    )
    main_for_step("timing_compare")
