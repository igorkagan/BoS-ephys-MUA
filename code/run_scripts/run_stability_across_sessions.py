#!/usr/bin/env python3
"""Re-run stability_across_sessions step only.

Usage:
    python -u code/run_scripts/run_stability_across_sessions.py Curius_BLOCKED
    python -u code/run_scripts/run_stability_across_sessions.py DUAL_NHP --monkey Curius
"""

from __future__ import annotations

import _bootstrap  # noqa: F401

from run_scripts._partial import main_for_step

if __name__ == "__main__":
    main_for_step("stability_across_sessions")
