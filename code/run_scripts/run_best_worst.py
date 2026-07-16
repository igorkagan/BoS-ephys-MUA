#!/usr/bin/env python3
"""Re-run best_worst step only. See _partial.py for usage."""

from __future__ import annotations

import _bootstrap  # noqa: F401

from run_scripts._partial import main_for_step

if __name__ == "__main__":
    main_for_step("best_worst")
