#!/usr/bin/env python3
"""Run the canonical comparisons pipeline step only."""

from __future__ import annotations

import _bootstrap  # noqa: F401

from run_scripts._partial import main_for_step


if __name__ == "__main__":
    main_for_step("comparisons")
