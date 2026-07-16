"""Shared --also-original flag for pipeline runners."""

from __future__ import annotations

import argparse

from run_pipeline.config import set_processing_modes


def add_processing_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--also-original",
        action="store_true",
        help="Also emit raw (non-z-scored) outputs under original/ (default: z-scored only)",
    )


def apply_processing_args(args: argparse.Namespace) -> None:
    set_processing_modes(also_original=bool(getattr(args, "also_original", False)))
