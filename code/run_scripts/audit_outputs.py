#!/usr/bin/env python3
"""Re-check pipeline/comparison manifests under an output tree."""

from __future__ import annotations

import _bootstrap  # noqa: F401

import argparse
import json
from pathlib import Path

from load_data.io import ARRAY_NAMES
from compare_conditions.compare import comparison_code_fingerprint
from run_pipeline.output_contracts import audit_manifest


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_root", type=Path)
    return parser.parse_args(argv)


def _audit_comparison_manifest(path: Path) -> tuple[list[str], list[str]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    root = path.parent
    file_tag = payload["file_tag"]
    required = [
        "comparison_manifest.json",
        "paired_channel_session.csv",
        "per_channel_comparison.csv",
        "summary.txt",
        "si_delta_heatmap.pdf",
        "waveform_r_heatmap.pdf",
        "delta_si_vs_mean_si_scatter.pdf",
        "median_delta_si_by_array.pdf",
        f"si_{file_tag}_scatter.pdf",
        f"si_{file_tag}_channel_median_scatter.pdf",
        f"combined/arrays_{file_tag}_combined.pdf",
    ]
    required.extend(
        f"combined/{file_tag}_{array_name}_combined.pdf"
        for array_name in ARRAY_NAMES
    )
    required.append(f"combined/arrays_{file_tag}_combined_pref.pdf")
    required.extend(
        f"combined/{file_tag}_{array_name}_combined_pref.pdf"
        for array_name in ARRAY_NAMES
    )
    missing = [name for name in required if not (root / name).is_file()]
    if payload.get("status") != "complete":
        missing.append("manifest status=complete")
    if payload.get("code_fingerprint_sha256") != comparison_code_fingerprint():
        missing.append("current comparison code fingerprint")
    for session_id in payload.get("used_session_ids", ()):
        stem = session_id.split(".")[0]
        missing.extend(
            f"session/{stem}_{array_name}_{file_tag}.pdf"
            for array_name in ARRAY_NAMES
            if not (root / "session" / f"{stem}_{array_name}_{file_tag}.pdf").is_file()
        )
    forbidden = [
        path.relative_to(root).as_posix()
        for path in sorted(root.rglob("*.png"))
    ]
    if payload["comparison_axis"] == "social_context":
        forbidden.extend(
            path.relative_to(root).as_posix()
            for pattern in ("*AgoB*", "*BgoA*")
            for path in sorted(root.rglob(pattern))
            if path.is_file() and path.name != "comparison_manifest.json"
        )
    return sorted(set(missing)), sorted(set(forbidden))


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    root = args.output_root.resolve()
    pipeline_manifests = sorted(root.rglob("pipeline_manifest.json"))
    comparison_manifests = sorted(root.rglob("comparison_manifest.json"))
    if not pipeline_manifests and not comparison_manifests:
        raise SystemExit(f"No pipeline/comparison manifests found under {root}")

    failures: list[str] = []
    for path in pipeline_manifests:
        audit = audit_manifest(path)
        if not audit.ok:
            failures.append(
                f"{path}: missing={list(audit.missing)}, forbidden={list(audit.forbidden)}"
            )
    for path in comparison_manifests:
        missing, forbidden = _audit_comparison_manifest(path)
        if missing or forbidden:
            failures.append(f"{path}: missing={missing}, forbidden={forbidden}")

    if failures:
        raise SystemExit("Output audit failed:\n" + "\n".join(failures))
    print(
        f"Output audit OK: {len(pipeline_manifests)} pipeline manifest(s), "
        f"{len(comparison_manifests)} comparison manifest(s)"
    )


if __name__ == "__main__":
    main()
