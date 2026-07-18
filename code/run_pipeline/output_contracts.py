"""Deterministic output manifests and post-run validation for pipeline branches."""

from __future__ import annotations

import json
import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from analyze_stability import combine as combine_step
from analyze_stability import stability_across_sessions as stability_step
from load_data.io import ARRAY_NAMES
from process_channels.features import ChannelSummary
from run_pipeline.config import dual_processing_modes
from run_pipeline.context import PipelineContext

PIPELINE_CONTRACT_VERSION = 1


def pipeline_code_fingerprint() -> str:
    digest = hashlib.sha256()
    root = Path(__file__).parent
    for path in (root / "context.py", root / "runner.py", Path(__file__)):
        digest.update(path.name.encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()


class CacheView(Protocol):
    summaries: dict[bool, list[ChannelSummary]]
    trials: dict[bool, dict[str, dict]]
    pooled: object | None


@dataclass(frozen=True)
class OutputAudit:
    manifest_path: Path
    missing: tuple[str, ...]
    forbidden: tuple[str, ...]

    @property
    def ok(self) -> bool:
        return not self.missing and not self.forbidden


def _mode_root(ctx: PipelineContext, zscore_mua: bool) -> Path:
    if not dual_processing_modes():
        return ctx.output_base
    return ctx.output_base / ("zscored" if zscore_mua else "original")


def _summary_sessions(summaries: list[ChannelSummary]) -> set[str]:
    return {summary.session_id for summary in summaries}


def expected_branch_outputs(
    ctx: PipelineContext,
    steps: list[str],
    cache: CacheView,
) -> list[Path]:
    """Return exact required artifacts for the data actually available in the cache."""
    expected: list[Path] = []

    if "session_lr" in steps:
        for zscore_mua, trials_by_session in cache.trials.items():
            root = _mode_root(ctx, zscore_mua)
            for session_id, channels in trials_by_session.items():
                if not channels:
                    continue
                event = ctx.alignment_event(session_id)
                expected.extend(
                    root / f"{session_id}_{event}_{array_name}_LR.pdf"
                    for array_name in ARRAY_NAMES
                )

    if "consistency" in steps:
        for zscore_mua, summaries in cache.summaries.items():
            if not summaries:
                continue
            root = _mode_root(ctx, zscore_mua) / "consistency"
            expected.extend(
                root / name
                for name in (
                    "channel_presence.csv",
                    "channel_stability.csv",
                    "si_heatmap.pdf",
                    "signed_sig_heatmap.pdf",
                    "session_similarity.pdf",
                )
            )
            expected.extend(root / f"si_stability_{name}.pdf" for name in ARRAY_NAMES)
            expected.extend(root / f"delta_consensus_{name}.pdf" for name in ARRAY_NAMES)

    z_summaries = cache.summaries.get(True, [])
    sessions_with_summaries = len(_summary_sessions(z_summaries))
    if (
        "stability_across_sessions" in steps
        and len(ctx.session_ids) >= stability_step.MIN_SESSIONS
        and sessions_with_summaries >= stability_step.MIN_SESSIONS
        and z_summaries
    ):
        root = _mode_root(ctx, True) / "stability_across_sessions"
        expected.extend(
            root / name
            for name in (
                "pairwise_channel_gap.csv",
                "gap_bin_summary.csv",
                "stable_channel_max_gap.csv",
                "recommended_session_clusters.csv",
                "best_combine_subset.csv",
                "channel_session_tradeoff.csv",
                "channel_session_explorer.html",
            )
        )

    if (
        cache.pooled is not None
        and len(ctx.session_ids) >= combine_step.MIN_SESSIONS
        and ("combine" in steps or "array_combined" in steps)
    ):
        root = _mode_root(ctx, True) / "combined"
        alignment = ctx.alignment_event_label(ctx.session_ids)
        if "combine" in steps:
            expected.extend(
                root / f"{ctx.condition_label}_{alignment}_{name}_combined_LR.pdf"
                for name in ARRAY_NAMES
            )
        if "array_combined" in steps:
            expected.append(
                root / f"{ctx.condition_label}_{alignment}_arrays_combined_LR.pdf"
            )

    return sorted(set(expected), key=lambda path: path.as_posix())


def _relative(path: Path, root: Path) -> str:
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return path.as_posix()


def validate_pipeline_outputs(
    ctx: PipelineContext,
    steps: list[str],
    cache: CacheView,
) -> OutputAudit:
    """Write a branch manifest and fail if deterministic required outputs are absent."""
    expected = expected_branch_outputs(ctx, steps, cache)
    missing = tuple(_relative(path, ctx.output_base) for path in expected if not path.is_file())
    forbidden_paths = sorted(ctx.output_base.rglob("*.png")) if ctx.output_base.exists() else []
    forbidden = tuple(_relative(path, ctx.output_base) for path in forbidden_paths)

    manifest = {
        "condition_key": ctx.condition_key,
        "condition_label": ctx.condition_label,
        "code_fingerprint_sha256": pipeline_code_fingerprint(),
        "contract_version": PIPELINE_CONTRACT_VERSION,
        "expected_outputs": [_relative(path, ctx.output_base) for path in expected],
        "filters": {str(k): list(v) for k, v in sorted(ctx.trial_filters.items())},
        "forbidden_outputs": list(forbidden),
        "layout": ctx.layout,
        "missing_outputs": list(missing),
        "output_base": str(ctx.output_base.resolve()),
        "recording_monkey": ctx.recording_monkey,
        "session_ids": list(ctx.session_ids),
        "session_parent": ctx.session_parent,
        "source_kind": ctx.source_kind,
        "status": "ok" if not missing and not forbidden else "failed",
        "steps": list(steps),
    }
    ctx.output_base.mkdir(parents=True, exist_ok=True)
    manifest_path = ctx.output_base / "pipeline_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    audit = OutputAudit(manifest_path, missing, forbidden)
    if not audit.ok:
        details = []
        if missing:
            details.append(f"missing={list(missing)}")
        if forbidden:
            details.append(f"forbidden={list(forbidden)}")
        raise RuntimeError(
            f"Output contract failed for {ctx.condition_label}: " + "; ".join(details)
        )
    return audit


def audit_manifest(manifest_path: Path) -> OutputAudit:
    """Re-check one existing pipeline_manifest.json without rerunning analysis."""
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    root = Path(payload["output_base"])
    expected = tuple(payload.get("expected_outputs", ()))
    missing = tuple(name for name in expected if not (root / name).is_file())
    if payload.get("code_fingerprint_sha256") != pipeline_code_fingerprint():
        missing = (*missing, "current pipeline code fingerprint")
    forbidden = tuple(
        path.relative_to(root).as_posix()
        for path in sorted(root.rglob("*.png"))
    ) if root.exists() else ()
    return OutputAudit(manifest_path, missing, forbidden)
