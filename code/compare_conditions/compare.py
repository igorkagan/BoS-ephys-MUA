#!/usr/bin/env python3
"""Generic two-operand neural comparison engine (delta is always B - A)."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Literal

import numpy as np
from scipy.stats import binomtest, wilcoxon

from load_data.cache import disk_cache_path, load_summaries_disk_cache, require_pooled_disk_cache
from run_pipeline.dual_nhp import dual_nhp_run_label
from process_channels.evoked import TASK_EVOKED_ALPHA, TASK_EVOKED_WINDOW_MS
from process_channels.features import ChannelSummary, extract_session_summaries
from load_data.io import (
    ARRAY_NAMES,
    channel_to_array,
    nominal_channel_list,
    window_indices,
)
from process_channels.preprocess import (
    alignment_event_for_recording,
    choice_config_for_actor_side,
    processing_label,
    recording_actor_side,
    recording_monkey_from_condition_label,
    solo_output_subdir_for_actor_side,
    trial_filters_for_go_seq,
    trial_filters_for_solo_go_seq,
)
from load_data.sessions import (
    DUAL_NHP_LIST_NAME,
    SessionListConfig,
    is_confederate_list,
    load_dual_nhp_configs,
    load_session_list,
)
from analyze_stability.metrics import assess_channel_stability, channel_task_evoked_all_sessions
from run_pipeline.config import dual_processing_modes
from compare_conditions.plots_comparison import (
    plot_timing_combined_outputs,
    plot_delta_si_heatmap,
    plot_delta_si_vs_mean_si_scatter,
    plot_median_delta_si_by_array,
    plot_session_array_timing_comparison,
    plot_si_scatter,
    plot_si_channel_median_scatter,
    plot_timing_deep_dive_channel,
    plot_waveform_r_heatmap,
)

DEFAULT_ALIGNMENT_EVENT = "A_InitialFixationReleaseTime_ms"
DEFAULT_PRE_POST_TAG = "pre1000ms.post1000ms"
DEFAULT_ANALYSIS_WINDOW_MS = (-500, 500)
DEFAULT_SMOOTH_MS = 50.0
DEFAULT_MIN_MATCHED_SESSIONS = 3
DEEP_DIVE_N = 10
COMPARISON_CONTRACT_VERSION = 1


def comparison_code_fingerprint() -> str:
    """Hash the comparison engine, plots, and pooled-trial implementation."""
    paths = (
        Path(__file__),
        Path(__file__).with_name("plots_comparison.py"),
        Path(__file__).parents[1] / "analyze_stability" / "combine.py",
    )
    digest = hashlib.sha256()
    for path in paths:
        digest.update(path.name.encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()

R_STABLE_THRESH = 0.5
ICC_STABLE_THRESH = 0.4
SIGN_CONCORDANCE_THRESH = 0.7
TUNED_SI_ABS_MIN = 0.10

DEEP_DIVE_POOL_RULE = (
    "all sessions matched; stable+task-evoked+median|SI|>="
    f"{TUNED_SI_ABS_MIN} in both conditions; similar10=min |median dSI|; different10=max"
)


@dataclass(frozen=True)
class ComparisonSpec:
    comparison_axis: Literal["go_sequence", "social_context"]
    label_a: str
    label_b: str
    fixed_dimensions: tuple[tuple[str, str], ...]
    file_tag: str

    def __post_init__(self) -> None:
        if not self.label_a or not self.label_b or self.label_a == self.label_b:
            raise ValueError("Comparison operands must have distinct non-empty labels")
        expected_fixed = "social_context" if self.comparison_axis == "go_sequence" else "go_sequence"
        fixed = dict(self.fixed_dimensions)
        if set(fixed) != {expected_fixed} or not fixed[expected_fixed]:
            raise ValueError(
                f"{self.comparison_axis} comparison must fix exactly {expected_fixed!r}"
            )
        expected_tag = f"{self.label_a}_vs_{self.label_b}"
        if self.file_tag != expected_tag:
            raise ValueError(f"file_tag must be spec-derived: {expected_tag!r}")

    @property
    def varied_dimension(self) -> str:
        return self.comparison_axis

    @property
    def delta_title(self) -> str:
        return f"ΔSI ({self.label_b} − {self.label_a})"

    @property
    def delta_definition(self) -> str:
        return f"B-A: SI({self.label_b}) - SI({self.label_a})"


@dataclass(frozen=True)
class ComparisonInputs:
    source_kind: str
    layout: str
    condition_key: str
    actor_side: str
    recording_monkey: str
    source_path_a: str
    source_path_b: str
    summary_source_mode_a: Literal["disk_cache", "raw_extraction"]
    summary_source_mode_b: Literal["disk_cache", "raw_extraction"]
    trial_data_root: str
    filters_a: tuple[tuple[str, tuple[str, ...]], ...]
    filters_b: tuple[tuple[str, tuple[str, ...]], ...]
    dimensions_a: tuple[tuple[str, str], ...]
    dimensions_b: tuple[tuple[str, str], ...]
    session_ids: tuple[str, ...]
    alignment_events: tuple[str, ...]
    choice_field: str
    choice_values: tuple[tuple[str, tuple[str, ...]], ...]
    processing_mode: str
    analysis_window_ms: tuple[float, float]
    smooth_ms: float

    def validate(self, spec: ComparisonSpec) -> None:
        dims_a, dims_b = dict(self.dimensions_a), dict(self.dimensions_b)
        filters_a, filters_b = dict(self.filters_a), dict(self.filters_b)
        required = {"go_sequence", "social_context"}
        if set(dims_a) != required or set(dims_b) != required:
            raise ValueError(f"Operand dimensions must be exactly {sorted(required)}")
        differences = {key for key in required if dims_a[key] != dims_b[key]}
        if differences != {spec.varied_dimension}:
            raise ValueError(
                f"Operands must differ only in {spec.varied_dimension!r}; differ in {sorted(differences)}"
            )
        for key, value in spec.fixed_dimensions:
            if dims_a[key] != value or dims_b[key] != value:
                raise ValueError(
                    f"Fixed dimension {key!r} must be {value!r}; "
                    f"got A={dims_a[key]!r}, B={dims_b[key]!r}"
                )
        if dims_a[spec.varied_dimension] != spec.label_a:
            raise ValueError("Operand A label leaks or disagrees with its varied dimension")
        if dims_b[spec.varied_dimension] != spec.label_b:
            raise ValueError("Operand B label leaks or disagrees with its varied dimension")
        for operand, dimensions, filters in (
            ("A", dims_a, filters_a),
            ("B", dims_b, filters_b),
        ):
            go_values = filters.get("go_seq_500_list")
            if go_values != (dimensions["go_sequence"],):
                raise ValueError(
                    f"Operand {operand} go-sequence filter {go_values!r} disagrees "
                    f"with dimension {dimensions['go_sequence']!r}"
                )
            trial_types = filters.get("TrialSubType_list", ())
            social_context = dimensions["social_context"]
            if social_context == "Dyadic":
                if trial_types != ("Dyadic",):
                    raise ValueError(
                        f"Operand {operand} trial-type filter {trial_types!r} is not Dyadic"
                    )
            elif not trial_types or any(
                not trial_type.startswith(social_context) for trial_type in trial_types
            ):
                raise ValueError(
                    f"Operand {operand} trial-type filter {trial_types!r} disagrees "
                    f"with social context {social_context!r}"
                )
        if not self.source_path_a or not self.source_path_b:
            raise ValueError("Both exact operand source paths are required")
        for operand, mode, source_path, dimensions in (
            ("A", self.summary_source_mode_a, self.source_path_a, dims_a),
            ("B", self.summary_source_mode_b, self.source_path_b, dims_b),
        ):
            if mode == "disk_cache":
                parts = Path(source_path).parts
                expected = (
                    dimensions["go_sequence"],
                    dimensions["social_context"],
                )
                if not all(any(value in part for part in parts) for value in expected):
                    raise ValueError(
                        f"Operand {operand} cache path {source_path!r} does not encode "
                        f"go/social dimensions {expected!r}"
                    )
        if not self.session_ids:
            raise ValueError("At least one session ID is required")

    def manifest(self, spec: ComparisonSpec) -> dict:
        self.validate(spec)
        return {
            "actor_side": self.actor_side,
            "alignment_events": list(self.alignment_events),
            "analysis_window_ms": list(self.analysis_window_ms),
            "choice": {
                "field": self.choice_field,
                "values": {k: list(v) for k, v in self.choice_values},
            },
            "comparison_axis": spec.comparison_axis,
            "code_fingerprint_sha256": comparison_code_fingerprint(),
            "contract_version": COMPARISON_CONTRACT_VERSION,
            "condition_key": self.condition_key,
            "delta_definition": spec.delta_definition,
            "file_tag": spec.file_tag,
            "filters": {
                "A": {k: list(v) for k, v in self.filters_a},
                "B": {k: list(v) for k, v in self.filters_b},
            },
            "fixed_dimensions": dict(spec.fixed_dimensions),
            "layout": self.layout,
            "operand_labels": {"A": spec.label_a, "B": spec.label_b},
            "operand_dimensions": {"A": dict(self.dimensions_a), "B": dict(self.dimensions_b)},
            "processing_mode": self.processing_mode,
            "recording_monkey": self.recording_monkey,
            "session_ids": list(self.session_ids),
            "smooth_ms": self.smooth_ms,
            "source_kind": self.source_kind,
            "source_paths": {"A": self.source_path_a, "B": self.source_path_b},
            "summary_source_modes": {
                "A": self.summary_source_mode_a,
                "B": self.summary_source_mode_b,
            },
            "trial_data_root": self.trial_data_root,
            "varied_dimension": spec.varied_dimension,
            "status": "planned",
        }


def _freeze_filters(filters: dict[str, list[str]]) -> tuple[tuple[str, tuple[str, ...]], ...]:
    return tuple(sorted((str(k), tuple(map(str, v))) for k, v in filters.items()))


def write_comparison_manifest(out_dir: Path, spec: ComparisonSpec, inputs: ComparisonInputs) -> Path:
    manifest_path = out_dir / "comparison_manifest.json"
    manifest_path.write_text(
        json.dumps(inputs.manifest(spec), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest_path


def finalize_comparison_manifest(
    out_dir: Path,
    *,
    used_session_ids: list[str],
    matched_pair_count: int,
    per_channel_count: int,
) -> None:
    """Mark a validated comparison complete with the actual analyzed population."""
    manifest_path = out_dir / "comparison_manifest.json"
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    payload.update(
        {
            "matched_session_channel_pairs": matched_pair_count,
            "per_channel_rows": per_channel_count,
            "status": "complete",
            "used_session_ids": list(used_session_ids),
        }
    )
    manifest_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def prepare_comparison_output(
    out_dir: Path,
    spec: ComparisonSpec,
    inputs: ComparisonInputs,
) -> Path:
    """Validate provenance, clean stale artifacts, and write manifest before analysis."""
    inputs.validate(spec)
    ensure_dir(out_dir)
    remove_stale_outputs(out_dir, spec=spec)
    return write_comparison_manifest(out_dir, spec, inputs)


def go_sequence_spec(social_context: str = "Dyadic") -> ComparisonSpec:
    return ComparisonSpec(
        comparison_axis="go_sequence",
        label_a="AgoB",
        label_b="BgoA",
        fixed_dimensions=(("social_context", social_context),),
        file_tag="AgoB_vs_BgoA",
    )


def social_context_spec(solo_subdir: str, go_sequence: str) -> ComparisonSpec:
    return ComparisonSpec(
        comparison_axis="social_context",
        label_a="Dyadic",
        label_b=solo_subdir,
        fixed_dimensions=(("go_sequence", go_sequence),),
        file_tag=f"Dyadic_vs_{solo_subdir}",
    )


# Deprecated Python API aliases.
first_second_spec = go_sequence_spec
dyadic_solo_spec = social_context_spec


def combined_pdf_names(file_tag: str, spec: ComparisonSpec | None = None) -> set[str]:
    """L/R combined PDFs plus pref/unpref twins in the same combined/ folder."""
    from analyze_stability.pref_unpref import (
        comparison_combined_pref_filenames,
        comparison_solo_locked_pref_filenames,
    )

    names = {
        f"{file_tag}_{array_name}_combined.pdf" for array_name in ARRAY_NAMES
    }
    names.add(f"arrays_{file_tag}_combined.pdf")
    names.update(comparison_combined_pref_filenames(file_tag))
    if spec is not None and spec.comparison_axis == "social_context":
        names.update(comparison_solo_locked_pref_filenames(file_tag))
    return names


def si_scatter_pdf(file_tag: str) -> str:
    return f"si_{file_tag}_scatter.pdf"


def si_channel_median_scatter_pdf(file_tag: str) -> str:
    return f"si_{file_tag}_channel_median_scatter.pdf"


@dataclass(frozen=True)
class MatchedRow:
    session_id: str
    channel: int
    array_name: str
    si_a: float
    si_b: float
    delta_si: float
    waveform_r: float
    sign_a: int
    sign_b: int
    sign_flip: bool
    task_evoked_a: bool
    task_evoked_b: bool


def parse_bool(text: str) -> bool:
    val = text.strip().lower()
    if val in {"1", "true", "t", "yes", "y"}:
        return True
    if val in {"0", "false", "f", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {text!r}")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare two neural-analysis operands (delta is always B - A).",
    )
    parser.add_argument("--monkey", required=True, choices=["Elmo", "Curius"])
    parser.add_argument("--session-lists", default="session_lists.m")
    parser.add_argument(
        "--curated-condition",
        default=None,
        help="Curated condition folder (e.g. Curius_BLOCKED); discover sessions from disk",
    )
    parser.add_argument(
        "--list-name",
        default=DUAL_NHP_LIST_NAME,
        help="Session list name (default: DUAL_NHP; confederate e.g. Curius_SHUFFLED_CONF)",
    )
    parser.add_argument("--min-matched-sessions", type=int, default=DEFAULT_MIN_MATCHED_SESSIONS)
    parser.add_argument("--zscore", type=parse_bool, default=True)
    parser.add_argument("--smooth-ms", type=float, default=DEFAULT_SMOOTH_MS)
    parser.add_argument("--analysis-window", type=float, nargs=2, default=DEFAULT_ANALYSIS_WINDOW_MS)
    parser.add_argument("--alignment-event", default=DEFAULT_ALIGNMENT_EVENT)
    parser.add_argument("--pre-post-tag", default=DEFAULT_PRE_POST_TAG)
    parser.add_argument("--task-evoked-only", type=parse_bool, default=False)
    parser.add_argument("--output-root", default=None)
    parser.add_argument(
        "--comparison-axis",
        choices=["go_sequence", "social_context"],
        default=None,
        help="Dimension varied between operands (default: go_sequence)",
    )
    parser.add_argument(
        "--go-seq",
        choices=["AgoB", "BgoA"],
        default=None,
        help="Required when --comparison-axis=social_context",
    )
    parser.add_argument(
        "--social-context",
        choices=["dyadic", "solo"],
        default=None,
        help="Context held fixed for go-sequence comparison (default: dyadic)",
    )
    parser.add_argument(
        "--trial-type",
        choices=["first_second", "dyadic_solo"],
        default=None,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--trial-timing",
        choices=["dyadic", "solo"],
        default=None,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--only-combined",
        action="store_true",
        help=(
            "Replot trial-pooled combined PDFs only "
            "(skip session overlays, heatmaps, deep dives, and global scatters)"
        ),
    )
    args = parser.parse_args(argv)
    legacy_axis = {
        "first_second": "go_sequence",
        "dyadic_solo": "social_context",
    }.get(args.trial_type)
    if legacy_axis:
        warnings.warn(
            "--trial-type is deprecated; use --comparison-axis",
            DeprecationWarning,
            stacklevel=2,
        )
        if args.comparison_axis and args.comparison_axis != legacy_axis:
            parser.error("--trial-type conflicts with --comparison-axis")
        args.comparison_axis = legacy_axis
    if args.trial_timing:
        warnings.warn(
            "--trial-timing is deprecated; use --social-context",
            DeprecationWarning,
            stacklevel=2,
        )
        if args.social_context and args.social_context != args.trial_timing:
            parser.error("--trial-timing conflicts with --social-context")
        args.social_context = args.trial_timing
    args.comparison_axis = args.comparison_axis or "go_sequence"
    args.social_context = args.social_context or "dyadic"
    return args



def summaries_by_key(
    summaries: Iterable[ChannelSummary],
) -> dict[tuple[str, int], ChannelSummary]:
    out: dict[tuple[str, int], ChannelSummary] = {}
    for s in summaries:
        out[(s.session_id, s.channel)] = s
    return out


def nest_summaries(
    summaries: Iterable[ChannelSummary],
) -> dict[str, dict[int, ChannelSummary]]:
    out: dict[str, dict[int, ChannelSummary]] = {}
    for s in summaries:
        out.setdefault(s.session_id, {})[s.channel] = s
    return out


def sign_nonzero(v: float) -> int:
    if v > 0:
        return 1
    if v < 0:
        return -1
    return 0


def waveform_r(a: ChannelSummary, b: ChannelSummary) -> float:
    da, db = a.diff, b.diff
    if da.shape != db.shape:
        return np.nan
    mask = np.isfinite(da) & np.isfinite(db)
    if mask.sum() < 3:
        return np.nan
    return float(np.corrcoef(da[mask], db[mask])[0, 1])


def bh_fdr(pvals: np.ndarray) -> np.ndarray:
    n = pvals.size
    if n == 0:
        return pvals.copy()
    order = np.argsort(pvals)
    ranked = pvals[order]
    adj = np.empty(n, dtype=float)
    prev = 1.0
    for i in range(n - 1, -1, -1):
        rank = i + 1
        val = min(prev, ranked[i] * n / rank)
        adj[i] = val
        prev = val
    out = np.empty(n, dtype=float)
    out[order] = np.clip(adj, 0.0, 1.0)
    return out


def run_wilcoxon(values: np.ndarray) -> float | None:
    if values.size < 1:
        return None
    try:
        _, p = wilcoxon(values, alternative="two-sided", zero_method="wilcox")
    except ValueError:
        return None
    return float(p)


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def unlink_case_variants(path: Path) -> None:
    """Delete any same-directory entry that case-folds to ``path.name``.

    On Windows (case-preserving, case-insensitive), overwriting
    ``si_AgoB_vs_BgoA_*.pdf`` when a lowercase legacy name already exists keeps
    the old directory-entry casing. Force-unlink first so the next write stores
    the intended spelling.
    """
    parent = path.parent
    if not parent.is_dir():
        return
    target = path.name.casefold()
    for existing in parent.iterdir():
        if existing.is_file() and existing.name.casefold() == target:
            existing.unlink(missing_ok=True)


def processing_scoped_output(out_dir: Path, zscore_mua: bool) -> Path:
    """Avoid raw/z-scored comparison overwrites when both modes are requested."""
    if zscore_mua and not dual_processing_modes():
        return out_dir
    return out_dir / ("zscored" if zscore_mua else "original")


def remove_stale_outputs(out_dir: Path, *, spec: ComparisonSpec | None = None) -> None:
    stale_names = [
        "per_channel_delta_si.csv",
        "best10_delta_channels.csv",
        "delta_si_distribution.pdf",
        "si_agob_vs_bgoa_scatter.png",
        "si_agob_vs_bgoa_scatter.pdf",
        "si_agob_vs_bgoa_channel_median_scatter.pdf",
        "per_channel_timing_consistency.csv",
        "delta_si_histogram.png",
        "per_channel_median_delta_si_map.png",
        "top_positive_delta_channels.png",
        "top_negative_delta_channels.png",
    ]
    for name in stale_names:
        unlink_case_variants(out_dir / name)
    if spec is not None:
        # Always clear prior SI scatter spellings for this spec so Windows
        # does not preserve a legacy lowercase directory entry.
        unlink_case_variants(out_dir / si_scatter_pdf(spec.file_tag))
        unlink_case_variants(out_dir / si_channel_median_scatter_pdf(spec.file_tag))
    for pattern in ("best10_delta_*.pdf", "*.png"):
        for path in out_dir.glob(pattern):
            path.unlink(missing_ok=True)
    if spec is not None and spec.comparison_axis == "social_context":
        for pattern in ("*AgoB*", "*BgoA*"):
            for path in out_dir.rglob(pattern):
                if path.is_file():
                    path.unlink(missing_ok=True)


def load_condition_summaries(
    output_base: Path,
    condition_label: str,
    session_ids: list[str],
    data_root: Path,
    trial_filters: dict[str, list[str]],
    choice_field: str,
    left_choice: list[str],
    right_choice: list[str],
    *,
    alignment_event: str = DEFAULT_ALIGNMENT_EVENT,
    recording_monkey: str = "",
    pre_post_tag: str,
    analysis_window: tuple[float, float],
    smooth_ms: float,
    zscore: bool,
    condition_key: str | None = None,
    layout: str = "flat",
) -> list[ChannelSummary]:
    cached = load_summaries_disk_cache(output_base, condition_label, zscore)
    if cached is not None:
        print(f"Loaded {len(cached)} summaries from disk cache ({condition_label})")
        return cached
    print(f"Disk cache miss for {condition_label}; extracting...")
    return extract_condition_summaries(
        session_ids,
        data_root,
        trial_filters,
        choice_field,
        left_choice,
        right_choice,
        alignment_event=alignment_event,
        recording_monkey=recording_monkey,
        pre_post_tag=pre_post_tag,
        analysis_window=analysis_window,
        smooth_ms=smooth_ms,
        zscore=zscore,
        condition_key=condition_key,
        layout=layout,
    )


def extract_condition_summaries(
    session_ids: list[str],
    data_root: Path,
    trial_filters: dict[str, list[str]],
    choice_field: str,
    left_choice: list[str],
    right_choice: list[str],
    *,
    alignment_event: str = DEFAULT_ALIGNMENT_EVENT,
    recording_monkey: str = "",
    pre_post_tag: str,
    analysis_window: tuple[float, float],
    smooth_ms: float,
    zscore: bool,
    condition_key: str | None = None,
    layout: str = "flat",
) -> list[ChannelSummary]:
    all_summaries: list[ChannelSummary] = []
    for sid in session_ids:
        if layout == "curated" and condition_key:
            session_dir = data_root / condition_key / sid
        else:
            session_dir = data_root / sid
        event = (
            alignment_event_for_recording(sid, recording_monkey)
            if recording_monkey
            else alignment_event
        )
        try:
            summaries = extract_session_summaries(
                session_dir,
                sid,
                event,
                pre_post_tag,
                trial_filters,
                choice_field,
                left_choice,
                right_choice,
                analysis_window,
                smooth_ms,
                zscore_mua=zscore,
            )
        except Exception as exc:
            warnings.warn(f"Skipping {sid}: {exc}")
            continue
        all_summaries.extend(summaries)
    return all_summaries


def build_matched_rows(
    summaries_a: list[ChannelSummary],
    summaries_b: list[ChannelSummary],
    *,
    task_evoked_only: bool,
) -> list[MatchedRow]:
    map_a = summaries_by_key(summaries_a)
    map_b = summaries_by_key(summaries_b)
    keys = sorted(set(map_a).intersection(map_b))
    rows: list[MatchedRow] = []

    for key in keys:
        a = map_a[key]
        b = map_b[key]
        if not np.isfinite(a.si) or not np.isfinite(b.si):
            continue
        if task_evoked_only and not (a.task_evoked and b.task_evoked):
            continue
        d = float(b.si - a.si)
        rows.append(
            MatchedRow(
                session_id=a.session_id,
                channel=a.channel,
                array_name=a.array_name,
                si_a=float(a.si),
                si_b=float(b.si),
                delta_si=d,
                waveform_r=waveform_r(a, b),
                sign_a=sign_nonzero(float(a.si)),
                sign_b=sign_nonzero(float(b.si)),
                sign_flip=sign_nonzero(float(a.si)) != sign_nonzero(float(b.si)),
                task_evoked_a=bool(a.task_evoked),
                task_evoked_b=bool(b.task_evoked),
            )
        )
    return rows


def build_session_channel_matrix(
    rows: list[MatchedRow],
    session_ids: list[str],
    channels: list[int],
    value_fn,
) -> np.ndarray:
    sid_to_idx = {s: i for i, s in enumerate(session_ids)}
    ch_to_idx = {c: i for i, c in enumerate(channels)}
    mat = np.full((len(session_ids), len(channels)), np.nan, dtype=float)
    for r in rows:
        si = sid_to_idx.get(r.session_id)
        ci = ch_to_idx.get(r.channel)
        if si is None or ci is None:
            continue
        val = value_fn(r)
        if np.isfinite(val):
            mat[si, ci] = val
    return mat


def write_paired_rows_csv(rows: list[MatchedRow], out_path: Path) -> None:
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "session_id",
                "channel",
                "array_name",
                "si_a",
                "si_b",
                "delta_si",
                "waveform_r",
                "sign_a",
                "sign_b",
                "sign_flip",
                "task_evoked_a",
                "task_evoked_b",
            ],
        )
        writer.writeheader()
        for r in rows:
            writer.writerow(
                {
                    "session_id": r.session_id,
                    "channel": r.channel,
                    "array_name": r.array_name,
                    "si_a": r.si_a,
                    "si_b": r.si_b,
                    "delta_si": r.delta_si,
                    "waveform_r": r.waveform_r,
                    "sign_a": r.sign_a,
                    "sign_b": r.sign_b,
                    "sign_flip": int(r.sign_flip),
                    "task_evoked_a": int(r.task_evoked_a),
                    "task_evoked_b": int(r.task_evoked_b),
                }
            )


def per_channel_stats(rows: list[MatchedRow], min_matched_sessions: int) -> list[dict]:
    by_ch: dict[int, list[MatchedRow]] = {}
    for r in rows:
        by_ch.setdefault(r.channel, []).append(r)

    out: list[dict] = []
    for ch, ch_rows in sorted(by_ch.items()):
        deltas = np.asarray([r.delta_si for r in ch_rows], dtype=float)
        wf = np.asarray([r.waveform_r for r in ch_rows], dtype=float)
        if deltas.size < min_matched_sessions:
            continue
        p = run_wilcoxon(deltas)
        n_pos = int(np.sum(deltas > 0))
        n_neg = int(np.sum(deltas < 0))
        n_zero = int(np.sum(deltas == 0))
        n_flip = int(np.sum([r.sign_flip for r in ch_rows]))
        si_a_vals = np.asarray([r.si_a for r in ch_rows], dtype=float)
        si_b_vals = np.asarray([r.si_b for r in ch_rows], dtype=float)
        array_name, _ = channel_to_array(ch)
        array_index = (ch - 1) // 32
        out.append(
            {
                "channel": ch,
                "array_name": array_name,
                "array_index": array_index,
                "n_matched_sessions": int(deltas.size),
                "delta_si_median": float(np.median(deltas)),
                "delta_si_mean": float(np.mean(deltas)),
                "delta_si_std": float(np.std(deltas, ddof=0)),
                "delta_si_median_abs": float(abs(np.median(deltas))),
                "si_a_median": float(np.median(si_a_vals)),
                "si_b_median": float(np.median(si_b_vals)),
                "si_a_median_abs": float(np.median(np.abs(si_a_vals))),
                "si_b_median_abs": float(np.median(np.abs(si_b_vals))),
                "waveform_r_median": float(np.nanmedian(wf)) if np.any(np.isfinite(wf)) else None,
                "n_pos": n_pos,
                "n_neg": n_neg,
                "n_zero": n_zero,
                "n_sign_flip": n_flip,
                "sign_flip_rate": float(n_flip / deltas.size),
                "wilcoxon_p": p,
            }
        )

    finite_idx = [i for i, row in enumerate(out) if row["wilcoxon_p"] is not None]
    if finite_idx:
        pvals = np.asarray([out[i]["wilcoxon_p"] for i in finite_idx], dtype=float)
        adj = bh_fdr(pvals)
        for i, q in zip(finite_idx, adj):
            out[i]["wilcoxon_fdr_bh_q"] = float(q)
    for row in out:
        row.setdefault("wilcoxon_fdr_bh_q", None)
    return out


def write_per_channel_csv(rows: list[dict], out_path: Path) -> None:
    fieldnames = [
        "channel",
        "array_name",
        "array_index",
        "n_matched_sessions",
        "delta_si_median",
        "delta_si_mean",
        "delta_si_std",
        "delta_si_median_abs",
        "si_a_median",
        "si_b_median",
        "si_a_median_abs",
        "si_b_median_abs",
        "waveform_r_median",
        "n_pos",
        "n_neg",
        "n_zero",
        "n_sign_flip",
        "sign_flip_rate",
        "wilcoxon_p",
        "wilcoxon_fdr_bh_q",
    ]
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def global_stats(
    rows: list[MatchedRow],
    per_channel: list[dict],
    *,
    deep_dive_pool_size: int = 0,
    similar10_count: int = 0,
    different10_count: int = 0,
) -> dict[str, float | int | None | str]:
    deltas = np.asarray([r.delta_si for r in rows], dtype=float)
    if deltas.size == 0:
        return {
            "n_pairs": 0,
            "median_delta_si": None,
            "mean_delta_si": None,
            "iqr_delta_si": None,
            "wilcoxon_p": None,
            "sign_test_p": None,
            "n_delta_pos": 0,
            "n_delta_neg": 0,
            "n_delta_zero": 0,
            "sign_flip_fraction": None,
            "n_channels_fdr_significant": 0,
            "deep_dive_pool_rule": DEEP_DIVE_POOL_RULE,
            "deep_dive_pool_size": deep_dive_pool_size,
            "similar10_count": similar10_count,
            "different10_count": different10_count,
        }
    p_w = run_wilcoxon(deltas)
    n_pos = int(np.sum(deltas > 0))
    n_neg = int(np.sum(deltas < 0))
    n_zero = int(np.sum(deltas == 0))
    n_nz = n_pos + n_neg
    p_sign = None
    if n_nz > 0:
        p_sign = float(binomtest(n_pos, n_nz, p=0.5, alternative="two-sided").pvalue)
    sign_flip_fraction = float(np.mean([r.sign_flip for r in rows]))
    iqr = float(np.percentile(deltas, 75) - np.percentile(deltas, 25))
    n_fdr = sum(
        1 for row in per_channel
        if row.get("wilcoxon_fdr_bh_q") is not None and row["wilcoxon_fdr_bh_q"] < 0.05
    )
    return {
        "n_pairs": int(deltas.size),
        "median_delta_si": float(np.median(deltas)),
        "mean_delta_si": float(np.mean(deltas)),
        "iqr_delta_si": iqr,
        "wilcoxon_p": p_w,
        "sign_test_p": p_sign,
        "n_delta_pos": n_pos,
        "n_delta_neg": n_neg,
        "n_delta_zero": n_zero,
        "sign_flip_fraction": sign_flip_fraction,
        "n_channels_fdr_significant": n_fdr,
        "deep_dive_pool_rule": DEEP_DIVE_POOL_RULE,
        "deep_dive_pool_size": deep_dive_pool_size,
        "similar10_count": similar10_count,
        "different10_count": different10_count,
    }


def write_summary_txt(stats: dict[str, float | int | None | str], out_path: Path) -> None:
    with out_path.open("w", encoding="utf-8") as f:
        for k, v in stats.items():
            f.write(f"{k}: {v}\n")


def _stack_channel_data(
    nest: dict[str, dict[int, ChannelSummary]],
    session_ids: list[str],
    channel: int,
) -> tuple[np.ndarray, np.ndarray] | None:
    traces: list[np.ndarray] = []
    si_vals: list[float] = []
    for sid in session_ids:
        summary = nest.get(sid, {}).get(channel)
        if summary is None:
            return None
        traces.append(summary.diff)
        si_vals.append(float(summary.si))
    return np.stack(traces, axis=0), np.asarray(si_vals, dtype=float)


def _qualifies_comparison_operand(
    nest: dict[str, dict[int, ChannelSummary]],
    session_ids: list[str],
    channel: int,
    array_name: str,
    condition_name: str,
) -> dict | None:
    stacked = _stack_channel_data(nest, session_ids, channel)
    if stacked is None:
        return None
    traces, si_vals = stacked
    task_ok, n_evoked, _ = channel_task_evoked_all_sessions(nest, session_ids, channel)
    if not task_ok:
        return None
    si_median_abs = float(np.nanmedian(np.abs(si_vals)))
    if not np.isfinite(si_median_abs) or si_median_abs < TUNED_SI_ABS_MIN:
        return None
    stab = assess_channel_stability(
        channel,
        array_name,
        traces,
        si_vals,
        R_STABLE_THRESH,
        ICC_STABLE_THRESH,
        SIGN_CONCORDANCE_THRESH,
        task_evoked=task_ok,
        n_sessions_task_evoked=n_evoked,
    )
    if not stab.stable:
        return None
    return {
        "condition": condition_name,
        "median_pairwise_r": stab.median_pairwise_r,
        "icc": stab.icc,
        "sign_concordance": stab.sign_concordance,
        "si_median_abs": si_median_abs,
    }


def build_comparison_deep_dive_pool(
    per_channel: list[dict],
    nest_a: dict[str, dict[int, ChannelSummary]],
    nest_b: dict[str, dict[int, ChannelSummary]],
    session_ids: list[str],
    *,
    label_a: str,
    label_b: str,
) -> list[dict]:
    n_sessions = len(session_ids)
    pool: list[dict] = []
    for row in per_channel:
        if int(row["n_matched_sessions"]) != n_sessions:
            continue
        ch = int(row["channel"])
        array_name = str(row["array_name"])
        qual_a = _qualifies_comparison_operand(nest_a, session_ids, ch, array_name, label_a)
        qual_b = _qualifies_comparison_operand(nest_b, session_ids, ch, array_name, label_b)
        if qual_a is None and qual_b is None:
            continue
        conditions = []
        if qual_a is not None:
            conditions.append(label_a)
        if qual_b is not None:
            conditions.append(label_b)
        pool.append(
            {
                **row,
                "qualifying_conditions": "+".join(conditions),
                "qual_a": qual_a,
                "qual_b": qual_b,
            }
        )
    return pool


def rank_similar_different(pool: list[dict], n: int = DEEP_DIVE_N) -> tuple[list[dict], list[dict]]:
    by_abs = sorted(pool, key=lambda r: (r["delta_si_median_abs"], -r.get("waveform_r_median") or -np.inf))
    similar = by_abs[: min(n, len(by_abs))]
    different = sorted(
        pool,
        key=lambda r: (-r["delta_si_median_abs"], r.get("sign_flip_rate") or np.inf),
    )[: min(n, len(pool))]
    return similar, different


def write_deep_dive_rank_csv(rows: list[dict], out_path: Path, list_name: str) -> None:
    fieldnames = [
        "rank",
        "list",
        "channel",
        "array_name",
        "delta_si_median",
        "delta_si_median_abs",
        "qualifying_conditions",
        "si_a_median_abs",
        "si_b_median_abs",
        "waveform_r_median",
        "sign_flip_rate",
        "n_matched_sessions",
    ]
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for rank, row in enumerate(rows, start=1):
            writer.writerow(
                {
                    "rank": rank,
                    "list": list_name,
                    "channel": row["channel"],
                    "array_name": row["array_name"],
                    "delta_si_median": row["delta_si_median"],
                    "delta_si_median_abs": row["delta_si_median_abs"],
                    "qualifying_conditions": row["qualifying_conditions"],
                    "si_a_median_abs": row["si_a_median_abs"],
                    "si_b_median_abs": row["si_b_median_abs"],
                    "waveform_r_median": row.get("waveform_r_median"),
                    "sign_flip_rate": row["sign_flip_rate"],
                    "n_matched_sessions": row["n_matched_sessions"],
                }
            )


def _plot_ranked_deep_dives(
    ranked: list[dict],
    *,
    list_label: str,
    monkey: str,
    nest_a: dict[str, dict[int, ChannelSummary]],
    nest_b: dict[str, dict[int, ChannelSummary]],
    session_ids: list[str],
    win_idx: np.ndarray,
    out_dir: Path,
    label_a: str,
    label_b: str,
) -> None:
    for rank, row in enumerate(ranked, start=1):
        ch = int(row["channel"])
        title = (
            f"{monkey} | {list_label} {rank}/{len(ranked)} | ch{ch:03d} ({row['array_name']}) | "
            f"median ΔSI={row['delta_si_median']:.3f} | "
            f"qual={row['qualifying_conditions']}"
        )
        plot_timing_deep_dive_channel(
            nest_a,
            nest_b,
            session_ids,
            ch,
            win_idx,
            title,
            out_dir / f"{list_label}_delta_rank{rank:02d}_ch{ch:03d}.pdf",
            label_a=label_a,
            label_b=label_b,
        )


# Deprecated Python API alias; generic code should use build_comparison_deep_dive_pool.
build_timing_deep_dive_pool = build_comparison_deep_dive_pool


def validate_outputs(
    out_dir: Path,
    used_session_ids: list[str],
    similar10: list[dict],
    different10: list[dict],
    *,
    spec: ComparisonSpec,
) -> None:
    session_pdfs = {path.name for path in (out_dir / "session").glob("*.pdf")}
    if not session_pdfs:
        raise RuntimeError("No session PDFs found")
    expected_session_pdfs = {
        f"{sid.split('.')[0]}_{array_name}_{spec.file_tag}.pdf"
        for sid in used_session_ids
        for array_name in ARRAY_NAMES
    }
    if session_pdfs != expected_session_pdfs:
        missing = sorted(expected_session_pdfs - session_pdfs)
        extra = sorted(session_pdfs - expected_session_pdfs)
        raise RuntimeError(
            f"Session PDF set mismatch; missing={missing}, extra={extra}"
        )
    pngs = list(out_dir.rglob("*.png"))
    if pngs:
        raise RuntimeError(f"Found {len(pngs)} PNG files; expected PDF-only outputs")
    required = [
        f"combined/arrays_{spec.file_tag}_combined.pdf",
        "si_delta_heatmap.pdf",
        "waveform_r_heatmap.pdf",
        "delta_si_vs_mean_si_scatter.pdf",
        si_scatter_pdf(spec.file_tag),
        si_channel_median_scatter_pdf(spec.file_tag),
        "median_delta_si_by_array.pdf",
        "summary.txt",
        "paired_channel_session.csv",
        "per_channel_comparison.csv",
        "comparison_manifest.json",
    ]
    for name in required:
        if not (out_dir / name).exists():
            raise RuntimeError(f"Missing expected output: {name}")
    expected_combined = combined_pdf_names(spec.file_tag, spec)
    actual_combined = {path.name for path in (out_dir / "combined").glob("*.pdf")}
    if actual_combined != expected_combined:
        raise RuntimeError(
            "Combined PDF set mismatch; "
            f"missing={sorted(expected_combined - actual_combined)}, "
            f"extra={sorted(actual_combined - expected_combined)}"
        )
    if similar10:
        first = similar10[0]
        ch = int(first["channel"])
        expected = out_dir / f"similar10_delta_rank01_ch{ch:03d}.pdf"
        if not expected.exists():
            raise RuntimeError(f"similar10 rank-1 PDF missing: {expected.name}")
    if different10:
        first = different10[0]
        ch = int(first["channel"])
        expected = out_dir / f"different10_delta_rank01_ch{ch:03d}.pdf"
        if not expected.exists():
            raise RuntimeError(f"different10 rank-1 PDF missing: {expected.name}")
    forbidden = {
        "per_channel_timing_consistency.csv",
    }
    # Legacy lowercase SI scatter names are only forbidden for social_context.
    # For go_sequence they case-fold to the canonical required outputs
    # (``si_AgoB_vs_BgoA_*.pdf``); on Windows a leftover lowercase directory
    # entry must be repaired by unlink+rewrite, not treated as a hard fail here.
    if spec.comparison_axis == "social_context":
        forbidden.update(
            {
                "si_agob_vs_bgoa_scatter.pdf",
                "si_agob_vs_bgoa_channel_median_scatter.pdf",
            }
        )
        forbidden.update(
            path.name
            for path in out_dir.rglob("*AgoB*")
            if path.is_file()
        )
        forbidden.update(
            path.name
            for path in out_dir.rglob("*BgoA*")
            if path.is_file()
        )
    # Compare directory-entry strings, not Path.exists/rglob(pattern): Windows
    # path matching is case-insensitive, while the canonical go-sequence tag
    # intentionally contains ``AgoB_vs_BgoA``.
    actual_names = {
        path.name
        for path in out_dir.rglob("*")
        if path.is_file()
    }
    present = sorted(forbidden & actual_names)
    if present:
        raise RuntimeError(f"Forbidden legacy/mislabeled outputs found: {', '.join(present)}")
    # Soft check: go_sequence SI scatter files should use canonical casing.
    if spec.comparison_axis == "go_sequence":
        for expected in (
            si_scatter_pdf(spec.file_tag),
            si_channel_median_scatter_pdf(spec.file_tag),
        ):
            matches = [
                name
                for name in actual_names
                if name.casefold() == expected.casefold()
            ]
            if matches and expected not in matches:
                raise RuntimeError(
                    f"SI scatter filename casing mismatch: found {matches!r}, "
                    f"expected {expected!r}"
                )


def comparison_filter_summary(
    filters_a: dict[str, list[str]],
    filters_b: dict[str, list[str]],
    label_a: str,
    label_b: str,
    *,
    comparison_axis: str | None = None,
) -> str:
    """Compact filter line: shared keys once; omit the varied comparison axis.

    Varied keys are already in the title as ``label_a vs label_b``:
    - social_context → drop ``TrialSubType*``
    - go_sequence → drop ``go_seq_500*``
    """
    omit_prefixes: tuple[str, ...] = ()
    if comparison_axis == "social_context":
        omit_prefixes = ("TrialSubType",)
    elif comparison_axis == "go_sequence":
        omit_prefixes = ("go_seq_500",)

    def _omit(key: str) -> bool:
        short = key.replace("_list", "")
        return any(short == prefix or short.startswith(prefix) for prefix in omit_prefixes)

    keys = [
        key
        for key in dict.fromkeys([*filters_a.keys(), *filters_b.keys()])
        if not _omit(key)
    ]
    parts: list[str] = []
    for key in keys:
        short = key.replace("_list", "")
        va = [str(v) for v in filters_a.get(key, [])]
        vb = [str(v) for v in filters_b.get(key, [])]
        if va == vb:
            parts.append(f"{short}={','.join(va)}")
        else:
            parts.append(
                f"{short} {label_a}={','.join(va)} | {label_b}={','.join(vb)}"
            )
    if parts:
        return "; ".join(parts)
    # All keys were the varied axis; fall back to empty rather than junk.
    return ""


def fixed_dimensions_summary(spec: ComparisonSpec) -> str:
    return "; ".join(f"{key}={value}" for key, value in spec.fixed_dimensions)


def alignment_label_for_sessions(
    session_ids: list[str],
    recording_monkey: str,
) -> str:
    events = {
        alignment_event_for_recording(sid, recording_monkey)
        for sid in session_ids
    }
    if len(events) == 1:
        return events.pop()
    return "actor-side fixation release"


def build_comparison_combined_suptitles(
    *,
    monkey: str,
    spec: ComparisonSpec,
    filters_a: dict[str, list[str]],
    filters_b: dict[str, list[str]],
    alignment_label: str,
    analysis_window_ms: tuple[float, float],
    smooth_ms: float,
    zscore_mua: bool,
    n_sessions: int,
    choice_field: str,
    left_choice: list[str],
    right_choice: list[str],
) -> tuple[str, str]:
    """Titles mirroring Dyadic/Solo ``combine`` / ``arrays`` conventions.

    Returns ``(per_array_channel_grid_base, arrays_mean_base)``.
    Per-array figures append `` | A1`` etc. in the plot helper.
    """
    from analyze_stability.arrays import ARRAY_MEAN_AGGREGATION_LABEL

    proc = processing_label(smooth_ms, zscore_mua=zscore_mua)
    fixed = fixed_dimensions_summary(spec)
    filt = comparison_filter_summary(
        filters_a,
        filters_b,
        spec.label_a,
        spec.label_b,
        comparison_axis=spec.comparison_axis,
    )
    win_lo, win_hi = analysis_window_ms
    meta_parts = [fixed]
    if filt:
        meta_parts.append(filt)
    meta_parts.append(
        f"L/R {choice_field} {'/'.join(left_choice)}/{'/'.join(right_choice)}"
    )
    meta_parts.append(proc)
    meta_parts.append(f"window {win_lo:g}:{win_hi:g} ms")
    meta_parts.append(f"n_sessions={n_sessions}")
    meta = " | ".join(meta_parts)
    pair = f"{monkey} | {spec.label_a} vs {spec.label_b}"
    channel_grids = (
        f"{pair} | combined sessions | {alignment_label}\n{meta}"
    )
    arrays_mean = (
        f"{pair} | {ARRAY_MEAN_AGGREGATION_LABEL} | {alignment_label}\n{meta}"
    )
    return channel_grids, arrays_mean


def comparison_suptitle(
    monkey: str,
    session_id: str,
    array_name: str,
    *,
    label_a: str,
    label_b: str,
    zscore_mua: bool,
    smooth_ms: float,
    n_a_l: int,
    n_a_r: int,
    n_b_l: int,
    n_b_r: int,
    fixed_summary: str = "",
    alignment_label: str = "",
) -> str:
    proc = "z-scored" if zscore_mua else "raw"
    smooth = f"smooth {smooth_ms} ms" if smooth_ms > 0 else "unsmoothed"
    te_lo, te_hi = TASK_EVOKED_WINDOW_MS
    head = f"{monkey} | {session_id} | {array_name} | {label_a} vs {label_b}"
    if fixed_summary:
        head = f"{head} | {fixed_summary}"
    if alignment_label:
        head = f"{head} | {alignment_label}"
    return (
        f"{head}\n"
        f"{proc} | {smooth} | "
        f"task-evoked {te_lo}:{te_hi} α={TASK_EVOKED_ALPHA} | "
        f"trials {label_a} L={n_a_l} R={n_a_r} | {label_b} L={n_b_l} R={n_b_r}"
    )


def run_level1_session_pdfs(
    session_ids: list[str],
    nest_a: dict[str, dict[int, ChannelSummary]],
    nest_b: dict[str, dict[int, ChannelSummary]],
    out_dir: Path,
    win_idx: np.ndarray,
    *,
    monkey: str,
    zscore_mua: bool,
    smooth_ms: float,
    spec: ComparisonSpec,
) -> None:
    session_dir = out_dir / "session"
    ensure_dir(session_dir)
    for sid in session_ids:
        by_ch_a = nest_a.get(sid, {})
        by_ch_b = nest_b.get(sid, {})
        if not by_ch_a and not by_ch_b:
            continue
        ref = next(iter(by_ch_a.values()), None) or next(iter(by_ch_b.values()))
        t_ms = ref.t_ms
        stem = sid.split(".")[0]
        for array_idx, array_name in enumerate(ARRAY_NAMES):
            ref_a = next(iter(by_ch_a.values()), None)
            ref_b = next(iter(by_ch_b.values()), None)
            n_al = ref_a.n_left if ref_a else 0
            n_ar = ref_a.n_right if ref_a else 0
            n_bl = ref_b.n_left if ref_b else 0
            n_br = ref_b.n_right if ref_b else 0
            suptitle = comparison_suptitle(
                monkey,
                sid,
                array_name,
                label_a=spec.label_a,
                label_b=spec.label_b,
                zscore_mua=zscore_mua,
                smooth_ms=smooth_ms,
                n_a_l=n_al,
                n_a_r=n_ar,
                n_b_l=n_bl,
                n_b_r=n_br,
                fixed_summary=fixed_dimensions_summary(spec),
                alignment_label=alignment_event_for_recording(sid, monkey),
            )
            out_path = session_dir / f"{stem}_{array_name}_{spec.file_tag}.pdf"
            plot_session_array_timing_comparison(
                array_idx,
                by_ch_a,
                by_ch_b,
                t_ms,
                win_idx,
                suptitle,
                out_path,
                zscore_mua=zscore_mua,
                label_a=spec.label_a,
                label_b=spec.label_b,
            )


def branch_output_dir_from_source_path(source_path: Path | str) -> Path:
    """Figure-folder for a branch from either that folder or ``.cache/summaries_*.npz``."""
    path = Path(source_path)
    if path.parent.name == ".cache":
        return path.parent.parent
    return path


def run_combined_pdfs(
    session_ids: list[str],
    out_dir: Path,
    *,
    spec: ComparisonSpec,
    filters_a: dict,
    filters_b: dict,
    source_path_a: Path,
    source_path_b: Path,
    condition_label_a: str,
    condition_label_b: str,
    pool_kwargs: dict,
    monkey: str,
    zscore_mua: bool,
    smooth_ms: float,
    analysis_window_ms: tuple[float, float],
    choice_field: str,
    left_choice: list[str],
    right_choice: list[str],
) -> None:
    alignment_label = alignment_label_for_sessions(
        session_ids, pool_kwargs["recording_monkey"],
    )
    channel_title, arrays_title = build_comparison_combined_suptitles(
        monkey=monkey,
        spec=spec,
        filters_a=filters_a,
        filters_b=filters_b,
        alignment_label=alignment_label,
        analysis_window_ms=analysis_window_ms,
        smooth_ms=smooth_ms,
        zscore_mua=zscore_mua,
        n_sessions=len(session_ids),
        choice_field=choice_field,
        left_choice=left_choice,
        right_choice=right_choice,
    )
    branch_a = branch_output_dir_from_source_path(source_path_a)
    branch_b = branch_output_dir_from_source_path(source_path_b)
    try:
        pooled_a = require_pooled_disk_cache(branch_a, condition_label_a, zscore_mua)
        pooled_b = require_pooled_disk_cache(branch_b, condition_label_b, zscore_mua)
    except FileNotFoundError as exc:
        warnings.warn(f"Skipping combined L/R PDFs for {out_dir.name}: {exc}")
    else:
        print(f"Combined trial-pooled PDFs -> {out_dir / 'combined'}")
        plot_timing_combined_outputs(
            pooled_a.left_by_ch, pooled_a.right_by_ch,
            pooled_b.left_by_ch, pooled_b.right_by_ch,
            pooled_a.t_ms, pooled_a.win_idx,
            out_dir / "combined",
            file_tag=spec.file_tag,
            label_a=spec.label_a,
            label_b=spec.label_b,
            suptitle_channel_grids=channel_title,
            suptitle_arrays=arrays_title,
            zscore_mua=zscore_mua,
        )
    from analyze_stability.pref_unpref import (
        write_comparison_pref_combined,
        write_paired_pref_session_combined,
    )

    summaries_a = load_summaries_disk_cache(branch_a, condition_label_a, zscore_mua) or []
    summaries_b = load_summaries_disk_cache(branch_b, condition_label_b, zscore_mua) or []
    write_comparison_pref_combined(
        summaries_a,
        summaries_b,
        out_dir / "combined",
        file_tag=spec.file_tag,
        label_a=spec.label_a,
        label_b=spec.label_b,
        suptitle_grids=f"{channel_title} | pref vs unpref",
        suptitle_arrays=f"{arrays_title} | pref vs unpref",
    )
    if spec.comparison_axis == "social_context":
        write_paired_pref_session_combined(
            summaries_a,
            summaries_b,
            out_dir / "combined",
            file_tag=spec.file_tag,
            label_a=spec.label_a,
            label_b=spec.label_b,
            suptitle=f"{channel_title} | Solo-locked pref | session mean then across sessions",
            lock="solo",
        )


def run_level2_outputs(
    rows: list[MatchedRow],
    session_ids: list[str],
    per_channel: list[dict],
    nest_a: dict[str, dict[int, ChannelSummary]],
    nest_b: dict[str, dict[int, ChannelSummary]],
    out_dir: Path,
    win_idx: np.ndarray,
    monkey: str,
    spec: ComparisonSpec,
) -> tuple[list[dict], list[dict], int]:
    channels = nominal_channel_list()
    delta_mat = build_session_channel_matrix(rows, session_ids, channels, lambda r: r.delta_si)
    wf_mat = build_session_channel_matrix(rows, session_ids, channels, lambda r: r.waveform_r)

    plot_delta_si_heatmap(
        delta_mat,
        session_ids,
        channels,
        f"{monkey} | {spec.delta_title} by session and channel",
        out_dir / "si_delta_heatmap.pdf",
        colorbar_label=spec.delta_title,
    )
    plot_waveform_r_heatmap(
        wf_mat,
        session_ids,
        channels,
        f"{monkey} | waveform r ({spec.label_a} vs {spec.label_b} L−R diff)",
        out_dir / "waveform_r_heatmap.pdf",
        colorbar_label=f"Waveform r ({spec.label_a} vs {spec.label_b} L−R diff)",
    )

    pool = build_comparison_deep_dive_pool(
        per_channel, nest_a, nest_b, session_ids,
        label_a=spec.label_a, label_b=spec.label_b,
    )
    print(f"Deep-dive pool (stable+task+tuned, all sessions): {len(pool)} channels")
    similar10, different10 = rank_similar_different(pool, n=DEEP_DIVE_N)

    write_deep_dive_rank_csv(similar10, out_dir / "similar10_delta_channels.csv", "similar10")
    write_deep_dive_rank_csv(different10, out_dir / "different10_delta_channels.csv", "different10")

    _plot_ranked_deep_dives(
        similar10,
        list_label="similar10",
        monkey=monkey,
        nest_a=nest_a,
        nest_b=nest_b,
        session_ids=session_ids,
        win_idx=win_idx,
        out_dir=out_dir,
        label_a=spec.label_a,
        label_b=spec.label_b,
    )
    _plot_ranked_deep_dives(
        different10,
        list_label="different10",
        monkey=monkey,
        nest_a=nest_a,
        nest_b=nest_b,
        session_ids=session_ids,
        win_idx=win_idx,
        out_dir=out_dir,
        label_a=spec.label_a,
        label_b=spec.label_b,
    )
    return similar10, different10, len(pool)


def channel_task_either_from_rows(rows: list[MatchedRow], per_channel: list[dict]) -> np.ndarray:
    task_by_ch: dict[int, bool] = {}
    for r in rows:
        if r.task_evoked_a or r.task_evoked_b:
            task_by_ch[r.channel] = True
    return np.asarray([task_by_ch.get(int(row["channel"]), False) for row in per_channel], dtype=bool)


def run_level3_outputs(
    rows: list[MatchedRow],
    per_channel: list[dict],
    out_dir: Path,
    monkey: str,
    spec: ComparisonSpec,
) -> None:
    deltas = np.asarray([r.delta_si for r in rows], dtype=float)
    si_a = np.asarray([r.si_a for r in rows], dtype=float)
    si_b = np.asarray([r.si_b for r in rows], dtype=float)
    mean_si = (si_a + si_b) / 2.0
    task_either = np.asarray(
        [r.task_evoked_a or r.task_evoked_b for r in rows],
        dtype=bool,
    )

    ch_median_a = np.asarray([r["si_a_median"] for r in per_channel], dtype=float)
    ch_median_b = np.asarray([r["si_b_median"] for r in per_channel], dtype=float)
    ch_task_either = channel_task_either_from_rows(rows, per_channel)

    plot_delta_si_vs_mean_si_scatter(
        mean_si,
        deltas,
        task_either,
        out_dir / "delta_si_vs_mean_si_scatter.pdf",
        suptitle=f"{monkey} | {spec.delta_title} vs mean SI (n={deltas.size} pairs)",
        label_a=spec.label_a,
        label_b=spec.label_b,
    )
    si_scatter_path = out_dir / si_scatter_pdf(spec.file_tag)
    unlink_case_variants(si_scatter_path)
    plot_si_scatter(
        si_a,
        si_b,
        task_either,
        si_scatter_path,
        suptitle=f"{monkey} | SI {spec.label_a} vs {spec.label_b} (n={si_a.size} pairs)",
        label_a=spec.label_a,
        label_b=spec.label_b,
    )
    si_median_path = out_dir / si_channel_median_scatter_pdf(spec.file_tag)
    unlink_case_variants(si_median_path)
    plot_si_channel_median_scatter(
        ch_median_a,
        ch_median_b,
        ch_task_either,
        si_median_path,
        suptitle=f"{monkey} | SI {spec.label_a} vs {spec.label_b} (channel medians, n={ch_median_a.size})",
        label_a=spec.label_a,
        label_b=spec.label_b,
    )
    plot_median_delta_si_by_array(
        per_channel,
        out_dir / "median_delta_si_by_array.pdf",
        suptitle=f"{monkey} | {spec.delta_title} by array (across sessions)",
        delta_label=spec.delta_title,
    )


def load_curated_timing_run(
    curated_condition: str,
    monkey: str,
) -> tuple[SessionListConfig, list[str], str]:
    """Return config for timing comparison on a curated condition."""
    from run_pipeline.curated import (
        curated_data_root,
        curated_output_base,
        discover_curated_sessions,
    )

    expected = recording_monkey_from_condition_label(curated_condition)
    if monkey != expected:
        raise ValueError(
            f"Monkey {monkey!r} does not match curated condition {curated_condition!r} "
            f"(expected {expected!r})"
        )
    data_root = curated_data_root()
    session_ids = discover_curated_sessions(curated_condition)
    cfg = SessionListConfig(
        list_name=curated_condition,
        root_folder=data_root,
        output_folder=curated_output_base(curated_condition),
        session_ids=session_ids,
        condition_key=curated_condition,
    )
    return cfg, session_ids, curated_condition


def load_timing_run(
    session_lists_path: str | Path,
    list_name: str,
    monkey: str,
) -> tuple[SessionListConfig, list[str], str]:
    """Return (config, session_ids, condition_key) for timing comparison."""
    path = Path(session_lists_path)
    if list_name == DUAL_NHP_LIST_NAME:
        cfg, split = load_dual_nhp_configs(path)
        session_ids = split[monkey]
        condition_key = DUAL_NHP_LIST_NAME
    elif is_confederate_list(list_name):
        cfg = load_session_list(list_name, path)
        expected = recording_monkey_from_condition_label(cfg.condition_key)
        if monkey != expected:
            raise ValueError(
                f"Monkey {monkey!r} does not match confederate list {list_name!r} "
                f"(expected {expected!r})"
            )
        session_ids = cfg.session_ids
        condition_key = cfg.condition_key
    else:
        raise ValueError(
            f"Timing comparison not supported for list {list_name!r}; "
            "use DUAL_NHP or a confederate list (Elmo/Curius_BLOCKED/SHUFFLED_*)"
        )
    return cfg, session_ids, condition_key


def run_comparison(
    args: argparse.Namespace,
    *,
    spec: ComparisonSpec,
    cfg: SessionListConfig,
    session_ids: list[str],
    condition_key: str,
    layout: str,
    summaries_a: list[ChannelSummary],
    summaries_b: list[ChannelSummary],
    filters_a: dict,
    filters_b: dict,
    out_dir: Path,
    inputs: ComparisonInputs,
) -> None:
    inputs.validate(spec)
    if not (out_dir / "comparison_manifest.json").exists():
        prepare_comparison_output(out_dir, spec, inputs)

    analysis_window = (float(args.analysis_window[0]), float(args.analysis_window[1]))
    actor_side = recording_actor_side(session_ids[0], args.monkey)
    choice = choice_config_for_actor_side(actor_side)

    print(f"Comparison axis: {spec.comparison_axis}")
    print(f"Monkey: {args.monkey}")
    print(f"Conditions: {spec.label_a} vs {spec.label_b}")
    print(f"Choice: field={choice.field}, left={choice.left}, right={choice.right}")
    print(f"Output: {out_dir}")
    print(f"Zscore: {args.zscore}, smooth_ms: {args.smooth_ms}, window: {analysis_window}")

    pool_kwargs = dict(
        data_root=cfg.root_folder,
        choice_field=choice.field,
        left_choice=choice.left,
        right_choice=choice.right,
        recording_monkey=args.monkey,
        layout=layout,
        condition_key=condition_key,
        pre_post_tag=args.pre_post_tag,
        analysis_window_ms=analysis_window,
        smooth_ms=float(args.smooth_ms),
        zscore_mua=bool(args.zscore),
    )

    if bool(getattr(args, "only_combined", False)):
        print("Combined trial-pooled PDFs only (--only-combined)...")
        run_combined_pdfs(
            session_ids,
            out_dir,
            spec=spec,
            filters_a=filters_a,
            filters_b=filters_b,
            source_path_a=Path(inputs.source_path_a),
            source_path_b=Path(inputs.source_path_b),
            condition_label_a=branch_output_dir_from_source_path(inputs.source_path_a).parent.name,
            condition_label_b=branch_output_dir_from_source_path(inputs.source_path_b).parent.name,
            pool_kwargs=pool_kwargs,
            monkey=args.monkey,
            zscore_mua=bool(args.zscore),
            smooth_ms=float(args.smooth_ms),
            analysis_window_ms=analysis_window,
            choice_field=choice.field,
            left_choice=list(choice.left),
            right_choice=list(choice.right),
        )
        print("Done (combined only).")
        return

    if not summaries_a and not summaries_b:
        print(f"No {spec.label_a} or {spec.label_b} summaries; skipping.")
        return

    rows = build_matched_rows(summaries_a, summaries_b, task_evoked_only=bool(args.task_evoked_only))
    print(f"Matched session-channel pairs: {len(rows)}")

    ref_summary = summaries_a[0] if summaries_a else summaries_b[0]
    win_idx = window_indices(ref_summary.t_ms, analysis_window)

    nest_a = nest_summaries(summaries_a)
    nest_b = nest_summaries(summaries_b)
    used_session_ids = [
        sid for sid in session_ids if sid in nest_a or sid in nest_b
    ]
    if not used_session_ids:
        raise RuntimeError(f"No sessions with usable {spec.label_a} or {spec.label_b} summaries.")
    if len(used_session_ids) < len(session_ids):
        skipped = set(session_ids) - set(used_session_ids)
        print(f"Using {len(used_session_ids)}/{len(session_ids)} sessions "
              f"(skipped: {', '.join(sorted(s.split('.')[0] for s in skipped))})")

    paired_csv = out_dir / "paired_channel_session.csv"
    write_paired_rows_csv(rows, paired_csv)

    per_ch = per_channel_stats(rows, int(args.min_matched_sessions))
    print(f"Channels with >= {args.min_matched_sessions} matched sessions: {len(per_ch)}")
    per_channel_csv = out_dir / "per_channel_comparison.csv"
    write_per_channel_csv(per_ch, per_channel_csv)

    print("Level 1: session × array PDFs...")
    run_level1_session_pdfs(
        session_ids,
        nest_a,
        nest_b,
        out_dir,
        win_idx,
        monkey=args.monkey,
        zscore_mua=bool(args.zscore),
        smooth_ms=float(args.smooth_ms),
        spec=spec,
    )

    print("Combined trial-pooled PDFs...")
    run_combined_pdfs(
        used_session_ids,
        out_dir,
        spec=spec,
        filters_a=filters_a,
        filters_b=filters_b,
        source_path_a=Path(inputs.source_path_a),
        source_path_b=Path(inputs.source_path_b),
        condition_label_a=branch_output_dir_from_source_path(inputs.source_path_a).parent.name,
        condition_label_b=branch_output_dir_from_source_path(inputs.source_path_b).parent.name,
        pool_kwargs=pool_kwargs,
        monkey=args.monkey,
        zscore_mua=bool(args.zscore),
        smooth_ms=float(args.smooth_ms),
        analysis_window_ms=analysis_window,
        choice_field=choice.field,
        left_choice=list(choice.left),
        right_choice=list(choice.right),
    )

    print("Level 2: heatmaps + similar10/different10 deep dives...")
    similar10, different10, pool_size = run_level2_outputs(
        rows,
        session_ids,
        per_ch,
        nest_a,
        nest_b,
        out_dir,
        win_idx,
        args.monkey,
        spec,
    )

    print("Level 3: global PDFs...")
    run_level3_outputs(rows, per_ch, out_dir, args.monkey, spec)

    gstats = global_stats(
        rows,
        per_ch,
        deep_dive_pool_size=pool_size,
        similar10_count=len(similar10),
        different10_count=len(different10),
    )
    summary_txt = out_dir / "summary.txt"
    write_summary_txt(gstats, summary_txt)

    validate_outputs(out_dir, used_session_ids, similar10, different10, spec=spec)
    finalize_comparison_manifest(
        out_dir,
        used_session_ids=used_session_ids,
        matched_pair_count=len(rows),
        per_channel_count=len(per_ch),
    )

    print("Done.")
    print(f"Saved: {paired_csv}")
    print(f"Saved: {per_channel_csv}")
    print(f"Saved: {summary_txt}")


def make_comparison_inputs(
    *,
    args: argparse.Namespace,
    spec: ComparisonSpec,
    session_ids: list[str],
    condition_key: str,
    layout: str,
    actor_side: str,
    choice,
    data_root: Path,
    source_path_a: Path,
    source_path_b: Path,
    filters_a: dict[str, list[str]],
    filters_b: dict[str, list[str]],
    dimensions_a: dict[str, str],
    dimensions_b: dict[str, str],
) -> ComparisonInputs:
    cache_a = disk_cache_path(source_path_a, "", bool(args.zscore))
    cache_b = disk_cache_path(source_path_b, "", bool(args.zscore))
    summary_source_a = cache_a if cache_a.is_file() else data_root
    summary_source_b = cache_b if cache_b.is_file() else data_root
    events = tuple(
        alignment_event_for_recording(sid, args.monkey)
        for sid in session_ids
    )
    inputs = ComparisonInputs(
        source_kind="curated" if args.curated_condition else "session_list",
        layout=layout,
        condition_key=condition_key,
        actor_side=actor_side,
        recording_monkey=args.monkey,
        source_path_a=str(summary_source_a.resolve()),
        source_path_b=str(summary_source_b.resolve()),
        summary_source_mode_a="disk_cache" if cache_a.is_file() else "raw_extraction",
        summary_source_mode_b="disk_cache" if cache_b.is_file() else "raw_extraction",
        trial_data_root=str(data_root.resolve()),
        filters_a=_freeze_filters(filters_a),
        filters_b=_freeze_filters(filters_b),
        dimensions_a=tuple(sorted(dimensions_a.items())),
        dimensions_b=tuple(sorted(dimensions_b.items())),
        session_ids=tuple(session_ids),
        alignment_events=events,
        choice_field=choice.field,
        choice_values=(("left", tuple(choice.left)), ("right", tuple(choice.right))),
        processing_mode="zscored" if args.zscore else "original",
        analysis_window_ms=(float(args.analysis_window[0]), float(args.analysis_window[1])),
        smooth_ms=float(args.smooth_ms),
    )
    inputs.validate(spec)
    return inputs


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    layout = "flat"
    if args.curated_condition:
        cfg, session_ids, condition_key = load_curated_timing_run(
            args.curated_condition, args.monkey,
        )
        layout = "curated"
    else:
        cfg, session_ids, condition_key = load_timing_run(
            args.session_lists, args.list_name, args.monkey,
        )
    if not session_ids:
        label = args.curated_condition or args.list_name
        raise RuntimeError(
            f"No sessions found for {args.monkey} in {label!r}."
        )

    actor_side = recording_actor_side(session_ids[0], args.monkey)
    choice = choice_config_for_actor_side(actor_side)
    output_root = Path(args.output_root) if args.output_root else cfg.output_folder
    solo_subdir = solo_output_subdir_for_actor_side(actor_side)
    analysis_window = (float(args.analysis_window[0]), float(args.analysis_window[1]))

    extract_kwargs = dict(
        session_ids=session_ids,
        data_root=cfg.root_folder,
        choice_field=choice.field,
        left_choice=choice.left,
        right_choice=choice.right,
        alignment_event=args.alignment_event,
        recording_monkey=args.monkey,
        pre_post_tag=args.pre_post_tag,
        analysis_window=analysis_window,
        smooth_ms=float(args.smooth_ms),
        zscore=bool(args.zscore),
        condition_key=condition_key,
        layout=layout,
    )

    if args.comparison_axis == "social_context":
        if args.go_seq is None:
            raise ValueError(
                "--go-seq is required when --comparison-axis=social_context"
            )
        go_seq = args.go_seq
        spec = social_context_spec(solo_subdir, go_seq)
        go_label = dual_nhp_run_label(args.monkey, go_seq)
        out_dir = processing_scoped_output(
            output_root / go_label / f"Dyadic_vs_{solo_subdir}_comparison",
            bool(args.zscore),
        )
        filters_dyadic = trial_filters_for_go_seq(condition_key, go_seq, actor_side)
        filters_solo = trial_filters_for_solo_go_seq(condition_key, go_seq, actor_side)
        branch_base = cfg.output_folder / go_label
        inputs = make_comparison_inputs(
            args=args,
            spec=spec,
            session_ids=session_ids,
            condition_key=condition_key,
            layout=layout,
            actor_side=actor_side,
            choice=choice,
            data_root=cfg.root_folder,
            source_path_a=branch_base / "Dyadic",
            source_path_b=branch_base / solo_subdir,
            filters_a=filters_dyadic,
            filters_b=filters_solo,
            dimensions_a={"go_sequence": go_seq, "social_context": "Dyadic"},
            dimensions_b={"go_sequence": go_seq, "social_context": solo_subdir},
        )
        prepare_comparison_output(out_dir, spec, inputs)
        if args.only_combined:
            summaries_a, summaries_b = [], []
        else:
            print(f"Loading Dyadic summaries from {branch_base / 'Dyadic'}...")
            summaries_a = load_condition_summaries(
                branch_base / "Dyadic", go_label, trial_filters=filters_dyadic, **extract_kwargs,
            )
            print(f"Dyadic summaries: {len(summaries_a)}")
            print(f"Loading {solo_subdir} summaries from {branch_base / solo_subdir}...")
            summaries_b = load_condition_summaries(
                branch_base / solo_subdir, go_label, trial_filters=filters_solo, **extract_kwargs,
            )
            print(f"{solo_subdir} summaries: {len(summaries_b)}")
        run_comparison(
            args,
            spec=spec,
            cfg=cfg,
            session_ids=session_ids,
            condition_key=condition_key,
            layout=layout,
            summaries_a=summaries_a,
            summaries_b=summaries_b,
            filters_a=filters_dyadic,
            filters_b=filters_solo,
            out_dir=out_dir,
            inputs=inputs,
        )
        return

    branch_subdir = solo_subdir if args.social_context == "solo" else "Dyadic"
    spec = go_sequence_spec(branch_subdir)
    if args.social_context == "solo":
        filters_a = trial_filters_for_solo_go_seq(condition_key, "AgoB", actor_side)
        filters_b = trial_filters_for_solo_go_seq(condition_key, "BgoA", actor_side)
    else:
        filters_a = trial_filters_for_go_seq(condition_key, "AgoB", actor_side)
        filters_b = trial_filters_for_go_seq(condition_key, "BgoA", actor_side)
    out_dir = processing_scoped_output(
        output_root / f"{args.monkey}_{branch_subdir}_first_second_comparison",
        bool(args.zscore),
    )
    agob_label = dual_nhp_run_label(args.monkey, "AgoB")
    bgoa_label = dual_nhp_run_label(args.monkey, "BgoA")
    agob_base = cfg.output_folder / agob_label / branch_subdir
    bgoa_base = cfg.output_folder / bgoa_label / branch_subdir
    inputs = make_comparison_inputs(
        args=args,
        spec=spec,
        session_ids=session_ids,
        condition_key=condition_key,
        layout=layout,
        actor_side=actor_side,
        choice=choice,
        data_root=cfg.root_folder,
        source_path_a=agob_base,
        source_path_b=bgoa_base,
        filters_a=filters_a,
        filters_b=filters_b,
        dimensions_a={"go_sequence": "AgoB", "social_context": branch_subdir},
        dimensions_b={"go_sequence": "BgoA", "social_context": branch_subdir},
    )
    prepare_comparison_output(out_dir, spec, inputs)

    if args.only_combined:
        summaries_a, summaries_b = [], []
    else:
        print("Loading AgoB summaries...")
        summaries_a = load_condition_summaries(
            agob_base, agob_label, trial_filters=filters_a, **extract_kwargs,
        )
        print(f"AgoB summaries: {len(summaries_a)}")

        print("Loading BgoA summaries...")
        summaries_b = load_condition_summaries(
            bgoa_base, bgoa_label, trial_filters=filters_b, **extract_kwargs,
        )
        print(f"BgoA summaries: {len(summaries_b)}")

    run_comparison(
        args,
        spec=spec,
        cfg=cfg,
        session_ids=session_ids,
        condition_key=condition_key,
        layout=layout,
        summaries_a=summaries_a,
        summaries_b=summaries_b,
        filters_a=filters_a,
        filters_b=filters_b,
        out_dir=out_dir,
        inputs=inputs,
    )


if __name__ == "__main__":
    main()
