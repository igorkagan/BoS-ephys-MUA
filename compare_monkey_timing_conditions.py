#!/usr/bin/env python3
"""Compare monkey tuning between first-vs-second timing conditions.

Conditions:
- AgoB (A first, B second)
- BgoA (B first, A second)

Perspective:
- Curius -> A-side choices (Al/Ar)
- Elmo -> B-side choices (Bl/Br)
"""

from __future__ import annotations

import argparse
import csv
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
from scipy.stats import binomtest, wilcoxon

from bos_mua.evoked import TASK_EVOKED_ALPHA, TASK_EVOKED_WINDOW_MS
from bos_mua.features import ChannelSummary, extract_session_summaries
from bos_mua.io import ARRAY_NAMES, channel_to_array, nominal_channel_list, window_indices
from bos_mua.preprocess import (
    dual_nhp_choice_config,
    recording_monkey_from_condition_label,
    trial_filters_for_go_seq,
)
from bos_mua.session_lists import (
    DUAL_NHP_LIST_NAME,
    SessionListConfig,
    is_confederate_list,
    load_dual_nhp_configs,
    load_session_list,
)
from bos_mua.stability import assess_channel_stability, channel_task_evoked_all_sessions
from bos_mua.viz_timing import (
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

R_STABLE_THRESH = 0.5
ICC_STABLE_THRESH = 0.4
SIGN_CONCORDANCE_THRESH = 0.7
TUNED_SI_ABS_MIN = 0.10

DEEP_DIVE_POOL_RULE = (
    "all sessions matched; stable+task-evoked+median|SI|>="
    f"{TUNED_SI_ABS_MIN} in AgoB or BgoA; similar10=min |median dSI|; different10=max"
)


@dataclass
class MatchedRow:
    session_id: str
    channel: int
    array_name: str
    si_agob: float
    si_bgoa: float
    delta_si: float
    waveform_r: float
    sign_agob: int
    sign_bgoa: int
    sign_flip: bool
    task_evoked_agob: bool
    task_evoked_bgoa: bool


def parse_bool(text: str) -> bool:
    val = text.strip().lower()
    if val in {"1", "true", "t", "yes", "y"}:
        return True
    if val in {"0", "false", "f", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {text!r}")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare AgoB vs BgoA tuning for one monkey perspective.",
    )
    parser.add_argument("--monkey", required=True, choices=["Elmo", "Curius"])
    parser.add_argument("--session-lists", default="session_lists.m")
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
    return parser.parse_args(argv)



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


def remove_stale_outputs(out_dir: Path) -> None:
    stale_names = [
        "per_channel_delta_si.csv",
        "best10_delta_channels.csv",
        "delta_si_distribution.pdf",
        "si_agob_vs_bgoa_scatter.png",
        "delta_si_histogram.png",
        "per_channel_median_delta_si_map.png",
        "top_positive_delta_channels.png",
        "top_negative_delta_channels.png",
    ]
    for name in stale_names:
        (out_dir / name).unlink(missing_ok=True)
    for pattern in ("best10_delta_*.pdf", "*.png"):
        for path in out_dir.glob(pattern):
            path.unlink(missing_ok=True)


def extract_condition_summaries(
    session_ids: list[str],
    data_root: Path,
    trial_filters: dict[str, list[str]],
    choice_field: str,
    left_choice: list[str],
    right_choice: list[str],
    *,
    alignment_event: str,
    pre_post_tag: str,
    analysis_window: tuple[float, float],
    smooth_ms: float,
    zscore: bool,
) -> list[ChannelSummary]:
    all_summaries: list[ChannelSummary] = []
    for sid in session_ids:
        session_dir = data_root / sid
        try:
            summaries = extract_session_summaries(
                session_dir,
                sid,
                alignment_event,
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
    agob: list[ChannelSummary],
    bgoa: list[ChannelSummary],
    *,
    task_evoked_only: bool,
) -> list[MatchedRow]:
    agob_map = summaries_by_key(agob)
    bgoa_map = summaries_by_key(bgoa)
    keys = sorted(set(agob_map).intersection(bgoa_map))
    rows: list[MatchedRow] = []

    for key in keys:
        a = agob_map[key]
        b = bgoa_map[key]
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
                si_agob=float(a.si),
                si_bgoa=float(b.si),
                delta_si=d,
                waveform_r=waveform_r(a, b),
                sign_agob=sign_nonzero(float(a.si)),
                sign_bgoa=sign_nonzero(float(b.si)),
                sign_flip=sign_nonzero(float(a.si)) != sign_nonzero(float(b.si)),
                task_evoked_agob=bool(a.task_evoked),
                task_evoked_bgoa=bool(b.task_evoked),
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
                "si_agob",
                "si_bgoa",
                "delta_si",
                "waveform_r",
                "sign_agob",
                "sign_bgoa",
                "sign_flip",
                "task_evoked_agob",
                "task_evoked_bgoa",
            ],
        )
        writer.writeheader()
        for r in rows:
            writer.writerow(
                {
                    "session_id": r.session_id,
                    "channel": r.channel,
                    "array_name": r.array_name,
                    "si_agob": r.si_agob,
                    "si_bgoa": r.si_bgoa,
                    "delta_si": r.delta_si,
                    "waveform_r": r.waveform_r,
                    "sign_agob": r.sign_agob,
                    "sign_bgoa": r.sign_bgoa,
                    "sign_flip": int(r.sign_flip),
                    "task_evoked_agob": int(r.task_evoked_agob),
                    "task_evoked_bgoa": int(r.task_evoked_bgoa),
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
        si_agob_vals = np.asarray([r.si_agob for r in ch_rows], dtype=float)
        si_bgoa_vals = np.asarray([r.si_bgoa for r in ch_rows], dtype=float)
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
                "si_agob_median": float(np.median(si_agob_vals)),
                "si_bgoa_median": float(np.median(si_bgoa_vals)),
                "si_agob_median_abs": float(np.median(np.abs(si_agob_vals))),
                "si_bgoa_median_abs": float(np.median(np.abs(si_bgoa_vals))),
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
        "si_agob_median",
        "si_bgoa_median",
        "si_agob_median_abs",
        "si_bgoa_median_abs",
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


def _qualifies_timing_condition(
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


def build_timing_deep_dive_pool(
    per_channel: list[dict],
    agob_nest: dict[str, dict[int, ChannelSummary]],
    bgoa_nest: dict[str, dict[int, ChannelSummary]],
    session_ids: list[str],
) -> list[dict]:
    n_sessions = len(session_ids)
    pool: list[dict] = []
    for row in per_channel:
        if int(row["n_matched_sessions"]) != n_sessions:
            continue
        ch = int(row["channel"])
        array_name = str(row["array_name"])
        qual_agob = _qualifies_timing_condition(agob_nest, session_ids, ch, array_name, "AgoB")
        qual_bgoa = _qualifies_timing_condition(bgoa_nest, session_ids, ch, array_name, "BgoA")
        if qual_agob is None and qual_bgoa is None:
            continue
        conditions = []
        if qual_agob is not None:
            conditions.append("AgoB")
        if qual_bgoa is not None:
            conditions.append("BgoA")
        pool.append(
            {
                **row,
                "qualifying_conditions": "+".join(conditions),
                "qual_agob": qual_agob,
                "qual_bgoa": qual_bgoa,
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
        "si_agob_median_abs",
        "si_bgoa_median_abs",
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
                    "si_agob_median_abs": row["si_agob_median_abs"],
                    "si_bgoa_median_abs": row["si_bgoa_median_abs"],
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
    agob_nest: dict[str, dict[int, ChannelSummary]],
    bgoa_nest: dict[str, dict[int, ChannelSummary]],
    session_ids: list[str],
    win_idx: np.ndarray,
    out_dir: Path,
) -> None:
    for rank, row in enumerate(ranked, start=1):
        ch = int(row["channel"])
        title = (
            f"{monkey} | {list_label} {rank}/{len(ranked)} | ch{ch:03d} ({row['array_name']}) | "
            f"median ΔSI={row['delta_si_median']:.3f} | "
            f"qual={row['qualifying_conditions']}"
        )
        plot_timing_deep_dive_channel(
            agob_nest,
            bgoa_nest,
            session_ids,
            ch,
            win_idx,
            title,
            out_dir / f"{list_label}_delta_rank{rank:02d}_ch{ch:03d}.pdf",
        )


def validate_outputs(
    out_dir: Path,
    used_session_ids: list[str],
    similar10: list[dict],
    different10: list[dict],
) -> None:
    session_pdfs = list((out_dir / "session").glob("*.pdf"))
    if not session_pdfs:
        raise RuntimeError("No session PDFs found")
    expected = len(used_session_ids) * len(ARRAY_NAMES)
    if len(session_pdfs) != expected:
        raise RuntimeError(
            f"Expected {expected} session PDFs "
            f"({len(used_session_ids)} sessions × {len(ARRAY_NAMES)} arrays), "
            f"found {len(session_pdfs)}"
        )
    pngs = list(out_dir.rglob("*.png"))
    if pngs:
        raise RuntimeError(f"Found {len(pngs)} PNG files; expected PDF-only outputs")
    required = [
        "si_delta_heatmap.pdf",
        "waveform_r_heatmap.pdf",
        "delta_si_vs_mean_si_scatter.pdf",
        "si_agob_vs_bgoa_scatter.pdf",
        "si_agob_vs_bgoa_channel_median_scatter.pdf",
        "median_delta_si_by_array.pdf",
        "summary.txt",
        "paired_channel_session.csv",
        "per_channel_timing_consistency.csv",
    ]
    for name in required:
        if not (out_dir / name).exists():
            raise RuntimeError(f"Missing expected output: {name}")
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


def timing_suptitle(
    monkey: str,
    session_id: str,
    array_name: str,
    *,
    zscore_mua: bool,
    smooth_ms: float,
    n_agob_l: int,
    n_agob_r: int,
    n_bgoa_l: int,
    n_bgoa_r: int,
) -> str:
    proc = "z-scored" if zscore_mua else "raw"
    smooth = f"smooth {smooth_ms} ms" if smooth_ms > 0 else "unsmoothed"
    te_lo, te_hi = TASK_EVOKED_WINDOW_MS
    return (
        f"{monkey} | {session_id} | {array_name} | AgoB vs BgoA\n"
        f"Dyadic RA1-4 | {proc} | {smooth} | "
        f"task-evoked {te_lo}:{te_hi} α={TASK_EVOKED_ALPHA} | "
        f"trials AgoB L={n_agob_l} R={n_agob_r} | BgoA L={n_bgoa_l} R={n_bgoa_r}"
    )


def run_level1_session_pdfs(
    session_ids: list[str],
    agob_nest: dict[str, dict[int, ChannelSummary]],
    bgoa_nest: dict[str, dict[int, ChannelSummary]],
    out_dir: Path,
    win_idx: np.ndarray,
    *,
    monkey: str,
    zscore_mua: bool,
    smooth_ms: float,
) -> None:
    session_dir = out_dir / "session"
    ensure_dir(session_dir)
    for sid in session_ids:
        agob_by_ch = agob_nest.get(sid, {})
        bgoa_by_ch = bgoa_nest.get(sid, {})
        if not agob_by_ch and not bgoa_by_ch:
            continue
        ref = next(iter(agob_by_ch.values()), None) or next(iter(bgoa_by_ch.values()))
        t_ms = ref.t_ms
        stem = sid.split(".")[0]
        for array_idx, array_name in enumerate(ARRAY_NAMES):
            ref_agob = next(iter(agob_by_ch.values()), None)
            ref_bgoa = next(iter(bgoa_by_ch.values()), None)
            n_al = ref_agob.n_left if ref_agob else 0
            n_ar = ref_agob.n_right if ref_agob else 0
            n_bl = ref_bgoa.n_left if ref_bgoa else 0
            n_br = ref_bgoa.n_right if ref_bgoa else 0
            suptitle = timing_suptitle(
                monkey,
                sid,
                array_name,
                zscore_mua=zscore_mua,
                smooth_ms=smooth_ms,
                n_agob_l=n_al,
                n_agob_r=n_ar,
                n_bgoa_l=n_bl,
                n_bgoa_r=n_br,
            )
            out_path = session_dir / f"{stem}_{array_name}_AgoB_vs_BgoA.pdf"
            plot_session_array_timing_comparison(
                array_idx,
                agob_by_ch,
                bgoa_by_ch,
                t_ms,
                win_idx,
                suptitle,
                out_path,
                zscore_mua=zscore_mua,
            )


def run_level2_outputs(
    rows: list[MatchedRow],
    session_ids: list[str],
    per_channel: list[dict],
    agob_nest: dict[str, dict[int, ChannelSummary]],
    bgoa_nest: dict[str, dict[int, ChannelSummary]],
    out_dir: Path,
    win_idx: np.ndarray,
    monkey: str,
) -> tuple[list[dict], list[dict], int]:
    channels = nominal_channel_list()
    delta_mat = build_session_channel_matrix(rows, session_ids, channels, lambda r: r.delta_si)
    wf_mat = build_session_channel_matrix(rows, session_ids, channels, lambda r: r.waveform_r)

    plot_delta_si_heatmap(
        delta_mat,
        session_ids,
        channels,
        f"{monkey} | ΔSI (BgoA − AgoB) by session and channel",
        out_dir / "si_delta_heatmap.pdf",
    )
    plot_waveform_r_heatmap(
        wf_mat,
        session_ids,
        channels,
        f"{monkey} | waveform r (AgoB vs BgoA L−R diff)",
        out_dir / "waveform_r_heatmap.pdf",
    )

    pool = build_timing_deep_dive_pool(per_channel, agob_nest, bgoa_nest, session_ids)
    print(f"Deep-dive pool (stable+task+tuned, all sessions): {len(pool)} channels")
    similar10, different10 = rank_similar_different(pool, n=DEEP_DIVE_N)

    write_deep_dive_rank_csv(similar10, out_dir / "similar10_delta_channels.csv", "similar10")
    write_deep_dive_rank_csv(different10, out_dir / "different10_delta_channels.csv", "different10")

    _plot_ranked_deep_dives(
        similar10,
        list_label="similar10",
        monkey=monkey,
        agob_nest=agob_nest,
        bgoa_nest=bgoa_nest,
        session_ids=session_ids,
        win_idx=win_idx,
        out_dir=out_dir,
    )
    _plot_ranked_deep_dives(
        different10,
        list_label="different10",
        monkey=monkey,
        agob_nest=agob_nest,
        bgoa_nest=bgoa_nest,
        session_ids=session_ids,
        win_idx=win_idx,
        out_dir=out_dir,
    )
    return similar10, different10, len(pool)


def channel_task_either_from_rows(rows: list[MatchedRow], per_channel: list[dict]) -> np.ndarray:
    task_by_ch: dict[int, bool] = {}
    for r in rows:
        if r.task_evoked_agob or r.task_evoked_bgoa:
            task_by_ch[r.channel] = True
    return np.asarray([task_by_ch.get(int(row["channel"]), False) for row in per_channel], dtype=bool)


def run_level3_outputs(
    rows: list[MatchedRow],
    per_channel: list[dict],
    out_dir: Path,
    monkey: str,
) -> None:
    deltas = np.asarray([r.delta_si for r in rows], dtype=float)
    si_agob = np.asarray([r.si_agob for r in rows], dtype=float)
    si_bgoa = np.asarray([r.si_bgoa for r in rows], dtype=float)
    mean_si = (si_agob + si_bgoa) / 2.0
    task_either = np.asarray(
        [r.task_evoked_agob or r.task_evoked_bgoa for r in rows],
        dtype=bool,
    )

    ch_median_agob = np.asarray([r["si_agob_median"] for r in per_channel], dtype=float)
    ch_median_bgoa = np.asarray([r["si_bgoa_median"] for r in per_channel], dtype=float)
    ch_task_either = channel_task_either_from_rows(rows, per_channel)

    plot_delta_si_vs_mean_si_scatter(
        mean_si,
        deltas,
        task_either,
        out_dir / "delta_si_vs_mean_si_scatter.pdf",
        suptitle=f"{monkey} | ΔSI vs mean SI (n={deltas.size} pairs)",
    )
    plot_si_scatter(
        si_agob,
        si_bgoa,
        task_either,
        out_dir / "si_agob_vs_bgoa_scatter.pdf",
        suptitle=f"{monkey} | SI comparison (session×channel pairs, n={si_agob.size})",
    )
    plot_si_channel_median_scatter(
        ch_median_agob,
        ch_median_bgoa,
        ch_task_either,
        out_dir / "si_agob_vs_bgoa_channel_median_scatter.pdf",
        suptitle=f"{monkey} | SI comparison (channel medians, n={ch_median_agob.size})",
    )
    plot_median_delta_si_by_array(
        per_channel,
        out_dir / "median_delta_si_by_array.pdf",
        suptitle=f"{monkey} | Median ΔSI by array (across sessions)",
    )


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


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    cfg, session_ids, condition_key = load_timing_run(
        args.session_lists, args.list_name, args.monkey,
    )
    if not session_ids:
        raise RuntimeError(
            f"No sessions found for {args.monkey} in list {args.list_name!r}."
        )

    choice = dual_nhp_choice_config(args.monkey)
    output_root = Path(args.output_root) if args.output_root else cfg.output_folder
    out_dir = output_root / f"{args.monkey}_first_second_comparison"
    ensure_dir(out_dir)
    remove_stale_outputs(out_dir)

    analysis_window = (float(args.analysis_window[0]), float(args.analysis_window[1]))
    print(f"Monkey: {args.monkey}")
    print(f"Choice: field={choice.field}, left={choice.left}, right={choice.right}")
    print(f"Output: {out_dir}")
    print(f"Zscore: {args.zscore}, smooth_ms: {args.smooth_ms}, window: {analysis_window}")
    print(f"Task evoked only: {args.task_evoked_only}")

    filters_agob = trial_filters_for_go_seq(condition_key, "AgoB")
    filters_bgoa = trial_filters_for_go_seq(condition_key, "BgoA")

    print("Extracting AgoB summaries...")
    agob = extract_condition_summaries(
        session_ids,
        cfg.root_folder,
        filters_agob,
        choice.field,
        choice.left,
        choice.right,
        alignment_event=args.alignment_event,
        pre_post_tag=args.pre_post_tag,
        analysis_window=analysis_window,
        smooth_ms=float(args.smooth_ms),
        zscore=bool(args.zscore),
    )
    print(f"AgoB summaries: {len(agob)}")

    print("Extracting BgoA summaries...")
    bgoa = extract_condition_summaries(
        session_ids,
        cfg.root_folder,
        filters_bgoa,
        choice.field,
        choice.left,
        choice.right,
        alignment_event=args.alignment_event,
        pre_post_tag=args.pre_post_tag,
        analysis_window=analysis_window,
        smooth_ms=float(args.smooth_ms),
        zscore=bool(args.zscore),
    )
    print(f"BgoA summaries: {len(bgoa)}")

    rows = build_matched_rows(agob, bgoa, task_evoked_only=bool(args.task_evoked_only))
    print(f"Matched session-channel pairs: {len(rows)}")

    ref_summary = agob[0] if agob else bgoa[0]
    win_idx = window_indices(ref_summary.t_ms, analysis_window)

    agob_nest = nest_summaries(agob)
    bgoa_nest = nest_summaries(bgoa)
    used_session_ids = [
        sid for sid in session_ids if sid in agob_nest or sid in bgoa_nest
    ]
    if not used_session_ids:
        raise RuntimeError("No sessions with usable AgoB or BgoA summaries.")
    if len(used_session_ids) < len(session_ids):
        skipped = set(session_ids) - set(used_session_ids)
        print(f"Using {len(used_session_ids)}/{len(session_ids)} sessions "
              f"(skipped: {', '.join(sorted(s.split('.')[0] for s in skipped))})")

    paired_csv = out_dir / "paired_channel_session.csv"
    write_paired_rows_csv(rows, paired_csv)

    per_ch = per_channel_stats(rows, int(args.min_matched_sessions))
    print(f"Channels with >= {args.min_matched_sessions} matched sessions: {len(per_ch)}")
    per_channel_csv = out_dir / "per_channel_timing_consistency.csv"
    write_per_channel_csv(per_ch, per_channel_csv)

    print("Level 1: session × array PDFs...")
    run_level1_session_pdfs(
        session_ids,
        agob_nest,
        bgoa_nest,
        out_dir,
        win_idx,
        monkey=args.monkey,
        zscore_mua=bool(args.zscore),
        smooth_ms=float(args.smooth_ms),
    )

    print("Level 2: heatmaps + similar10/different10 deep dives...")
    similar10, different10, pool_size = run_level2_outputs(
        rows,
        session_ids,
        per_ch,
        agob_nest,
        bgoa_nest,
        out_dir,
        win_idx,
        args.monkey,
    )

    print("Level 3: global PDFs...")
    run_level3_outputs(rows, per_ch, out_dir, args.monkey)

    gstats = global_stats(
        rows,
        per_ch,
        deep_dive_pool_size=pool_size,
        similar10_count=len(similar10),
        different10_count=len(different10),
    )
    summary_txt = out_dir / "summary.txt"
    write_summary_txt(gstats, summary_txt)

    validate_outputs(out_dir, used_session_ids, similar10, different10)

    print("Done.")
    print(f"Saved: {paired_csv}")
    print(f"Saved: {per_channel_csv}")
    print(f"Saved: {summary_txt}")


if __name__ == "__main__":
    main()
