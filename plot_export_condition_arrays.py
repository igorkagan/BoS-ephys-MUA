#!/usr/bin/env python3
"""Array-average L/R plots from per-array combined session data (DUAL_NHP export).

Uses the same pooled per-channel traces as combine_sessions (what goes into
{condition}_{event}_{A1}_combined_LR.pdf, etc.): z-scored, smoothed, all
sessions' trials concatenated per channel.

For each array (A1–A5):
  1. Per channel: mean trace across pooled L (and R) trials  [= one subplot line
     in the existing *_A1_combined_LR.pdf]
  2. Average those channel-mean traces → array L/R line
  3. Shaded band = SE across channels at each time sample

Output:
    {dual_nhp_root}/{Monkey}_{AgoB|BgoA}/figures/zscored/combined/
        {condition}_{event}_arrays_combined_LR.pdf   (5 panels: A1–A5)

Usage:
    python -u plot_export_condition_arrays.py --all
    python -u plot_export_condition_arrays.py Elmo_AgoB
"""

from __future__ import annotations

import argparse
import warnings
from dataclasses import replace
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import combine_sessions as cs
from bos_mua.dual_nhp import (
    ALL_DUAL_NHP_RUNS,
    DualNhpRunKey,
    build_dual_nhp_context,
    dual_nhp_run_label,
)
from bos_mua.io import ARRAY_NAMES, array_nominal_channels, filter_summary
from bos_mua.pipeline_runner import verify_sessions
from bos_mua.preprocess import processing_label, resolve_condition_output_dir
from bos_mua.run_context import apply_pipeline_context
from bos_mua.session_lists import load_dual_nhp_configs
from bos_mua.viz_lr import configure_array_time_axis

DEFAULT_DUAL_NHP_ROOT = Path(
    r"S:\taskcontroller\SCP_DATA\SCP-CTRL-01\MUA_export_per_session\DUAL_NHP"
)

FIG_SIZE_IN = (14.0, 4.5)
DPI = 150


ExportCondition = DualNhpRunKey
EXPORT_CONDITIONS = ALL_DUAL_NHP_RUNS
EXPORT_CONDITION_BY_NAME = {run.label: run for run in EXPORT_CONDITIONS}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    names = ", ".join(EXPORT_CONDITION_BY_NAME)
    parser = argparse.ArgumentParser(
        description="Array-mean combined L/R plots (mean over channels, SE across channels).",
    )
    parser.add_argument("condition", nargs="?", help=f"e.g. Elmo_AgoB ({names})")
    parser.add_argument("--all", action="store_true", help="Run all four conditions")
    parser.add_argument("--dual-nhp-root", type=Path, default=DEFAULT_DUAL_NHP_ROOT)
    parser.add_argument("--session-lists", default="session_lists.m")
    return parser.parse_args(argv)


def resolve_conditions(args: argparse.Namespace) -> list[ExportCondition]:
    if args.all and args.condition:
        raise SystemExit("Pass either a condition name or --all, not both.")
    if args.all:
        return list(EXPORT_CONDITIONS)
    if not args.condition:
        parser = argparse.ArgumentParser(
            description="Array-mean combined L/R plots (mean over channels, SE across channels).",
        )
        parser.print_help()
        raise SystemExit("Provide a condition name or --all.")
    if args.condition not in EXPORT_CONDITION_BY_NAME:
        raise SystemExit(
            f"Unknown condition {args.condition!r}. "
            f"Expected one of: {', '.join(EXPORT_CONDITION_BY_NAME)}"
        )
    return [EXPORT_CONDITION_BY_NAME[args.condition]]


def build_export_context(
    condition: ExportCondition,
    dual_nhp_root: Path,
    session_lists_path: Path,
):
    cfg, split = load_dual_nhp_configs(session_lists_path)
    ctx = build_dual_nhp_context(cfg, condition.monkey, condition.go_seq, split[condition.monkey])
    label = dual_nhp_run_label(condition.monkey, condition.go_seq)
    return replace(ctx, output_base=dual_nhp_root / label)


def channel_combined_mean_trace(trial_parts: list[np.ndarray]) -> np.ndarray | None:
    """Mean L or R trace for one channel (pooled trials across sessions)."""
    if not trial_parts:
        return None
    trials = np.vstack(trial_parts)
    if trials.size == 0:
        return None
    return np.nanmean(trials, axis=0)


def mean_and_se_across_channels(channel_traces: list[np.ndarray]) -> tuple[np.ndarray | None, np.ndarray | None, int]:
    """Mean trace and SE across channel-mean traces (ddof=1)."""
    if not channel_traces:
        return None, None, 0
    stack = np.stack(channel_traces, axis=0)
    mean = np.nanmean(stack, axis=0)
    n_ch = stack.shape[0]
    if n_ch <= 1:
        se = np.zeros_like(mean)
    else:
        se = np.nanstd(stack, axis=0, ddof=1) / np.sqrt(n_ch)
    return mean, se, n_ch


def array_combined_stats(
    array_index: int,
    left_by_ch: dict[int, list[np.ndarray]],
    right_by_ch: dict[int, list[np.ndarray]],
) -> tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None, np.ndarray | None, int, int]:
    left_traces: list[np.ndarray] = []
    right_traces: list[np.ndarray] = []

    for ch_num in array_nominal_channels(array_index):
        left_mean = channel_combined_mean_trace(left_by_ch.get(ch_num, []))
        right_mean = channel_combined_mean_trace(right_by_ch.get(ch_num, []))
        if left_mean is not None:
            left_traces.append(left_mean)
        if right_mean is not None:
            right_traces.append(right_mean)

    left_mean, left_se, n_left_ch = mean_and_se_across_channels(left_traces)
    right_mean, right_se, n_right_ch = mean_and_se_across_channels(right_traces)
    return left_mean, left_se, right_mean, right_se, n_left_ch, n_right_ch


def plot_mean_se(
    ax,
    t_ms: np.ndarray,
    mean: np.ndarray,
    se: np.ndarray,
    color: str,
    label: str,
) -> None:
    ax.plot(t_ms, mean, color=color, linewidth=1.2, label=label, zorder=3)
    ax.fill_between(
        t_ms,
        mean - se,
        mean + se,
        color=color,
        alpha=cs.SD_ALPHA,
        linewidth=0,
        zorder=2,
    )


def plot_array_panel(
    ax,
    t_ms: np.ndarray,
    win_idx: np.ndarray,
    array_name: str,
    left_mean: np.ndarray | None,
    left_se: np.ndarray | None,
    right_mean: np.ndarray | None,
    right_se: np.ndarray | None,
    n_left_ch: int,
    n_right_ch: int,
) -> None:
    if left_mean is None and right_mean is None:
        cs._mark_empty_axis(ax, array_name, "no data")
        return

    if left_mean is not None and left_se is not None:
        plot_mean_se(ax, t_ms, left_mean, left_se, cs.LEFT_COLOR, "Left")
    if right_mean is not None and right_se is not None:
        plot_mean_se(ax, t_ms, right_mean, right_se, cs.RIGHT_COLOR, "Right")

    if win_idx.size:
        ax.axvspan(t_ms[win_idx[0]], t_ms[win_idx[-1]], color="0.85", alpha=0.35, zorder=0)
    ax.axvline(0, color="0.5", linewidth=0.6, linestyle="--")
    ax.set_title(array_name, fontsize=9)
    ax.text(
        0.02,
        0.98,
        f"nCh L={n_left_ch}, R={n_right_ch}\nmean ± SE (across ch)",
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=7,
        bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.8, edgecolor="none"),
    )
    ax.tick_params(labelsize=6)
    ax.set_xlim(t_ms[0], t_ms[-1])


def make_arrays_combined_figure(
    t_ms: np.ndarray,
    win_idx: np.ndarray,
    left_by_ch: dict[int, list[np.ndarray]],
    right_by_ch: dict[int, list[np.ndarray]],
    suptitle: str,
) -> plt.Figure:
    fig, axes = plt.subplots(1, len(ARRAY_NAMES), figsize=FIG_SIZE_IN, sharex=True, sharey=False)

    for array_index, (array_name, ax) in enumerate(zip(ARRAY_NAMES, np.atleast_1d(axes))):
        left_mean, left_se, right_mean, right_se, n_l, n_r = array_combined_stats(
            array_index, left_by_ch, right_by_ch,
        )
        plot_array_panel(ax, t_ms, win_idx, array_name, left_mean, left_se, right_mean, right_se, n_l, n_r)
        if array_index == 0:
            ax.set_ylabel("MUA (z)", fontsize=8)

    configure_array_time_axis(axes, t_ms, time_panel=0)
    fig.suptitle(suptitle, fontsize=10, y=1.02)
    fig.tight_layout()
    return fig


def plot_condition_array_combined(ctx: PipelineContext) -> None:
    apply_pipeline_context(ctx)
    session_ids = verify_sessions(ctx)

    if len(session_ids) < cs.MIN_SESSIONS:
        raise ValueError(
            f"{ctx.condition_label}: need >={cs.MIN_SESSIONS} sessions, found {len(session_ids)}"
        )

    t_ms, win_idx, left_by_ch, right_by_ch, _, _ = cs.collect_pooled_data(
        ctx.condition_label,
        session_ids,
        ctx.trial_filters,
    )

    output_dir = resolve_condition_output_dir(
        ctx.output_base / "figures",
        True,
        ctx.condition_label,
        "combined",
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    proc_label = processing_label(cs.GAUSSIAN_SMOOTH_MS, zscore_mua=True)
    suptitle = (
        f"{ctx.condition_label} | array mean of combined channel traces | {cs.ALIGNMENT_EVENT}\n"
        f"{filter_summary(ctx.trial_filters)} | "
        f"L/R {ctx.choice_field} {ctx.left_choice}/{ctx.right_choice} | "
        f"{proc_label} | mean ± SE across channels | n_sessions={len(session_ids)}"
    )

    fig = make_arrays_combined_figure(t_ms, win_idx, left_by_ch, right_by_ch, suptitle)
    out_path = output_dir / f"{ctx.condition_label}_{cs.ALIGNMENT_EVENT}_arrays_combined_LR.pdf"
    fig.savefig(out_path, format="pdf", dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out_path}")


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    session_lists_path = Path(args.session_lists)
    if not session_lists_path.exists():
        raise FileNotFoundError(f"Session lists file not found: {session_lists_path}")

    dual_nhp_root = args.dual_nhp_root
    if not dual_nhp_root.exists():
        raise FileNotFoundError(f"DUAL_NHP root not found: {dual_nhp_root}")

    for condition in resolve_conditions(args):
        print("\n" + "#" * 72)
        print(f"# {condition.label}")
        print(f"# -> {dual_nhp_root / condition.label / 'figures' / 'zscored' / 'combined'}")
        print("#" * 72)
        ctx = build_export_context(condition, dual_nhp_root, session_lists_path)
        if not ctx.session_ids:
            warnings.warn(f"No DUAL_NHP sessions for {condition.monkey}")
            continue
        try:
            plot_condition_array_combined(ctx)
        except Exception as exc:
            warnings.warn(f"Skipping {condition.label}: {exc}")


if __name__ == "__main__":
    main()
