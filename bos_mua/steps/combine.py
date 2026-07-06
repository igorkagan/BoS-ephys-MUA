#!/usr/bin/env python3
"""Pool L/R MUA trials across sessions per nominal channel (z-scored only).

Usage:
    pip install -r requirements.txt
    python combine_sessions.py
"""

from __future__ import annotations

import warnings
from collections import defaultdict
from pathlib import Path

from bos_mua.viz.lr import configure_array_time_axis
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat
from scipy.stats import wilcoxon

from bos_mua.features import run_mann_whitney
from bos_mua.io import (
    ARRAY_NAMES,
    CHANNELS_PER_ARRAY,
    array_nominal_channels,
    channel_files_by_number,
    channel_label,
    choice_mask,
    build_base_mask,
    discover_sessions,
    filter_summary,
    gaussian_smooth_trials,
    load_trial_labels,
    load_time_vector,
    trial_window_means,
    window_indices,
)
from bos_mua.preprocess import (
    MONKEY_CONDITIONS,
    processing_label,
    recording_monkey,
    resolve_condition_output_dir,
    trial_filters_for_condition,
    zscore_channel_trials,
    zscore_reference_mask,
)
from bos_mua.run_context import resolve_session_dir, session_ids_for_run

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DATA_ROOT = r"S:\taskcontroller\SCP_DATA\SCP-CTRL-01\MUA_curated_sessions"

ALIGNMENT_EVENT = "A_InitialFixationReleaseTime_ms"
PRE_POST_TAG = "pre1000ms.post1000ms"

TRIAL_FILTERS = None  # None = per-condition defaults via trial_filters_for_condition
CHOICE_FIELD = "A_LR_pos_list"
LEFT_CHOICE = ["Al"]
RIGHT_CHOICE = ["Ar"]
INVALID_LABELS = frozenset({"NONE", "None", "none", ""})

ANALYSIS_WINDOW_MS = (-500, 500)
GAUSSIAN_SMOOTH_MS = 50
ALPHA = 0.05
MIN_TRIALS_PER_GROUP = 3
MIN_SESSIONS = 2  # paired Wilcoxon: use all sessions with valid L/R pair for this channel

SUBPLOT_GRID = (4, 8)
OUTPUT_DIR = r"./figures"
FIG_SIZE_IN = (11, 8.5)
DPI = 150
Y_AXIS_PAD_FRACTION = 0.05

LEFT_COLOR = "#d62728"
RIGHT_COLOR = "#1f77b4"
SD_ALPHA = 0.25
MWU_COLOR = "#2ca02c"
WILCOXON_COLOR = "#9467bd"

# ---------------------------------------------------------------------------


def format_p_value(p: float) -> str:
    if p < 0.001:
        return "p<0.001"
    return f"p={p:.3f}"


def significance_marker(p: float, alpha: float) -> str:
    if p >= alpha:
        return ""
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    return "*"


def run_paired_wilcoxon(
    session_left_means: list[float],
    session_right_means: list[float],
    min_sessions: int = MIN_SESSIONS,
) -> tuple[float | None, str]:
    if len(session_left_means) < min_sessions:
        return None, "n/a"

    left = np.asarray(session_left_means, dtype=float)
    right = np.asarray(session_right_means, dtype=float)
    valid = np.isfinite(left) & np.isfinite(right)
    left = left[valid]
    right = right[valid]

    if len(left) < min_sessions:
        return None, "n/a"

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            _, p = wilcoxon(left, right, alternative="two-sided", zero_method="wilcox")
        except ValueError:
            return None, "n/a"

    return float(p), format_p_value(float(p))


def plot_condition(ax, t_ms: np.ndarray, trials: np.ndarray, color: str, label: str) -> None:
    if trials.size == 0:
        return
    mean = np.nanmean(trials, axis=0)
    sd = np.nanstd(trials, axis=0, ddof=1)
    ax.plot(t_ms, mean, color=color, linewidth=1.0, label=label, zorder=3)
    ax.fill_between(
        t_ms, mean - sd, mean + sd, color=color, alpha=SD_ALPHA, linewidth=0, zorder=2,
    )


def _set_tight_ylim(ax, *trial_sets: np.ndarray) -> None:
    ymin, ymax = np.inf, -np.inf
    for trials in trial_sets:
        if trials.size == 0:
            continue
        mean = np.nanmean(trials, axis=0)
        sd = np.nanstd(trials, axis=0, ddof=1)
        ymin = min(ymin, np.nanmin(mean - sd))
        ymax = max(ymax, np.nanmax(mean + sd))

    if not np.isfinite(ymin) or not np.isfinite(ymax):
        return
    pad = Y_AXIS_PAD_FRACTION * (ymax - ymin) if ymax > ymin else 0.1
    ax.set_ylim(ymin - pad, ymax + pad)


def plot_channel_subplot(
    ax,
    t_ms: np.ndarray,
    left_trials: np.ndarray,
    right_trials: np.ndarray,
    win_idx: np.ndarray,
    session_left_means: list[float],
    session_right_means: list[float],
    title: str,
) -> None:
    plot_condition(ax, t_ms, left_trials, LEFT_COLOR, "Left")
    plot_condition(ax, t_ms, right_trials, RIGHT_COLOR, "Right")

    if win_idx.size:
        ax.axvspan(t_ms[win_idx[0]], t_ms[win_idx[-1]], color="0.85", alpha=0.35, zorder=0)
    ax.axvline(0, color="0.5", linewidth=0.6, linestyle="--")

    left_rates = trial_window_means(left_trials, win_idx)
    right_rates = trial_window_means(right_trials, win_idx)
    mwu_p, _ = run_mann_whitney(left_rates, right_rates, min_trials=MIN_TRIALS_PER_GROUP)
    wilcox_p, _ = run_paired_wilcoxon(session_left_means, session_right_means)

    mwu_line = format_p_value(mwu_p) if mwu_p is not None else "n/a"
    if mwu_p is not None and mwu_p < ALPHA:
        mwu_line += f" {significance_marker(mwu_p, ALPHA)}"

    wilcox_line = format_p_value(wilcox_p) if wilcox_p is not None else "n/a"
    n_sess = len(session_left_means)
    if wilcox_p is not None and wilcox_p < ALPHA:
        wilcox_line += f" {significance_marker(wilcox_p, ALPHA)}"
    wilcox_line += f"  n={n_sess}"

    n_left = len(left_trials)
    n_right = len(right_trials)

    stat_y = 0.98
    line_dy = 0.085

    ax.text(
        0.02, stat_y, mwu_line,
        transform=ax.transAxes, va="top", ha="left", fontsize=7, color=MWU_COLOR,
        bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.75, edgecolor="none"),
    )
    ax.text(
        0.02, stat_y - line_dy, wilcox_line,
        transform=ax.transAxes, va="top", ha="left", fontsize=7, color=WILCOXON_COLOR,
    )
    ax.text(
        0.02, stat_y - 2 * line_dy, f"nL={n_left}, nR={n_right}",
        transform=ax.transAxes, va="top", ha="left", fontsize=6, color="0.35",
    )

    ax.set_title(title, fontsize=8)
    ax.tick_params(labelsize=6)
    ax.set_xlim(t_ms[0], t_ms[-1])
    _set_tight_ylim(ax, left_trials, right_trials)


def _mark_empty_axis(ax, title: str, reason: str) -> None:
    ax.set_title(f"{title}\n[{reason}]", fontsize=8)
    ax.text(0.5, 0.5, reason, transform=ax.transAxes, ha="center", va="center", fontsize=8, color="0.45")
    ax.set_xticks([])
    ax.set_yticks([])


def load_channel_lr_trials(
    ch_path: Path,
    left_mask: np.ndarray,
    right_mask: np.ndarray,
    t_ms: np.ndarray,
    *,
    zscore_reference: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    mua = loadmat(ch_path)["cur_output_data"]
    mua = zscore_channel_trials(mua, reference_mask=zscore_reference)
    row_ok = ~np.all(np.isnan(mua), axis=1)

    left_trials = gaussian_smooth_trials(mua[left_mask & row_ok], t_ms, GAUSSIAN_SMOOTH_MS)
    right_trials = gaussian_smooth_trials(mua[right_mask & row_ok], t_ms, GAUSSIAN_SMOOTH_MS)
    return left_trials, right_trials


def collect_pooled_data(
    condition: str,
    session_ids: list[str],
    trial_filters: dict,
) -> tuple[np.ndarray, np.ndarray, dict[int, list[np.ndarray]], dict[int, list[np.ndarray]], dict[int, list[float]], dict[int, list[float]]]:
    """Return time vector, win_idx, pooled trials and session means per channel."""
    t_ms: np.ndarray | None = None
    win_idx: np.ndarray | None = None

    left_by_ch: dict[int, list[np.ndarray]] = defaultdict(list)
    right_by_ch: dict[int, list[np.ndarray]] = defaultdict(list)
    sess_left_by_ch: dict[int, list[float]] = defaultdict(list)
    sess_right_by_ch: dict[int, list[float]] = defaultdict(list)

    for session_id in session_ids:
        session_dir = resolve_session_dir(
            session_id, data_root=DATA_ROOT, condition_folder=condition,
        )
        event_dir = session_dir / ALIGNMENT_EVENT
        if not event_dir.exists():
            warnings.warn(f"Skipping {session_id}: missing event dir {event_dir}")
            continue

        try:
            labels = load_trial_labels(session_dir, session_id)
            base_mask = build_base_mask(labels, trial_filters, invalid_labels=INVALID_LABELS)
            left_mask = choice_mask(labels, base_mask, LEFT_CHOICE, field=CHOICE_FIELD)
            right_mask = choice_mask(labels, base_mask, RIGHT_CHOICE, field=CHOICE_FIELD)
            monkey = recording_monkey(session_id=session_id, condition_label=condition)
            zscore_ref = zscore_reference_mask(labels, monkey)
        except Exception as exc:
            warnings.warn(f"Skipping {session_id}: {exc}")
            continue

        session_t_ms = load_time_vector(event_dir, session_id, ALIGNMENT_EVENT, PRE_POST_TAG)
        session_win_idx = window_indices(session_t_ms, ANALYSIS_WINDOW_MS)
        if session_win_idx.size == 0:
            warnings.warn(f"Skipping {session_id}: empty analysis window")
            continue

        if t_ms is None:
            t_ms = session_t_ms
            win_idx = session_win_idx
        elif not np.allclose(t_ms, session_t_ms):
            warnings.warn(f"Skipping {session_id}: time vector mismatch")
            continue

        channel_paths = channel_files_by_number(event_dir)

        for ch_num, ch_path in channel_paths.items():
            try:
                left_trials, right_trials = load_channel_lr_trials(
                    ch_path, left_mask, right_mask, session_t_ms,
                    zscore_reference=zscore_ref,
                )
            except Exception as exc:
                warnings.warn(f"Skipping ch{ch_num:03d} in {session_id}: {exc}")
                continue

            # Pooled traces: any session with trials on that side.
            if left_trials.size:
                left_by_ch[ch_num].append(left_trials)
            if right_trials.size:
                right_by_ch[ch_num].append(right_trials)

            left_rates = trial_window_means(left_trials, session_win_idx)
            right_rates = trial_window_means(right_trials, session_win_idx)
            n_left = int(np.sum(~np.isnan(left_rates))) if left_rates.size else 0
            n_right = int(np.sum(~np.isnan(right_rates))) if right_rates.size else 0

            # Session-paired Wilcoxon: include this session if both sides have
            # enough trials here; missing channel or one-sided sessions in other
            # days do not exclude the remaining paired sessions.
            if n_left >= MIN_TRIALS_PER_GROUP and n_right >= MIN_TRIALS_PER_GROUP:
                sess_left_by_ch[ch_num].append(float(np.nanmean(left_rates)))
                sess_right_by_ch[ch_num].append(float(np.nanmean(right_rates)))

    if t_ms is None or win_idx is None:
        raise ValueError(f"No usable sessions for {condition}")

    return t_ms, win_idx, left_by_ch, right_by_ch, sess_left_by_ch, sess_right_by_ch


def make_array_figure(
    array_index: int,
    t_ms: np.ndarray,
    win_idx: np.ndarray,
    left_by_ch: dict[int, list[np.ndarray]],
    right_by_ch: dict[int, list[np.ndarray]],
    sess_left_by_ch: dict[int, list[float]],
    sess_right_by_ch: dict[int, list[float]],
    suptitle: str,
) -> plt.Figure:
    array_name = ARRAY_NAMES[array_index]
    ch_nums = array_nominal_channels(array_index)
    n_rows, n_cols = SUBPLOT_GRID
    fig, axes = plt.subplots(n_rows, n_cols, figsize=FIG_SIZE_IN, sharex=True, sharey=False)
    axes_flat = axes.ravel()

    for panel_idx, ax in enumerate(axes_flat):
        ch_num = ch_nums[panel_idx]
        title = channel_label(ch_num, array_name, panel_idx + 1)

        left_parts = left_by_ch.get(ch_num, [])
        right_parts = right_by_ch.get(ch_num, [])

        if not left_parts and not right_parts:
            _mark_empty_axis(ax, title, "no data")
            continue

        left_trials = np.vstack(left_parts) if left_parts else np.empty((0, len(t_ms)))
        right_trials = np.vstack(right_parts) if right_parts else np.empty((0, len(t_ms)))

        if left_trials.size == 0 and right_trials.size == 0:
            _mark_empty_axis(ax, title, "no data")
            continue

        plot_channel_subplot(
            ax, t_ms, left_trials, right_trials, win_idx,
            sess_left_by_ch.get(ch_num, []),
            sess_right_by_ch.get(ch_num, []),
            title,
        )

        if panel_idx % n_cols == 0:
            ax.set_ylabel("MUA (z)", fontsize=7)

    configure_array_time_axis(axes, t_ms)

    fig.suptitle(suptitle, fontsize=10, y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    return fig


def plot_condition_combined(
    condition: str,
    output_dir: Path,
    session_ids: list[str] | None = None,
) -> None:
    if session_ids is None:
        session_ids = session_ids_for_run(DATA_ROOT, condition)
    if not session_ids:
        raise FileNotFoundError(f"No sessions for {condition}")

    trial_filters = TRIAL_FILTERS or trial_filters_for_condition(condition)
    print(f"\n=== {condition} | {len(session_ids)} session(s) -> {output_dir} ===")

    t_ms, win_idx, left_by_ch, right_by_ch, sess_left_by_ch, sess_right_by_ch = collect_pooled_data(
        condition, session_ids, trial_filters,
    )

    n_left_total = sum(t.shape[0] for parts in left_by_ch.values() for t in parts)
    n_right_total = sum(t.shape[0] for parts in right_by_ch.values() for t in parts)
    print(f"  Pooled trials: L={n_left_total}, R={n_right_total}")

    summary = filter_summary(trial_filters)
    proc_label = processing_label(GAUSSIAN_SMOOTH_MS, zscore_mua=True)
    suptitle_base = (
        f"{condition} | combined sessions | {ALIGNMENT_EVENT}\n"
        f"{summary} | {proc_label} | "
        f"window {ANALYSIS_WINDOW_MS[0]}:{ANALYSIS_WINDOW_MS[1]} ms | "
        f"n_sessions={len(session_ids)}"
    )

    for array_index, array_name in enumerate(ARRAY_NAMES):
        fig = make_array_figure(
            array_index, t_ms, win_idx,
            left_by_ch, right_by_ch, sess_left_by_ch, sess_right_by_ch,
            f"{suptitle_base} | {array_name}",
        )
        out_path = output_dir / f"{condition}_{ALIGNMENT_EVENT}_{array_name}_combined_LR.pdf"
        fig.savefig(out_path, format="pdf", dpi=DPI, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved {out_path.name}")


def main() -> None:
    for condition in MONKEY_CONDITIONS:
        condition_dir = Path(DATA_ROOT) / condition
        if not condition_dir.exists():
            warnings.warn(f"Condition folder not found: {condition}")
            continue
        output_dir = resolve_condition_output_dir(OUTPUT_DIR, True, condition, "combined")
        output_dir.mkdir(parents=True, exist_ok=True)
        try:
            plot_condition_combined(condition, output_dir)
        except Exception as exc:
            warnings.warn(f"Skipping {condition}: {exc}")


if __name__ == "__main__":
    main()
