#!/usr/bin/env python3
"""Plot left vs right MUA timecourses for one session, all channels.

Usage:
    pip install -r requirements.txt
    python plot_session_lr_mua.py
"""

from __future__ import annotations

import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat
from scipy.stats import mannwhitneyu

from bos_mua.io import (
    ARRAY_NAMES,
    CHANNELS_PER_ARRAY,
    array_channel_files,
    channel_label,
    channel_number,
    choice_mask,
    build_base_mask,
    discover_channel_files,
    discover_sessions,
    filter_summary,
    gaussian_smooth_trials,
    load_trial_labels,
    load_time_vector,
    trial_window_means,
    window_indices,
)
from bos_mua.preprocess import processing_label, resolve_figures_dir, zscore_channel_trials

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DATA_ROOT = r"S:\taskcontroller\SCP_DATA\SCP-CTRL-01\MUA_curated_sessions"
CONDITION_FOLDER = "Elmo_BLOCKED"
SESSION_ID = None  # None = all sessions in CONDITION_FOLDER; or set a single session ID
ALIGNMENT_EVENT = "A_InitialFixationReleaseTime_ms"
PRE_POST_TAG = "pre1000ms.post1000ms"

TRIAL_FILTERS = {
    "TrialSubType_list": ["Dyadic"],
    "conf_predictability_list": ["Blocked"],
    "go_seq_500_list": ["AgoB"],
    "A_Reward_list": ["RA1", "RA2", "RA3", "RA4"],
}
LEFT_CHOICE = ["Al"]
RIGHT_CHOICE = ["Ar"]
INVALID_LABELS = {"NONE", "None", "none", ""}

ANALYSIS_WINDOW_MS = (-500, 500)
GAUSSIAN_SMOOTH_MS = 50
ZSCORE_MUA = False  # ignored when RUN_BOTH_PROCESSING is True
RUN_BOTH_PROCESSING = True
ALPHA = 0.05
MIN_TRIALS_PER_GROUP = 3

SUBPLOT_GRID = (4, 8)
OUTPUT_DIR = r"./figures"
FIG_SIZE_IN = (11, 8.5)
DPI = 150
Y_AXIS_PAD_FRACTION = 0.05

LEFT_COLOR = "#d62728"
RIGHT_COLOR = "#1f77b4"
SD_ALPHA = 0.25

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


def run_mann_whitney(left_rates: np.ndarray, right_rates: np.ndarray) -> tuple[float | None, str]:
    left = left_rates[~np.isnan(left_rates)]
    right = right_rates[~np.isnan(right_rates)]

    if len(left) < MIN_TRIALS_PER_GROUP or len(right) < MIN_TRIALS_PER_GROUP:
        return None, "n/a"

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            _, p = mannwhitneyu(left, right, alternative="two-sided")
        except ValueError:
            return None, "n/a"

    return float(p), format_p_value(float(p))


def plot_condition(ax, t_ms: np.ndarray, trials: np.ndarray, color: str, label: str) -> None:
    if trials.size == 0:
        return
    mean = np.nanmean(trials, axis=0)
    sd = np.nanstd(trials, axis=0, ddof=1)
    ax.plot(t_ms, mean, color=color, linewidth=1.0, label=label)
    ax.fill_between(t_ms, mean - sd, mean + sd, color=color, alpha=SD_ALPHA, linewidth=0)


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
    title: str,
) -> None:
    plot_condition(ax, t_ms, left_trials, LEFT_COLOR, "Left")
    plot_condition(ax, t_ms, right_trials, RIGHT_COLOR, "Right")

    if win_idx.size:
        ax.axvspan(t_ms[win_idx[0]], t_ms[win_idx[-1]], color="0.85", alpha=0.35, zorder=0)
    ax.axvline(0, color="0.5", linewidth=0.6, linestyle="--")

    left_rates = trial_window_means(left_trials, win_idx)
    right_rates = trial_window_means(right_trials, win_idx)
    p_value, p_text = run_mann_whitney(left_rates, right_rates)

    stat_lines = [p_text]
    if p_value is not None and p_value < ALPHA:
        stat_lines.append(significance_marker(p_value, ALPHA))

    ax.text(
        0.02, 0.98, "\n".join(stat_lines),
        transform=ax.transAxes, va="top", ha="left", fontsize=7,
        bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.75, edgecolor="none"),
    )

    ax.set_title(title, fontsize=8)
    ax.tick_params(labelsize=6)
    ax.set_xlim(t_ms[0], t_ms[-1])
    _set_tight_ylim(ax, left_trials, right_trials)


def make_array_figure(
    array_name: str,
    channel_files: list[Path],
    t_ms: np.ndarray,
    win_idx: np.ndarray,
    left_mask: np.ndarray,
    right_mask: np.ndarray,
    suptitle: str,
    zscore_mua: bool,
) -> plt.Figure:
    n_rows, n_cols = SUBPLOT_GRID
    fig, axes = plt.subplots(n_rows, n_cols, figsize=FIG_SIZE_IN, sharex=True, sharey=False)
    axes_flat = axes.ravel()

    for panel_idx, ax in enumerate(axes_flat):
        if panel_idx >= len(channel_files):
            ax.axis("off")
            continue

        ch_path = channel_files[panel_idx]
        ch_num = channel_number(ch_path)
        mua = loadmat(ch_path)["cur_output_data"]
        if zscore_mua:
            mua = zscore_channel_trials(mua)
        row_ok = ~np.all(np.isnan(mua), axis=1)

        left_trials = gaussian_smooth_trials(mua[left_mask & row_ok], t_ms, GAUSSIAN_SMOOTH_MS)
        right_trials = gaussian_smooth_trials(mua[right_mask & row_ok], t_ms, GAUSSIAN_SMOOTH_MS)
        title = channel_label(ch_num, array_name, panel_idx + 1)

        plot_channel_subplot(ax, t_ms, left_trials, right_trials, win_idx, title)

        if panel_idx % n_cols == 0:
            ax.set_ylabel("MUA (z)" if zscore_mua else "MUA", fontsize=7)
        if panel_idx >= (n_rows - 1) * n_cols:
            ax.set_xlabel("Time (ms)", fontsize=7)

    fig.suptitle(suptitle, fontsize=10, y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    return fig


def plot_session(session_id: str, output_dir: Path, zscore_mua: bool) -> None:
    session_dir = Path(DATA_ROOT) / CONDITION_FOLDER / session_id
    event_dir = session_dir / ALIGNMENT_EVENT

    if not session_dir.exists():
        raise FileNotFoundError(f"Session directory not found: {session_dir}")
    if not event_dir.exists():
        raise FileNotFoundError(f"Event directory not found: {event_dir}")

    labels = load_trial_labels(session_dir, session_id)
    base_mask = build_base_mask(labels, TRIAL_FILTERS, invalid_labels=frozenset(INVALID_LABELS))
    left_mask = choice_mask(labels, base_mask, LEFT_CHOICE)
    right_mask = choice_mask(labels, base_mask, RIGHT_CHOICE)

    n_left = int(np.sum(left_mask))
    n_right = int(np.sum(right_mask))
    print(f"{session_id}: trials L={n_left}, R={n_right}")

    t_ms = load_time_vector(event_dir, session_id, ALIGNMENT_EVENT, PRE_POST_TAG)
    win_idx = window_indices(t_ms, ANALYSIS_WINDOW_MS)
    if win_idx.size == 0:
        raise ValueError(f"No time points in analysis window {ANALYSIS_WINDOW_MS}")

    channel_files = discover_channel_files(event_dir)

    summary = filter_summary(TRIAL_FILTERS)
    proc_label = processing_label(GAUSSIAN_SMOOTH_MS, zscore_mua)
    suptitle_base = (
        f"{session_id} | {CONDITION_FOLDER} | {ALIGNMENT_EVENT}\n"
        f"{summary} | {proc_label} | "
        f"window {ANALYSIS_WINDOW_MS[0]}:{ANALYSIS_WINDOW_MS[1]} ms | "
        f"trials L={n_left}, R={n_right}"
    )

    n_arrays = int(np.ceil(len(channel_files) / CHANNELS_PER_ARRAY))
    for array_index in range(n_arrays):
        array_name = ARRAY_NAMES[array_index] if array_index < len(ARRAY_NAMES) else f"A{array_index + 1}"
        ch_files = array_channel_files(channel_files, array_index)
        if not ch_files:
            continue

        fig = make_array_figure(
            array_name, ch_files, t_ms, win_idx, left_mask, right_mask,
            f"{suptitle_base} | {array_name}",
            zscore_mua,
        )

        out_path = output_dir / f"{session_id}_{ALIGNMENT_EVENT}_{array_name}_LR.pdf"
        fig.savefig(out_path, format="pdf", dpi=DPI, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved {out_path.name}")


def main() -> None:
    condition_dir = Path(DATA_ROOT) / CONDITION_FOLDER
    session_ids = [SESSION_ID] if SESSION_ID else discover_sessions(condition_dir)
    if not session_ids:
        raise FileNotFoundError(f"No sessions in {condition_dir}")

    modes = (False, True) if RUN_BOTH_PROCESSING else (ZSCORE_MUA,)
    for zscore_mua in modes:
        output_dir = resolve_figures_dir(OUTPUT_DIR, zscore_mua)
        output_dir.mkdir(parents=True, exist_ok=True)
        label = "z-scored" if zscore_mua else "original"
        print(f"\n=== {label} | Plotting {len(session_ids)} session(s) -> {output_dir} ===")
        for session_id in session_ids:
            try:
                plot_session(session_id, output_dir, zscore_mua)
            except Exception as exc:
                warnings.warn(f"Skipping {session_id} ({label}): {exc}")


if __name__ == "__main__":
    main()
