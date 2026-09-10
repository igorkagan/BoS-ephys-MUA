"""Shared L/R session MUA plotting (timecourses, MWU, task-evoked)."""

from __future__ import annotations

import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat
from scipy.stats import mannwhitneyu

from process_channels.evoked import (
    TASK_EVOKED_ALPHA,
    TASK_EVOKED_BIN_MS,
    TASK_EVOKED_WINDOW_MS,
    session_task_evoked,
    task_evoked_anova_pvalues,
)
from load_data.io import (
    ARRAY_NAMES,
    array_nominal_channels,
    channel_label,
    gaussian_smooth_trials,
    trial_window_means,
)
from process_channels.preprocess import zscore_channel_trials
LEFT_COLOR = "#d62728"
RIGHT_COLOR = "#1f77b4"
SD_ALPHA = 0.25
Y_AXIS_PAD_FRACTION = 0.05
TIME_TICK_STEP_MS = 1000


def format_mwu_p(p: float) -> str:
    if p < 0.001:
        return "p<0.001"
    return f"p={p:.3f}"


def format_task_evoked_p(p: float | None) -> str:
    if p is None:
        return "n/a"
    if p < 0.001:
        return "<0.001"
    return f"{p:.3f}"


def significance_marker(p: float, alpha: float) -> str:
    if p >= alpha:
        return ""
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    return "*"


def run_mann_whitney(
    left_rates: np.ndarray,
    right_rates: np.ndarray,
    *,
    min_trials: int = 3,
) -> tuple[float | None, str]:
    left = left_rates[~np.isnan(left_rates)]
    right = right_rates[~np.isnan(right_rates)]

    if len(left) < min_trials or len(right) < min_trials:
        return None, "n/a"

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            _, p = mannwhitneyu(left, right, alternative="two-sided")
        except ValueError:
            return None, "n/a"

    return float(p), format_mwu_p(float(p))


def trial_trace_sd(trials: np.ndarray) -> np.ndarray:
    """Sample SD without warnings; one trial has a zero-width uncertainty band."""
    if trials.shape[0] <= 1:
        return np.zeros(trials.shape[1], dtype=float)
    return np.nanstd(trials, axis=0, ddof=1)


def plot_condition(ax, t_ms: np.ndarray, trials: np.ndarray, color: str, label: str) -> None:
    if trials.size == 0:
        return
    mean = np.nanmean(trials, axis=0)
    sd = trial_trace_sd(trials)
    ax.plot(t_ms, mean, color=color, linewidth=1.0, label=label)
    ax.fill_between(t_ms, mean - sd, mean + sd, color=color, alpha=SD_ALPHA, linewidth=0)


def _set_tight_ylim(ax, *trial_sets: np.ndarray) -> None:
    ymin, ymax = np.inf, -np.inf
    for trials in trial_sets:
        if trials.size == 0:
            continue
        mean = np.nanmean(trials, axis=0)
        sd = trial_trace_sd(trials)
        ymin = min(ymin, np.nanmin(mean - sd))
        ymax = max(ymax, np.nanmax(mean + sd))

    if not np.isfinite(ymin) or not np.isfinite(ymax):
        return
    pad = Y_AXIS_PAD_FRACTION * (ymax - ymin) if ymax > ymin else 0.1
    ax.set_ylim(ymin - pad, ymax + pad)


def task_evoked_annotation_text(
    *,
    task_evoked: bool,
    p_left: float | None,
    p_right: float | None,
) -> str:
    return (
        f"task: {'yes' if task_evoked else 'no'}\n"
        f"pL: {format_task_evoked_p(p_left)}\n"
        f"pR: {format_task_evoked_p(p_right)}"
    )


def annotate_task_evoked_text(ax, text: str) -> None:
    ax.text(
        0.02,
        0.02,
        text,
        transform=ax.transAxes,
        va="bottom",
        ha="left",
        fontsize=5,
        bbox=dict(boxstyle="round,pad=0.15", facecolor="white", alpha=0.85, edgecolor="none"),
    )


def annotate_task_evoked(
    ax,
    left_trials: np.ndarray,
    right_trials: np.ndarray,
    t_ms: np.ndarray,
    *,
    min_trials: int = 3,
    alpha: float = TASK_EVOKED_ALPHA,
) -> None:
    p_l, p_r = task_evoked_anova_pvalues(
        left_trials,
        right_trials,
        t_ms,
        bin_ms=TASK_EVOKED_BIN_MS,
        min_trials=min_trials,
    )
    evoked = session_task_evoked(p_l, p_r, alpha=alpha)
    annotate_task_evoked_text(
        ax,
        task_evoked_annotation_text(task_evoked=evoked, p_left=p_l, p_right=p_r),
    )


def plot_channel_subplot(
    ax,
    t_ms: np.ndarray,
    left_trials: np.ndarray,
    right_trials: np.ndarray,
    win_idx: np.ndarray,
    title: str,
    *,
    mwu_alpha: float = 0.05,
    min_trials: int = 3,
    show_task_evoked: bool = True,
    task_evoked_alpha: float = TASK_EVOKED_ALPHA,
    evoked_p_left: float | None = None,
    evoked_p_right: float | None = None,
    task_evoked: bool | None = None,
) -> None:
    plot_condition(ax, t_ms, left_trials, LEFT_COLOR, "Left")
    plot_condition(ax, t_ms, right_trials, RIGHT_COLOR, "Right")

    if win_idx.size:
        ax.axvspan(t_ms[win_idx[0]], t_ms[win_idx[-1]], color="0.85", alpha=0.35, zorder=0)
    ax.axvline(0, color="0.5", linewidth=0.6, linestyle="--")

    left_rates = trial_window_means(left_trials, win_idx)
    right_rates = trial_window_means(right_trials, win_idx)
    p_value, p_text = run_mann_whitney(left_rates, right_rates, min_trials=min_trials)

    stat_lines = [p_text]
    if p_value is not None and p_value < mwu_alpha:
        stat_lines.append(significance_marker(p_value, mwu_alpha))

    ax.text(
        0.02,
        0.98,
        "\n".join(stat_lines),
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=7,
        bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.75, edgecolor="none"),
    )

    if show_task_evoked:
        if evoked_p_left is not None or evoked_p_right is not None or task_evoked is not None:
            if task_evoked is None:
                task_evoked = session_task_evoked(
                    evoked_p_left, evoked_p_right, alpha=task_evoked_alpha,
                )
            annotate_task_evoked_text(
                ax,
                task_evoked_annotation_text(
                    task_evoked=bool(task_evoked),
                    p_left=evoked_p_left,
                    p_right=evoked_p_right,
                ),
            )
        else:
            annotate_task_evoked(
                ax,
                left_trials,
                right_trials,
                t_ms,
                min_trials=min_trials,
                alpha=task_evoked_alpha,
            )

    ax.set_title(title, fontsize=8)
    ax.tick_params(labelsize=6)
    ax.set_xlim(t_ms[0], t_ms[-1])
    _set_tight_ylim(ax, left_trials, right_trials)


def mark_empty_axis(ax, title: str, reason: str) -> None:
    ax.set_title(f"{title}\n[{reason}]", fontsize=8)
    ax.text(0.5, 0.5, reason, transform=ax.transAxes, ha="center", va="center", fontsize=8, color="0.45")
    ax.set_yticks([])


def configure_array_time_axis(
    axes,
    t_ms: np.ndarray,
    *,
    time_panel: int = 0,
    tick_step_ms: int = TIME_TICK_STEP_MS,
) -> None:
    flat = np.asarray(axes).ravel()
    ax0 = flat[time_panel]
    lo, hi = float(t_ms[0]), float(t_ms[-1])
    ax0.set_xlim(lo, hi)
    first = tick_step_ms * int(np.ceil(lo / tick_step_ms))
    ticks = np.arange(first, hi + 1e-9, tick_step_ms)
    if ticks.size == 0 or ticks[0] > lo:
        ticks = np.concatenate(([lo], ticks))
    if ticks[-1] < hi:
        ticks = np.concatenate((ticks, [hi]))
    ax0.set_xticks(ticks)
    ax0.set_xticklabels([f"{int(t)}" for t in ticks])
    ax0.tick_params(axis="x", labelsize=6, labelbottom=True)
    ax0.set_xlabel("Time (ms)", fontsize=7)
    for ax in flat[time_panel + 1:]:
        ax.tick_params(axis="x", labelbottom=False)
        ax.set_xlabel("")


def build_session_lr_suptitle(
    session_id: str,
    condition_label: str,
    alignment_event: str,
    filter_summary: str,
    *,
    zscore_mua: bool,
    gaussian_smooth_ms: float,
    n_left: int,
    n_right: int,
    task_evoked_window_ms: tuple[float, float] = TASK_EVOKED_WINDOW_MS,
    task_evoked_alpha: float = TASK_EVOKED_ALPHA,
) -> str:
    te_lo, te_hi = task_evoked_window_ms
    proc = "z-scored" if zscore_mua else "raw"
    smooth = f"smooth {gaussian_smooth_ms} ms" if gaussian_smooth_ms > 0 else "unsmoothed"
    return (
        f"{session_id} | {condition_label} | {alignment_event}\n"
        f"{filter_summary} | {proc} | {smooth} | "
        f"task-evoked {te_lo}:{te_hi} α={task_evoked_alpha} | "
        f"trials L={n_left}, R={n_right}"
    )


def load_smoothed_lr_trials(
    ch_path: Path,
    left_mask: np.ndarray,
    right_mask: np.ndarray,
    t_ms: np.ndarray,
    *,
    gaussian_smooth_ms: float,
    zscore_mua: bool,
    zscore_reference: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    mua = loadmat(ch_path)["cur_output_data"]
    if zscore_mua:
        mua = zscore_channel_trials(mua, reference_mask=zscore_reference)
    row_ok = ~np.all(np.isnan(mua), axis=1)
    left_trials = gaussian_smooth_trials(mua[left_mask & row_ok], t_ms, gaussian_smooth_ms)
    right_trials = gaussian_smooth_trials(mua[right_mask & row_ok], t_ms, gaussian_smooth_ms)
    return left_trials, right_trials


def make_session_array_figure(
    array_index: int,
    channel_paths: dict[int, Path],
    t_ms: np.ndarray,
    win_idx: np.ndarray,
    left_mask: np.ndarray,
    right_mask: np.ndarray,
    suptitle: str,
    *,
    gaussian_smooth_ms: float,
    zscore_mua: bool,
    zscore_reference: np.ndarray | None = None,
    subplot_grid: tuple[int, int] = (4, 8),
    fig_size: tuple[float, float] = (11, 8.5),
    mwu_alpha: float = 0.05,
    min_trials: int = 3,
    show_task_evoked: bool = True,
    task_evoked_alpha: float = TASK_EVOKED_ALPHA,
    cached_trials: dict[int, tuple[np.ndarray, np.ndarray]] | None = None,
    cached_summaries: dict[int, "ChannelSummary"] | None = None,
) -> plt.Figure:
    array_name = ARRAY_NAMES[array_index]
    ch_nums = array_nominal_channels(array_index)
    n_rows, n_cols = subplot_grid
    fig, axes = plt.subplots(n_rows, n_cols, figsize=fig_size, sharex=True, sharey=False)
    y_label = "MUA (z)" if zscore_mua else "MUA"

    for panel_idx, ax in enumerate(np.asarray(axes).ravel()):
        ch_num = ch_nums[panel_idx]
        title = channel_label(ch_num, array_name, panel_idx + 1)
        ch_path = channel_paths.get(ch_num)

        if cached_trials is not None:
            if ch_num not in cached_trials:
                mark_empty_axis(ax, title, "no data")
                continue
            left_trials, right_trials = cached_trials[ch_num]
        elif ch_path is None:
            mark_empty_axis(ax, title, "missing")
            continue
        else:
            left_trials, right_trials = load_smoothed_lr_trials(
                ch_path,
                left_mask,
                right_mask,
                t_ms,
                gaussian_smooth_ms=gaussian_smooth_ms,
                zscore_mua=zscore_mua,
                zscore_reference=zscore_reference,
            )

        n_left = 0 if left_trials.size == 0 else int(left_trials.shape[0])
        n_right = 0 if right_trials.size == 0 else int(right_trials.shape[0])
        if n_left < min_trials or n_right < min_trials:
            mark_empty_axis(ax, title, "no data")
            continue

        summary = cached_summaries.get(ch_num) if cached_summaries else None
        plot_channel_subplot(
            ax,
            t_ms,
            left_trials,
            right_trials,
            win_idx,
            title,
            mwu_alpha=mwu_alpha,
            min_trials=min_trials,
            show_task_evoked=show_task_evoked,
            task_evoked_alpha=task_evoked_alpha,
            evoked_p_left=summary.evoked_p_left if summary else None,
            evoked_p_right=summary.evoked_p_right if summary else None,
            task_evoked=summary.task_evoked if summary else None,
        )

        if panel_idx % n_cols == 0:
            ax.set_ylabel(y_label, fontsize=7)

    configure_array_time_axis(axes, t_ms)
    fig.suptitle(suptitle, fontsize=10, y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    return fig
