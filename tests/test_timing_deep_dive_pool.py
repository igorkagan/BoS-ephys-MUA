"""Tests for timing deep-dive channel selection."""

from __future__ import annotations

import numpy as np

from compare_monkey_timing_conditions import build_timing_deep_dive_pool, rank_similar_different
from bos_mua.features import ChannelSummary


def _summary(session_id: str, channel: int, si: float, task: bool) -> ChannelSummary:
    t = np.linspace(-100, 100, 21)
    diff = np.sin(t / 20) * si
    return ChannelSummary(
        session_id=session_id,
        channel=channel,
        array_name="A1",
        index_in_array=1,
        n_left=10,
        n_right=10,
        t_ms=t,
        mean_left=diff * 0.5,
        mean_right=-diff * 0.5,
        diff=diff,
        si=si,
        mwu_p=0.01,
        pref_side="L",
        evoked_p_left=0.0001 if task else 0.5,
        evoked_p_right=0.0001 if task else 0.5,
        task_evoked=task,
    )


def test_rank_similar_different_order() -> None:
    pool = [
        {"channel": 1, "delta_si_median_abs": 0.01, "delta_si_median": 0.01, "waveform_r_median": 0.9, "sign_flip_rate": 0.0},
        {"channel": 2, "delta_si_median_abs": 0.50, "delta_si_median": 0.50, "waveform_r_median": 0.8, "sign_flip_rate": 0.25},
        {"channel": 3, "delta_si_median_abs": 0.20, "delta_si_median": -0.20, "waveform_r_median": 0.7, "sign_flip_rate": 0.5},
    ]
    similar, different = rank_similar_different(pool, n=2)
    assert [r["channel"] for r in similar] == [1, 3]
    assert [r["channel"] for r in different] == [2, 3]
