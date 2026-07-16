"""Tests for plot_export_condition_arrays."""

from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np

from process_channels.preprocess import trial_filters_for_dual_nhp_go_seq
from analyze_stability.arrays import (
    EXPORT_CONDITIONS,
    array_combined_stats,
    build_export_context,
    channel_combined_mean_trace,
    mean_and_se_across_channels,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
SESSION_LISTS = REPO_ROOT / "session_lists.m"
DUAL_NHP_ROOT = Path(r"S:/taskcontroller/SCP_DATA/SCP-CTRL-01/MUA_export_per_session/DUAL_NHP")


class PlotExportConditionArraysTests(unittest.TestCase):
    def test_export_conditions(self) -> None:
        labels = {c.label for c in EXPORT_CONDITIONS}
        self.assertEqual(
            labels,
            {"Elmo_AgoB", "Elmo_BgoA", "Curius_AgoB", "Curius_BgoA"},
        )

    def test_build_export_context(self) -> None:
        cond = next(c for c in EXPORT_CONDITIONS if c.label == "Elmo_AgoB")
        ctx = build_export_context(cond, DUAL_NHP_ROOT, SESSION_LISTS)
        self.assertEqual(len(ctx.session_ids), 4)
        self.assertEqual(ctx.trial_filters["go_seq_500_list"], ["AgoB"])
        self.assertEqual(ctx.choice_field, "B_LR_pos_list")
        self.assertEqual(ctx.trial_filters["B_Reward_list"], ["RB1", "RB2", "RB3", "RB4"])

    def test_channel_combined_mean_trace(self) -> None:
        trace = channel_combined_mean_trace([np.array([[1.0, 3.0], [3.0, 5.0]])])
        np.testing.assert_allclose(trace, [2.0, 4.0])
        self.assertIsNone(channel_combined_mean_trace([]))

    def test_mean_and_se_across_channels(self) -> None:
        traces = [np.array([0.0, 2.0]), np.array([2.0, 4.0]), np.array([4.0, 6.0])]
        mean, se, n = mean_and_se_across_channels(traces)
        np.testing.assert_allclose(mean, [2.0, 4.0])
        np.testing.assert_allclose(se, [2.0 / np.sqrt(3), 2.0 / np.sqrt(3)])
        self.assertEqual(n, 3)

    def test_array_combined_stats(self) -> None:
        t_len = 3
        left_by_ch = {
            1: [np.ones((2, t_len))],
            2: [np.full((2, t_len), 3.0)],
        }
        right_by_ch = {
            1: [np.full((2, t_len), 5.0)],
            2: [np.full((2, t_len), 7.0)],
        }
        left_mean, left_se, right_mean, right_se, n_l, n_r = array_combined_stats(
            0, left_by_ch, right_by_ch,
        )
        np.testing.assert_allclose(left_mean, [2.0, 2.0, 2.0])
        np.testing.assert_allclose(right_mean, [6.0, 6.0, 6.0])
        self.assertEqual(n_l, 2)
        self.assertEqual(n_r, 2)
        self.assertIsNotNone(left_se)
        self.assertIsNotNone(right_se)

    def test_trial_filters_for_go_seq(self) -> None:
        self.assertEqual(
            trial_filters_for_dual_nhp_go_seq("BgoA", "B")["go_seq_500_list"],
            ["BgoA"],
        )


if __name__ == "__main__":
    unittest.main()
