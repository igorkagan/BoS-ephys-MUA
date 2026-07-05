"""Tests for cross-session task-evoked stability gate."""

from __future__ import annotations

import unittest
from unittest.mock import MagicMock

import numpy as np

from bos_mua.stability import (
    assess_channel_stability,
    channel_task_evoked_all_sessions,
    passes_tuned_stable_gate,
    rank_best_worst_channels,
)


class TaskEvokedStabilityTests(unittest.TestCase):
    def test_channel_task_evoked_all_sessions(self) -> None:
        lookup = {
            "s1": {1: MagicMock(task_evoked=True)},
            "s2": {1: MagicMock(task_evoked=True)},
            "s3": {1: MagicMock(task_evoked=False)},
        }
        ok, n_ev, n_data = channel_task_evoked_all_sessions(lookup, ["s1", "s2", "s3"], 1)
        self.assertFalse(ok)
        self.assertEqual(n_ev, 2)
        self.assertEqual(n_data, 3)

        ok2, _, _ = channel_task_evoked_all_sessions(lookup, ["s1", "s2"], 1)
        self.assertTrue(ok2)

    def test_stable_requires_task_evoked(self) -> None:
        traces = np.tile(np.linspace(0, 1, 20), (3, 1))
        si = np.array([0.2, 0.25, 0.22])
        stab = assess_channel_stability(
            1,
            "A1",
            traces,
            si,
            r_thresh=0.1,
            icc_thresh=0.1,
            sign_thresh=0.5,
            task_evoked=False,
            n_sessions_task_evoked=0,
        )
        self.assertFalse(stab.stable)

        stab_ok = assess_channel_stability(
            1,
            "A1",
            traces,
            si,
            r_thresh=0.1,
            icc_thresh=0.1,
            sign_thresh=0.5,
            task_evoked=True,
            n_sessions_task_evoked=3,
        )
        self.assertTrue(stab_ok.stable)

    def test_best10_only_stable(self) -> None:
        stabilities = [
            assess_channel_stability(
                1, "A1", np.ones((3, 10)), np.array([0.2, 0.2, 0.2]),
                0.1, 0.1, 0.5, task_evoked=True, n_sessions_task_evoked=3,
            ),
            assess_channel_stability(
                2, "A1", np.ones((3, 10)) * 2, np.array([0.3, 0.3, 0.3]),
                0.1, 0.1, 0.5, task_evoked=False, n_sessions_task_evoked=1,
            ),
        ]
        stabilities[0].median_pairwise_r = 0.9
        stabilities[0].stable = True
        stabilities[1].median_pairwise_r = 0.99
        stabilities[1].stable = False
        best, _ = rank_best_worst_channels(stabilities, n=10)
        self.assertEqual(len(best), 1)
        self.assertEqual(best[0].channel, 1)

    def test_tuned_stable_requires_task_evoked(self) -> None:
        stab = assess_channel_stability(
            5, "A1", np.ones((3, 10)), np.array([0.2, 0.21, 0.19]),
            0.1, 0.1, 0.5, task_evoked=False, n_sessions_task_evoked=0,
        )
        stab.sign_concordance = 1.0
        stab.si_std = 0.01
        stab.si_median_abs = 0.5
        self.assertFalse(passes_tuned_stable_gate(stab, 0.7, 0.3, 0.1))


if __name__ == "__main__":
    unittest.main()
