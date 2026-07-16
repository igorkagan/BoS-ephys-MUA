"""Tests for task-evoked Friedman helper."""

from __future__ import annotations

import unittest

import numpy as np

from process_channels.evoked import (
    TASK_EVOKED_BIN_MS,
    TASK_EVOKED_WINDOW_MS,
    bin_trials_by_time,
    session_task_evoked,
    task_evoked_anova_pvalues,
)
from load_data.io import window_indices


class TaskEvokedAnovaTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.t_ms = np.arange(-1000, 1001, dtype=float)

    def test_bin_count_task_window(self) -> None:
        win_idx = window_indices(self.t_ms, TASK_EVOKED_WINDOW_MS)
        trials = np.ones((5, self.t_ms.size))
        binned = bin_trials_by_time(self.t_ms[win_idx], trials[:, win_idx], TASK_EVOKED_BIN_MS)
        self.assertEqual(binned.shape, (5, 30))

    def test_flat_trace_not_evoked(self) -> None:
        trials = np.ones((10, self.t_ms.size))
        p_l, p_r = task_evoked_anova_pvalues(trials, trials, self.t_ms)
        self.assertIsNotNone(p_l)
        self.assertGreater(p_l, 0.05)
        self.assertFalse(session_task_evoked(p_l, p_r))

    def test_modulated_trace_evoked(self) -> None:
        ramp = np.tile(np.linspace(0, 1, self.t_ms.size), (10, 1))
        p_l, _ = task_evoked_anova_pvalues(ramp, np.empty((0, self.t_ms.size)), self.t_ms)
        self.assertIsNotNone(p_l)
        self.assertLess(p_l, 0.05)
        self.assertTrue(session_task_evoked(p_l, None))


if __name__ == "__main__":
    unittest.main()
