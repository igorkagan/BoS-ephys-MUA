"""Both-sides trial minima for PSTH extract, session plots, and decode."""

from __future__ import annotations

import unittest
from pathlib import Path
from unittest import mock

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from analyze_decoding.config import MIN_TRIALS_PER_CONDITION
from analyze_decoding.session_decode import enough_trials_per_side
from analyze_stability.plots_lr import make_session_array_figure
from process_channels.features import process_channel_mua


class EnoughTrialsPerSideTests(unittest.TestCase):
    def test_decode_min_needs_both_sides(self) -> None:
        self.assertTrue(enough_trials_per_side(5, 5))
        self.assertFalse(enough_trials_per_side(4, 5))
        self.assertFalse(enough_trials_per_side(5, 4))
        self.assertFalse(
            enough_trials_per_side(3, 5, min_trials=MIN_TRIALS_PER_CONDITION)
        )


class ProcessChannelMinTrialsTests(unittest.TestCase):
    def _call(self, n_left: int, n_right: int, n_other: int = 2):
        n_trials = n_left + n_right + n_other
        rng = np.random.default_rng(0)
        mua = rng.random((n_trials, 21))
        t_ms = np.linspace(-1000, 1000, 21)
        left = np.zeros(n_trials, dtype=bool)
        right = np.zeros(n_trials, dtype=bool)
        left[:n_left] = True
        right[n_left : n_left + n_right] = True
        return process_channel_mua(
            mua,
            ch_num=1,
            session_id="sess",
            t_ms=t_ms,
            win_idx=np.arange(21),
            left_mask=left,
            right_mask=right,
            gaussian_smooth_ms=0.0,
            min_trials=3,
        )

    def test_drops_when_one_side_below_min(self) -> None:
        self.assertIsNone(self._call(2, 6))
        self.assertIsNone(self._call(6, 2))
        self.assertIsNone(self._call(0, 8))

    def test_keeps_when_both_sides_meet_min(self) -> None:
        result = self._call(3, 3)
        self.assertIsNotNone(result)
        self.assertEqual(result.summary.n_left, 3)
        self.assertEqual(result.summary.n_right, 3)


class SessionPlotCacheTests(unittest.TestCase):
    def tearDown(self) -> None:
        plt.close("all")

    def test_missing_cached_channel_does_not_reload_mat(self) -> None:
        t_ms = np.linspace(-100, 100, 5)
        with mock.patch("analyze_stability.plots_lr.load_smoothed_lr_trials") as load:
            fig = make_session_array_figure(
                0,
                {1: Path("fake.mat")},
                t_ms,
                np.arange(5),
                np.array([True, False]),
                np.array([False, True]),
                "t",
                gaussian_smooth_ms=0.0,
                zscore_mua=True,
                cached_trials={},
                subplot_grid=(1, 1),
            )
        load.assert_not_called()
        self.assertIsNotNone(fig)

    def test_one_sided_cached_channel_is_empty(self) -> None:
        t_ms = np.linspace(-100, 100, 5)
        left = np.ones((2, 5))
        right = np.ones((6, 5))
        with mock.patch("analyze_stability.plots_lr.plot_channel_subplot") as plot:
            make_session_array_figure(
                0,
                {1: Path("fake.mat")},
                t_ms,
                np.arange(5),
                np.array([True, False]),
                np.array([False, True]),
                "t",
                gaussian_smooth_ms=0.0,
                zscore_mua=True,
                cached_trials={1: (left, right)},
                subplot_grid=(1, 1),
                min_trials=3,
            )
        plot.assert_not_called()


if __name__ == "__main__":
    unittest.main()
