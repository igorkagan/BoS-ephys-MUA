"""Tests for analyze_stability.plots_consistency helpers."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from analyze_stability.plots_consistency import (
    _deep_dive_subplot_grid,
    plot_deep_dive_channel,
    write_stability_csv,
)
from process_channels.features import ChannelSummary


def _summary(session_id: str, channel: int, *, left: float, right: float) -> ChannelSummary:
    t_ms = np.linspace(-100, 100, 5)
    mean_left = np.full_like(t_ms, left, dtype=float)
    mean_right = np.full_like(t_ms, right, dtype=float)
    diff = mean_left - mean_right
    return ChannelSummary(
        session_id=session_id,
        channel=channel,
        array_name="A1",
        index_in_array=channel,
        n_left=3,
        n_right=3,
        t_ms=t_ms,
        mean_left=mean_left,
        mean_right=mean_right,
        diff=diff,
        si=0.1,
        mwu_p=0.5,
        pref_side="L",
        evoked_p_left=0.01,
        evoked_p_right=0.2,
        task_evoked=True,
    )


class DeepDiveSubplotGridTests(unittest.TestCase):
    def test_ten_sessions_two_rows(self) -> None:
        nrows, ncols, _ = _deep_dive_subplot_grid(10)
        self.assertEqual((nrows, ncols), (2, 5))

    def test_twelve_sessions_three_rows(self) -> None:
        nrows, ncols, figsize = _deep_dive_subplot_grid(12)
        self.assertEqual((nrows, ncols), (3, 5))
        self.assertEqual(figsize[1], 7.5)

    def test_fifteen_sessions_three_rows(self) -> None:
        nrows, ncols, _ = _deep_dive_subplot_grid(15)
        self.assertEqual((nrows, ncols), (3, 5))


class DeepDiveChannelPlotTests(unittest.TestCase):
    def test_first_missing_session_keeps_y_ticks_on_column_zero(self) -> None:
        session_ids = ["s1", "s2"]
        lookup = {
            "s2": {1: _summary("s2", 1, left=1.0, right=0.0)},
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            out_path = Path(tmpdir) / "best10_ch001.pdf"
            plot_deep_dive_channel(
                lookup,
                session_ids,
                1,
                np.arange(5),
                "test",
                out_path,
            )
            self.assertTrue(out_path.exists())


class WriteStabilityCsvTests(unittest.TestCase):
    def test_empty_stabilities_still_writes_header(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            out_path = Path(tmpdir) / "channel_stability.csv"
            write_stability_csv([], out_path)
            self.assertTrue(out_path.is_file())
            text = out_path.read_text(encoding="utf-8")
            self.assertIn("channel", text)
            self.assertIn("stable", text)
            self.assertGreaterEqual(len(text.strip().splitlines()), 1)


if __name__ == "__main__":
    unittest.main()
