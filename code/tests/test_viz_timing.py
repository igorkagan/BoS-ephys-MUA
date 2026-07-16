"""Tests for compare_conditions.plots_comparison layout helpers."""

from __future__ import annotations

import tempfile
import unittest
import inspect
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from analyze_stability.arrays import array_combined_stats
from compare_conditions.plots_comparison import (
    SESSION_ROW_HEIGHT_IN,
    _timing_deep_dive_subplot_grid,
    add_timing_trace_legend,
    plot_delta_si_heatmap,
    plot_si_scatter,
    plot_timing_overlay_subplot,
    validate_figure_legends_inside_canvas,
    plot_waveform_r_heatmap,
    plot_arrays_timing_combined,
    plot_timing_combined_outputs,
)


def _trial_parts(left_val: float, right_val: float, n_trials: int = 2) -> tuple[list[np.ndarray], list[np.ndarray]]:
    t_len = 5
    left = [np.full(t_len, left_val, dtype=float) for _ in range(n_trials)]
    right = [np.full(t_len, right_val, dtype=float) for _ in range(n_trials)]
    return left, right


class TimingDeepDiveSubplotGridTests(unittest.TestCase):
    def test_generic_plot_helpers_require_explicit_labels(self) -> None:
        for func in (plot_timing_overlay_subplot, plot_si_scatter):
            signature = inspect.signature(func)
            self.assertIs(signature.parameters["label_a"].default, inspect.Parameter.empty)
            self.assertIs(signature.parameters["label_b"].default, inspect.Parameter.empty)
        for func in (plot_delta_si_heatmap, plot_waveform_r_heatmap):
            signature = inspect.signature(func)
            self.assertIs(
                signature.parameters["colorbar_label"].default,
                inspect.Parameter.empty,
            )

    def test_bottom_legend_uses_explicit_labels_inside_figure(self) -> None:
        fig = plt.figure()
        try:
            add_timing_trace_legend(fig, "Dyadic", "SoloA")
            legend = fig.legends[0]
            self.assertEqual(
                [text.get_text() for text in legend.get_texts()],
                ["Dyadic L", "Dyadic R", "SoloA L", "SoloA R"],
            )
            self.assertGreaterEqual(legend.get_bbox_to_anchor().y0, 0)
            validate_figure_legends_inside_canvas(fig)
        finally:
            plt.close(fig)

    def test_single_row_up_to_ten(self) -> None:
        for n in (1, 5, 10):
            nrows, ncols, _ = _timing_deep_dive_subplot_grid(n)
            self.assertEqual((nrows, ncols), (1, n))

    def test_fifteen_sessions_three_by_five(self) -> None:
        nrows, ncols, figsize = _timing_deep_dive_subplot_grid(15)
        self.assertEqual((nrows, ncols), (3, 5))
        self.assertGreaterEqual(figsize[1], SESSION_ROW_HEIGHT_IN * nrows)

    def test_twenty_sessions_two_by_ten(self) -> None:
        nrows, ncols, _ = _timing_deep_dive_subplot_grid(20)
        self.assertEqual((nrows, ncols), (2, 10))

    def test_thirty_sessions_three_by_ten(self) -> None:
        nrows, ncols, _ = _timing_deep_dive_subplot_grid(30)
        self.assertEqual((nrows, ncols), (3, 10))

    def test_never_more_than_ten_columns(self) -> None:
        for n in range(1, 51):
            _nrows, ncols, _ = _timing_deep_dive_subplot_grid(n)
            self.assertLessEqual(ncols, 10)


class ArrayTimingCombinedTests(unittest.TestCase):
    def test_array_combined_stats_pools_channels(self) -> None:
        left_a, right_a = _trial_parts(1.0, 0.0)
        left_b, right_b = _trial_parts(0.0, 2.0)
        left_by_ch_a = {1: left_a, 2: [np.full(5, 3.0)]}
        right_by_ch_a = {1: right_a}
        left_by_ch_b = {1: left_b}
        right_by_ch_b = {1: right_b, 2: [np.full(5, 4.0)]}
        stats_a = array_combined_stats(0, left_by_ch_a, right_by_ch_a)
        stats_b = array_combined_stats(0, left_by_ch_b, right_by_ch_b)
        a_l_mean, _, a_r_mean, _, n_a_l, n_a_r = stats_a
        b_l_mean, _, b_r_mean, _, n_b_l, n_b_r = stats_b
        self.assertEqual(n_a_l, 2)
        self.assertEqual(n_a_r, 1)
        self.assertEqual(n_b_l, 1)
        self.assertEqual(n_b_r, 2)
        assert a_l_mean is not None and b_r_mean is not None
        np.testing.assert_allclose(a_l_mean, 2.0)
        np.testing.assert_allclose(b_r_mean, 3.0)
        assert a_r_mean is None or np.allclose(a_r_mean, 0.0)
        assert b_l_mean is None or np.allclose(b_l_mean, 0.0)

    def test_plot_timing_combined_outputs_writes_pdfs(self) -> None:
        t_ms = np.linspace(-100, 100, 5)
        win_idx = np.arange(len(t_ms))
        left_a, right_a = _trial_parts(1.0, 0.0)
        left_b, right_b = _trial_parts(0.0, 1.0)
        left_by_ch_a = {1: left_a}
        right_by_ch_a = {1: right_a}
        left_by_ch_b = {1: left_b}
        right_by_ch_b = {1: right_b}
        with tempfile.TemporaryDirectory() as tmpdir:
            combined_dir = Path(tmpdir) / "combined"
            plot_timing_combined_outputs(
                left_by_ch_a,
                right_by_ch_a,
                left_by_ch_b,
                right_by_ch_b,
                t_ms,
                win_idx,
                combined_dir,
                file_tag="AgoB_vs_BgoA",
                label_a="AgoB",
                label_b="BgoA",
                suptitle_channel_grids="test channel grids",
                suptitle_arrays="test arrays mean",
                zscore_mua=True,
            )
            summary_pdf = combined_dir / "arrays_AgoB_vs_BgoA_combined.pdf"
            self.assertTrue(summary_pdf.exists())
            self.assertGreater(summary_pdf.stat().st_size, 0)
            array_pdfs = list(combined_dir.glob("AgoB_vs_BgoA_A*_combined.pdf"))
            self.assertEqual(len(array_pdfs), 5)

    def test_plot_arrays_timing_combined_writes_pdf(self) -> None:
        t_ms = np.linspace(-100, 100, 5)
        win_idx = np.arange(len(t_ms))
        left_a, right_a = _trial_parts(1.0, 0.0)
        left_b, right_b = _trial_parts(0.0, 1.0)
        with tempfile.TemporaryDirectory() as tmpdir:
            out_path = Path(tmpdir) / "arrays_Dyadic_vs_SoloA_combined.pdf"
            plot_arrays_timing_combined(
                {1: left_a},
                {1: right_a},
                {1: left_b},
                {1: right_b},
                t_ms,
                win_idx,
                "test",
                out_path,
                label_a="Dyadic",
                label_b="SoloA",
                zscore_mua=True,
            )
            self.assertTrue(out_path.exists())
            self.assertGreater(out_path.stat().st_size, 0)


if __name__ == "__main__":
    unittest.main()
