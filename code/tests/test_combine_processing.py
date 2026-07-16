"""Processing-mode contracts for pooled comparison trials."""

from __future__ import annotations

import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from analyze_stability.combine import load_channel_lr_trials


class CombinedTrialProcessingTests(unittest.TestCase):
    def test_raw_mode_does_not_zscore(self) -> None:
        mua = np.asarray(
            [
                [10.0, 20.0],
                [30.0, 40.0],
                [50.0, 60.0],
            ]
        )
        with (
            patch("analyze_stability.combine.loadmat", return_value={"cur_output_data": mua}),
            patch(
                "analyze_stability.combine.gaussian_smooth_trials",
                side_effect=lambda trials, *_args, **_kwargs: trials,
            ),
            patch("analyze_stability.combine.zscore_channel_trials") as zscore,
        ):
            left, right = load_channel_lr_trials(
                Path("fake.mat"),
                np.asarray([True, False, False]),
                np.asarray([False, True, False]),
                np.asarray([0.0, 1.0]),
                zscore_mua=False,
            )
        zscore.assert_not_called()
        np.testing.assert_array_equal(left, mua[[0]])
        np.testing.assert_array_equal(right, mua[[1]])

    def test_zscore_mode_uses_reference_mask(self) -> None:
        mua = np.arange(6, dtype=float).reshape(3, 2)
        transformed = mua + 100.0
        reference = np.asarray([True, True, False])
        with (
            patch("analyze_stability.combine.loadmat", return_value={"cur_output_data": mua}),
            patch(
                "analyze_stability.combine.gaussian_smooth_trials",
                side_effect=lambda trials, *_args, **_kwargs: trials,
            ),
            patch(
                "analyze_stability.combine.zscore_channel_trials",
                return_value=transformed,
            ) as zscore,
        ):
            left, _ = load_channel_lr_trials(
                Path("fake.mat"),
                np.asarray([True, False, False]),
                np.asarray([False, True, False]),
                np.asarray([0.0, 1.0]),
                zscore_reference=reference,
                zscore_mua=True,
            )
        zscore.assert_called_once()
        np.testing.assert_array_equal(
            zscore.call_args.kwargs["reference_mask"], reference,
        )
        np.testing.assert_array_equal(left, transformed[[0]])


if __name__ == "__main__":
    unittest.main()
