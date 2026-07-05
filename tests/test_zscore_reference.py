"""Tests for actor-trial z-score reference masks."""

from __future__ import annotations

import unittest

import numpy as np

from bos_mua.preprocess import (
    recording_monkey,
    zscore_channel_trials,
    zscore_reference_mask,
)


class ZscoreReferenceMaskTests(unittest.TestCase):
    def test_curius_actor_mask(self) -> None:
        labels = {
            "TrialSubType_list": np.array(
                ["Dyadic", "SoloARewardAB", "SoloBRewardAB", "None", "SoloA"],
                dtype=object,
            )
        }
        mask = zscore_reference_mask(labels, "Curius")
        np.testing.assert_array_equal(mask, [True, True, False, False, True])

    def test_elmo_actor_mask(self) -> None:
        labels = {
            "TrialSubType_list": np.array(
                ["Dyadic", "SoloARewardAB", "SoloBRewardAB", "SoloA", "SoloB"],
                dtype=object,
            )
        }
        mask = zscore_reference_mask(labels, "Elmo")
        np.testing.assert_array_equal(mask, [True, False, True, False, True])

    def test_zscore_uses_reference_only(self) -> None:
        # Row 0: high dyadic (reference). Row 1: near-zero passive solo (excluded).
        mua = np.array([[10.0, 12.0], [0.0, 0.0], [10.0, 12.0]])
        ref = np.array([True, False, True])
        z = zscore_channel_trials(mua, reference_mask=ref)
        self.assertAlmostEqual(float(np.mean(z[0])), 0.0, places=5)
        self.assertAlmostEqual(float(np.mean(z[2])), 0.0, places=5)
        # Passive row z-scored with same mu/sd as actor reference, not driven to ~0 mean.
        self.assertGreater(abs(float(np.mean(z[1]))), 1.0)

    def test_recording_monkey_dual_nhp_suffix(self) -> None:
        self.assertEqual(
            recording_monkey(session_id="20230630T115937B.A_Curius.B_Elmo.SCP_01"),
            "Elmo",
        )
        self.assertEqual(
            recording_monkey(session_id="20230630T115937.A_Curius.B_Elmo.SCP_01"),
            "Curius",
        )

    def test_recording_monkey_curated_condition(self) -> None:
        self.assertEqual(
            recording_monkey(
                session_id="20210401T124246.A_Elmo.B_KN.SCP_01",
                condition_label="Elmo_BLOCKED",
            ),
            "Elmo",
        )


if __name__ == "__main__":
    unittest.main()
