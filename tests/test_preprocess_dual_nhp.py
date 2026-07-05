"""Tests for DUAL_NHP trial filters and choice config."""

from __future__ import annotations

import unittest

from bos_mua.preprocess import (
    DUAL_NHP_GO_SEQS,
    dual_nhp_choice_config,
    trial_filters_for_dual_nhp_go_seq,
    trial_filters_for_dual_nhp_monkey,
    trial_filters_for_go_seq,
)


class PreprocessDualNhpTests(unittest.TestCase):
    def test_go_seqs(self) -> None:
        self.assertEqual(DUAL_NHP_GO_SEQS, ("AgoB", "BgoA"))

    def test_trial_filters_for_dual_nhp_go_seq(self) -> None:
        agob = trial_filters_for_dual_nhp_go_seq("AgoB")
        bgoa = trial_filters_for_dual_nhp_go_seq("BgoA")
        self.assertEqual(agob["go_seq_500_list"], ["AgoB"])
        self.assertEqual(bgoa["go_seq_500_list"], ["BgoA"])
        self.assertEqual(agob["TrialSubType_list"], ["Dyadic"])
        self.assertNotIn("conf_predictability_list", agob)

    def test_trial_filters_for_dual_nhp_monkey_accepts_go_seq(self) -> None:
        self.assertEqual(
            trial_filters_for_dual_nhp_monkey("Curius", "BgoA")["go_seq_500_list"],
            ["BgoA"],
        )
        self.assertEqual(
            trial_filters_for_dual_nhp_monkey("Elmo", "AgoB")["go_seq_500_list"],
            ["AgoB"],
        )

    def test_dual_nhp_choice_config(self) -> None:
        curius = dual_nhp_choice_config("Curius")
        elmo = dual_nhp_choice_config("Elmo")
        self.assertEqual(curius.field, "A_LR_pos_list")
        self.assertEqual(curius.left, ["Al"])
        self.assertEqual(elmo.field, "B_LR_pos_list")
        self.assertEqual(elmo.left, ["Bl"])

    def test_trial_filters_for_go_seq_confederate(self) -> None:
        bgoa = trial_filters_for_go_seq("Curius_SHUFFLED", "BgoA")
        self.assertEqual(bgoa["go_seq_500_list"], ["BgoA"])
        self.assertEqual(bgoa["conf_predictability_list"], ["Shuffled"])
        blocked = trial_filters_for_go_seq("Elmo_BLOCKED", "AgoB")
        self.assertEqual(blocked["conf_predictability_list"], ["Blocked"])
        dual = trial_filters_for_go_seq("DUAL_NHP", "AgoB")
        self.assertNotIn("conf_predictability_list", dual)


if __name__ == "__main__":
    unittest.main()
