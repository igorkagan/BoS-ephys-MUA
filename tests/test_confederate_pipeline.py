"""Tests for confederate session-list go-seq pipeline."""

from __future__ import annotations

import unittest
from pathlib import Path

from bos_mua.confederate import build_confederate_context, iter_confederate_runs
from bos_mua.session_lists import (
    is_confederate_list,
    load_session_list,
)
from compare_monkey_timing_conditions import load_timing_run

REPO = Path(__file__).resolve().parents[1]


class ConfederatePipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.session_lists = REPO / "session_lists.m"

    def test_is_confederate_list(self) -> None:
        self.assertTrue(is_confederate_list("Curius_SHUFFLED_CONF"))
        self.assertTrue(is_confederate_list("Elmo_BLOCKED_CONF"))
        self.assertFalse(is_confederate_list("DUAL_NHP"))
        self.assertFalse(is_confederate_list("ElmoBLOCKED_SpikeSortedSessions"))

    def test_iter_curius_shuffled_conf_runs(self) -> None:
        cfg = load_session_list("Curius_SHUFFLED_CONF", self.session_lists)
        contexts = list(iter_confederate_runs(cfg))
        self.assertEqual(len(contexts), 2)
        self.assertEqual(
            sorted(c.condition_label for c in contexts),
            ["Curius_AgoB", "Curius_BgoA"],
        )

    def test_curius_bgoa_filters(self) -> None:
        cfg = load_session_list("Curius_SHUFFLED_CONF", self.session_lists)
        ctx = build_confederate_context(cfg, "BgoA")
        self.assertEqual(ctx.trial_filters["go_seq_500_list"], ["BgoA"])
        self.assertEqual(ctx.trial_filters["conf_predictability_list"], ["Shuffled"])
        self.assertEqual(ctx.choice_field, "A_LR_pos_list")
        self.assertIn("Curius_SHUFFLED_CONF", str(ctx.output_base))
        self.assertTrue(str(ctx.output_base).endswith("Curius_BgoA"))

    def test_load_timing_run_confederate(self) -> None:
        cfg, session_ids, condition_key = load_timing_run(
            self.session_lists, "Curius_SHUFFLED_CONF", "Curius",
        )
        self.assertEqual(condition_key, "Curius_SHUFFLED")
        self.assertEqual(len(session_ids), 15)
        self.assertEqual(cfg.list_name, "Curius_SHUFFLED_CONF")


if __name__ == "__main__":
    unittest.main()
