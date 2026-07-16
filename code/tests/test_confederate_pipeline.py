"""Tests for confederate session-list go-seq pipeline."""

from __future__ import annotations

import unittest
from pathlib import Path

from run_pipeline.confederate import build_confederate_context, iter_confederate_runs
from load_data.sessions import (
    is_confederate_list,
    load_session_list,
)
from compare_conditions.compare import load_timing_run
from run_pipeline.curated import build_curated_context

REPO = Path(__file__).resolve().parents[2]


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
        self.assertEqual(ctx.source_kind, "session_list")
        self.assertEqual(ctx.condition_key, "Curius_SHUFFLED")
        self.assertIsNone(ctx.session_parent)
        self.assertEqual(ctx.session_dir(ctx.session_ids[0]), cfg.root_folder / ctx.session_ids[0])
        self.assertIn("Curius_SHUFFLED_CONF", str(ctx.output_base))
        self.assertTrue(str(ctx.output_base).endswith("Curius_BgoA"))

    def test_elmo_blocked_conf_uses_actor_a_lr(self) -> None:
        cfg = load_session_list("Elmo_BLOCKED_CONF", self.session_lists)
        ctx = build_confederate_context(cfg, "AgoB")
        self.assertEqual(ctx.choice_field, "A_LR_pos_list")
        self.assertEqual(ctx.left_choice, ["Al"])
        self.assertEqual(ctx.trial_filters["A_Reward_list"], ["RA1", "RA2", "RA3", "RA4"])

    def test_load_timing_run_confederate(self) -> None:
        cfg, session_ids, condition_key = load_timing_run(
            self.session_lists, "Curius_SHUFFLED_CONF", "Curius",
        )
        self.assertEqual(condition_key, "Curius_SHUFFLED")
        self.assertEqual(len(session_ids), 15)
        self.assertEqual(cfg.list_name, "Curius_SHUFFLED_CONF")

    def test_curated_and_export_contexts_share_filters_for_same_condition(self) -> None:
        cfg = load_session_list("Elmo_BLOCKED_CONF", self.session_lists)
        export = build_confederate_context(cfg, "AgoB")
        curated = build_curated_context(
            "Elmo_BLOCKED",
            "AgoB",
            [cfg.session_ids[0]],
            data_root=Path("/curated"),
            output_base=Path("/figures"),
        )
        self.assertEqual(export.condition_key, curated.condition_key)
        self.assertEqual(export.trial_filters, curated.trial_filters)
        self.assertEqual(export.choice_field, curated.choice_field)
        self.assertEqual(export.left_choice, curated.left_choice)
        self.assertNotEqual(export.session_dir(cfg.session_ids[0]), curated.session_dir(cfg.session_ids[0]))


if __name__ == "__main__":
    unittest.main()
