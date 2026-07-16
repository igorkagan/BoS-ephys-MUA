"""Tests for DUAL_NHP pipeline run iteration and session grouping."""

from __future__ import annotations

import unittest
from pathlib import Path

from run_pipeline.dual_nhp import (
    ALL_DUAL_NHP_RUNS,
    build_dual_nhp_context,
    dual_nhp_run_label,
    iter_dual_nhp_runs,
    resolve_go_seqs,
)
from process_channels.preprocess import choice_config_for_recording
from process_channels.preprocess import recording_monkey_from_session_id
from load_data.sessions import (
    load_dual_nhp_configs,
    split_sessions_by_recording_monkey,
)

REPO = Path(__file__).resolve().parents[2]


class DualNhpPipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.session_lists = REPO / "session_lists.m"

    def test_four_standard_runs(self) -> None:
        self.assertEqual(len(ALL_DUAL_NHP_RUNS), 4)
        labels = {run.label for run in ALL_DUAL_NHP_RUNS}
        self.assertEqual(
            labels,
            {"Curius_AgoB", "Curius_BgoA", "Elmo_AgoB", "Elmo_BgoA"},
        )

    def test_resolve_go_seqs(self) -> None:
        self.assertEqual(resolve_go_seqs("all"), ["AgoB", "BgoA"])
        self.assertEqual(resolve_go_seqs("BgoA"), ["BgoA"])

    def test_iter_all_dual_nhp_runs(self) -> None:
        contexts = list(iter_dual_nhp_runs(self.session_lists))
        self.assertEqual(len(contexts), 4)
        self.assertEqual(
            sorted(c.condition_label for c in contexts),
            ["Curius_AgoB", "Curius_BgoA", "Elmo_AgoB", "Elmo_BgoA"],
        )

    def test_curius_bgoa_uses_a_lr(self) -> None:
        ctx = next(
            c for c in iter_dual_nhp_runs(self.session_lists)
            if c.condition_label == "Curius_BgoA"
        )
        self.assertEqual(ctx.trial_filters["go_seq_500_list"], ["BgoA"])
        self.assertEqual(ctx.choice_field, "A_LR_pos_list")
        self.assertEqual(ctx.left_choice, ["Al"])
        self.assertEqual(ctx.recording_monkey, "Curius")

    def test_elmo_agob_uses_b_lr(self) -> None:
        ctx = next(
            c for c in iter_dual_nhp_runs(self.session_lists)
            if c.condition_label == "Elmo_AgoB"
        )
        self.assertEqual(ctx.trial_filters["go_seq_500_list"], ["AgoB"])
        self.assertEqual(ctx.choice_field, "B_LR_pos_list")
        self.assertEqual(ctx.trial_filters["B_Reward_list"], ["RB1", "RB2", "RB3", "RB4"])
        self.assertEqual(ctx.left_choice, ["Bl"])
        self.assertEqual(ctx.recording_monkey, "Elmo")

    def test_single_monkey_filter(self) -> None:
        contexts = list(iter_dual_nhp_runs(self.session_lists, monkey="Elmo"))
        self.assertEqual(len(contexts), 2)
        self.assertTrue(all(c.recording_monkey == "Elmo" for c in contexts))

    def test_paired_and_confederate_session_assignment(self) -> None:
        _, split = load_dual_nhp_configs(self.session_lists)
        self.assertEqual(len(split["Curius"]), 4)
        self.assertEqual(len(split["Elmo"]), 4)
        self.assertTrue(
            all(
                recording_monkey_from_session_id(s) == "Curius"
                for s in split["Curius"]
            )
        )
        confederate = split_sessions_by_recording_monkey(
            ["20210401T124246.A_Elmo.B_KN.SCP_01"]
        )
        self.assertEqual(confederate["Elmo"], ["20210401T124246.A_Elmo.B_KN.SCP_01"])
        self.assertEqual(confederate["Curius"], [])

    def test_build_context_output_path(self) -> None:
        cfg, split = load_dual_nhp_configs(self.session_lists)
        ctx = build_dual_nhp_context(cfg, "Curius", "AgoB", split["Curius"])
        self.assertEqual(ctx.condition_label, dual_nhp_run_label("Curius", "AgoB"))
        self.assertEqual(ctx.condition_key, "DUAL_NHP")
        self.assertEqual(ctx.source_kind, "session_list")
        self.assertEqual(
            choice_config_for_recording(split["Curius"][0], "Curius").field,
            ctx.choice_field,
        )
        self.assertEqual(ctx.trial_filters["A_Reward_list"], ["RA1", "RA2", "RA3", "RA4"])


if __name__ == "__main__":
    unittest.main()
