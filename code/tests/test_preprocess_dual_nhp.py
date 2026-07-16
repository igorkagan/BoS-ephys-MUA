"""Tests for DUAL_NHP trial filters and choice config."""

from __future__ import annotations

import unittest
from pathlib import Path

from process_channels.preprocess import (
    DUAL_NHP_GO_SEQS,
    alignment_event_for_actor_side,
    alignment_event_for_recording,
    branch_specs_for_run,
    choice_config_for_actor_side,
    choice_config_for_recording,
    recording_actor_side,
    solo_output_subdir_for_actor_side,
    trial_filters_for_dual_nhp_go_seq,
    trial_filters_for_dual_nhp_monkey,
    trial_filters_for_go_seq,
    trial_filters_for_solo_from_dyadic,
    trial_filters_for_solo_go_seq,
)
from run_pipeline.context import PipelineContext
from run_pipeline.curated import build_curated_context

CURIUS_DUAL = "20230630T115937.A_Curius.B_Elmo.SCP_01"
ELMO_DUAL_B = "20230630T115937B.A_Curius.B_Elmo.SCP_01"
ELMO_A_Elmo = "20201204T125624.A_Elmo.B_FS.SCP_01"


class PreprocessDualNhpTests(unittest.TestCase):
    def test_go_seqs(self) -> None:
        self.assertEqual(DUAL_NHP_GO_SEQS, ("AgoB", "BgoA"))

    def test_trial_filters_for_dual_nhp_go_seq(self) -> None:
        agob = trial_filters_for_dual_nhp_go_seq("AgoB", "A")
        bgoa = trial_filters_for_dual_nhp_go_seq("BgoA", "B")
        self.assertEqual(agob["go_seq_500_list"], ["AgoB"])
        self.assertEqual(bgoa["go_seq_500_list"], ["BgoA"])
        self.assertEqual(agob["TrialSubType_list"], ["Dyadic"])
        self.assertEqual(agob["A_Reward_list"], ["RA1", "RA2", "RA3", "RA4"])
        self.assertEqual(bgoa["B_Reward_list"], ["RB1", "RB2", "RB3", "RB4"])
        self.assertNotIn("conf_predictability_list", agob)

    def test_trial_filters_for_dual_nhp_monkey_accepts_go_seq(self) -> None:
        self.assertEqual(
            trial_filters_for_dual_nhp_monkey(
                "Curius", "BgoA", session_id=CURIUS_DUAL,
            )["go_seq_500_list"],
            ["BgoA"],
        )
        self.assertEqual(
            trial_filters_for_dual_nhp_monkey(
                "Elmo", "AgoB", session_id=ELMO_DUAL_B,
            )["go_seq_500_list"],
            ["AgoB"],
        )
        self.assertIn(
            "B_Reward_list",
            trial_filters_for_dual_nhp_monkey("Elmo", "AgoB", session_id=ELMO_DUAL_B),
        )

    def test_choice_config_from_actor_side(self) -> None:
        curius = choice_config_for_actor_side("A")
        elmo_b = choice_config_for_actor_side("B")
        self.assertEqual(curius.field, "A_LR_pos_list")
        self.assertEqual(curius.left, ["Al"])
        self.assertEqual(elmo_b.field, "B_LR_pos_list")
        self.assertEqual(elmo_b.left, ["Bl"])

    def test_choice_config_for_recording(self) -> None:
        self.assertEqual(
            choice_config_for_recording(CURIUS_DUAL, "Curius").field,
            "A_LR_pos_list",
        )
        self.assertEqual(
            choice_config_for_recording(ELMO_DUAL_B, "Elmo").field,
            "B_LR_pos_list",
        )
        self.assertEqual(
            choice_config_for_recording(ELMO_A_Elmo, "Elmo").field,
            "A_LR_pos_list",
        )

    def test_choice_config_for_recording(self) -> None:
        curius = choice_config_for_recording(CURIUS_DUAL, "Curius")
        elmo = choice_config_for_recording(ELMO_DUAL_B, "Elmo")
        self.assertEqual(curius.field, "A_LR_pos_list")
        self.assertEqual(elmo.field, "B_LR_pos_list")

    def test_trial_filters_for_go_seq_confederate(self) -> None:
        bgoa = trial_filters_for_go_seq("Curius_SHUFFLED", "BgoA", "A")
        self.assertEqual(bgoa["go_seq_500_list"], ["BgoA"])
        self.assertEqual(bgoa["conf_predictability_list"], ["Shuffled"])
        self.assertEqual(bgoa["A_Reward_list"], ["RA1", "RA2", "RA3", "RA4"])
        blocked = trial_filters_for_go_seq("Elmo_BLOCKED", "AgoB", "A")
        self.assertEqual(blocked["conf_predictability_list"], ["Blocked"])
        self.assertEqual(blocked["A_Reward_list"], ["RA1", "RA2", "RA3", "RA4"])
        dual = trial_filters_for_go_seq("DUAL_NHP", "AgoB", "B")
        self.assertNotIn("conf_predictability_list", dual)
        self.assertEqual(dual["B_Reward_list"], ["RB1", "RB2", "RB3", "RB4"])

    def test_curated_elmo_context_uses_actor_side(self) -> None:
        ctx = build_curated_context(
            "Elmo_BLOCKED",
            "AgoB",
            [ELMO_A_Elmo],
            data_root=Path("/data"),
            output_base=Path("/out"),
        )
        self.assertEqual(ctx.choice_field, "A_LR_pos_list")
        self.assertEqual(ctx.left_choice, ["Al"])
        self.assertEqual(ctx.source_kind, "curated")
        self.assertEqual(ctx.layout, "curated")
        self.assertEqual(ctx.session_parent, "Elmo_BLOCKED")
        self.assertEqual(ctx.session_dir(ELMO_A_Elmo), Path("/data/Elmo_BLOCKED") / ELMO_A_Elmo)
        self.assertEqual(ctx.trial_filters["A_Reward_list"], ["RA1", "RA2", "RA3", "RA4"])
        self.assertNotIn("B_Reward_list", ctx.trial_filters)

    def test_recording_actor_side(self) -> None:
        self.assertEqual(recording_actor_side(CURIUS_DUAL, "Curius"), "A")
        self.assertEqual(recording_actor_side(ELMO_DUAL_B, "Elmo"), "B")
        self.assertEqual(recording_actor_side(ELMO_A_Elmo, "Elmo"), "A")

    def test_alignment_event_for_actor_side(self) -> None:
        self.assertEqual(
            alignment_event_for_actor_side("A"),
            "A_InitialFixationReleaseTime_ms",
        )
        self.assertEqual(
            alignment_event_for_actor_side("B"),
            "B_InitialFixationReleaseTime_ms",
        )

    def test_alignment_event_for_recording(self) -> None:
        self.assertEqual(
            alignment_event_for_recording(CURIUS_DUAL, "Curius"),
            "A_InitialFixationReleaseTime_ms",
        )
        self.assertEqual(
            alignment_event_for_recording(ELMO_DUAL_B, "Elmo"),
            "B_InitialFixationReleaseTime_ms",
        )
        self.assertEqual(
            alignment_event_for_recording(ELMO_A_Elmo, "Elmo"),
            "A_InitialFixationReleaseTime_ms",
        )

    def test_pipeline_context_alignment_event(self) -> None:
        ctx = build_curated_context(
            "Elmo_BLOCKED",
            "AgoB",
            [ELMO_A_Elmo, ELMO_DUAL_B],
            data_root=Path("/data"),
            output_base=Path("/out"),
        )
        self.assertEqual(
            ctx.alignment_event(ELMO_A_Elmo),
            "A_InitialFixationReleaseTime_ms",
        )
        self.assertEqual(
            ctx.alignment_event(ELMO_DUAL_B),
            "B_InitialFixationReleaseTime_ms",
        )
        self.assertEqual(
            ctx.alignment_event_label([ELMO_A_Elmo]),
            "A_InitialFixationReleaseTime_ms",
        )
        self.assertEqual(
            ctx.alignment_event_label([ELMO_A_Elmo, ELMO_DUAL_B]),
            "actor-side fixation release",
        )

    def test_trial_filters_for_solo_go_seq(self) -> None:
        curius_a = trial_filters_for_solo_go_seq("Curius_SHUFFLED", "AgoB", "A")
        self.assertEqual(curius_a["TrialSubType_list"], ["SoloARewardAB", "SoloA"])
        self.assertEqual(curius_a["A_Reward_list"], ["RA1", "RA2", "RA3", "RA4"])
        self.assertEqual(curius_a["go_seq_500_list"], ["AgoB"])
        self.assertNotIn("B_Reward_list", curius_a)
        self.assertNotIn("conf_predictability_list", curius_a)

        elmo_a = trial_filters_for_solo_go_seq("Elmo_BLOCKED", "BgoA", "A")
        self.assertEqual(elmo_a["TrialSubType_list"], ["SoloARewardAB", "SoloA"])
        self.assertEqual(elmo_a["A_Reward_list"], ["RA1", "RA2", "RA3", "RA4"])
        self.assertEqual(elmo_a["go_seq_500_list"], ["BgoA"])
        self.assertNotIn("conf_predictability_list", elmo_a)

        elmo_b = trial_filters_for_solo_go_seq("DUAL_NHP", "BgoA", "B")
        self.assertEqual(elmo_b["TrialSubType_list"], ["SoloBRewardAB", "SoloB"])
        self.assertEqual(elmo_b["B_Reward_list"], ["RB1", "RB2", "RB3", "RB4"])
        self.assertEqual(elmo_b["go_seq_500_list"], ["BgoA"])

    def test_solo_filters_keep_go_seq_drop_blocked(self) -> None:
        import numpy as np
        from load_data.io import build_base_mask, choice_mask

        labels = {
            "TrialSubType_list": np.array(
                ["Dyadic", "SoloA", "SoloA", "SoloA", "SoloA"],
                dtype=object,
            ),
            "go_seq_500_list": np.array(["AgoB", "AgoB", "BgoA", "AgoB", "AgoB"], dtype=object),
            "conf_predictability_list": np.array(
                ["Blocked", "Free", "Free", "Free", "Free"],
                dtype=object,
            ),
            "A_Reward_list": np.array(["RA2", "RA2", "RA2", "RA0", "RA2"], dtype=object),
            "A_LR_pos_list": np.array(["Al", "Al", "Ar", "Al", "Ar"], dtype=object),
        }
        dyadic = trial_filters_for_go_seq("Elmo_BLOCKED", "AgoB", "A")
        solo = trial_filters_for_solo_from_dyadic(dyadic, "A")
        self.assertEqual(solo["go_seq_500_list"], ["AgoB"])
        self.assertNotIn("conf_predictability_list", solo)
        base = build_base_mask(labels, solo)
        self.assertEqual(int(base.sum()), 2)  # AgoB solo with RA2, not RA0
        self.assertEqual(
            int(choice_mask(labels, base, ["Al"], field="A_LR_pos_list").sum()),
            1,
        )
        self.assertEqual(
            int(choice_mask(labels, base, ["Ar"], field="A_LR_pos_list").sum()),
            1,
        )

    def test_branch_specs_for_run(self) -> None:
        ctx = PipelineContext(
            data_root=Path("/data"),
            session_ids=[CURIUS_DUAL],
            output_base=Path("/out/Curius_AgoB"),
            condition_label="Curius_AgoB",
            condition_key="Curius_AgoB",
            layout="flat",
            trial_filters=trial_filters_for_go_seq("Curius_SHUFFLED", "AgoB", "A"),
            recording_monkey="Curius",
        )
        specs = branch_specs_for_run(ctx)
        self.assertEqual(len(specs), 2)
        self.assertEqual(specs[0].name, "dyadic")
        self.assertEqual(specs[0].output_subdir, "Dyadic")
        self.assertEqual(specs[1].output_subdir, "SoloA")
        self.assertEqual(specs[1].trial_filters["TrialSubType_list"], ["SoloARewardAB", "SoloA"])

        elmo_ctx = PipelineContext(
            data_root=Path("/data"),
            session_ids=[ELMO_A_Elmo],
            output_base=Path("/out/Elmo_AgoB"),
            condition_label="Elmo_AgoB",
            condition_key="Elmo_AgoB",
            layout="flat",
            trial_filters=trial_filters_for_go_seq("Elmo_BLOCKED", "AgoB", "A"),
            recording_monkey="Elmo",
        )
        elmo_specs = branch_specs_for_run(elmo_ctx)
        self.assertEqual(elmo_specs[1].output_subdir, "SoloA")
        self.assertEqual(elmo_specs[1].trial_filters["TrialSubType_list"], ["SoloARewardAB", "SoloA"])
        self.assertEqual(elmo_specs[1].trial_filters["A_Reward_list"], ["RA1", "RA2", "RA3", "RA4"])

        dyadic_only = branch_specs_for_run(ctx, include_solo=False)
        self.assertEqual(len(dyadic_only), 1)


if __name__ == "__main__":
    unittest.main()
