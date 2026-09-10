"""Session-list (flat export) decode path + filter key."""

from __future__ import annotations

import unittest
from pathlib import Path

from analyze_decoding.data import decode_session_dir, filter_condition_key, trial_filters_for_branch
from analyze_decoding.paths import decoding_dir, timing_branch_dir
from load_data.sessions import load_session_list
from process_channels.preprocess import (
    alignment_event_for_actor_side,
    recording_actor_side,
    recording_monkey_for_decode,
    recording_monkey_from_session_id,
    solo_output_subdir_for_actor_side,
)
from run_scripts.run_decode_session_list import actor_trial_types, iter_decode_jobs, parse_args

REPO = Path(__file__).resolve().parents[2]
CURIUS_DUAL = "20230630T115937.A_Curius.B_Elmo.SCP_01"
ELMO_DUAL_B = "20230630T115937B.A_Curius.B_Elmo.SCP_01"


class DecodeSessionListTests(unittest.TestCase):
    def test_filter_key_maps_conf_list_to_blocked(self) -> None:
        self.assertEqual(filter_condition_key("Elmo_BLOCKED_CONF"), "Elmo_BLOCKED")
        self.assertEqual(filter_condition_key("Elmo_BLOCKED"), "Elmo_BLOCKED")
        self.assertEqual(filter_condition_key("DUAL_NHP"), "DUAL_NHP")

    def test_conf_list_gets_blocked_predictability_filter(self) -> None:
        filt = trial_filters_for_branch("Elmo_BLOCKED_CONF", "AgoB", "Dyadic", "A")
        self.assertEqual(filt["conf_predictability_list"], ["Blocked"])
        self.assertEqual(filt["go_seq_500_list"], ["AgoB"])

    def test_session_dir_curated_vs_flat(self) -> None:
        root = Path("/data")
        sid = "20210401T124246.A_Elmo.B_KN.SCP_01"
        self.assertEqual(
            decode_session_dir(root, sid, "Elmo_BLOCKED"),
            root / "Elmo_BLOCKED" / sid,
        )
        self.assertEqual(
            decode_session_dir(root, sid, "Elmo_BLOCKED_CONF", session_parent=""),
            root / sid,
        )

    def test_elmo_blocked_conf_sessions_are_flat_under_export_root(self) -> None:
        cfg = load_session_list("Elmo_BLOCKED_CONF", REPO / "session_lists.m")
        sid = cfg.session_ids[0]
        session_dir = decode_session_dir(
            cfg.root_folder, sid, cfg.list_name, session_parent="",
        )
        self.assertEqual(session_dir, cfg.root_folder / sid)
        nested = decode_session_dir(cfg.root_folder, sid, cfg.list_name)
        self.assertNotEqual(session_dir, nested)
        if not cfg.root_folder.is_dir():
            self.skipTest(f"export root missing: {cfg.root_folder}")
        self.assertTrue(session_dir.is_dir(), f"missing flat session dir {session_dir}")
        self.assertFalse(nested.is_dir())

    def test_dual_nhp_monkey_from_session_id(self) -> None:
        self.assertEqual(recording_monkey_for_decode("DUAL_NHP", session_id=ELMO_DUAL_B), "Elmo")
        self.assertEqual(recording_monkey_for_decode("DUAL_NHP", session_id=CURIUS_DUAL), "Curius")
        self.assertEqual(
            recording_monkey_for_decode("DUAL_NHP", recording_monkey="Elmo"),
            "Elmo",
        )
        with self.assertRaises(ValueError):
            recording_monkey_for_decode("DUAL_NHP")
        self.assertEqual(
            recording_monkey_for_decode("Elmo_BLOCKED_CONF"),
            "Elmo",
        )

    def test_dual_nhp_elmo_b_filters_and_alignment(self) -> None:
        monkey = recording_monkey_from_session_id(ELMO_DUAL_B)
        actor = recording_actor_side(ELMO_DUAL_B, monkey)
        self.assertEqual(monkey, "Elmo")
        self.assertEqual(actor, "B")
        self.assertEqual(alignment_event_for_actor_side(actor), "B_InitialFixationReleaseTime_ms")
        self.assertEqual(solo_output_subdir_for_actor_side(actor), "SoloB")
        filt = trial_filters_for_branch("DUAL_NHP", "AgoB", "SoloB", actor)
        self.assertEqual(filt["TrialSubType_list"], ["SoloBRewardAB", "SoloB"])
        self.assertEqual(filt["B_Reward_list"], ["RB1", "RB2", "RB3", "RB4"])
        self.assertEqual(filt["go_seq_500_list"], ["AgoB"])
        self.assertNotIn("A_Reward_list", filt)
        self.assertNotIn("conf_predictability_list", filt)
        dyadic = trial_filters_for_branch("DUAL_NHP", "BgoA", "Dyadic", actor)
        self.assertEqual(dyadic["TrialSubType_list"], ["Dyadic"])
        self.assertEqual(dyadic["B_Reward_list"], ["RB1", "RB2", "RB3", "RB4"])
        self.assertEqual(dyadic["go_seq_500_list"], ["BgoA"])

    def test_dual_nhp_curius_a_filters(self) -> None:
        monkey = recording_monkey_from_session_id(CURIUS_DUAL)
        actor = recording_actor_side(CURIUS_DUAL, monkey)
        self.assertEqual(monkey, "Curius")
        self.assertEqual(actor, "A")
        self.assertEqual(alignment_event_for_actor_side(actor), "A_InitialFixationReleaseTime_ms")
        filt = trial_filters_for_branch("DUAL_NHP", "AgoB", "SoloA", actor)
        self.assertEqual(filt["TrialSubType_list"], ["SoloARewardAB", "SoloA"])
        self.assertEqual(filt["A_Reward_list"], ["RA1", "RA2", "RA3", "RA4"])
        self.assertNotIn("B_Reward_list", filt)

    def test_dual_nhp_decode_output_paths(self) -> None:
        root = Path("/data")
        elmo_solo = decoding_dir(
            "DUAL_NHP",
            "AgoB",
            "SoloB",
            "actor_choice",
            figures_root=root,
            recording_monkey="Elmo",
        )
        self.assertEqual(
            elmo_solo,
            root / "DUAL_NHP" / "Elmo_AgoB" / "SoloB" / "decoding" / "actor_choice",
        )
        curius_dyadic = decoding_dir(
            "DUAL_NHP",
            "BgoA",
            "Dyadic",
            "same_diff",
            figures_root=root,
            recording_monkey="Curius",
        )
        self.assertEqual(
            curius_dyadic,
            root / "DUAL_NHP" / "Curius_BgoA" / "Dyadic" / "decoding" / "same_diff",
        )
        from_sid = timing_branch_dir(
            "DUAL_NHP",
            "AgoB",
            "Dyadic",
            figures_root=root,
            session_id=ELMO_DUAL_B,
        )
        self.assertEqual(from_sid, root / "DUAL_NHP" / "Elmo_AgoB" / "Dyadic")
        with self.assertRaises(ValueError):
            timing_branch_dir("DUAL_NHP", "AgoB", "Dyadic", figures_root=root)

    def test_cli_accepts_dual_nhp_and_solob(self) -> None:
        args = parse_args(["DUAL_NHP", "--mode", "dyadic_all", "--force"])
        self.assertEqual(args.list_name, "DUAL_NHP")
        self.assertTrue(args.force)
        args_actor = parse_args(["DUAL_NHP", "--mode", "actor", "--trial-type", "SoloB"])
        self.assertEqual(args_actor.trial_type, "SoloB")

    def test_actor_trial_types_both_follows_monkey_solo(self) -> None:
        self.assertEqual(
            actor_trial_types("both", recording_monkey=None, session_ids=["x"]),
            ("Dyadic", "SoloA"),
        )
        self.assertEqual(
            actor_trial_types("both", recording_monkey="Elmo", session_ids=[ELMO_DUAL_B]),
            ("Dyadic", "SoloB"),
        )
        self.assertEqual(
            actor_trial_types("both", recording_monkey="Curius", session_ids=[CURIUS_DUAL]),
            ("Dyadic", "SoloA"),
        )
        self.assertEqual(
            actor_trial_types("SoloB", recording_monkey="Elmo", session_ids=[ELMO_DUAL_B]),
            ("SoloB",),
        )

    def test_iter_decode_jobs_splits_dual_nhp(self) -> None:
        jobs = list(iter_decode_jobs("DUAL_NHP", REPO / "session_lists.m"))
        monkeys = [m for _, m, _ in jobs]
        self.assertEqual(monkeys, ["Curius", "Elmo"])
        by_monkey = {m: sids for _, m, sids in jobs}
        self.assertEqual(len(by_monkey["Curius"]), 4)
        self.assertEqual(len(by_monkey["Elmo"]), 4)
        elmo_only = list(
            iter_decode_jobs("DUAL_NHP", REPO / "session_lists.m", monkey="Elmo")
        )
        self.assertEqual(len(elmo_only), 1)
        self.assertEqual(elmo_only[0][1], "Elmo")


if __name__ == "__main__":
    unittest.main()
