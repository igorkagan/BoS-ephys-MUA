"""Session-list (flat export) decode path + filter key."""

from __future__ import annotations

import unittest
from pathlib import Path

from analyze_decoding.data import decode_session_dir, filter_condition_key, trial_filters_for_branch
from load_data.sessions import load_session_list

REPO = Path(__file__).resolve().parents[2]


class DecodeSessionListTests(unittest.TestCase):
    def test_filter_key_maps_conf_list_to_blocked(self) -> None:
        self.assertEqual(filter_condition_key("Elmo_BLOCKED_CONF"), "Elmo_BLOCKED")
        self.assertEqual(filter_condition_key("Elmo_BLOCKED"), "Elmo_BLOCKED")

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


if __name__ == "__main__":
    unittest.main()
