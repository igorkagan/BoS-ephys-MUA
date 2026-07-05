"""Tests for session_lists.m parser."""

from __future__ import annotations

import unittest
from pathlib import Path

from bos_mua.session_lists import (
    condition_key_from_list_name,
    list_available_session_lists,
    load_dual_nhp_configs,
    load_session_list,
    monkey_for_dual_session,
    split_dual_nhp_sessions,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
SESSION_LISTS = REPO_ROOT / "session_lists.m"

CURius_SESSION = "20230623T124557U.A_Curius.B_Elmo.SCP_01"
ELMO_SESSION = "20230623T124557B.A_Curius.B_Elmo.SCP_01"


class SessionListsParserTests(unittest.TestCase):
    def test_condition_key_from_list_name(self) -> None:
        self.assertEqual(condition_key_from_list_name("Elmo_BLOCKED_CONF"), "Elmo_BLOCKED")
        self.assertEqual(
            condition_key_from_list_name("Elmo_BLOCKED_CONF_not_top_10"),
            "Elmo_BLOCKED",
        )

    def test_load_root_folder(self) -> None:
        cfg = load_session_list("Elmo_BLOCKED_CONF", SESSION_LISTS)
        self.assertIn("MUA_export_per_session", str(cfg.root_folder))

    def test_elmo_blocked_conf_matches_selection_list(self) -> None:
        cfg = load_session_list("Elmo_BLOCKED_CONF", SESSION_LISTS)
        self.assertEqual(len(cfg.session_ids), 40)
        self.assertEqual(cfg.session_ids[0], "20201204T125624.A_Elmo.B_FS.SCP_01")
        self.assertEqual(cfg.output_folder, cfg.root_folder / "Elmo_BLOCKED_CONF")

    def test_not_top_10_skips_commented_sessions(self) -> None:
        cfg = load_session_list("Elmo_BLOCKED_CONF_not_top_10", SESSION_LISTS)
        self.assertEqual(len(cfg.session_ids), 17)
        self.assertNotIn("20201218T130348.A_Elmo.B_FS.SCP_01", cfg.session_ids)
        self.assertIn("20210127T130717.A_Elmo.B_FS.SCP_01", cfg.session_ids)

    def test_list_available_includes_named_arrays(self) -> None:
        names = list_available_session_lists(SESSION_LISTS)
        self.assertIn("Elmo_BLOCKED_CONF", names)
        self.assertIn("Elmo_BLOCKED_CONF_not_top_10", names)
        self.assertIn("ElmoBLOCKED_SpikeSortedSessions", names)
        self.assertIn("DUAL_NHP", names)

    def test_monkey_for_dual_session(self) -> None:
        self.assertEqual(monkey_for_dual_session(ELMO_SESSION), "Elmo")
        self.assertEqual(monkey_for_dual_session(CURius_SESSION), "Curius")
        self.assertEqual(
            monkey_for_dual_session("20230630T115937.A_Curius.B_Elmo.SCP_01"),
            "Curius",
        )

    def test_load_dual_nhp_list(self) -> None:
        cfg = load_session_list("DUAL_NHP", SESSION_LISTS)
        self.assertEqual(len(cfg.session_ids), 8)
        self.assertEqual(cfg.condition_key, "DUAL_NHP")
        self.assertEqual(cfg.output_folder, cfg.root_folder / "DUAL_NHP")

    def test_split_dual_nhp_sessions(self) -> None:
        cfg, split = load_dual_nhp_configs(SESSION_LISTS)
        self.assertEqual(len(split["Curius"]), 4)
        self.assertEqual(len(split["Elmo"]), 4)
        self.assertIn(CURius_SESSION, split["Curius"])
        self.assertIn(ELMO_SESSION, split["Elmo"])
        self.assertEqual(len(cfg.session_ids), 8)
        self.assertEqual(
            split["Curius"],
            split_dual_nhp_sessions(cfg.session_ids)["Curius"],
        )


if __name__ == "__main__":
    unittest.main()
