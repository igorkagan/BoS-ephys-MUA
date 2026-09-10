"""Tests for confederate session-overview peel reasons and Excel columns."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

from load_data.session_overview import (
    _branch_note_label,
    _branch_status,
    _peel_empty_reason,
    sessions_frame,
    write_overview_excel,
)


class PeelFilterTests(unittest.TestCase):
    def test_peel_names_reward_cut_in_plain_language(self) -> None:
        labels = {
            "TrialSubType_list": np.array(["Dyadic", "SoloA"], dtype=object),
            "go_seq_500_list": np.array(["AgoB", "AgoB"], dtype=object),
            "A_Reward_list": np.array(["RA0", "RA1"], dtype=object),
        }
        reason = _peel_empty_reason(
            labels,
            {"TrialSubType_list": ["Dyadic"], "A_Reward_list": ["RA1", "RA2", "RA3", "RA4"]},
        )
        self.assertIn("no rewarded A trials", reason)
        self.assertIn("1 left before this cut", reason)
        self.assertNotIn("A_Reward_list", reason)

    def test_peel_reports_missing_column_in_plain_language(self) -> None:
        labels = {"TrialSubType_list": np.array(["Dyadic"], dtype=object)}
        reason = _peel_empty_reason(labels, {"conf_predictability_list": ["Blocked"]})
        self.assertEqual(reason, "trialinfo has no Blocked/Shuffled column")

    def test_peel_no_shuffled_is_no_trials(self) -> None:
        labels = {
            "TrialSubType_list": np.array(["Dyadic", "Dyadic"], dtype=object),
            "go_seq_500_list": np.array(["AgoB", "AgoB"], dtype=object),
            "conf_predictability_list": np.array(["Blocked", "Blocked"], dtype=object),
        }
        reason = _peel_empty_reason(
            labels,
            {
                "TrialSubType_list": ["Dyadic"],
                "go_seq_500_list": ["AgoB"],
                "conf_predictability_list": ["Shuffled"],
            },
        )
        self.assertTrue(reason.startswith("no trials"))
        self.assertNotIn("paired", reason)
        self.assertNotIn("tagged", reason)

    def test_peel_no_solo_trials(self) -> None:
        labels = {
            "TrialSubType_list": np.array(["Dyadic", "Dyadic"], dtype=object),
        }
        reason = _peel_empty_reason(
            labels,
            {"TrialSubType_list": ["SoloA", "SoloARewardAB"]},
        )
        self.assertEqual(reason, "no trials")


class BranchNoteTests(unittest.TestCase):
    def test_labels_are_dyadic_blocked_or_shuffled(self) -> None:
        self.assertEqual(_branch_note_label("Dyadic", "Elmo_BLOCKED"), "Dyadic Blocked")
        self.assertEqual(_branch_note_label("Dyadic", "Curius_SHUFFLED"), "Dyadic Shuffled")
        self.assertEqual(_branch_note_label("SoloA", "Elmo_BLOCKED"), "Solo")

    def test_too_few_left_right_is_plain(self) -> None:
        choice = SimpleNamespace(field="A_LR_pos_list", left=["Al"], right=["Ar"])
        labels = {
            "TrialSubType_list": np.array(["SoloA", "SoloA"], dtype=object),
            "A_LR_pos_list": np.array(["Al", "Ar"], dtype=object),
        }
        out = _branch_status(
            labels,
            {"TrialSubType_list": ["SoloA", "SoloARewardAB"]},
            choice,
            tag="solo_AgoB",
            condition_key="Elmo_BLOCKED",
        )
        note = str(out["solo_AgoB_exclusion"])
        self.assertIn("AgoB Solo: left=1, right=1 (need ≥3 each)", note)
        self.assertNotIn("cache", note.lower())
        self.assertNotIn("locked", note.lower())
        self.assertNotIn("TrialSubType", note)

    def test_dyadic_prefix_includes_blocked(self) -> None:
        choice = SimpleNamespace(field="A_LR_pos_list", left=["Al"], right=["Ar"])
        labels = {
            "TrialSubType_list": np.array(["Dyadic"] * 4, dtype=object),
            "conf_predictability_list": np.array(["Shuffled"] * 4, dtype=object),
            "A_LR_pos_list": np.array(["Al", "Al", "Ar", "Ar"], dtype=object),
        }
        out = _branch_status(
            labels,
            {"TrialSubType_list": ["Dyadic"], "conf_predictability_list": ["Blocked"]},
            choice,
            tag="dyadic_AgoB",
            condition_key="Elmo_BLOCKED",
        )
        self.assertIn("AgoB Dyadic Blocked: no trials", str(out["dyadic_AgoB_exclusion"]))


class WorkbookTests(unittest.TestCase):
    def test_write_excel_has_three_sheets(self) -> None:
        rows = [
            {
                "dataset": "Elmo_BLOCKED_CONF",
                "monkey": "Elmo",
                "session_dir_exists": True,
                "trialinfo_exists": True,
                "dyadic_AgoB_mua_ok": True,
                "solo_AgoB_mua_ok": False,
                "dyadic_AgoB_in_cache": True,
                "solo_AgoB_in_cache": False,
                "locked_AgoB_included": False,
                "dyadic_AgoB_decode_ok": True,
                "decode_dyadic_AgoB_same_diff": True,
                "dyadic_BgoA_mua_ok": False,
                "solo_BgoA_mua_ok": False,
                "dyadic_BgoA_in_cache": False,
                "solo_BgoA_in_cache": False,
                "locked_BgoA_included": False,
                "dyadic_BgoA_decode_ok": False,
                "decode_dyadic_BgoA_same_diff": False,
                "notes": "AgoB Solo: left=1, right=2 (need ≥3 each)",
            }
        ]
        table = sessions_frame(rows)
        with tempfile.TemporaryDirectory() as tmp:
            path = write_overview_excel(table, Path(tmp) / "overview.xlsx")
            self.assertTrue(path.is_file())
            sheets = pd.ExcelFile(path).sheet_names
            self.assertEqual(sheets, ["sessions", "dataset_summary", "column_key"])


if __name__ == "__main__":
    unittest.main()
