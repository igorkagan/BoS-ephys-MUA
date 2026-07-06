"""Tests for confederate session audit logic (synthetic labels, no S: drive)."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from bos_mua.session_audit import (
    REWARDED_A,
    REWARDED_B,
    audit_labels,
    audit_session,
    format_markdown,
    list_monkey_from_list_name,
    reward_field_for_trial_type,
    session_id_prefix,
    session_pair_from_id,
    write_csv,
)


def _labels(**fields: list[str]) -> dict[str, np.ndarray]:
    return {name: np.array(values, dtype=object) for name, values in fields.items()}


class RewardFieldAssignmentTests(unittest.TestCase):
    def test_solo_a_types_use_a_reward(self) -> None:
        self.assertEqual(reward_field_for_trial_type("SoloA", "Curius"), "A_Reward_list")
        self.assertEqual(reward_field_for_trial_type("SoloARewardAB", "Elmo"), "A_Reward_list")

    def test_solo_b_types_use_b_reward(self) -> None:
        self.assertEqual(reward_field_for_trial_type("SoloB", "Curius"), "B_Reward_list")
        self.assertEqual(reward_field_for_trial_type("SoloBRewardAB", "Elmo"), "B_Reward_list")

    def test_dyadic_semisolo_follow_list_monkey(self) -> None:
        self.assertEqual(reward_field_for_trial_type("Dyadic", "Curius"), "A_Reward_list")
        self.assertEqual(reward_field_for_trial_type("SemiSolo", "Elmo"), "B_Reward_list")

    def test_list_monkey_from_list_name(self) -> None:
        self.assertEqual(list_monkey_from_list_name("Curius_SHUFFLED_CONF"), "Curius")
        self.assertEqual(list_monkey_from_list_name("Elmo_BLOCKED_CONF"), "Elmo")


class AuditLabelsTests(unittest.TestCase):
    def test_ok_session_counts_rewarded_trials(self) -> None:
        labels = _labels(
            TrialSubType_list=[
                "Dyadic",
                "Dyadic",
                "SoloA",
                "SemiSolo",
                "SoloARewardAB",
                "SoloB",
                "SoloBRewardAB",
                "None",
            ],
            A_Reward_list=["RA1", "RA0", "RA2", "RA3", "RA4", "RA1", "RA2", "RA0"],
            B_Reward_list=["RB0", "RB0", "RB0", "RB0", "RB0", "RB3", "RB4", "RB0"],
        )
        row = audit_labels(
            labels,
            session_id="20230607T115959.A_Curius.B_VC.SCP_01",
            dataset="Curius_SHUFFLED_CONF",
            list_monkey="Curius",
        )
        self.assertEqual(row.session_id, "20230607T115959")
        self.assertEqual(row.session_pair, "A_Curius.B_VC")
        self.assertEqual(row.problem, "-")
        self.assertEqual(row.n_Dyadic, "1")
        self.assertEqual(row.n_SemiSolo, "1")
        self.assertEqual(row.n_SoloA, "1")
        self.assertEqual(row.n_SoloARewardAB, "1")
        self.assertEqual(row.n_SoloB, "1")
        self.assertEqual(row.n_SoloBRewardAB, "1")

    def test_absent_type_shows_dash_not_zero(self) -> None:
        labels = _labels(
            TrialSubType_list=["Dyadic", "Dyadic"],
            A_Reward_list=["RA1", "RA0"],
        )
        row = audit_labels(
            labels,
            session_id="20230607T115959.A_Curius.B_VC.SCP_01",
            dataset="Curius_SHUFFLED_CONF",
            list_monkey="Curius",
        )
        self.assertEqual(row.n_Dyadic, "1")
        self.assertEqual(row.n_SemiSolo, "-")
        self.assertEqual(row.n_SoloA, "-")

    def test_zero_rewarded_shows_zero(self) -> None:
        labels = _labels(
            TrialSubType_list=["SoloA", "SoloA"],
            A_Reward_list=["RA0", "None"],
        )
        row = audit_labels(
            labels,
            session_id="20230607T115959.A_Curius.B_VC.SCP_01",
            dataset="Curius_SHUFFLED_CONF",
            list_monkey="Curius",
        )
        self.assertEqual(row.n_SoloA, "0")
        self.assertIn("suspicious: 2 SoloA trials but 0 rewarded", row.problem)

    def test_missing_reward_field_reports_problem(self) -> None:
        labels = _labels(
            TrialSubType_list=["Dyadic", "SoloA"],
            A_Reward_list=["RA1", "RA2"],
        )
        row = audit_labels(
            labels,
            session_id="20230607T115959.A_Elmo.B_FS.SCP_01",
            dataset="Elmo_BLOCKED_CONF",
            list_monkey="Elmo",
        )
        self.assertEqual(row.n_Dyadic, "-")
        self.assertEqual(row.n_SoloA, "1")
        self.assertIn("B_Reward_list absent from trialinfo.4python.mat", row.problem)

    def test_elmo_dyadic_uses_b_reward(self) -> None:
        labels = _labels(
            TrialSubType_list=["Dyadic", "Dyadic"],
            B_Reward_list=["RB1", "RB2"],
        )
        row = audit_labels(
            labels,
            session_id="20230607T115959.A_Elmo.B_FS.SCP_01",
            dataset="Elmo_BLOCKED_CONF",
            list_monkey="Elmo",
        )
        self.assertEqual(row.n_Dyadic, "2")
        self.assertEqual(row.problem, "-")

    def test_multiple_problems_semicolon_separated(self) -> None:
        labels = _labels(
            TrialSubType_list=["Dyadic", "SoloB"],
        )
        row = audit_labels(
            labels,
            session_id="20230607T115959.A_Curius.B_VC.SCP_01",
            dataset="Curius_SHUFFLED_CONF",
            list_monkey="Curius",
        )
        self.assertIn("A_Reward_list absent from trialinfo.4python.mat", row.problem)
        self.assertIn("B_Reward_list absent from trialinfo.4python.mat", row.problem)
        self.assertIn(";", row.problem)

    def test_missing_trial_subtype_field(self) -> None:
        labels = _labels(A_Reward_list=["RA1"])
        row = audit_labels(
            labels,
            session_id="20230607T115959.A_Curius.B_VC.SCP_01",
            dataset="Curius_SHUFFLED_CONF",
            list_monkey="Curius",
        )
        self.assertEqual(
            row.problem,
            "TrialSubType_list absent from trialinfo.4python.mat",
        )
        for col in row.count_values():
            self.assertEqual(col, "-")

    def test_rewarded_sets(self) -> None:
        self.assertEqual(REWARDED_A, frozenset({"RA1", "RA2", "RA3", "RA4"}))
        self.assertEqual(REWARDED_B, frozenset({"RB1", "RB2", "RB3", "RB4"}))


class AuditSessionDiskTests(unittest.TestCase):
    def test_missing_session_dir(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            row = audit_session(
                "20230608T102315.A_Curius.B_VC.SCP_01",
                dataset="Curius_SHUFFLED_CONF",
                session_dir=Path(tmp) / "missing",
                list_monkey="Curius",
            )
            self.assertEqual(row.problem, "session dir missing")


class OutputFormatTests(unittest.TestCase):
    def test_session_id_prefix(self) -> None:
        self.assertEqual(
            session_id_prefix("20230607T115959.A_Curius.B_VC.SCP_01"),
            "20230607T115959",
        )

    def test_format_markdown_and_write_csv(self) -> None:
        labels = _labels(
            TrialSubType_list=["Dyadic"],
            A_Reward_list=["RA1"],
        )
        row = audit_labels(
            labels,
            session_id="20230607T115959.A_Curius.B_VC.SCP_01",
            dataset="Curius_SHUFFLED_CONF",
            list_monkey="Curius",
        )
        md = format_markdown([row], title="Test")
        self.assertIn("## Test", md)
        self.assertIn("20230607T115959", md)
        self.assertIn("n_Dyadic", md)

        with tempfile.TemporaryDirectory() as tmp:
            path = write_csv([row], Path(tmp) / "audit.csv")
            text = path.read_text(encoding="utf-8")
            self.assertIn("session_id,session_pair,dataset,problem", text)
            self.assertIn("20230607T115959", text)


class SessionPairTests(unittest.TestCase):
    def test_conf_session_pair(self) -> None:
        self.assertEqual(
            session_pair_from_id("20230621T100721.A_Curius.B_MK.SCP_01"),
            "A_Curius.B_MK",
        )

    def test_dual_nhp_session_pair(self) -> None:
        self.assertEqual(
            session_pair_from_id("20230623T124557B.A_Curius.B_Elmo.SCP_01"),
            "A_Curius.B_Elmo",
        )


class DualNhpAuditTests(unittest.TestCase):
    def test_elmo_export_dyadic_uses_rb_tiers(self) -> None:
        labels = _labels(
            TrialSubType_list=["Dyadic", "Dyadic", "Dyadic"],
            B_Reward_list=["RB1", "RB2", "RB0"],
        )
        row = audit_labels(
            labels,
            session_id="20230623T124557B.A_Curius.B_Elmo.SCP_01",
            dataset="DUAL_NHP",
            list_monkey=None,
        )
        self.assertEqual(row.n_Dyadic, "2")
        self.assertEqual(row.problem, "-")

    def test_curius_export_dyadic_uses_ra_tiers(self) -> None:
        labels = _labels(
            TrialSubType_list=["Dyadic", "Dyadic"],
            A_Reward_list=["RA1", "RA4"],
        )
        row = audit_labels(
            labels,
            session_id="20230623T124557U.A_Curius.B_Elmo.SCP_01",
            dataset="DUAL_NHP",
            list_monkey=None,
        )
        self.assertEqual(row.n_Dyadic, "2")


class SanityCheckTests(unittest.TestCase):
    def test_wrong_tier_prefix_flags_suspicious(self) -> None:
        labels = _labels(
            TrialSubType_list=["Dyadic", "Dyadic", "Dyadic"],
            B_Reward_list=["RA1", "RA2", "RA3"],
        )
        row = audit_labels(
            labels,
            session_id="20210401T124246.A_Elmo.B_KN.SCP_01",
            dataset="Elmo_BLOCKED_CONF",
            list_monkey="Elmo",
        )
        self.assertEqual(row.n_Dyadic, "0")
        self.assertIn("suspicious: 3 Dyadic trials but 0 rewarded", row.problem)


if __name__ == "__main__":
    unittest.main()
