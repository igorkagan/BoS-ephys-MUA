"""Curated trialinfo.4python.mat vs export parity (pipeline trial counts)."""

from __future__ import annotations

import unittest
import os
from pathlib import Path

import numpy as np

from load_data.io import load_trial_labels
from load_data.sessions import load_session_list
from load_data.trial_selection_counts import (
    conf_list_for_curated_condition,
    curated_sessions_with_export_counterpart,
    pipeline_trial_counts,
)
from process_channels.preprocess import MONKEY_CONDITIONS, recording_monkey_from_condition_label
from run_pipeline.curated import curated_data_root, discover_curated_sessions

REPO_ROOT = Path(__file__).resolve().parents[2]
SESSION_LISTS = REPO_ROOT / "session_lists.m"
N_SESSIONS_PER_CONDITION = 10

DATA_AVAILABLE = (
    os.environ.get("BOS_RUN_DATA_TESTS") == "1"
    and curated_data_root().is_dir()
)


def _labels(**fields: list[str]) -> dict[str, np.ndarray]:
    return {name: np.array(values, dtype=object) for name, values in fields.items()}


def _compare_trialinfo_sources(
    *,
    condition: str,
    session_id: str,
    curated_dir: Path,
    export_dir: Path,
) -> None:
    recording_monkey = recording_monkey_from_condition_label(condition)
    curated_labels = load_trial_labels(curated_dir, session_id)
    export_labels = load_trial_labels(export_dir, session_id)

    curated_counts = pipeline_trial_counts(
        curated_labels,
        condition_key=condition,
        session_id=session_id,
        recording_monkey=recording_monkey,
    )
    export_counts = pipeline_trial_counts(
        export_labels,
        condition_key=condition,
        session_id=session_id,
        recording_monkey=recording_monkey,
    )

    self_assert = unittest.TestCase()
    self_assert.assertEqual(
        curated_counts["n_trials"],
        export_counts["n_trials"],
        f"{condition} {session_id}: n_trials curated={curated_counts['n_trials']} "
        f"export={export_counts['n_trials']}",
    )
    for key in curated_counts:
        if key == "n_trials":
            continue
        self_assert.assertEqual(
            curated_counts[key],
            export_counts[key],
            f"{condition} {session_id}: {key} curated={curated_counts[key]} "
            f"export={export_counts[key]}",
        )


class PipelineTrialCountsUnitTests(unittest.TestCase):
    def test_pipeline_trial_counts_synthetic(self) -> None:
        labels = _labels(
            TrialSubType_list=["Dyadic", "Dyadic", "SoloA", "SoloB"],
            go_seq_500_list=["AgoB", "BgoA", "AgoB", "BgoA"],
            conf_predictability_list=["Blocked", "Blocked", "Blocked", "Blocked"],
            A_Reward_list=["RA1", "RA2", "RA3", "RA0"],
            B_Reward_list=["RB0", "RB0", "RB0", "RB4"],
            A_LR_pos_list=["Al", "Ar", "Al", "Ar"],
            B_LR_pos_list=["Bl", "Br", "Bl", "Br"],
        )
        counts = pipeline_trial_counts(
            labels,
            condition_key="Elmo_BLOCKED",
            session_id="20201204T125624.A_Elmo.B_FS.SCP_01",
            recording_monkey="Elmo",
        )
        self.assertEqual(counts["n_trials"], 4)
        self.assertEqual(counts["dyadic_AgoB_total"], 1)
        self.assertEqual(counts["dyadic_BgoA_total"], 1)
        self.assertEqual(counts["solo_AgoB_total"], 1)
        self.assertEqual(counts["solo_BgoA_total"], 0)

    def test_conf_list_mapping(self) -> None:
        self.assertEqual(conf_list_for_curated_condition("Elmo_BLOCKED"), "Elmo_BLOCKED_CONF")

    def test_curated_sessions_with_export_counterpart(self) -> None:
        curated = ["a", "b", "c", "d"]
        conf = ["b", "c", "e"]
        picked = curated_sessions_with_export_counterpart(
            curated, conf, n=2, conf_list_name="Test_CONF",
        )
        self.assertEqual(picked, ["b", "c"])

    def test_curated_sessions_with_export_counterpart_raises(self) -> None:
        with self.assertRaises(ValueError):
            curated_sessions_with_export_counterpart(
                ["a", "b"], ["c", "d"], n=2, conf_list_name="Test_CONF",
            )


@unittest.skipUnless(DATA_AVAILABLE, f"curated data root not available: {curated_data_root()}")
class CuratedTrialinfoVsExportIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.curated_root = curated_data_root()
        cls.session_lists = SESSION_LISTS
        if not cls.session_lists.is_file():
            raise unittest.SkipTest(f"session_lists.m not found: {cls.session_lists}")

    def _sessions_for_condition(self, condition: str) -> list[str]:
        conf_name = conf_list_for_curated_condition(condition)
        conf_cfg = load_session_list(conf_name, self.session_lists)
        if not conf_cfg.root_folder.is_dir():
            raise unittest.SkipTest(f"export root not available: {conf_cfg.root_folder}")
        curated_ids = discover_curated_sessions(condition, data_root=self.curated_root)
        return curated_sessions_with_export_counterpart(
            curated_ids,
            conf_cfg.session_ids,
            n=N_SESSIONS_PER_CONDITION,
            conf_list_name=conf_name,
        )

    def _assert_condition_parity(self, condition: str) -> None:
        conf_name = conf_list_for_curated_condition(condition)
        export_root = load_session_list(conf_name, self.session_lists).root_folder
        for session_id in self._sessions_for_condition(condition):
            with self.subTest(condition=condition, session_id=session_id):
                curated_dir = self.curated_root / condition / session_id
                export_dir = export_root / session_id
                self.assertTrue(
                    curated_dir.is_dir(),
                    f"missing curated folder: {curated_dir}",
                )
                self.assertTrue(
                    export_dir.is_dir(),
                    f"missing export folder: {export_dir}",
                )
                _compare_trialinfo_sources(
                    condition=condition,
                    session_id=session_id,
                    curated_dir=curated_dir,
                    export_dir=export_dir,
                )

    def test_elmo_blocked_parity(self) -> None:
        self._assert_condition_parity("Elmo_BLOCKED")

    def test_elmo_shuffled_parity(self) -> None:
        self._assert_condition_parity("Elmo_SHUFFLED")

    def test_curius_blocked_parity(self) -> None:
        self._assert_condition_parity("Curius_BLOCKED")

    def test_curius_shuffled_parity(self) -> None:
        self._assert_condition_parity("Curius_SHUFFLED")

    def test_all_conditions_have_ten_sessions(self) -> None:
        for condition in MONKEY_CONDITIONS:
            with self.subTest(condition=condition):
                sessions = self._sessions_for_condition(condition)
                self.assertEqual(len(sessions), N_SESSIONS_PER_CONDITION)


if __name__ == "__main__":
    unittest.main()
