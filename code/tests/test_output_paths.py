"""Tests for branch output path resolution."""

from __future__ import annotations

import unittest
from pathlib import Path

from process_channels.preprocess import resolve_condition_output_dir, resolve_consistency_dir
from run_pipeline.config import reset_processing_modes, set_processing_modes
from run_pipeline.context import PipelineContext, set_active_context


class OutputPathTests(unittest.TestCase):
    def setUp(self) -> None:
        set_active_context(
            PipelineContext(
                data_root=Path("/data"),
                session_ids=["s1"],
                output_base=Path("/out"),
                condition_label="Elmo_AgoB",
                condition_key="Elmo_BLOCKED",
                layout="flat",
                trial_filters={},
            ),
        )

    def tearDown(self) -> None:
        set_active_context(None)
        reset_processing_modes()

    def test_zscore_only_flat_under_branch(self) -> None:
        base = Path("/out/Elmo_AgoB/Dyadic")
        self.assertEqual(
            resolve_condition_output_dir(base, True, "Elmo_AgoB"),
            base,
        )
        self.assertEqual(
            resolve_consistency_dir(base, True, "Elmo_AgoB"),
            base / "consistency",
        )
        self.assertEqual(
            resolve_condition_output_dir(base, True, "Elmo_AgoB", "combined"),
            base / "combined",
        )

    def test_also_original_mode_subdirs(self) -> None:
        set_processing_modes(also_original=True)
        base = Path("/out/Elmo_AgoB/Dyadic")
        self.assertEqual(
            resolve_condition_output_dir(base, False, "Elmo_AgoB"),
            base / "original",
        )
        self.assertEqual(
            resolve_condition_output_dir(base, True, "Elmo_AgoB"),
            base / "zscored",
        )
        self.assertEqual(
            resolve_consistency_dir(base, True, "Elmo_AgoB"),
            base / "zscored" / "consistency",
        )


if __name__ == "__main__":
    unittest.main()
