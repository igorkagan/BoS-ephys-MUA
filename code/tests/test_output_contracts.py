"""Offline tests for explicit contexts and deterministic output contracts."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from process_channels.features import ChannelSummary
from run_pipeline.config import reset_processing_modes
from run_pipeline.context import PipelineContext
from run_pipeline.output_contracts import (
    audit_manifest,
    expected_branch_outputs,
    validate_pipeline_outputs,
)


def _summary(session_id: str) -> ChannelSummary:
    t_ms = np.asarray([-1.0, 0.0, 1.0])
    return ChannelSummary(
        session_id=session_id,
        channel=1,
        array_name="A1",
        index_in_array=1,
        n_left=3,
        n_right=3,
        t_ms=t_ms,
        mean_left=np.ones(3),
        mean_right=np.zeros(3),
        diff=np.ones(3),
        si=0.5,
        mwu_p=0.5,
        pref_side="L",
        evoked_p_left=0.5,
        evoked_p_right=0.5,
        task_evoked=True,
    )


class PipelineContextContractTests(unittest.TestCase):
    def test_curated_context_requires_explicit_parent(self) -> None:
        with self.assertRaisesRegex(ValueError, "session_parent"):
            PipelineContext(
                data_root=Path("/data"),
                session_ids=["s1"],
                output_base=Path("/out"),
                condition_label="Elmo_AgoB",
                condition_key="Elmo_BLOCKED",
                layout="curated",
                trial_filters={},
                source_kind="curated",
            )

    def test_session_list_cannot_smuggle_curated_parent(self) -> None:
        with self.assertRaisesRegex(ValueError, "no session_parent"):
            PipelineContext(
                data_root=Path("/data"),
                session_ids=["s1"],
                output_base=Path("/out"),
                condition_label="Elmo_AgoB",
                condition_key="Elmo_BLOCKED",
                layout="flat",
                trial_filters={},
                session_parent="Elmo_BLOCKED",
            )


class OutputManifestTests(unittest.TestCase):
    def tearDown(self) -> None:
        reset_processing_modes()

    def _fixture(self, root: Path):
        session_ids = [
            f"s{i}.A_Elmo.B_X.SCP_01"
            for i in range(1, 4)
        ]
        ctx = PipelineContext(
            data_root=root / "data",
            session_ids=session_ids,
            output_base=root / "out",
            condition_label="Elmo_AgoB",
            condition_key="Elmo_BLOCKED",
            layout="flat",
            trial_filters={"go_seq_500_list": ["AgoB"]},
            recording_monkey="Elmo",
        )
        cache = SimpleNamespace(
            summaries={True: [_summary(session_id) for session_id in session_ids]},
            trials={
                True: {
                    session_id: {1: (np.ones((3, 3)), np.ones((3, 3)))}
                    for session_id in session_ids
                }
            },
            pooled=object(),
        )
        return ctx, cache

    def test_expected_outputs_are_data_driven_and_exact(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            ctx, cache = self._fixture(Path(tmp))
            expected = expected_branch_outputs(
                ctx, ["session_lr", "combine", "array_combined"], cache,
            )
            names = {path.name for path in expected}
            self.assertEqual(sum(name.endswith("_LR.pdf") for name in names), 21)
            self.assertIn(
                "Elmo_AgoB_A_InitialFixationReleaseTime_ms_arrays_combined_LR.pdf",
                names,
            )

    def test_pref_unpref_expected_combined_pref_pdfs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            ctx, cache = self._fixture(Path(tmp))
            expected = expected_branch_outputs(ctx, ["pref_unpref"], cache)
            names = {path.name for path in expected}
            self.assertEqual(sum(name.endswith("_combined_pref.pdf") for name in names), 6)
            self.assertIn(
                "Elmo_AgoB_A_InitialFixationReleaseTime_ms_arrays_combined_pref.pdf",
                names,
            )
            self.assertFalse(any("pref_unpref" in path.as_posix() for path in expected))
            self.assertTrue(all("combined" in path.as_posix() for path in expected))

    def test_validation_writes_recheckable_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            ctx, cache = self._fixture(Path(tmp))
            steps = ["session_lr", "combine", "array_combined"]
            for path in expected_branch_outputs(ctx, steps, cache):
                path.parent.mkdir(parents=True, exist_ok=True)
                path.touch()
            result = validate_pipeline_outputs(ctx, steps, cache)
            self.assertTrue(result.ok)
            payload = json.loads(result.manifest_path.read_text(encoding="utf-8"))
            self.assertEqual(payload["status"], "ok")
            self.assertEqual(payload["condition_key"], "Elmo_BLOCKED")
            self.assertEqual(len(payload["code_fingerprint_sha256"]), 64)
            self.assertTrue(audit_manifest(result.manifest_path).ok)
            expected_branch_outputs(ctx, steps, cache)[0].unlink()
            self.assertFalse(audit_manifest(result.manifest_path).ok)

    def test_validation_fails_on_missing_and_png(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            ctx, cache = self._fixture(Path(tmp))
            ctx.output_base.mkdir(parents=True)
            (ctx.output_base / "stale.png").touch()
            with self.assertRaisesRegex(RuntimeError, "Output contract failed"):
                validate_pipeline_outputs(ctx, ["session_lr"], cache)
            payload = json.loads(
                (ctx.output_base / "pipeline_manifest.json").read_text(encoding="utf-8")
            )
            self.assertEqual(payload["status"], "failed")


if __name__ == "__main__":
    unittest.main()
