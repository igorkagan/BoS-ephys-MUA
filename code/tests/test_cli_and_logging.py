"""Offline smoke tests for CLI construction and pipeline logging."""

from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from compare_conditions.compare import parse_args as parse_comparison_args
from run_pipeline.run_log import _Tee, pipeline_run_log
from run_scripts.plan_pipeline import build_plan, parse_args as parse_plan_args
from run_scripts.run_session_list import main as session_list_main


class _FakeStream:
    def __init__(self) -> None:
        self.data = ""

    def write(self, data: str) -> int:
        self.data += data
        return len(data)

    def flush(self) -> None:
        return None

    def fileno(self) -> int:
        return 17

    def isatty(self) -> bool:
        return True


class PipelineLogTests(unittest.TestCase):
    def test_tee_delegates_terminal_contract(self) -> None:
        stream = _FakeStream()
        log = _FakeStream()
        tee = _Tee(stream, log)
        self.assertEqual(tee.fileno(), 17)
        self.assertTrue(tee.isatty())
        tee.write("hello")
        self.assertEqual(stream.data, "hello")
        self.assertEqual(log.data, "hello")

    def test_argparse_works_inside_pipeline_log_and_stdout_restores(self) -> None:
        original = sys.stdout
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with pipeline_run_log(root):
                parser = argparse.ArgumentParser()
                parser.add_argument("--value")
                self.assertEqual(parser.parse_args(["--value", "ok"]).value, "ok")
                print("logged")
            self.assertIs(sys.stdout, original)
            self.assertIn("logged", (root / "pipeline.log").read_text(encoding="utf-8"))


class CliContractTests(unittest.TestCase):
    def test_comparison_cli_uses_explicit_axis_arguments(self) -> None:
        args = parse_comparison_args(
            [
                "--monkey",
                "Elmo",
                "--comparison-axis",
                "social_context",
                "--go-seq",
                "AgoB",
            ]
        )
        self.assertEqual(args.comparison_axis, "social_context")
        self.assertEqual(args.go_seq, "AgoB")

    def test_module_entrypoints_render_help(self) -> None:
        for module in (
            "run_scripts.run_curated",
            "run_scripts.run_session_list",
            "run_scripts.run_decode_session_list",
            "run_scripts.run_comparisons",
            "run_scripts.plan_pipeline",
            "run_scripts.audit_outputs",
        ):
            with self.subTest(module=module):
                result = subprocess.run(
                    [sys.executable, "-m", module, "--help"],
                    cwd=Path(__file__).resolve().parents[1],
                    capture_output=True,
                    text=True,
                    timeout=30,
                    check=False,
                )
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertIn("usage:", result.stdout.lower())

    def test_plan_pipeline_obeys_selected_go_sequence(self) -> None:
        repo = Path(__file__).resolve().parents[2]
        plan = build_plan(
            parse_plan_args(
                [
                    "DUAL_NHP",
                    "--session-lists",
                    str(repo / "session_lists.m"),
                    "--go-seq",
                    "AgoB",
                ]
            )
        )
        self.assertTrue(plan["runs"])
        self.assertTrue(plan["comparisons"])
        self.assertTrue(
            all(item["comparison_axis"] == "social_context" for item in plan["comparisons"])
        )
        self.assertTrue(
            all(item["go_seq"] == "AgoB" for item in plan["comparisons"])
        )

    def test_reduced_legacy_list_is_fail_closed(self) -> None:
        repo = Path(__file__).resolve().parents[2]
        with self.assertRaisesRegex(SystemExit, "Refusing the reduced legacy"):
            session_list_main(
                [
                    "ElmoBLOCKED_SpikeSortedSessions",
                    "--session-lists",
                    str(repo / "session_lists.m"),
                    "--steps",
                    "session_lr",
                ]
            )


if __name__ == "__main__":
    unittest.main()
