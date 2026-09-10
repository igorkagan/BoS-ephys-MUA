"""Offline contract tests for the generic comparison engine."""

from __future__ import annotations

import json
import tempfile
import unittest
import warnings
from pathlib import Path
from unittest.mock import patch

import numpy as np

from compare_conditions.compare import (
    ComparisonInputs,
    ComparisonSpec,
    branch_output_dir_from_source_path,
    build_comparison_combined_suptitles,
    build_matched_rows,
    combined_pdf_names,
    dyadic_solo_spec,
    first_second_spec,
    finalize_comparison_manifest,
    remove_stale_outputs,
    processing_scoped_output,
    unlink_case_variants,
    validate_outputs,
    write_comparison_manifest,
    main as comparison_main,
)
from load_data.sessions import SessionListConfig
from load_data.io import ARRAY_NAMES
from process_channels.features import ChannelSummary
from run_pipeline.runner import (
    ALL_STEPS,
    _run_comparison_suite,
    _run_comparisons,
    parse_steps,
)
from run_pipeline.config import reset_processing_modes, set_processing_modes


def _summary(si: float) -> ChannelSummary:
    values = np.asarray([0.0, si, 2.0 * si])
    return ChannelSummary(
        session_id="session-1",
        channel=1,
        array_name="A1",
        index_in_array=1,
        n_left=2,
        n_right=2,
        t_ms=np.asarray([-1.0, 0.0, 1.0]),
        mean_left=values,
        mean_right=-values,
        diff=2.0 * values,
        si=si,
        mwu_p=0.5,
        pref_side="L",
        evoked_p_left=0.5,
        evoked_p_right=0.5,
        task_evoked=True,
    )


def _touch_combined_pdfs(out: Path, spec: ComparisonSpec) -> None:
    combined = out / "combined"
    combined.mkdir(exist_ok=True)
    for name in combined_pdf_names(spec.file_tag, spec):
        (combined / name).touch()


def _inputs(
    *,
    dimensions_a: dict[str, str],
    dimensions_b: dict[str, str],
) -> ComparisonInputs:
    def filters(dimensions: dict[str, str]):
        social = dimensions["social_context"]
        trial_types = ("Dyadic",) if social == "Dyadic" else (social,)
        return (
            ("TrialSubType_list", trial_types),
            ("go_seq_500_list", (dimensions["go_sequence"],)),
        )

    source_a = (
        f"C:/cache/Elmo_{dimensions_a['go_sequence']}/"
        f"{dimensions_a['social_context']}/.cache/summaries_zscore.npz"
    )
    source_b = (
        f"C:/cache/Elmo_{dimensions_b['go_sequence']}/"
        f"{dimensions_b['social_context']}/.cache/summaries_zscore.npz"
    )
    return ComparisonInputs(
        source_kind="session_list",
        layout="flat",
        condition_key="Elmo_BLOCKED",
        actor_side="A",
        recording_monkey="Elmo",
        source_path_a=source_a,
        source_path_b=source_b,
        summary_source_mode_a="disk_cache",
        summary_source_mode_b="disk_cache",
        trial_data_root="C:/data",
        filters_a=filters(dimensions_a),
        filters_b=filters(dimensions_b),
        dimensions_a=tuple(sorted(dimensions_a.items())),
        dimensions_b=tuple(sorted(dimensions_b.items())),
        session_ids=("session-1",),
        alignment_events=("A_Event",),
        choice_field="choice",
        choice_values=(("left", ("Al",)), ("right", ("Ar",))),
        processing_mode="zscored",
        analysis_window_ms=(-500.0, 500.0),
        smooth_ms=50.0,
    )


class ComparisonSpecTests(unittest.TestCase):
    def test_branch_output_dir_from_summaries_cache_path(self) -> None:
        branch = Path("figures/Elmo_BLOCKED/Elmo_AgoB/Dyadic")
        cache_file = branch / ".cache" / "summaries_zscore.npz"
        self.assertEqual(branch_output_dir_from_source_path(cache_file), branch)
        self.assertEqual(branch_output_dir_from_source_path(branch), branch)
    def tearDown(self) -> None:
        reset_processing_modes()

    def test_specs_are_explicit_and_delta_is_b_minus_a(self) -> None:
        first_second = first_second_spec("Dyadic")
        social = dyadic_solo_spec("SoloA", "AgoB")
        self.assertEqual(first_second.comparison_axis, "go_sequence")
        self.assertEqual(dict(first_second.fixed_dimensions), {"social_context": "Dyadic"})
        self.assertEqual(social.comparison_axis, "social_context")
        self.assertEqual(dict(social.fixed_dimensions), {"go_sequence": "AgoB"})
        self.assertEqual(social.delta_definition, "B-A: SI(SoloA) - SI(Dyadic)")

    def test_spec_rejects_non_derived_file_tag(self) -> None:
        with self.assertRaisesRegex(ValueError, "file_tag"):
            ComparisonSpec(
                comparison_axis="social_context",
                label_a="Dyadic",
                label_b="SoloA",
                fixed_dimensions=(("go_sequence", "AgoB"),),
                file_tag="AgoB_vs_BgoA",
            )

    def test_operand_validation_rejects_cross_axis_leakage(self) -> None:
        spec = dyadic_solo_spec("SoloA", "AgoB")
        valid = _inputs(
            dimensions_a={"go_sequence": "AgoB", "social_context": "Dyadic"},
            dimensions_b={"go_sequence": "AgoB", "social_context": "SoloA"},
        )
        valid.validate(spec)
        leaked = _inputs(
            dimensions_a={"go_sequence": "AgoB", "social_context": "Dyadic"},
            dimensions_b={"go_sequence": "BgoA", "social_context": "SoloA"},
        )
        with self.assertRaisesRegex(ValueError, "differ only"):
            leaked.validate(spec)

    def test_operand_validation_rejects_mislabeled_social_operand(self) -> None:
        spec = dyadic_solo_spec("SoloA", "AgoB")
        mislabeled = _inputs(
            dimensions_a={"go_sequence": "AgoB", "social_context": "Dyadic"},
            dimensions_b={"go_sequence": "AgoB", "social_context": "BgoA"},
        )
        with self.assertRaisesRegex(ValueError, "label leaks"):
            mislabeled.validate(spec)

    def test_operand_validation_rejects_filter_leakage(self) -> None:
        spec = dyadic_solo_spec("SoloA", "AgoB")
        valid = _inputs(
            dimensions_a={"go_sequence": "AgoB", "social_context": "Dyadic"},
            dimensions_b={"go_sequence": "AgoB", "social_context": "SoloA"},
        )
        leaked = ComparisonInputs(
            **{
                **valid.__dict__,
                "filters_b": (
                    ("TrialSubType_list", ("SoloA",)),
                    ("go_seq_500_list", ("BgoA",)),
                ),
            }
        )
        with self.assertRaisesRegex(ValueError, "go-sequence filter"):
            leaked.validate(spec)

    def test_dual_processing_modes_cannot_overwrite_each_other(self) -> None:
        base = Path("/out/comparison")
        self.assertEqual(processing_scoped_output(base, True), base)
        self.assertEqual(processing_scoped_output(base, False), base / "original")
        set_processing_modes(also_original=True)
        self.assertEqual(processing_scoped_output(base, True), base / "zscored")
        self.assertEqual(processing_scoped_output(base, False), base / "original")


class ComparisonNumericsAndManifestTests(unittest.TestCase):
    def test_delta_is_b_minus_a_for_both_axes(self) -> None:
        for spec in (first_second_spec(), dyadic_solo_spec("SoloA", "AgoB")):
            with self.subTest(axis=spec.comparison_axis):
                rows = build_matched_rows([_summary(-0.25)], [_summary(0.5)], task_evoked_only=False)
                self.assertEqual(len(rows), 1)
                self.assertAlmostEqual(rows[0].delta_si, 0.75)
                self.assertAlmostEqual(rows[0].si_a, -0.25)
                self.assertAlmostEqual(rows[0].si_b, 0.5)

    def test_manifest_is_deterministic_and_complete(self) -> None:
        spec = dyadic_solo_spec("SoloA", "AgoB")
        inputs = _inputs(
            dimensions_a={"go_sequence": "AgoB", "social_context": "Dyadic"},
            dimensions_b={"go_sequence": "AgoB", "social_context": "SoloA"},
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = write_comparison_manifest(Path(tmp), spec, inputs)
            first = path.read_bytes()
            write_comparison_manifest(Path(tmp), spec, inputs)
            self.assertEqual(path.read_bytes(), first)
            manifest = json.loads(first)
            finalize_comparison_manifest(
                Path(tmp),
                used_session_ids=["session-1"],
                matched_pair_count=1,
                per_channel_count=1,
            )
            finalized = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(manifest["comparison_axis"], "social_context")
        self.assertEqual(len(manifest["code_fingerprint_sha256"]), 64)
        self.assertEqual(manifest["operand_labels"], {"A": "Dyadic", "B": "SoloA"})
        self.assertIn("Elmo_AgoB/SoloA/.cache", manifest["source_paths"]["B"])
        self.assertEqual(manifest["delta_definition"], "B-A: SI(SoloA) - SI(Dyadic)")
        self.assertNotIn("git_revision", manifest)
        self.assertNotIn("AgoB_vs_BgoA", json.dumps(manifest))
        self.assertEqual(finalized["status"], "complete")
        self.assertEqual(finalized["used_session_ids"], ["session-1"])

    def test_stale_cleanup_removes_legacy_names(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            stale = out / "per_channel_timing_consistency.csv"
            stale.write_text("old", encoding="utf-8")
            mislabeled = out / "si_agob_vs_bgoa_scatter.pdf"
            mislabeled.write_text("old", encoding="utf-8")
            leaked = out / "old_AgoB_summary.csv"
            leaked.write_text("old", encoding="utf-8")
            remove_stale_outputs(out, spec=dyadic_solo_spec("SoloA", "AgoB"))
            self.assertFalse(stale.exists())
            self.assertFalse(mislabeled.exists())
            self.assertFalse(leaked.exists())

    def test_exact_outputs_are_spec_derived_and_reject_social_leakage(self) -> None:
        spec = dyadic_solo_spec("SoloA", "AgoB")
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            (out / "session").mkdir()
            for array_name in ARRAY_NAMES:
                (out / "session" / f"session-1_{array_name}_{spec.file_tag}.pdf").touch()
            _touch_combined_pdfs(out, spec)
            for name in (
                "si_delta_heatmap.pdf",
                "waveform_r_heatmap.pdf",
                "delta_si_vs_mean_si_scatter.pdf",
                f"si_{spec.file_tag}_scatter.pdf",
                f"si_{spec.file_tag}_channel_median_scatter.pdf",
                "median_delta_si_by_array.pdf",
                "summary.txt",
                "paired_channel_session.csv",
                "per_channel_comparison.csv",
                "comparison_manifest.json",
            ):
                (out / name).touch()
            validate_outputs(out, ["session-1.full"], [], [], spec=spec)
            (out / "social_AgoB_leak.csv").touch()
            with self.assertRaisesRegex(RuntimeError, "Forbidden"):
                validate_outputs(out, ["session-1.full"], [], [], spec=spec)

    def test_go_sequence_canonical_case_is_not_legacy_on_windows(self) -> None:
        spec = first_second_spec("Dyadic")
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            (out / "session").mkdir()
            for array_name in ARRAY_NAMES:
                (out / "session" / f"session-1_{array_name}_{spec.file_tag}.pdf").touch()
            _touch_combined_pdfs(out, spec)
            for name in (
                "si_delta_heatmap.pdf",
                "waveform_r_heatmap.pdf",
                "delta_si_vs_mean_si_scatter.pdf",
                f"si_{spec.file_tag}_scatter.pdf",
                f"si_{spec.file_tag}_channel_median_scatter.pdf",
                "median_delta_si_by_array.pdf",
                "summary.txt",
                "paired_channel_session.csv",
                "per_channel_comparison.csv",
                "comparison_manifest.json",
            ):
                (out / name).touch()
            validate_outputs(out, ["session-1.full"], [], [], spec=spec)

    def test_unlink_case_variants_restores_canonical_casing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            legacy = out / "si_agob_vs_bgoa_scatter.pdf"
            legacy.write_text("old", encoding="utf-8")
            canonical = out / "si_AgoB_vs_BgoA_scatter.pdf"
            # Windows keeps legacy casing on overwrite without unlink.
            canonical.write_text("new", encoding="utf-8")
            names_before = {p.name for p in out.iterdir()}
            self.assertIn("si_agob_vs_bgoa_scatter.pdf", names_before)
            unlink_case_variants(canonical)
            canonical.write_text("new", encoding="utf-8")
            names_after = {p.name for p in out.iterdir()}
            self.assertEqual(names_after, {"si_AgoB_vs_BgoA_scatter.pdf"})

    def test_go_sequence_rejects_wrong_si_scatter_casing(self) -> None:
        spec = first_second_spec("Dyadic")
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            (out / "session").mkdir()
            for array_name in ARRAY_NAMES:
                (out / "session" / f"session-1_{array_name}_{spec.file_tag}.pdf").touch()
            _touch_combined_pdfs(out, spec)
            for name in (
                "si_delta_heatmap.pdf",
                "waveform_r_heatmap.pdf",
                "delta_si_vs_mean_si_scatter.pdf",
                "si_agob_vs_bgoa_scatter.pdf",
                "si_agob_vs_bgoa_channel_median_scatter.pdf",
                "median_delta_si_by_array.pdf",
                "summary.txt",
                "paired_channel_session.csv",
                "per_channel_comparison.csv",
                "comparison_manifest.json",
            ):
                (out / name).touch()
            with self.assertRaisesRegex(RuntimeError, "casing mismatch"):
                validate_outputs(out, ["session-1.full"], [], [], spec=spec)


class ComparisonRunnerTests(unittest.TestCase):
    def tearDown(self) -> None:
        reset_processing_modes()

    def test_canonical_steps_expose_comparisons_only(self) -> None:
        self.assertIn("comparisons", ALL_STEPS)
        self.assertIn("pref_unpref", ALL_STEPS)
        self.assertNotIn("timing_compare", ALL_STEPS)
        self.assertEqual(parse_steps("comparisons"), ["comparisons"])
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            self.assertEqual(parse_steps("timing_compare"), ["comparisons"])
        self.assertTrue(any(issubclass(w.category, DeprecationWarning) for w in caught))

    def test_comparison_runner_builds_expected_argv(self) -> None:
        with patch("compare_conditions.compare.main") as main:
            _run_comparisons(
                monkey="Elmo",
                social_context="solo",
                comparison_axis="social_context",
                go_seq="AgoB",
                session_lists_path=Path("session_lists.m"),
                list_name="DUAL_NHP",
            )
        argv = main.call_args.args[0]
        self.assertIn("--comparison-axis", argv)
        self.assertIn("social_context", argv)
        self.assertIn("--social-context", argv)
        self.assertIn("--go-seq", argv)
        self.assertIn("AgoB", argv)
        self.assertIn("--zscore", argv)
        self.assertIn("True", argv)
        self.assertNotIn("--only-combined", argv)

    def test_comparison_runner_forwards_only_combined(self) -> None:
        with patch("compare_conditions.compare.main") as main:
            _run_comparisons(
                monkey="Elmo",
                curated_condition="Elmo_BLOCKED",
                only_combined=True,
            )
        argv = main.call_args.args[0]
        self.assertIn("--only-combined", argv)

    def test_combined_suptitles_mirror_dyadic_solo_conventions(self) -> None:
        from analyze_stability.arrays import ARRAY_MEAN_AGGREGATION_LABEL

        spec = dyadic_solo_spec("SoloA", "AgoB")
        filters_a = {
            "TrialSubType_list": ["Dyadic"],
            "go_seq_500_list": ["AgoB"],
            "A_Reward_list": ["RA1"],
            "conf_predictability_list": ["Blocked"],
        }
        filters_b = {
            "TrialSubType_list": ["SoloA"],
            "go_seq_500_list": ["AgoB"],
            "A_Reward_list": ["RA1"],
            "conf_predictability_list": ["Blocked"],
        }
        grids, arrays = build_comparison_combined_suptitles(
            monkey="Elmo",
            spec=spec,
            filters_a=filters_a,
            filters_b=filters_b,
            alignment_label="A_InitialFixationReleaseTime_ms",
            analysis_window_ms=(-500.0, 500.0),
            smooth_ms=50.0,
            zscore_mua=True,
            n_sessions=10,
            choice_field="A_LR_pos_list",
            left_choice=["Al"],
            right_choice=["Ar"],
        )
        self.assertIn("Dyadic vs SoloA", grids)
        self.assertIn("combined sessions", grids)
        self.assertIn("A_InitialFixationReleaseTime_ms", grids)
        self.assertIn("go_sequence=AgoB", grids)
        self.assertIn("A_Reward=RA1", grids)
        self.assertIn("conf_predictability=Blocked", grids)
        self.assertNotIn("TrialSubType", grids)
        self.assertNotIn("TrialSubType", arrays)
        self.assertNotIn(ARRAY_MEAN_AGGREGATION_LABEL, grids)
        self.assertIn(ARRAY_MEAN_AGGREGATION_LABEL, arrays)
        self.assertIn("window -500:500 ms", arrays)
        self.assertIn("n_sessions=10", arrays)

        go_spec = first_second_spec("Dyadic")
        go_grids, go_arrays = build_comparison_combined_suptitles(
            monkey="Elmo",
            spec=go_spec,
            filters_a={
                "TrialSubType_list": ["Dyadic"],
                "go_seq_500_list": ["AgoB"],
            },
            filters_b={
                "TrialSubType_list": ["Dyadic"],
                "go_seq_500_list": ["BgoA"],
            },
            alignment_label="A_InitialFixationReleaseTime_ms",
            analysis_window_ms=(-500.0, 500.0),
            smooth_ms=50.0,
            zscore_mua=True,
            n_sessions=10,
            choice_field="A_LR_pos_list",
            left_choice=["Al"],
            right_choice=["Ar"],
        )
        self.assertIn("AgoB vs BgoA", go_grids)
        self.assertIn("social_context=Dyadic", go_grids)
        self.assertIn("TrialSubType=Dyadic", go_grids)
        self.assertNotIn("go_seq_500", go_grids)
        self.assertNotIn(ARRAY_MEAN_AGGREGATION_LABEL, go_grids)
        self.assertIn(ARRAY_MEAN_AGGREGATION_LABEL, go_arrays)

    @patch("run_pipeline.runner._run_comparisons")
    def test_suite_all_runs_exact_four_cell_matrix(self, run) -> None:
        _run_comparison_suite(
            monkey="Elmo",
            suite_label="Elmo_BLOCKED",
            include_solo=True,
            go_seq="all",
            curated_condition="Elmo_BLOCKED",
            session_ids=["s1.A_Elmo.B_X.SCP_01"],
        )
        calls = [call.kwargs for call in run.call_args_list]
        self.assertEqual(len(calls), 4)
        self.assertEqual(
            [(c["comparison_axis"], c["social_context"], c.get("go_seq")) for c in calls],
            [
                ("go_sequence", "dyadic", None),
                ("go_sequence", "solo", None),
                ("social_context", "dyadic", "AgoB"),
                ("social_context", "dyadic", "BgoA"),
            ],
        )

    @patch("run_pipeline.runner._run_comparisons")
    def test_suite_single_go_sequence_cannot_leak_bgoa(self, run) -> None:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            _run_comparison_suite(
                monkey="Elmo",
                suite_label="Elmo_BLOCKED",
                include_solo=True,
                go_seq="AgoB",
                curated_condition="Elmo_BLOCKED",
                session_ids=["s1.A_Elmo.B_X.SCP_01"],
            )
        self.assertEqual(run.call_count, 1)
        self.assertEqual(run.call_args.kwargs["comparison_axis"], "social_context")
        self.assertEqual(run.call_args.kwargs["go_seq"], "AgoB")

    @patch("run_pipeline.runner._run_comparisons")
    def test_suite_dyadic_only_never_runs_social_comparison(self, run) -> None:
        _run_comparison_suite(
            monkey="Elmo",
            suite_label="Elmo_BLOCKED",
            include_solo=False,
            go_seq="all",
            curated_condition="Elmo_BLOCKED",
            session_ids=["s1.A_Elmo.B_X.SCP_01"],
        )
        self.assertEqual(run.call_count, 1)
        self.assertEqual(run.call_args.kwargs["social_context"], "dyadic")
        self.assertEqual(run.call_args.kwargs["comparison_axis"], "go_sequence")

    @patch("run_pipeline.runner._run_comparisons")
    def test_suite_runs_each_request_in_both_processing_modes(self, run) -> None:
        set_processing_modes(also_original=True)
        _run_comparison_suite(
            monkey="Elmo",
            suite_label="Elmo_BLOCKED",
            include_solo=False,
            go_seq="all",
            curated_condition="Elmo_BLOCKED",
            session_ids=["s1.A_Elmo.B_X.SCP_01"],
        )
        self.assertEqual(run.call_count, 2)
        self.assertEqual(
            {call.kwargs["zscore"] for call in run.call_args_list},
            {False, True},
        )

    def test_social_context_main_resolves_same_go_sequence_distinct_branches(self) -> None:
        session_id = "s1.A_Elmo.B_X.SCP_01"
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cfg = SessionListConfig(
                list_name="Elmo_BLOCKED",
                root_folder=root / "data",
                output_folder=root / "figures",
                session_ids=[session_id],
                condition_key="Elmo_BLOCKED",
            )
            with (
                patch(
                    "compare_conditions.compare.load_curated_timing_run",
                    return_value=(cfg, [session_id], "Elmo_BLOCKED"),
                ),
                patch(
                    "compare_conditions.compare.load_condition_summaries",
                    side_effect=[[_summary(-0.2)], [_summary(0.4)]],
                ) as load,
                patch("compare_conditions.compare.run_comparison") as run,
            ):
                comparison_main(
                    [
                        "--monkey",
                        "Elmo",
                        "--curated-condition",
                        "Elmo_BLOCKED",
                        "--comparison-axis",
                        "social_context",
                        "--go-seq",
                        "AgoB",
                    ]
                )
        self.assertEqual(load.call_count, 2)
        self.assertEqual(load.call_args_list[0].args[0].parts[-2:], ("Elmo_AgoB", "Dyadic"))
        self.assertEqual(load.call_args_list[1].args[0].parts[-2:], ("Elmo_AgoB", "SoloA"))
        filters_a = run.call_args.kwargs["filters_a"]
        filters_b = run.call_args.kwargs["filters_b"]
        self.assertEqual(filters_a["go_seq_500_list"], ["AgoB"])
        self.assertEqual(filters_b["go_seq_500_list"], ["AgoB"])
        self.assertEqual(filters_a["TrialSubType_list"], ["Dyadic"])
        self.assertTrue(all(v.startswith("SoloA") for v in filters_b["TrialSubType_list"]))
        inputs = run.call_args.kwargs["inputs"]
        self.assertEqual(
            dict(inputs.dimensions_a),
            {"go_sequence": "AgoB", "social_context": "Dyadic"},
        )
        self.assertEqual(
            dict(inputs.dimensions_b),
            {"go_sequence": "AgoB", "social_context": "SoloA"},
        )


if __name__ == "__main__":
    unittest.main()
