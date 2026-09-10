"""Tests for per-session preferred vs unpreferred gating and averages."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import numpy as np

from analyze_stability.pref_unpref import (
    branch_combined_pref_filenames,
    channel_combined_trace,
    comparison_combined_pref_filenames,
    comparison_solo_locked_pref_filenames,
    grand_from_locked_sessions,
    grand_mean_across_sessions,
    is_tuned,
    locked_session_traces,
    paired_session_ids,
    pref_unpref_means,
    remap_pref_unpref,
    run_from_summaries,
    session_tuned_mean,
    traces_by_channel,
    write_comparison_pref_combined,
    write_paired_pref_session_combined,
)
from process_channels.features import ChannelSummary
from run_pipeline.context import PipelineContext, set_active_context


def _summary(
    session_id: str,
    channel: int,
    *,
    left: float,
    right: float,
    mwu_p: float | None,
    pref_side: str,
    si: float | None = None,
) -> ChannelSummary:
    t_ms = np.linspace(-100, 100, 5)
    mean_left = np.full_like(t_ms, left, dtype=float)
    mean_right = np.full_like(t_ms, right, dtype=float)
    if si is None:
        denom = abs(left) + abs(right)
        si = (left - right) / denom if denom else 0.0
    return ChannelSummary(
        session_id=session_id,
        channel=channel,
        array_name="A1",
        index_in_array=channel,
        n_left=5,
        n_right=5,
        t_ms=t_ms,
        mean_left=mean_left,
        mean_right=mean_right,
        diff=mean_left - mean_right,
        si=si,
        mwu_p=mwu_p,
        pref_side=pref_side,
        evoked_p_left=0.01,
        evoked_p_right=0.2,
        task_evoked=True,
    )


class PrefUnprefGateTests(unittest.TestCase):
    def test_mwu_p_below_alpha_is_tuned(self) -> None:
        self.assertTrue(is_tuned(_summary("s", 1, left=2, right=0, mwu_p=0.04, pref_side="L")))

    def test_mwu_p_at_or_above_alpha_is_not_tuned(self) -> None:
        self.assertFalse(is_tuned(_summary("s", 1, left=2, right=0, mwu_p=0.05, pref_side="L")))
        self.assertFalse(is_tuned(_summary("s", 1, left=2, right=0, mwu_p=0.06, pref_side="L")))

    def test_missing_p_or_pref_side_is_not_tuned(self) -> None:
        self.assertFalse(is_tuned(_summary("s", 1, left=2, right=0, mwu_p=None, pref_side="L")))
        self.assertFalse(is_tuned(_summary("s", 1, left=2, right=0, mwu_p=0.01, pref_side="")))

    def test_weak_si_still_tuned_if_mwu_significant(self) -> None:
        self.assertTrue(
            is_tuned(_summary("s", 1, left=1.05, right=1.0, mwu_p=0.01, pref_side="L", si=0.05))
        )

    def test_left_pref_keeps_l_as_pref(self) -> None:
        pref, unpref = pref_unpref_means(
            _summary("s", 1, left=3.0, right=1.0, mwu_p=0.01, pref_side="L"),
        )
        np.testing.assert_allclose(pref, 3.0)
        np.testing.assert_allclose(unpref, 1.0)

    def test_right_pref_swaps(self) -> None:
        pref, unpref = pref_unpref_means(
            _summary("s", 1, left=1.0, right=4.0, mwu_p=0.01, pref_side="R"),
        )
        np.testing.assert_allclose(pref, 4.0)
        np.testing.assert_allclose(unpref, 1.0)

    def test_session_mean_is_equal_weight_of_channels(self) -> None:
        a = _summary("s", 1, left=2.0, right=0.0, mwu_p=0.01, pref_side="L")
        b = _summary("s", 2, left=4.0, right=0.0, mwu_p=0.01, pref_side="L")
        result = session_tuned_mean([a, b])
        self.assertIsNotNone(result)
        _, mean_pref, mean_unpref = result
        np.testing.assert_allclose(mean_pref, 3.0)
        np.testing.assert_allclose(mean_unpref, 0.0)

    def test_untuned_channel_excluded_from_session_mean(self) -> None:
        tuned = _summary("s", 1, left=2.0, right=0.0, mwu_p=0.01, pref_side="L")
        ns = _summary("s", 2, left=100.0, right=0.0, mwu_p=0.4, pref_side="L")
        _, mean_pref, _ = session_tuned_mean([tuned, ns])
        np.testing.assert_allclose(mean_pref, 2.0)

    def test_grand_mean_and_sem_across_two_sessions(self) -> None:
        pref_a = np.array([0.0, 2.0, 4.0])
        pref_b = np.array([2.0, 4.0, 6.0])
        unpref_a = np.zeros(3)
        unpref_b = np.zeros(3)
        mean_p, sem_p, mean_u, sem_u = grand_mean_across_sessions(
            [pref_a, pref_b], [unpref_a, unpref_b],
        )
        np.testing.assert_allclose(mean_p, [1.0, 3.0, 5.0])
        np.testing.assert_allclose(mean_u, 0.0)
        expected_sem = np.std([0.0, 2.0], ddof=1) / np.sqrt(2)
        np.testing.assert_allclose(sem_p[0], expected_sem)
        np.testing.assert_allclose(sem_u, 0.0)

    def test_grand_mean_none_when_no_sessions(self) -> None:
        self.assertIsNone(grand_mean_across_sessions([], []))

    def test_channel_combined_equal_weight_across_gated_sessions(self) -> None:
        a = _summary("s1", 1, left=2.0, right=0.0, mwu_p=0.01, pref_side="L")
        b = _summary("s2", 1, left=4.0, right=0.0, mwu_p=0.01, pref_side="L")
        ns = _summary("s3", 1, left=100.0, right=0.0, mwu_p=0.4, pref_side="L")
        other = _summary("s1", 2, left=9.0, right=0.0, mwu_p=0.01, pref_side="L")
        trace = channel_combined_trace([a, b, ns, other], 1)
        self.assertIsNotNone(trace)
        np.testing.assert_allclose(trace.mean_pref, 3.0)
        np.testing.assert_allclose(trace.mean_unpref, 0.0)
        self.assertEqual(trace.n_sessions, 2)

    def test_right_pref_session_swaps_before_channel_average(self) -> None:
        left_pref = _summary("s1", 1, left=2.0, right=0.0, mwu_p=0.01, pref_side="L")
        right_pref = _summary("s2", 1, left=0.0, right=4.0, mwu_p=0.01, pref_side="R")
        trace = channel_combined_trace([left_pref, right_pref], 1)
        np.testing.assert_allclose(trace.mean_pref, 3.0)
        np.testing.assert_allclose(trace.mean_unpref, 0.0)

    def test_untuned_channel_absent_from_traces_by_channel(self) -> None:
        ns = _summary("s", 1, left=2.0, right=0.0, mwu_p=0.4, pref_side="L")
        self.assertEqual(traces_by_channel([ns]), {})


class PrefUnprefWriteTests(unittest.TestCase):
    def tearDown(self) -> None:
        set_active_context(None)

    def test_run_writes_combined_pref_next_to_lr(self) -> None:
        sid1 = "20201204T125624.A_Elmo.B_FS.SCP_01"
        sid2 = "20201208T125551.A_Elmo.B_FS.SCP_01"
        summaries = [
            _summary(sid1, 1, left=2.0, right=0.0, mwu_p=0.01, pref_side="L"),
            _summary(sid1, 2, left=0.0, right=2.0, mwu_p=0.02, pref_side="R"),
            _summary(sid2, 1, left=3.0, right=0.0, mwu_p=0.01, pref_side="L"),
            _summary(sid2, 2, left=1.0, right=0.0, mwu_p=0.4, pref_side="L"),
        ]
        with tempfile.TemporaryDirectory() as tmp:
            ctx = PipelineContext(
                data_root=Path(tmp) / "data",
                session_ids=[sid1, sid2],
                output_base=Path(tmp) / "out",
                condition_label="Elmo_AgoB",
                condition_key="Elmo_BLOCKED",
                layout="flat",
                trial_filters={"go_seq_500_list": ["AgoB"]},
                recording_monkey="Elmo",
            )
            dest = run_from_summaries(ctx, summaries)
            self.assertEqual(dest, Path(tmp) / "out" / "combined")
            event = "A_InitialFixationReleaseTime_ms"
            names = branch_combined_pref_filenames("Elmo_AgoB", event)
            for name in names:
                self.assertTrue((dest / name).is_file(), name)
            self.assertFalse((dest.parent / "pref_unpref").exists())
            self.assertFalse(any(dest.glob("*_tuned_mean.pdf")))
            self.assertFalse(any(dest.glob("across_sessions_mean.pdf")))

    def test_comparison_write_emits_overlay_pref_pdfs(self) -> None:
        sid = "20201204T125624.A_Elmo.B_FS.SCP_01"
        summaries_a = [_summary(sid, 1, left=2.0, right=0.0, mwu_p=0.01, pref_side="L")]
        summaries_b = [_summary(sid, 1, left=0.0, right=3.0, mwu_p=0.01, pref_side="R")]
        with tempfile.TemporaryDirectory() as tmp:
            dest = Path(tmp) / "combined"
            write_comparison_pref_combined(
                summaries_a,
                summaries_b,
                dest,
                file_tag="Dyadic_vs_SoloA",
                label_a="Dyadic",
                label_b="SoloA",
                suptitle_grids="test grids",
                suptitle_arrays="test arrays",
            )
            for name in comparison_combined_pref_filenames("Dyadic_vs_SoloA"):
                self.assertTrue((dest / name).is_file(), name)
            self.assertEqual(sum(1 for _ in dest.glob("*.pdf")), 6)


class SoloLockedPairedTests(unittest.TestCase):
    def test_solo_pref_remaps_flipped_dyadic(self) -> None:
        sid = "s1"
        dyadic = _summary(sid, 1, left=2.0, right=8.0, mwu_p=0.4, pref_side="R")
        solo = _summary(sid, 1, left=5.0, right=1.0, mwu_p=0.01, pref_side="L")
        traces = locked_session_traces([dyadic], [solo], lock="solo")
        self.assertEqual(len(traces), 1)
        np.testing.assert_allclose(traces[0].mean_pref_a, 2.0)
        np.testing.assert_allclose(traces[0].mean_unpref_a, 8.0)
        np.testing.assert_allclose(traces[0].mean_pref_b, 5.0)
        np.testing.assert_allclose(traces[0].mean_unpref_b, 1.0)

    def test_ns_solo_drops_channel_even_if_dyadic_tuned(self) -> None:
        sid = "s1"
        dyadic = _summary(sid, 1, left=8.0, right=0.0, mwu_p=0.01, pref_side="L")
        solo = _summary(sid, 1, left=5.0, right=1.0, mwu_p=0.4, pref_side="L")
        self.assertEqual(locked_session_traces([dyadic], [solo], lock="solo"), [])

    def test_session_missing_from_solo_is_dropped(self) -> None:
        dyadic = [
            _summary("s1", 1, left=2.0, right=0.0, mwu_p=0.01, pref_side="L"),
            _summary("s2", 1, left=3.0, right=0.0, mwu_p=0.01, pref_side="L"),
        ]
        solo = [_summary("s1", 1, left=4.0, right=0.0, mwu_p=0.01, pref_side="L")]
        self.assertEqual(paired_session_ids(dyadic, solo), ["s1"])
        traces = locked_session_traces(dyadic, solo, lock="solo")
        self.assertEqual([row.session_id for row in traces], ["s1"])

    def test_session_mean_is_equal_weight_of_all_gated_channels(self) -> None:
        sid = "s1"
        dyadic = [
            _summary(sid, 1, left=2.0, right=0.0, mwu_p=0.4, pref_side="L"),
            _summary(sid, 2, left=6.0, right=0.0, mwu_p=0.4, pref_side="L"),
        ]
        solo = [
            _summary(sid, 1, left=1.0, right=0.0, mwu_p=0.01, pref_side="L"),
            _summary(sid, 2, left=3.0, right=0.0, mwu_p=0.01, pref_side="L"),
        ]
        traces = locked_session_traces(dyadic, solo, lock="solo")
        self.assertEqual(traces[0].n_channels, 2)
        np.testing.assert_allclose(traces[0].mean_pref_a, 4.0)
        np.testing.assert_allclose(traces[0].mean_pref_b, 2.0)

    def test_grand_mean_equal_weight_sessions(self) -> None:
        dyadic = [
            _summary("s1", 1, left=2.0, right=0.0, mwu_p=0.4, pref_side="L"),
            _summary("s2", 1, left=6.0, right=0.0, mwu_p=0.4, pref_side="L"),
        ]
        solo = [
            _summary("s1", 1, left=1.0, right=0.0, mwu_p=0.01, pref_side="L"),
            _summary("s2", 1, left=3.0, right=0.0, mwu_p=0.01, pref_side="L"),
        ]
        grand = grand_from_locked_sessions(locked_session_traces(dyadic, solo, lock="solo"))
        self.assertIsNotNone(grand)
        np.testing.assert_allclose(grand.mean_pref_a, 4.0)
        np.testing.assert_allclose(grand.mean_pref_b, 2.0)
        self.assertEqual(grand.n_sessions, 2)

    def test_remap_uses_explicit_side(self) -> None:
        row = _summary("s", 1, left=2.0, right=9.0, mwu_p=0.4, pref_side="R")
        pref, unpref = remap_pref_unpref(row, "L")
        np.testing.assert_allclose(pref, 2.0)
        np.testing.assert_allclose(unpref, 9.0)

    def test_write_emits_solo_locked_pdfs(self) -> None:
        sid = "20201204T125624.A_Curius.B_FS.SCP_01"
        dyadic = [_summary(sid, 1, left=2.0, right=0.0, mwu_p=0.4, pref_side="L")]
        solo = [_summary(sid, 1, left=3.0, right=0.0, mwu_p=0.01, pref_side="L")]
        with tempfile.TemporaryDirectory() as tmp:
            dest = Path(tmp) / "combined"
            write_paired_pref_session_combined(
                dyadic,
                solo,
                dest,
                file_tag="Dyadic_vs_SoloA",
                label_a="Dyadic",
                label_b="SoloA",
                suptitle="test locked",
            )
            for name in comparison_solo_locked_pref_filenames("Dyadic_vs_SoloA"):
                self.assertTrue((dest / name).is_file(), name)
            self.assertTrue((dest / "Dyadic_vs_SoloA_sessions_mean_pref_solo_locked.npz").is_file())


if __name__ == "__main__":
    unittest.main()
