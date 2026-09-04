"""Tests for pipeline RunDataCache and single-read enforcement."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from load_data.cache import (
    RunDataCache,
    load_pooled_disk_cache,
    load_summaries_disk_cache,
    require_pooled_disk_cache,
    write_pooled_disk_cache,
    write_summaries_disk_cache,
)
from process_channels.features import ChannelSummary
from run_pipeline.runner import run_pipeline_steps
from process_channels.preprocess import TrialBranchSpec, trial_filters_for_solo_from_dyadic
from run_pipeline.context import PipelineContext
from run_pipeline.config import reset_processing_modes, set_processing_modes


def _fake_summary(session_id: str, channel: int, t_ms: np.ndarray) -> ChannelSummary:
    mean = np.zeros_like(t_ms)
    return ChannelSummary(
        session_id=session_id,
        channel=channel,
        array_name="A1",
        index_in_array=channel,
        n_left=5,
        n_right=5,
        t_ms=t_ms,
        mean_left=mean,
        mean_right=mean,
        diff=mean,
        si=0.1,
        mwu_p=0.5,
        pref_side="L",
        evoked_p_left=0.01,
        evoked_p_right=0.2,
        task_evoked=True,
    )


def _make_mua(n_trials: int = 6, n_time: int = 21) -> np.ndarray:
    rng = np.random.default_rng(0)
    return rng.random((n_trials, n_time))


class PipelineCacheTests(unittest.TestCase):
    def _build_ctx(self, tmp: Path, session_ids: list[str]) -> PipelineContext:
        return PipelineContext(
            data_root=tmp / "data",
            session_ids=session_ids,
            output_base=tmp / "out",
            condition_label="Test_AgoB",
            condition_key="Test_AgoB",
            layout="flat",
            trial_filters={"go_seq_500_list": ["AgoB"]},
            choice_field="A_LR_pos_list",
            left_choice=["Al"],
            right_choice=["Ar"],
            recording_monkey="Curius",
        )

    def _setup_session_tree(
        self,
        ctx: PipelineContext,
        channels: list[int],
        *,
        alignment_event: str = "A_InitialFixationReleaseTime_ms",
        labels: dict | None = None,
    ) -> tuple[np.ndarray, dict]:
        t_ms = np.linspace(-1000, 1000, 21)
        if labels is None:
            labels = {
                "A_LR_pos_list": np.array(["Al", "Al", "Al", "Ar", "Ar", "Ar"]),
                "TrialSubType_list": np.array(["Dyadic"] * 6),
                "go_seq_500_list": np.array(["AgoB"] * 6),
                "conf_predictability_list": np.array(["Blocked"] * 6),
                "A_Reward_list": np.array(["RA1"] * 6),
            }
        for session_id in ctx.session_ids:
            event_dir = ctx.session_dir(session_id) / alignment_event
            event_dir.mkdir(parents=True)
            for ch in channels:
                name = (
                    f"{session_id}.{alignment_event}.ch{ch:03d}."
                    "MUA.pre1000ms.post1000ms.event_aligned_data.mat"
                )
                (event_dir / name).touch()

        return t_ms, labels

    @mock.patch("load_data.cache.load_trial_labels")
    @mock.patch("load_data.cache.load_time_vector")
    @mock.patch("load_data.cache.zscore_reference_mask", return_value=np.ones(6, dtype=bool))
    @mock.patch("load_data.cache.zscore_channel_trials", side_effect=lambda mua, **_: mua * 0.5)
    @mock.patch("load_data.cache.RunDataCache._load_channel_mat")
    def test_one_loadmat_per_session_channel(
        self,
        mock_loadmat,
        _zscore,
        _zref,
        mock_time,
        mock_labels,
    ) -> None:
        mock_loadmat.side_effect = lambda _path: _make_mua()

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            ctx = self._build_ctx(tmp, ["sess1.A_Curius.B_X.SCP_01", "sess2.A_Curius.B_X.SCP_01"])
            t_ms, labels = self._setup_session_tree(ctx, [1, 2])
            mock_time.return_value = t_ms
            mock_labels.return_value = labels

            cache = RunDataCache(ctx, ctx.session_ids)
            mock_loadmat.side_effect = lambda _path: _make_mua()
            cache.populate(["session_lr", "consistency", "combine", "array_combined"])

            self.assertEqual(mock_loadmat.call_count, 4)  # 2 sessions × 2 channels
            cache.ensure_pooled("dyadic")
            self.assertIsNotNone(cache.pooled)
            self.assertIn(True, cache.summaries)
            self.assertGreater(len(cache.summaries[True]), 0)

    @mock.patch("load_data.cache.load_trial_labels")
    @mock.patch("load_data.cache.load_time_vector")
    @mock.patch("load_data.cache.zscore_reference_mask", return_value=np.ones(6, dtype=bool))
    @mock.patch("load_data.cache.zscore_channel_trials", side_effect=lambda mua, **_: mua * 0.5)
    @mock.patch("load_data.cache.RunDataCache._load_channel_mat")
    def test_b_side_cache_uses_b_alignment_event(
        self,
        mock_loadmat,
        _zscore,
        _zref,
        mock_time,
        mock_labels,
    ) -> None:
        mock_loadmat.side_effect = lambda _path: _make_mua()
        elmo_session = "20230630T115937B.A_Curius.B_Elmo.SCP_01"

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            ctx = PipelineContext(
                data_root=tmp / "data",
                session_ids=[elmo_session],
                output_base=tmp / "out",
                condition_label="Elmo_AgoB",
                condition_key="Elmo_AgoB",
                layout="flat",
                trial_filters={
                    "TrialSubType_list": ["Dyadic"],
                    "go_seq_500_list": ["AgoB"],
                    "B_Reward_list": ["RB1", "RB2", "RB3", "RB4"],
                },
                choice_field="B_LR_pos_list",
                left_choice=["Bl"],
                right_choice=["Br"],
                recording_monkey="Elmo",
            )
            b_labels = {
                "B_LR_pos_list": np.array(["Bl", "Bl", "Bl", "Br", "Br", "Br"]),
                "TrialSubType_list": np.array(["Dyadic"] * 6),
                "go_seq_500_list": np.array(["AgoB"] * 6),
                "B_Reward_list": np.array(["RB1"] * 6),
            }
            t_ms, labels = self._setup_session_tree(
                ctx,
                [1],
                alignment_event="B_InitialFixationReleaseTime_ms",
                labels=b_labels,
            )
            mock_time.return_value = t_ms
            mock_labels.return_value = labels

            cache = RunDataCache(ctx, ctx.session_ids)
            cache.populate(["session_lr", "consistency"])

            self.assertEqual(mock_loadmat.call_count, 1)
            self.assertGreater(len(cache.summaries[True]), 0)
            loaded_path = mock_loadmat.call_args.args[0]
            self.assertIn("B_InitialFixationReleaseTime_ms", str(loaded_path))

    @mock.patch("load_data.cache.load_trial_labels")
    @mock.patch("load_data.cache.load_time_vector")
    @mock.patch("load_data.cache.zscore_reference_mask", return_value=np.ones(6, dtype=bool))
    @mock.patch("load_data.cache.zscore_channel_trials", side_effect=lambda mua, **_: mua * 0.5)
    @mock.patch("load_data.cache.RunDataCache._load_channel_mat")
    @mock.patch("analyze_stability.combine.plot_from_pooled")
    @mock.patch("analyze_stability.arrays.plot_condition_array_combined")
    def test_combine_and_array_combined_share_pooled(
        self,
        mock_array,
        mock_combine,
        mock_loadmat,
        _zscore,
        _zref,
        mock_time,
        mock_labels,
    ) -> None:
        mock_loadmat.side_effect = lambda _path: _make_mua()

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            ctx = self._build_ctx(tmp, ["sess1.A_Curius.B_X.SCP_01"])
            t_ms, labels = self._setup_session_tree(ctx, [1])
            mock_time.return_value = t_ms
            mock_labels.return_value = labels

            cache = RunDataCache(ctx, ctx.session_ids)
            cache.populate(["combine", "array_combined"])
            cache.ensure_pooled("dyadic")
            pooled = cache.pooled
            self.assertIsNotNone(pooled)

            from analyze_stability import combine as cs
            from analyze_stability.arrays import plot_condition_array_combined

            cs.plot_from_pooled(pooled, ctx.condition_label, tmp / "comb", ctx.session_ids)
            plot_condition_array_combined(ctx, pooled=pooled)

            mock_combine.assert_called_once()
            self.assertIs(mock_combine.call_args.args[0], pooled)
            mock_array.assert_called_once()
            self.assertIs(mock_array.call_args.kwargs["pooled"], pooled)

    @mock.patch("analyze_stability.consistency.run_from_cache")
    @mock.patch("analyze_stability.deep_dives.run_from_cache")
    @mock.patch("load_data.cache.RunDataCache.populate")
    @mock.patch("run_pipeline.runner.verify_sessions")
    def test_best_worst_skipped_when_consistency_ran(
        self,
        mock_verify,
        _populate,
        mock_best,
        mock_consistency,
    ) -> None:
        mock_verify.return_value = [
            "sess1.A_Curius.B_X.SCP_01",
            "sess2.A_Curius.B_X.SCP_01",
            "sess3.A_Curius.B_X.SCP_01",
        ]
        ctx = PipelineContext(
            data_root=Path("/tmp/data"),
            session_ids=mock_verify.return_value,
            output_base=Path("/tmp/out"),
            condition_label="Test_AgoB",
            condition_key="Test_AgoB",
            layout="flat",
            trial_filters={},
            recording_monkey="Curius",
        )
        run_pipeline_steps(ctx, ["consistency", "best_worst"])
        mock_consistency.assert_called_once()
        mock_best.assert_not_called()

    def test_disk_cache_roundtrip(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            t_ms = np.linspace(-500, 500, 11)
            summaries = [_fake_summary("s1", 1, t_ms), _fake_summary("s1", 2, t_ms)]
            write_summaries_disk_cache(tmp, "Test_AgoB", True, summaries)
            loaded = load_summaries_disk_cache(tmp, "Test_AgoB", True)
            self.assertIsNotNone(loaded)
            assert loaded is not None
            self.assertEqual(len(loaded), 2)
            self.assertEqual(loaded[0].session_id, "s1")
            self.assertEqual(loaded[0].channel, 1)
            np.testing.assert_allclose(loaded[0].t_ms, t_ms)

    def test_pooled_disk_cache_roundtrip(self) -> None:
        from load_data.cache import PooledTrialData

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            t_ms = np.linspace(-500, 500, 5)
            win_idx = np.arange(5, dtype=np.int32)
            pooled = PooledTrialData(
                t_ms=t_ms,
                win_idx=win_idx,
                left_by_ch={1: [np.ones((2, 5)), np.full((3, 5), 2.0)]},
                right_by_ch={1: [np.zeros((2, 5))]},
                sess_left_by_ch={1: [1.0, 2.0]},
                sess_right_by_ch={1: [0.5]},
            )
            write_pooled_disk_cache(tmp, "Test_AgoB", True, pooled)
            loaded = load_pooled_disk_cache(tmp, "Test_AgoB", True)
            self.assertIsNotNone(loaded)
            assert loaded is not None
            np.testing.assert_allclose(loaded.t_ms, t_ms)
            self.assertEqual(len(loaded.left_by_ch[1]), 2)
            self.assertEqual(loaded.left_by_ch[1][0].shape, (2, 5))
            self.assertEqual(loaded.left_by_ch[1][1].shape, (3, 5))
            self.assertEqual(loaded.sess_left_by_ch[1], [1.0, 2.0])

    def test_require_pooled_disk_cache_missing_path(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            with self.assertRaises(FileNotFoundError) as ctx:
                require_pooled_disk_cache(tmp, "Test_AgoB", True)
            self.assertIn("pooled_zscore.npz", str(ctx.exception))
            self.assertIn(str(tmp / ".cache"), str(ctx.exception))

    @mock.patch("load_data.cache.load_trial_labels")
    @mock.patch("load_data.cache.load_time_vector")
    @mock.patch("load_data.cache.zscore_reference_mask", return_value=np.ones(6, dtype=bool))
    @mock.patch("load_data.cache.zscore_channel_trials", side_effect=lambda mua, **_: mua * 0.5)
    @mock.patch("load_data.cache.RunDataCache._load_channel_mat")
    def test_agob_bgoa_one_loadmat_per_channel(
        self,
        mock_loadmat,
        _zscore,
        _zref,
        mock_time,
        mock_labels,
    ) -> None:
        mock_loadmat.side_effect = lambda _path: _make_mua()
        labels = {
            "A_LR_pos_list": np.array(["Al", "Al", "Al", "Ar", "Ar", "Ar"]),
            "TrialSubType_list": np.array(["Dyadic"] * 6),
            "go_seq_500_list": np.array(
                ["AgoB", "AgoB", "AgoB", "BgoA", "BgoA", "BgoA"],
            ),
            "conf_predictability_list": np.array(["Blocked"] * 6),
            "A_Reward_list": np.array(["RA1"] * 6),
        }
        branch_specs = [
            TrialBranchSpec(
                "Elmo_AgoB__dyadic",
                "Elmo_AgoB/Dyadic",
                {
                    "TrialSubType_list": ["Dyadic"],
                    "go_seq_500_list": ["AgoB"],
                    "conf_predictability_list": ["Blocked"],
                    "A_Reward_list": ["RA1", "RA2", "RA3", "RA4"],
                },
                condition_label="Elmo_AgoB",
            ),
            TrialBranchSpec(
                "Elmo_BgoA__dyadic",
                "Elmo_BgoA/Dyadic",
                {
                    "TrialSubType_list": ["Dyadic"],
                    "go_seq_500_list": ["BgoA"],
                    "conf_predictability_list": ["Blocked"],
                    "A_Reward_list": ["RA1", "RA2", "RA3", "RA4"],
                },
                condition_label="Elmo_BgoA",
            ),
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            ctx = PipelineContext(
                data_root=tmp / "data",
                session_ids=["sess1.A_Curius.B_X.SCP_01"],
                output_base=tmp / "out",
                condition_label="Elmo_BLOCKED",
                condition_key="Elmo_BLOCKED",
                layout="flat",
                trial_filters={},
                choice_field="A_LR_pos_list",
                left_choice=["Al"],
                right_choice=["Ar"],
                recording_monkey="Curius",
            )
            t_ms, _ = self._setup_session_tree(ctx, [1, 2])
            mock_time.return_value = t_ms
            mock_labels.return_value = labels

            cache = RunDataCache(ctx, ctx.session_ids)
            cache.populate(["session_lr", "combine"], branch_specs)

            self.assertEqual(mock_loadmat.call_count, 2)
            self.assertEqual(len(cache.branch("Elmo_AgoB__dyadic").summaries[True]), 2)
            self.assertEqual(len(cache.branch("Elmo_BgoA__dyadic").summaries[True]), 2)
            self.assertEqual(cache.branch("Elmo_AgoB__dyadic").trials.get(True, {}), {})
            self.assertIsNone(cache.branch("Elmo_AgoB__dyadic").pooled)
            cache.ensure_pooled("Elmo_AgoB__dyadic")
            cache.ensure_pooled("Elmo_BgoA__dyadic")
            self.assertIsNotNone(cache.branch("Elmo_AgoB__dyadic").pooled)
            self.assertIsNotNone(cache.branch("Elmo_BgoA__dyadic").pooled)
            cache.release_branch("Elmo_AgoB__dyadic")
            self.assertIsNone(cache.branch("Elmo_AgoB__dyadic").pooled)
            self.assertEqual(cache.branch("Elmo_AgoB__dyadic").summaries.get(True, []), [])
            self.assertIsNotNone(cache.branch("Elmo_BgoA__dyadic").pooled)

    @mock.patch("load_data.cache.load_trial_labels")
    @mock.patch("load_data.cache.load_time_vector")
    @mock.patch("load_data.cache.zscore_reference_mask", return_value=np.ones(6, dtype=bool))
    @mock.patch("load_data.cache.zscore_channel_trials", side_effect=lambda mua, **_: mua * 0.5)
    @mock.patch("load_data.cache.RunDataCache._load_channel_mat")
    def test_also_original_populates_raw_summaries(
        self,
        mock_loadmat,
        _zscore,
        _zref,
        mock_time,
        mock_labels,
    ) -> None:
        set_processing_modes(also_original=True)
        mock_loadmat.side_effect = lambda _path: _make_mua()

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            ctx = self._build_ctx(tmp, ["sess1.A_Curius.B_X.SCP_01"])
            t_ms, labels = self._setup_session_tree(ctx, [1])
            mock_time.return_value = t_ms
            mock_labels.return_value = labels

            cache = RunDataCache(ctx, ctx.session_ids)
            cache.populate(["session_lr", "consistency"])

            self.assertIn(False, cache.summaries)
            self.assertIn(True, cache.summaries)
            self.assertGreater(len(cache.summaries[False]), 0)
            self.assertGreater(len(cache.summaries[True]), 0)
        reset_processing_modes()

    @mock.patch("load_data.cache.load_trial_labels")
    @mock.patch("load_data.cache.load_time_vector")
    @mock.patch("load_data.cache.zscore_reference_mask", return_value=np.ones(6, dtype=bool))
    @mock.patch("load_data.cache.zscore_channel_trials", side_effect=lambda mua, **_: mua * 0.5)
    @mock.patch("load_data.cache.RunDataCache._load_channel_mat")
    def test_multi_branch_one_loadmat_per_channel(
        self,
        mock_loadmat,
        _zscore,
        _zref,
        mock_time,
        mock_labels,
    ) -> None:
        mock_loadmat.side_effect = lambda _path: _make_mua()

        dyadic_filters = {
            "TrialSubType_list": ["Dyadic"],
            "go_seq_500_list": ["AgoB"],
            "A_Reward_list": ["RA1", "RA2", "RA3", "RA4"],
        }
        solo_filters = trial_filters_for_solo_from_dyadic(dyadic_filters, "A")
        branch_specs = [
            TrialBranchSpec("dyadic", "Dyadic", dyadic_filters),
            TrialBranchSpec("solo_a", "SoloA", solo_filters),
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            ctx = self._build_ctx(tmp, ["sess1.A_Curius.B_X.SCP_01"])
            ctx = PipelineContext(
                data_root=ctx.data_root,
                session_ids=ctx.session_ids,
                output_base=ctx.output_base,
                condition_label=ctx.condition_label,
                condition_key=ctx.condition_key,
                layout=ctx.layout,
                trial_filters=dyadic_filters,
                choice_field=ctx.choice_field,
                left_choice=ctx.left_choice,
                right_choice=ctx.right_choice,
                recording_monkey=ctx.recording_monkey,
            )
            t_ms, _ = self._setup_session_tree(ctx, [1, 2])
            labels = {
                "A_LR_pos_list": np.array(["Al", "Al", "Al", "Al", "Al", "Al"]),
                "TrialSubType_list": np.array(
                    ["Dyadic", "Dyadic", "Dyadic", "SoloA", "SoloA", "SoloA"],
                ),
                "go_seq_500_list": np.array(["AgoB"] * 6),
                "conf_predictability_list": np.array(["Blocked"] * 6),
                "A_Reward_list": np.array(["RA1"] * 6),
            }
            mock_time.return_value = t_ms
            mock_labels.return_value = labels

            cache = RunDataCache(ctx, ctx.session_ids)
            cache.populate(["session_lr", "consistency"], branch_specs)

            self.assertEqual(mock_loadmat.call_count, 2)
            self.assertEqual(len(cache.branch("dyadic").summaries[True]), 2)
            self.assertEqual(len(cache.branch("solo_a").summaries[True]), 2)
            dyadic_n_left = cache.branch("dyadic").summaries[True][0].n_left
            solo_n_left = cache.branch("solo_a").summaries[True][0].n_left
            self.assertEqual(dyadic_n_left, 3)
            self.assertEqual(solo_n_left, 3)

    def test_solo_disk_cache_path(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            t_ms = np.linspace(-500, 500, 11)
            summaries = [_fake_summary("s1", 1, t_ms)]
            solo_base = tmp / "Curius_AgoB" / "SoloA"
            write_summaries_disk_cache(solo_base, "Curius_AgoB", True, summaries)
            loaded = load_summaries_disk_cache(solo_base, "Curius_AgoB", True)
            self.assertIsNotNone(loaded)
            assert loaded is not None
            self.assertEqual(len(loaded), 1)


if __name__ == "__main__":
    unittest.main()
