"""Two-sample decode comparison: cluster perm + same_diff cache loader."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from analyze_decoding.compare_lists import (
    compare_output_dir,
    compare_same_diff_panels,
    load_same_diff_combined,
    list_name,
    same_diff_combined_npz,
    save_same_diff_compare,
)
from analyze_decoding.stats_cluster import cluster_p_two_sample, welch_t_curves


def _fake_same_diff_npz(path: Path, *, n_sess: int, n_bins: int, offset: float) -> None:
    t = np.linspace(-1000.0, 1000.0, n_bins)
    rng = np.random.default_rng(0)
    curves = rng.normal(0.5 + offset, 0.02, size=(n_sess, n_bins))
    mean = np.mean(curves, axis=0)
    payload = {
        "go_seq": "AgoB",
        "align_0": "A_fixspotRel",
        "align_1": "B_fixspotRel",
    }
    for pfx in ("A", "B"):
        payload[f"{pfx}_mean"] = mean
        payload[f"{pfx}_ci_low"] = mean - 0.05
        payload[f"{pfx}_ci_high"] = mean + 0.05
        payload[f"{pfx}_cluster_mask"] = np.zeros(n_bins, dtype=bool)
        payload[f"{pfx}_session_curves"] = curves
        payload[f"{pfx}_bin_centers_ms"] = t
        payload[f"{pfx}_n_sessions"] = n_sess
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **payload)


class TwoSampleClusterTests(unittest.TestCase):
    def test_identical_groups_no_clusters(self) -> None:
        rng = np.random.default_rng(1)
        a = rng.normal(0.5, 0.05, size=(16, 21))
        result = cluster_p_two_sample(a, a.copy(), n_perm=200, seed=0)
        self.assertFalse(np.any(result.mask))
        self.assertEqual(result.clusters, [])

    def test_mid_window_bump_detected(self) -> None:
        rng = np.random.default_rng(2)
        n_sess, n_bins = 24, 21
        a = rng.normal(0.50, 0.04, size=(n_sess, n_bins))
        b = rng.normal(0.50, 0.04, size=(n_sess, n_bins))
        b[:, 8:13] += 0.28
        result = cluster_p_two_sample(a, b, n_perm=500, seed=0)
        self.assertTrue(np.any(result.mask[8:13]))
        self.assertFalse(np.any(result.mask[:4]))
        self.assertFalse(np.any(result.mask[-4:]))
        self.assertTrue(np.isfinite(result.threshold))
        self.assertTrue(any(c.p_value < 0.05 for c in result.clusters))

    def test_welch_t_sign(self) -> None:
        rng = np.random.default_rng(0)
        a = rng.normal(0.8, 0.02, size=(10, 5))
        b = rng.normal(0.4, 0.02, size=(10, 5))
        t = welch_t_curves(a, b)
        self.assertTrue(np.all(t > 0))

    def test_too_few_sessions(self) -> None:
        a = np.ones((1, 8))
        b = np.ones((5, 8))
        result = cluster_p_two_sample(a, b, n_perm=50)
        self.assertEqual(result.mask.size, 8)
        self.assertFalse(np.any(result.mask))


class SameDiffLoaderTests(unittest.TestCase):
    def test_paths(self) -> None:
        root = Path("/data")
        self.assertEqual(list_name("Curius", True), "Curius_BLOCKED_CONF")
        p = same_diff_combined_npz(root, "Curius", "AgoB", blocked=False)
        self.assertEqual(
            p,
            root
            / "Curius_SHUFFLED_CONF"
            / "Curius_AgoB"
            / "Dyadic"
            / "decoding"
            / "same_diff"
            / "combined"
            / "mean_ci_decode.npz",
        )
        out = compare_output_dir(root, "Elmo", "AgoB", "same_diff")
        self.assertEqual(
            out,
            root
            / "decode_compare"
            / "Elmo_BLOCKED_vs_SHUFFLED"
            / "Elmo_AgoB"
            / "Dyadic"
            / "decoding"
            / "same_diff",
        )

    def test_load_and_compare_fake_npz(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            blk = tmp_path / "blocked.npz"
            shf = tmp_path / "shuffled.npz"
            _fake_same_diff_npz(blk, n_sess=8, n_bins=11, offset=0.0)
            _fake_same_diff_npz(shf, n_sess=8, n_bins=11, offset=0.0)
            blocked = load_same_diff_combined(blk)
            shuffled = load_same_diff_combined(shf)
            self.assertEqual(set(blocked), {"A_fixspotRel", "B_fixspotRel"})
            self.assertEqual(blocked["A_fixspotRel"].n_sessions_used, 8)
            results = compare_same_diff_panels(
                blocked, shuffled, n_perm=100, seed=0,
            )
            out = tmp_path / "mean_ci_compare.npz"
            save_same_diff_compare(out, results, go_seq="AgoB")
            z = np.load(out, allow_pickle=True)
            self.assertIn("A_t_welch", z.files)
            self.assertIn("A_cluster_mask", z.files)
            self.assertEqual(int(z["A_blocked_n_sessions"]), 8)


def _fake_choice_ab_own(path: Path, *, n_sess: int, n_bins: int, offset: float) -> None:
    t = np.linspace(-1000.0, 1000.0, n_bins)
    rng = np.random.default_rng(3)
    curves = rng.normal(0.5 + offset, 0.02, size=(n_sess, n_bins))
    mean = np.mean(curves, axis=0)
    payload = {"align_0": "A_InitialFixationReleaseTime_ms", "align_1": "B_InitialFixationReleaseTime_ms"}
    for pfx in ("A__A_choice", "A__B_choice", "B__A_choice", "B__B_choice"):
        other = 0.0 if pfx == "A__A_choice" else 0.9
        payload[f"{pfx}_mean"] = mean if pfx == "A__A_choice" else np.full(n_bins, other)
        payload[f"{pfx}_ci_low"] = mean - 0.05
        payload[f"{pfx}_ci_high"] = mean + 0.05
        payload[f"{pfx}_cluster_mask"] = np.zeros(n_bins, dtype=bool)
        payload[f"{pfx}_session_curves"] = curves if pfx == "A__A_choice" else np.full((n_sess, n_bins), other)
        payload[f"{pfx}_bin_centers_ms"] = t
        payload[f"{pfx}_n_sessions"] = n_sess
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **payload)


def _fake_actor_combined(path: Path, *, n_sess: int, n_bins: int, offset: float) -> None:
    t = np.linspace(-1000.0, 1000.0, n_bins)
    rng = np.random.default_rng(4)
    curves = rng.normal(0.5 + offset, 0.02, size=(n_sess, n_bins))
    mean = np.mean(curves, axis=0)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        bin_centers_ms=t,
        mean=mean,
        ci_low=mean - 0.05,
        ci_high=mean + 0.05,
        session_curves=curves,
        session_ids=np.array([f"s{i}" for i in range(n_sess)], dtype=object),
        n_sessions_used=n_sess,
        cluster_mask=np.zeros(n_bins, dtype=bool),
        cluster_p=np.array([], dtype=float),
        cluster_threshold=np.nan,
    )


class ActorOwnLoaderTests(unittest.TestCase):
    def test_paths(self) -> None:
        from analyze_decoding.compare_lists import (
            actor_own_dyadic_npz,
            actor_own_output_dir,
            actor_own_solo_npz,
            own_action_prefix,
        )

        root = Path("/data")
        self.assertEqual(own_action_prefix("A"), "A__A_choice")
        self.assertEqual(
            actor_own_dyadic_npz(root, "Curius", "BgoA", blocked=True),
            root
            / "Curius_BLOCKED_CONF"
            / "Curius_BgoA"
            / "Dyadic"
            / "decoding"
            / "choice_ab_grid"
            / "combined"
            / "mean_ci_decode.npz",
        )
        self.assertEqual(
            actor_own_solo_npz(root, "Elmo", "AgoB", blocked=False),
            root
            / "Elmo_SHUFFLED_CONF"
            / "Elmo_AgoB"
            / "SoloA"
            / "decoding"
            / "actor_choice"
            / "combined"
            / "mean_ci_decode.npz",
        )
        self.assertEqual(
            actor_own_output_dir(root, "Elmo", "AgoB", blocked=True),
            root
            / "decode_compare"
            / "Elmo_BLOCKED_Dyadic_vs_SoloA"
            / "Elmo_AgoB"
            / "decoding"
            / "actor_choice",
        )

    def test_load_own_panel_and_compare(self) -> None:
        from analyze_decoding.compare_lists import (
            compare_two_combined,
            load_choice_ab_own_action,
            save_actor_own_compare,
        )
        from analyze_decoding.io_cache import load_combined_result

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            dy = tmp_path / "dy.npz"
            so = tmp_path / "so.npz"
            _fake_choice_ab_own(dy, n_sess=8, n_bins=11, offset=0.10)
            _fake_actor_combined(so, n_sess=6, n_bins=11, offset=0.00)
            dyadic = load_choice_ab_own_action(dy)
            solo = load_combined_result(so)
            self.assertEqual(dyadic.n_sessions_used, 8)
            self.assertEqual(solo.n_sessions_used, 6)
            self.assertTrue(np.mean(dyadic.mean) > np.mean(solo.mean))
            cmp = compare_two_combined(dyadic, solo, n_perm=50, seed=0)
            out = tmp_path / "mean_ci_compare.npz"
            save_actor_own_compare(
                out, {**cmp, "dyadic": dyadic, "solo": solo}, go_seq="AgoB", list_tag="BLOCKED",
            )
            z = np.load(out, allow_pickle=True)
            self.assertIn("t_welch", z.files)
            self.assertEqual(int(z["dyadic_n_sessions"]), 8)
            self.assertEqual(int(z["solo_n_sessions"]), 6)


if __name__ == "__main__":
    unittest.main()
