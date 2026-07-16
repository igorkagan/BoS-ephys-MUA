"""Tests for session datetime parsing and temporal stability helpers."""

from __future__ import annotations

import unittest

import numpy as np

from load_data.io import parse_session_datetime, session_gap_days
from analyze_stability.temporal_core import (
    build_session_compatibility_graph,
    greedy_session_clusters,
    invert_tradeoff,
    max_clique,
    pairwise_channel_rows,
    pick_sweet_spot,
    recommend_combinable_gap,
    summarize_pairs_by_gap_bin,
    sweep_channel_session_tradeoff,
    TradeoffRow,
)
from analyze_stability.plots_tradeoff import build_explorer_figure, write_explorer_html


class SessionTimeTests(unittest.TestCase):
    def test_parse_datetime_strips_u_b_suffix(self) -> None:
        dt = parse_session_datetime("20230623T124557U.A_Curius.B_Elmo.SCP_01")
        self.assertEqual(dt.strftime("%Y%m%dT%H%M%S"), "20230623T124557")

    def test_gap_same_day_pair(self) -> None:
        gap = session_gap_days(
            "20230623T124557U.A_Curius.B_Elmo.SCP_01",
            "20230623T124557B.A_Curius.B_Elmo.SCP_01",
        )
        self.assertAlmostEqual(gap, 0.0, places=5)

    def test_gap_multi_day(self) -> None:
        gap = session_gap_days(
            "20230623T124557.A_Curius.B_Elmo.SCP_01",
            "20230630T115937.A_Curius.B_Elmo.SCP_01",
        )
        self.assertAlmostEqual(gap, 7.0, places=1)


class TemporalStabilityTests(unittest.TestCase):
    def test_pairwise_rows_and_recommendation(self) -> None:
        session_ids = ["20230601T120000.A_Curius.B_X.SCP_01", "20230602T120000.A_Curius.B_X.SCP_01",
                       "20230610T120000.A_Curius.B_X.SCP_01"]
        t = np.linspace(-100, 100, 11)
        diff_tensor = np.zeros((3, 1, len(t)))
        diff_tensor[0, 0] = t
        diff_tensor[1, 0] = t * 0.95
        diff_tensor[2, 0] = np.sin(t / 10)
        si_matrix = np.array([[0.2], [0.18], [-0.1]], dtype=float)

        rows = pairwise_channel_rows(session_ids, [1], diff_tensor, si_matrix)
        self.assertEqual(len(rows), 3)

        stable = {1}
        rec = recommend_combinable_gap(rows, stable, median_r_threshold=0.5, p25_r_threshold=0.3)
        self.assertGreater(rec.n_pair_rows_used, 0)
        bins = summarize_pairs_by_gap_bin([r for r in rows if r.channel in stable])
        self.assertTrue(bins)

    def test_greedy_clusters(self) -> None:
        ids = [
            "20230601T120000.A_Curius.B_X.SCP_01",
            "20230602T120000.A_Curius.B_X.SCP_01",
            "20230620T120000.A_Curius.B_X.SCP_01",
        ]
        clusters = greedy_session_clusters(ids, max_gap_days=3.0)
        self.assertEqual(len(clusters), 2)
        self.assertEqual(len(clusters[0]), 2)


class TradeoffTests(unittest.TestCase):
    def test_max_clique_toy_graph(self) -> None:
        adj = {
            "a": {"b", "c"},
            "b": {"a", "c"},
            "c": {"a", "b"},
            "d": {"e"},
            "e": {"d"},
        }
        clique = max_clique(adj)
        self.assertEqual(len(clique), 3)
        self.assertEqual(set(clique), {"a", "b", "c"})

    def test_sweep_shrinks_with_bad_channel(self) -> None:
        session_ids = [
            "20230601T120000.A_Curius.B_X.SCP_01",
            "20230602T120000.A_Curius.B_X.SCP_01",
            "20230603T120000.A_Curius.B_X.SCP_01",
        ]
        t = np.linspace(-100, 100, 11)
        diff_tensor = np.zeros((3, 2, len(t)))
        diff_tensor[:, 0, :] = t  # ch1: all sessions correlate
        diff_tensor[0, 1] = t
        diff_tensor[1, 1] = t
        diff_tensor[2, 1] = np.sin(t / 5)  # ch2: breaks session 3
        si_matrix = np.zeros((3, 2), dtype=float)

        rows = pairwise_channel_rows(session_ids, [1, 2], diff_tensor, si_matrix)
        tradeoff = sweep_channel_session_tradeoff(session_ids, [1, 2], rows, r_min=0.5)
        self.assertEqual(tradeoff[0].max_n_sessions, 3)
        self.assertEqual(tradeoff[1].max_n_sessions, 2)

    def test_invert_and_sweet_spot(self) -> None:
        rows = [
            TradeoffRow(1, (1,), 3, ("s1", "s2", "s3"), 3, 0.9),
            TradeoffRow(2, (1, 2), 2, ("s1", "s2"), 4, 0.8),
            TradeoffRow(3, (1, 2, 3), 2, ("s1", "s2"), 6, 0.7),
        ]
        inv = invert_tradeoff(rows)
        self.assertEqual(inv[3], 1)
        self.assertEqual(inv[2], 3)
        sweet = pick_sweet_spot(rows)
        self.assertEqual(sweet.coverage, 6)

    def test_explorer_html_smoke(self) -> None:
        rows = [
            TradeoffRow(1, (1,), 2, ("s1", "s2"), 2, 0.9),
            TradeoffRow(2, (1, 2), 2, ("s1", "s2"), 4, 0.85),
        ]
        inv = invert_tradeoff(rows)
        sweet = pick_sweet_spot(rows)
        fig = build_explorer_figure(rows, sweet, inv, title="test")
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "explorer.html"
            write_explorer_html(fig, path)
            self.assertTrue(path.exists())
            self.assertGreater(path.stat().st_size, 500)


if __name__ == "__main__":
    unittest.main()
