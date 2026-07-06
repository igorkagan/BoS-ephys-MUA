"""Tests for bos_mua.viz_consistency helpers."""

from __future__ import annotations

import unittest

from bos_mua.viz.consistency import _deep_dive_subplot_grid


class DeepDiveSubplotGridTests(unittest.TestCase):
    def test_ten_sessions_two_rows(self) -> None:
        nrows, ncols, _ = _deep_dive_subplot_grid(10)
        self.assertEqual((nrows, ncols), (2, 5))

    def test_twelve_sessions_three_rows(self) -> None:
        nrows, ncols, figsize = _deep_dive_subplot_grid(12)
        self.assertEqual((nrows, ncols), (3, 5))
        self.assertEqual(figsize[1], 7.5)

    def test_fifteen_sessions_three_rows(self) -> None:
        nrows, ncols, _ = _deep_dive_subplot_grid(15)
        self.assertEqual((nrows, ncols), (3, 5))


if __name__ == "__main__":
    unittest.main()
