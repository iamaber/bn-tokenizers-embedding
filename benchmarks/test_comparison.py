"""Checks for the quality metric accounting, independent of trained models."""

import unittest
from pathlib import Path

from compare_frameworks import Candidate, quality


class QualityTests(unittest.TestCase):
    def setUp(self) -> None:
        self.candidate = Candidate(
            "character fixture",
            Path("unused"),
            256,
            lambda text: list(map(ord, text)),
            lambda texts: [list(map(ord, text)) for text in texts],
            lambda ids: "".join(chr(value) if value else "?" for value in ids),
            0,
            "fixture",
        )

    def test_unknowns_and_round_trips_are_distinct(self) -> None:
        result = quality(self.candidate, ["abc", "a\0b", ""])
        self.assertEqual(result["tokens"], 6)
        self.assertEqual(result["exact_normalized_round_trips"], 2)
        self.assertEqual(result["texts_with_unknowns"], 1)
        self.assertAlmostEqual(result["unknown_token_percent"], 100 / 6)

    def test_empty_text_has_no_unknowns(self) -> None:
        result = quality(self.candidate, [""])
        self.assertEqual(result["round_trip_percent"], 100)
        self.assertEqual(result["unknown_token_percent"], 0)

    def test_nearest_rank_and_budget_boundary(self) -> None:
        result = quality(self.candidate, ["a" * length for length in range(1, 21)])
        self.assertEqual(result["sequence_p95"], 19)
        result = quality(self.candidate, ["a" * 510, "a" * 511])
        self.assertEqual(result["over_510_token_percent"], 50)


if __name__ == "__main__":
    unittest.main()
