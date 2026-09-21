import csv
import math
import tempfile
import unittest
from pathlib import Path

from evaluate import bm25, evaluate, terms
from metrics import query_metrics, summarize
from prepare import prepare, read_rows, write_rows


class MetricsTests(unittest.TestCase):
    def test_graded_ranking(self) -> None:
        got = query_metrics(["b", "a"], {"a": 3, "b": 1})
        self.assertEqual(got["recall_at_10"], 1)
        self.assertAlmostEqual(got["ndcg_at_10"], (1 + 7 / math.log2(3)) / (7 + 1 / math.log2(3)))
        self.assertEqual(query_metrics([], {"a": 1})["ndcg_at_10"], 0)
        self.assertEqual(
            query_metrics([str(i) for i in range(10)] + ["a"], {"a": 1})["recall_at_10"],
            0,
        )
        for ranking, qrels in [
            (["a", "a"], {"a": 1}),
            (["a"], {"a": 0}),
            (["a"], {"a": -1}),
            (["a"], {"a": True}),
        ]:
            with self.assertRaises(ValueError):
                query_metrics(ranking, qrels)

    def test_group_bootstrap_does_not_overweight_variants(self) -> None:
        rows = [
            {"group": group, "metrics": query_metrics(rank, {"a": 1})}
            for group, rank in [("one", ["a"]), ("one", ["a"]), ("two", [])]
        ]
        result = summarize(rows, repeats=100)
        self.assertEqual(result["groups"], 2)
        self.assertEqual(result["metrics"]["recall_at_10"]["mean"], 0.5)
        self.assertEqual(result, summarize(rows, repeats=100))

    def test_bm25_retains_bengali_marks_and_avoids_zero_score_ties(self) -> None:
        self.assertEqual(terms("ভালো, ফোন!"), ["ভালো", "ফোন"])
        ranking, info = bm25(
            [{"id": "a", "text": "ভালো ফোন"}, {"id": "b", "text": "অন্য কিছু"}],
            [{"text": "ভালো"}, {"text": "valo"}],
        )
        self.assertEqual(ranking, [["a"], []])
        self.assertEqual(info["no_matching_term_queries"], 1)


class PilotTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        (self.root / "processed").mkdir()
        (self.root / "raw/banglatlit").mkdir(parents=True)
        records = []
        pairs = [
            ("one", "ami valo achi", "আমি ভালো আছি"),
            ("two", "tumi kemon acho", "তুমি কেমন আছো"),
            ("three", "notun ekta boi", "নতুন একটা বই"),
        ]
        for group, roman, bengali in pairs:
            for text, bucket in [(roman, "banglish"), (bengali, "bangla")]:
                records.append(
                    {
                        "text": text,
                        "bucket": bucket,
                        "source": "banglatlit",
                        "group": group,
                        "attribution": "fixture",
                        "split": "validation",
                    }
                )
        write_rows(self.root / "processed/validation.jsonl", records)
        write_rows(
            self.root / "processed/train.jsonl",
            [{"text": "separate training", "group": "training", "split": "train"}],
        )
        for split in ("train", "validation"):
            with (self.root / f"raw/banglatlit/{split}.csv").open("w", newline="") as stream:
                writer = csv.writer(stream)
                writer.writerow(["id", "text_transliterated", "text_bengali"])
                if split == "validation":
                    writer.writerows(pairs)

    def test_reproducible_pilot_and_review_queue(self) -> None:
        first, second = self.root / "pilot1", self.root / "pilot2"
        a = prepare(self.root, first, 2, 3, 42)
        b = prepare(self.root, second, 2, 3, 42)
        self.assertEqual(a["files"], b["files"])
        self.assertFalse(a["frozen_test_opened"])
        result = evaluate(first, self.root / "report.json", None)
        self.assertEqual(result["results"]["bm25"]["all"]["queries"], 2)
        for row in read_rows(first / "review-pool.jsonl"):
            self.assertIsNone(row["adjudicated_grade"])
        with self.assertRaises(ValueError):
            prepare(self.root, first, 2, 3, 42)
        with self.assertRaises(ValueError):
            evaluate(first, self.root / "report.json", None)

    def test_training_overlap_is_rejected(self) -> None:
        write_rows(
            self.root / "processed/train.jsonl",
            [{"text": "আমি ভালো আছি", "group": "different", "split": "train"}],
        )
        with self.assertRaisesRegex(ValueError, "overlaps"):
            prepare(self.root, self.root / "pilot", 2, 3, 42)

    def test_merged_groups_do_not_create_false_pairs(self) -> None:
        path = self.root / "processed/validation.jsonl"
        rows = read_rows(path)
        for row in rows:
            if row["group"] == "two":
                row["group"] = "one"
        write_rows(path, rows)
        pilot = self.root / "pilot"
        prepare(self.root, pilot, 2, 3, 42)
        documents = {r["id"]: r["text"] for r in read_rows(pilot / "documents.jsonl")}
        queries = {r["id"]: r["text"] for r in read_rows(pilot / "queries.jsonl")}
        expected = {
            "ami valo achi": "আমি ভালো আছি",
            "tumi kemon acho": "তুমি কেমন আছো",
            "notun ekta boi": "নতুন একটা বই",
        }
        for row in read_rows(pilot / "qrels.jsonl"):
            self.assertEqual(documents[row["document_id"]], expected[queries[row["query_id"]]])

    def test_split_and_checksum_guards(self) -> None:
        pilot = self.root / "pilot"
        prepare(self.root, pilot, 2, 3, 42)
        (pilot / "queries.jsonl").write_text("{}\n")
        with self.assertRaisesRegex(ValueError, "checksum"):
            evaluate(pilot, self.root / "report.json", None)
        path = self.root / "processed/validation.jsonl"
        rows = read_rows(path)
        rows[0]["split"] = "test"
        write_rows(path, rows)
        with self.assertRaisesRegex(ValueError, "validation records"):
            prepare(self.root, self.root / "bad", 2, 3, 42)


if __name__ == "__main__":
    unittest.main()
