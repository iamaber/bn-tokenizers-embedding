import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from evaluate_encoder import evaluate
from prepare import read_rows, sha256, write_rows
from run_evaluations import run


class ConstantEncoder:
    custom_path = None

    def __init__(self, path, max_length):
        pass

    def encode(self, texts, batch_size=16):
        return torch.tensor([[1.0, 0.0]] * len(texts)), {"seconds": 0.0}

    def close(self):
        pass


class EncoderEvaluationTests(unittest.TestCase):
    def test_ties_are_stable_and_evaluation_preserves_frozen_labels(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dataset = root / "dataset"
            dataset.mkdir()
            write_rows(
                dataset / "documents.jsonl",
                [
                    {"id": "z", "text": "late document"},
                    {"id": "a", "text": "early document"},
                ],
            )
            write_rows(
                dataset / "queries.jsonl",
                [
                    {"id": "q", "group": "g", "bucket": "english", "text": "query"},
                ],
            )
            write_rows(
                dataset / "qrels.jsonl",
                [
                    {"query_id": "q", "document_id": "a", "grade": 2},
                ],
            )
            manifest = {"files": {p.name: sha256(p) for p in dataset.iterdir()}}
            (dataset / "manifest.json").write_text(json.dumps(manifest))
            before = {p.name: sha256(p) for p in dataset.iterdir()}
            with patch("evaluate_encoder.Encoder", ConstantEncoder):
                result = evaluate(dataset, [dataset], root / "result", repeats=3)
                self.assertEqual(result["results"]["dataset"]["all"]["metrics"]["ndcg_at_10"], 1)
                self.assertEqual(
                    read_rows(root / "result/dataset/rankings.jsonl")[0]["documents"], ["a", "z"]
                )
                self.assertEqual(result["runtime"]["queries_per_pass"], 1)
                with self.assertRaisesRegex(ValueError, "exists"):
                    evaluate(dataset, [dataset], root / "result")
            self.assertEqual(before, {p.name: sha256(p) for p in dataset.iterdir()})
            (dataset / "qrels.jsonl").write_text("changed")
            with self.assertRaisesRegex(ValueError, "checksum"):
                evaluate(dataset, [dataset], root / "bad", repeats=0)
            self.assertFalse((root / "bad").exists())

    def test_resume_rejects_changed_measurement_implementation(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dataset = root / "domains/bl-bn"
            dataset.mkdir(parents=True)
            write_rows(dataset / "documents.jsonl", [{"id": "d", "text": "document"}])
            write_rows(
                dataset / "queries.jsonl",
                [
                    {"id": "q", "group": "g", "bucket": "english", "text": "query"},
                ],
            )
            write_rows(
                dataset / "qrels.jsonl",
                [
                    {"query_id": "q", "document_id": "d", "grade": 2},
                ],
            )
            (dataset / "manifest.json").write_text(
                json.dumps(
                    {
                        "files": {p.name: sha256(p) for p in dataset.iterdir()},
                    }
                )
            )
            with patch("evaluate_encoder.Encoder", ConstantEncoder):
                evaluate(dataset, [dataset], root / "suite/unchanged", repeats=3)
            self.assertEqual(
                len(run(dataset, [], root / "domains", root / "suite", repeats=3)["runs"]), 1
            )
            path = root / "suite/unchanged/report.json"
            report = json.loads(path.read_text())
            report["implementation_sha256"] = {}
            path.write_text(json.dumps(report))
            with self.assertRaisesRegex(ValueError, "implementation differs"):
                run(dataset, [], root / "domains", root / "suite", repeats=3)


if __name__ == "__main__":
    unittest.main()
