import copy
import json
import tempfile
import unittest
from pathlib import Path

from prepare import sha256, write_rows
from run_evaluations import run
from seed_analysis import DIRECTIONS, paired_seeds, selection_scores


class SeedAnalysisTests(unittest.TestCase):
    def cohort(self):
        return {
            seed: [
                {
                    "query_id": q,
                    "group": q,
                    "metrics": {
                        "recall_at_10": 0.2,
                        "ndcg_at_10": 0.2,
                        "mrr_at_10": 0.2,
                    },
                }
                for q in ("a", "b")
            ]
            for seed in (17, 29, 43)
        }

    def test_identical_runs_have_zero_uncertainty_and_seed_variation_is_sampled(self):
        reference = self.cohort()
        zero = paired_seeds(reference, reference, repeats=400)
        for metric in zero["metrics"].values():
            self.assertEqual(metric["bootstrap_95_percentile"], [0.0, 0.0])
        candidate = copy.deepcopy(reference)
        for seed, delta in zip((17, 29, 43), (0.1, 0.2, 0.3), strict=True):
            for row in candidate[seed]:
                row["metrics"] = {key: value + delta for key, value in row["metrics"].items()}
        result = paired_seeds(reference, candidate, repeats=400)
        effect = result["metrics"]["ndcg_at_10"]
        self.assertAlmostEqual(effect["mean"], 0.2)
        self.assertLess(effect["bootstrap_95_percentile"][0], effect["mean"])
        self.assertGreater(effect["bootstrap_95_percentile"][1], effect["mean"])

    def test_missing_seed_duplicate_query_and_changed_groups_rejected(self):
        reference = self.cohort()
        for mutation in ("missing", "duplicate", "group", "nonfinite"):
            candidate = copy.deepcopy(reference)
            if mutation == "missing":
                del candidate[43]
            elif mutation == "duplicate":
                candidate[17].append(copy.deepcopy(candidate[17][0]))
            elif mutation == "group":
                candidate[29][0]["group"] = "different"
            else:
                candidate[43][0]["metrics"]["ndcg_at_10"] = float("nan")
            with self.assertRaises(ValueError):
                paired_seeds(reference, candidate, repeats=40)

    def test_test_evaluation_requires_frozen_development_selection(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dataset = root / "test/bl-bn"
            dataset.mkdir(parents=True)
            for name in ("documents", "queries", "qrels"):
                write_rows(dataset / f"{name}.jsonl", [{"fixture": name}])
            manifest = {"split": "test", "files": {p.name: sha256(p) for p in dataset.iterdir()}}
            (dataset / "manifest.json").write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, "frozen development selection"):
                run(root / "base", [], root / "test", root / "output")
            self.assertFalse((root / "output").exists())

    def test_domain_selection_ignores_sentence_pilot_and_rejects_test(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            report = {"results": {"pilot-development": {"dataset": "not-read-for-selection"}}}
            for direction in DIRECTIONS:
                dataset = root / "domains" / direction
                ranking = root / "results" / direction
                dataset.mkdir(parents=True)
                ranking.mkdir(parents=True)
                write_rows(dataset / "documents.jsonl", [{"id": "d", "text": "document"}])
                write_rows(
                    dataset / "queries.jsonl",
                    [
                        {
                            "id": "q",
                            "group": "g",
                            "bucket": "english",
                            "split": "development",
                        }
                    ],
                )
                write_rows(
                    dataset / "qrels.jsonl",
                    [
                        {
                            "query_id": "q",
                            "document_id": "d",
                            "grade": 2,
                        }
                    ],
                )
                (dataset / "manifest.json").write_text(
                    json.dumps(
                        {
                            "files": {p.name: sha256(p) for p in dataset.iterdir()},
                        }
                    )
                )
                write_rows(ranking / "rankings.jsonl", [{"query_id": "q", "documents": ["d"]}])
                report["results"][direction] = {
                    "dataset": str(dataset),
                    "dataset_manifest_sha256": sha256(dataset / "manifest.json"),
                    "ranking_sha256": sha256(ranking / "rankings.jsonl"),
                }
            scores = selection_scores(root / "results", report)
            self.assertEqual(scores.keys(), DIRECTIONS)
            dataset = root / "domains/bl-bn"
            queries = [{"id": "q", "group": "g", "bucket": "english", "split": "test"}]
            write_rows(dataset / "queries.jsonl", queries)
            path = dataset / "manifest.json"
            manifest = json.loads(path.read_text())
            manifest["files"]["queries.jsonl"] = sha256(dataset / "queries.jsonl")
            path.write_text(json.dumps(manifest))
            report["results"]["bl-bn"]["dataset_manifest_sha256"] = sha256(path)
            with self.assertRaisesRegex(ValueError, "development-only"):
                selection_scores(root / "results", report)


if __name__ == "__main__":
    unittest.main()
