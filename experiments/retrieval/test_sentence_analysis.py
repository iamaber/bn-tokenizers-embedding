import json
import tempfile
import unittest
from pathlib import Path

from compare import score_rankings
from prepare import sha256, write_rows
from sentence_analysis import analyze, coverage, exact_scores


class SentenceAnalysisTests(unittest.TestCase):
    def make_suite(self, root, reviewed_grade=2):
        datasets = {}
        for key, grade in (("retrieval-pilot-v1", 1), ("retrieval-reviewed-v1", reviewed_grade)):
            dataset = root / key
            dataset.mkdir()
            write_rows(dataset / "documents.jsonl", [{"id": "d", "text": "fixture"}])
            write_rows(dataset / "queries.jsonl", [{"id": "q", "group": "g", "bucket": "banglish", "text": "fixture"}])
            write_rows(dataset / "qrels.jsonl", [{"query_id": "q", "document_id": "d", "grade": grade}])
            (dataset / "manifest.json").write_text(json.dumps({"files": {p.name: sha256(p) for p in dataset.iterdir()}}))
            datasets[key] = sha256(dataset / "manifest.json")
        names = ["unchanged"] + [f"{b}/{a}-seed{s}" for b in ("equal-data", "equal-compute") for a in ("original", "unigram-32000", "bpe-32000") for s in (17, 29, 43)]
        runs = []
        for name in names:
            directory = root / "runs" / name
            report = {"results": {}}
            for key, digest in datasets.items():
                path = directory / key
                path.mkdir(parents=True)
                write_rows(path / "rankings.jsonl", [{"query_id": "q", "documents": ["d"]}])
                write_rows(path / "scores.jsonl", score_rankings(root / key, path / "rankings.jsonl"))
                report["results"][key] = {"dataset": str(root / key), "dataset_manifest_sha256": digest, "ranking_sha256": sha256(path / "rankings.jsonl"), "scores_sha256": sha256(path / "scores.jsonl")}
            (directory / "report.json").write_text(json.dumps(report))
            runs.append({"name": name, "directory": str(directory), "report_sha256": sha256(directory / "report.json")})
        development = root / "development.json"
        development.write_text("{}")
        selection = root / "selection.json"
        selection.write_text(json.dumps({"selection": "unchanged", "selection_frozen_before_reserved_test": True, "evaluation_suite": str(development), "evaluation_suite_sha256": sha256(development)}))
        suite = root / "suite.json"
        suite.write_text(json.dumps({"runs": runs, "datasets": datasets, "frozen_selection_sha256": sha256(selection)}))
        return suite, selection

    def test_raw_grade1_labels_keep_graded_view_and_reviewed_has_exact_view(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            suite, selection = self.make_suite(root)
            result = analyze(suite, selection, root / "output")
            self.assertEqual(set(result["datasets"]["retrieval-pilot-v1"]["label_views"]), {"graded"})
            reviewed = result["datasets"]["retrieval-reviewed-v1"]
            self.assertEqual(set(reviewed["label_views"]), {"graded", "exact_grade2"})
            self.assertEqual(reviewed["label_views"]["exact_grade2"]["summary"]["unchanged"]["metrics"]["mrr_at_10"]["mean"], 1)

    def test_reviewed_without_exact_positive_rejected_before_outputs(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            suite, selection = self.make_suite(root, reviewed_grade=1)
            with self.assertRaisesRegex(ValueError, "grade-2 positive"):
                analyze(suite, selection, root / "output")
            self.assertFalse((root / "output").exists())

    def test_partial_match_and_unjudged_candidate_are_distinguished(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            write_rows(root / 'documents.jsonl', [{'id': d} for d in ('exact', 'partial', 'unknown')])
            write_rows(root / 'queries.jsonl', [{'id': 'q', 'group': 'g', 'bucket': 'banglish'}])
            write_rows(root / 'qrels.jsonl', [
                {'query_id': 'q', 'document_id': 'exact', 'grade': 2},
                {'query_id': 'q', 'document_id': 'partial', 'grade': 1},
            ])
            ranking = root / 'ranking.jsonl'
            write_rows(ranking, [{'query_id': 'q', 'documents': ['partial', 'unknown', 'exact']}])
            self.assertEqual(score_rankings(root, ranking)[0]['metrics']['mrr_at_10'], 1)
            self.assertAlmostEqual(exact_scores(root, ranking)[0]['metrics']['mrr_at_10'], 1/3)
            self.assertEqual(coverage(root, ranking)['unjudged_top10_pairs'], 1)
            self.assertAlmostEqual(coverage(root, ranking)['unjudged_top10_fraction'], 1/3)

    def test_changed_frozen_choice_rejected_before_outputs(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            suite, selection, output = root/'suite.json', root/'selection.json', root/'output'
            suite.write_text(json.dumps({
                'datasets': {'retrieval-pilot-v1': 'a', 'retrieval-reviewed-v1': 'b'},
                'frozen_selection_sha256': 'original choice hash',
            }))
            selection.write_text(json.dumps({'selection': 'changed'}))
            with self.assertRaisesRegex(ValueError, 'frozen domain choice'):
                analyze(suite, selection, output)
            self.assertFalse(output.exists())


if __name__ == '__main__':
    unittest.main()
