"""Expanded pooling rejects altered rankings and incomplete or changed annotations."""

import json
import tempfile
import unittest
from pathlib import Path

from expand_pool import EXPECTED_MODELS, build, import_labels
from prepare import read_rows, sha256, write_rows


class PoolTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.source = self.root / "reviewed"
        self.source.mkdir()
        write_rows(self.source / "queries.jsonl", [{"id": "q", "text": "query", "group": "g"}])
        write_rows(self.source / "documents.jsonl", [
            {"id": str(i), "text": f"document {i}"} for i in range(10)
        ])
        self.previous = {"query_id": "q", "document_id": "0", "grade": 2,
                         "rationale": "prior judgment", "label_origin": "agent_reviewed"}
        write_rows(self.source / "qrels.jsonl", [self.previous])
        (self.source / "manifest.json").write_text(json.dumps({
            "files": {n: sha256(self.source / n) for n in (
                "queries.jsonl", "documents.jsonl", "qrels.jsonl"
            )}
        }))
        run = self.root / "run"
        rankings = run / "reviewed" / "rankings.jsonl"
        rankings.parent.mkdir(parents=True)
        write_rows(rankings, [{"query_id": "q", "documents": list(map(str, range(10)))}])
        (run / "report.json").write_text(json.dumps({"results": {"reviewed": {
            "dataset_manifest_sha256": sha256(self.source / "manifest.json"),
            "ranking_sha256": sha256(rankings)
        }}}))
        self.suite = self.root / "suite.json"
        self.suite.write_text(json.dumps({
            "datasets": {"reviewed": sha256(self.source / "manifest.json")},
            "runs": [{"name": name, "directory": str(run),
                      "report_sha256": sha256(run / "report.json")} for name in EXPECTED_MODELS]
        }))
        self.pool = self.root / "pool"
        build(self.source, self.suite, self.pool)
        self.labels = self.root / "labels.jsonl"
        write_rows(self.labels, [
            {"query_id": r["query_id"], "document_id": r["document_id"], "grade": 0,
             "rationale": "different intent", "reviewer_id": "test", "reviewer_origin": "agent"}
            for r in read_rows(self.pool / "packet.jsonl")
        ])

    def tearDown(self):
        self.temp.cleanup()

    def test_union_preserves_previous_label(self):
        output = self.root / "expanded"
        result = import_labels(self.pool, self.labels, output)
        self.assertEqual(result["new_labels"], 9)
        self.assertEqual(read_rows(output / "qrels.jsonl")[0], self.previous)
        self.assertEqual(len(read_rows(output / "qrels.jsonl")), 10)
        self.assertEqual(result["reviewer_ids"], ["test"])
        self.assertEqual(result["judgment_origins"], {"agent": 10})

    def test_missing_or_unresolved_labels_fail_before_creating_dataset(self):
        rows = read_rows(self.labels)
        for invalid in (rows[:-1], [dict(rows[0], grade=None), *rows[1:]]):
            write_rows(self.labels, invalid)
            with self.assertRaises(ValueError):
                import_labels(self.pool, self.labels, self.root / "expanded")
            self.assertFalse((self.root / "expanded").exists())

    def test_changed_packet_and_ranking_are_rejected(self):
        packet = self.pool / "packet.jsonl"
        original = packet.read_bytes()
        packet.write_bytes(original + b"\n")
        with self.assertRaisesRegex(ValueError, "packet changed"):
            import_labels(self.pool, self.labels, self.root / "expanded")
        packet.write_bytes(original)
        ranking = self.root / "run/reviewed/rankings.jsonl"
        ranking.write_text("[]\n")
        with self.assertRaisesRegex(ValueError, "provenance changed"):
            import_labels(self.pool, self.labels, self.root / "expanded")
        self.assertFalse((self.root / "expanded").exists())

    def test_missing_cohort_is_rejected(self):
        specification = json.loads(self.suite.read_text())
        for runs in ([], specification["runs"][:-1]):
            self.suite.write_text(json.dumps(dict(specification, runs=runs)))
            with self.assertRaisesRegex(ValueError, "19 models"):
                build(self.source, self.suite, self.root / "bad-pool")
            self.assertFalse((self.root / "bad-pool").exists())

    def test_null_nonstring_or_blank_provenance_is_rejected(self):
        rows = read_rows(self.labels)
        for field in ("reviewer_id", "rationale"):
            for value in (None, 17, "  "):
                write_rows(self.labels, [dict(rows[0], **{field: value}), *rows[1:]])
                with self.assertRaisesRegex(ValueError, "reviewer required"):
                    import_labels(self.pool, self.labels, self.root / "expanded")
                self.assertFalse((self.root / "expanded").exists())


if __name__ == "__main__":
    unittest.main()
