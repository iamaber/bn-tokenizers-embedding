import json
import unittest

from annotation_archive import archive, restore
from prepare import read_rows, sha256, write_rows
from review import import_review
import test_review


class AnnotationArchiveTests(unittest.TestCase):
    def setUp(self):
        test_review.ReviewTests.setUp(self)

    def test_text_free_archive_restores_exact_agent_judgments_and_binds_source(self):
        judgments, pairs = test_review.ReviewTests.labels(self)
        compact = self.root / "compact"
        result = archive(self.pilot, judgments, pairs, compact)
        self.assertTrue(result["annotation_coverage_complete"])
        self.assertNotIn('"query":', (compact / "judgments.jsonl").read_text())
        self.assertNotIn('"source_banglish":', (compact / "pairs.jsonl").read_text())
        restored = self.root / "restored"
        restore(self.pilot, compact, restored)
        reviewed = self.root / "reviewed"
        frozen = import_review(
            self.pilot, restored / "reviewer-1.jsonl", restored / "pair-review.jsonl", reviewed
        )
        self.assertFalse(frozen["human_review_completed"])
        self.assertEqual(frozen["queries"], 2)
        source = self.pilot / "review-pool.jsonl"
        source.write_text(source.read_text() + "\n")
        with self.assertRaisesRegex(ValueError, "different frozen pilot"):
            restore(self.pilot, compact, self.root / "changed")
        self.assertFalse((self.root / "changed").exists())

    def test_partial_archive_keeps_unknown_labels_unresolved(self):
        judgments, pairs = test_review.ReviewTests.labels(self)
        rows = read_rows(judgments)
        rows[0]["grade"] = None
        write_rows(judgments, rows)
        compact = self.root / "partial"
        manifest = archive(self.pilot, judgments, pairs, compact)
        self.assertFalse(manifest["annotation_coverage_complete"])
        restored = self.root / "restored"
        restore(self.pilot, compact, restored)
        self.assertTrue(any(r["grade"] is None for r in read_rows(restored / "reviewer-1.jsonl")))
        with self.assertRaisesRegex(ValueError, "invalid judgment"):
            import_review(
                self.pilot,
                restored / "reviewer-1.jsonl",
                restored / "pair-review.jsonl",
                self.root / "bad",
            )
        path = compact / "manifest.json"
        damaged = json.loads(path.read_text())
        damaged["files"]["pairs.jsonl"] = "wrong"
        path.write_text(json.dumps(damaged))
        with self.assertRaisesRegex(ValueError, "checksum"):
            restore(self.pilot, compact, self.root / "damaged")

    def test_rebound_archive_cannot_duplicate_ids_or_inject_corpus_text(self):
        judgments, pairs = test_review.ReviewTests.labels(self)
        compact = self.root / "compact"
        archive(self.pilot, judgments, pairs, compact)
        original = read_rows(compact / "judgments.jsonl")
        for name, rows in (
            ("duplicate", original + [original[0]]),
            ("unknown", [dict(original[0], document_id="missing")]),
            ("injected", [dict(original[0], query="replacement corpus text")]),
        ):
            with self.subTest(name=name):
                write_rows(compact / "judgments.jsonl", rows)
                manifest = json.loads((compact / "manifest.json").read_text())
                manifest["files"]["judgments.jsonl"] = sha256(compact / "judgments.jsonl")
                (compact / "manifest.json").write_text(json.dumps(manifest))
                output = self.root / name
                with self.assertRaises(ValueError):
                    restore(self.pilot, compact, output)
                self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
