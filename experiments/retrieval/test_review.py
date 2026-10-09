import unittest
from pathlib import Path

import test_retrieval
from analyze import failure_audit
from evaluate import evaluate
from prepare import read_rows, write_rows
from review import export, import_review


class ReviewTests(unittest.TestCase):
    def setUp(self) -> None:
        test_retrieval.PilotTests.setUp(self)
        from prepare import prepare

        self.pilot = self.root / "pilot-review"
        prepare(self.root, self.pilot, 2, 3, 42)
        evaluate(self.pilot, self.root / "report.json", None)

    def labels(self) -> tuple[Path, Path]:
        positives = {(r["query_id"], r["document_id"])
                     for r in read_rows(self.pilot / "qrels.jsonl")}
        grades = [
            dict(r, grade=2 if (r["query_id"], r["document_id"]) in positives else 0,
                 reviewer_id="fixture-agent", reviewer_origin="agent", rationale="fixture judgment")
            for r in read_rows(self.pilot / "review-pool.jsonl")
        ]
        pairs = [dict(r, pair_correct=True, privacy_review_passed=True,
                      reviewer="fixture-agent", reviewer_origin="agent")
                 for r in read_rows(self.pilot / "authoring.jsonl")]
        judgments, pair_review = self.root / "grades.jsonl", self.root / "pairs.jsonl"
        write_rows(judgments, grades)
        write_rows(pair_review, pairs)
        return judgments, pair_review

    def test_blinded_packets_and_explicit_agent_provenance(self) -> None:
        output = self.root / "packets"
        export(self.pilot, output, calibration=1)
        rows = read_rows(output / "reviewer-1.jsonl")
        self.assertTrue(all(r["grade"] is None and "rank" not in r for r in rows))
        judgments, pair_review = self.labels()
        reviewed = self.root / "reviewed"
        manifest = import_review(self.pilot, judgments, pair_review, reviewed)
        self.assertFalse(manifest["human_review_completed"])
        self.assertFalse(manifest["independent_review_verified"])
        self.assertEqual({r["grade"] for r in read_rows(reviewed / "qrels.jsonl")}, {2})
        with self.assertRaisesRegex(ValueError, "exists"):
            import_review(self.pilot, judgments, pair_review, reviewed)

    def test_unresolved_duplicate_and_invalid_grades_are_rejected(self) -> None:
        for grade in (None, True, -1, 3):
            judgments, pair_review = self.labels()
            rows = read_rows(judgments)
            rows[0]["grade"] = grade
            write_rows(judgments, rows)
            with self.assertRaisesRegex(ValueError, "invalid judgment"):
                import_review(self.pilot, judgments, pair_review, self.root / "bad")
            self.assertFalse((self.root / "bad").exists())
        judgments, pair_review = self.labels()
        rows = read_rows(judgments)
        write_rows(judgments, rows + rows[:1])
        with self.assertRaisesRegex(ValueError, "duplicate"):
            import_review(self.pilot, judgments, pair_review, self.root / "bad")

    def test_exclusions_are_versioned_and_incomplete_reviews_rejected(self) -> None:
        judgments, pair_review = self.labels()
        pairs = read_rows(pair_review)
        pairs[0].update(pair_correct=False, exclusion_reason="wrong pair")
        write_rows(pair_review, pairs)
        manifest = import_review(self.pilot, judgments, pair_review, self.root / "excluded")
        self.assertEqual(manifest["queries"], 1)
        self.assertEqual(manifest["exclusions"][0]["reason"], "wrong pair")
        rows = read_rows(judgments)
        write_rows(judgments, rows[:-1])
        with self.assertRaisesRegex(ValueError, "incomplete"):
            import_review(self.pilot, judgments, pair_review, self.root / "bad")

    def test_changed_pair_and_missing_positive_rejected(self) -> None:
        judgments, pair_review = self.labels()
        pairs = read_rows(pair_review)
        pairs[0]["group"] = "invented"
        write_rows(pair_review, pairs)
        with self.assertRaisesRegex(ValueError, "changed"):
            import_review(self.pilot, judgments, pair_review, self.root / "bad")
        judgments, pair_review = self.labels()
        rows = read_rows(judgments)
        for row in rows:
            row["grade"] = 0
        write_rows(judgments, rows)
        with self.assertRaisesRegex(ValueError, "positive"):
            import_review(self.pilot, judgments, pair_review, self.root / "bad")

    def test_changed_judgment_text_rejected_and_private_text_not_copied(self) -> None:
        judgments, pair_review = self.labels()
        rows = read_rows(judgments)
        rows[0]["query"] = "DIFFERENT TEXT"
        write_rows(judgments, rows)
        with self.assertRaisesRegex(ValueError, "changed"):
            import_review(self.pilot, judgments, pair_review, self.root / "bad")
        judgments, pair_review = self.labels()
        pairs = read_rows(pair_review)
        excluded = pairs[0]
        pairs[0].update(privacy_review_passed=False, exclusion_reason="privacy")
        write_rows(pair_review, pairs)
        output = self.root / "private-excluded"
        import_review(self.pilot, judgments, pair_review, output)
        for name in ("queries", "documents", "judgments", "pair-review"):
            rows = read_rows(output / f"{name}.jsonl")
            self.assertFalse(any(r.get("group") == excluded["group"] for r in rows))
            self.assertNotIn(excluded["source_banglish"], (output / f"{name}.jsonl").read_text())

    def test_failure_audit_does_not_invent_causes(self) -> None:
        summary = failure_audit(self.pilot, self.pilot / "ranking-bm25.jsonl", self.root / "audit")
        self.assertEqual(summary["misses_at_10"], 2)
        self.assertFalse(summary["causal_categories_completed"])
        rows = read_rows(self.root / "audit/failure-review.jsonl")
        self.assertTrue(all(r["primary_failure_category"] is None for r in rows))

    def test_old_unhashed_authoring_cannot_change_privacy_group(self) -> None:
        import json

        manifest_path = self.pilot / "manifest.json"
        manifest = json.loads(manifest_path.read_text())
        manifest["files"].pop("authoring.jsonl")
        manifest_path.write_text(json.dumps(manifest))
        judgments, pair_review = self.labels()
        source = read_rows(self.pilot / "authoring.jsonl")
        source[0]["group"] = "invented"
        write_rows(self.pilot / "authoring.jsonl", source)
        pairs = read_rows(pair_review)
        pairs[0].update(group="invented", privacy_review_passed=False, exclusion_reason="privacy")
        write_rows(pair_review, pairs)
        with self.assertRaisesRegex(ValueError, "frozen source"):
            import_review(self.pilot, judgments, pair_review, self.root / "bad")


if __name__ == "__main__":
    unittest.main()
