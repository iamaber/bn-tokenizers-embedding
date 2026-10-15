"""Protect the fresh benchmark's split and training-contamination boundaries."""

import json
import tempfile
import unittest
from pathlib import Path

from source_benchmark import audit_overlap, validate


class SourceBenchmarkTests(unittest.TestCase):
    def records(self):
        return [
            dict(id=f"{domain}-{split}", domain=domain, split=split,
                 source_group=f"{domain}-{split}", author_origin="agent",
                 bn=f"{domain} {split} passage bn", bl=f"{domain} {split} passage bl",
                 en=f"{domain} {split} passage en", q_bn=f"{domain} {split} query bn",
                 q_bl=f"{domain} {split} query bl", q_en=f"{domain} {split} query en",
                 q_bl_variants=[f"{domain} {split} variant"])
            for domain in ("business", "banking", "technology", "science", "conversation", "qa")
            for split in ("development", "test")
        ]

    def test_source_must_not_cross_split(self):
        rows = self.records()
        validate(rows)
        rows[1]["source_group"] = rows[0]["source_group"]
        with self.assertRaisesRegex(ValueError, "source group crosses"):
            validate(rows)

    def test_duplicate_surface_across_intents_rejected(self):
        rows = self.records()
        rows[1]["q_bl_variants"] = rows[0]["q_bl_variants"]
        with self.assertRaisesRegex(ValueError, "duplicate text"):
            validate(rows)

    def test_containment_and_hard_negative_overlap_are_detected(self):
        rows = self.records()
        rows[0]["en"] = "one two three four five six seven eight nine ten eleven twelve"
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "pairs.jsonl"
            path.write_text(json.dumps({"negative": "prefix " + rows[0]["en"] + " suffix",
                                        "query": rows[1]["q_bl_variants"][0].upper()}) + "\n")
            result = audit_overlap(rows, [path])
        self.assertEqual({r["kind"] for r in result["matches"]},
                         {"canonical", "eight-word-shingle-containment"})


if __name__ == "__main__":
    unittest.main()
