import json
import tempfile
import unittest
from pathlib import Path

from domains import build
from prepare import read_rows, sha256, verify_manifest, write_rows


class DomainTests(unittest.TestCase):
    def test_all_directions_are_group_disjoint_and_queries_are_not_copies(self) -> None:
        material = Path(__file__).with_name("domain_material.jsonl")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            training = root / "train.jsonl"
            write_rows(
                training, [{"text": "unrelated fixture", "split": "train", "group": "fixture"}]
            )
            output = root / "benchmark"
            manifest = build(material, training, output)
            self.assertEqual(manifest["intent_groups"], 60)
            dev = read_rows(output / "development/bl-bn/queries.jsonl")
            test = read_rows(output / "test/bl-bn/queries.jsonl")
            self.assertFalse({r["group"] for r in dev} & {r["group"] for r in test})
            self.assertEqual(len(dev), 42)
            self.assertEqual(len(test), 18)
            for split in ("development", "test"):
                for direction in (
                    "bl-bn",
                    "bn-bl",
                    "en-bn",
                    "bn-en",
                    "bl-en",
                    "en-bl",
                    "en-en",
                    "bn-bn",
                    "bl-bl",
                ):
                    frozen = verify_manifest(output / split / direction)
                    self.assertFalse(frozen["human_review_completed"])
            with self.assertRaisesRegex(ValueError, "exists"):
                build(material, training, output)
            first = read_rows(material)[0]
            write_rows(training, [{"text": first["bn"], "split": "train", "group": "overlap"}])
            with self.assertRaisesRegex(ValueError, "overlaps"):
                build(material, training, root / "contaminated")
            self.assertFalse((root / "contaminated").exists())

    def test_nontraining_audit_input_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            training = root / "train.jsonl"
            write_rows(training, [{"text": "fixture", "split": "test", "group": "fixture"}])
            with self.assertRaisesRegex(ValueError, "training records"):
                build(Path(__file__).with_name("domain_material.jsonl"), training, root / "out")

    def test_complete_review_coverage_and_alternate_answers_are_preserved(self):
        material = Path(__file__).with_name("domain_material.jsonl")
        source_labels = Path(__file__).with_name("domain_intent_judgments_v1.jsonl")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            training = root / "train.jsonl"
            write_rows(training, [{"text": "fixture", "split": "train", "group": "fixture"}])
            # Alternate-answer labels must be used rather than overwritten by own-ID matches.
            labels = read_rows(source_labels)
            labels[1]["grade"] = 1
            local = root / "judgments.jsonl"
            write_rows(local, labels)
            local.with_suffix(".manifest.json").write_text(
                json.dumps(
                    {
                        "material_sha256": sha256(material),
                        "intent_judgments_sha256": sha256(local),
                    }
                )
            )
            output = root / "reviewed"
            result = build(material, training, output, judgments=local)
            for split, size in (("development", 42), ("test", 18)):
                frozen = verify_manifest(output / split / "bl-bn")
                self.assertEqual(frozen["judgments"], size * size)
                self.assertEqual(
                    frozen["unjudged_documents"], "none within this controlled candidate set"
                )
            self.assertTrue(
                any(r["grade"] == 1 for r in read_rows(output / "development/bl-bn/qrels.jsonl"))
            )
            write_rows(local, labels[:-1])
            local.with_suffix(".manifest.json").write_text(
                json.dumps(
                    {
                        "material_sha256": sha256(material),
                        "intent_judgments_sha256": sha256(local),
                    }
                )
            )
            with self.assertRaisesRegex(ValueError, "cover every"):
                build(material, training, root / "incomplete", judgments=local)
            self.assertFalse((root / "incomplete").exists())
            dev_id = next(
                i for i, split in result["split_by_intent"].items() if split == "development"
            )
            test_id = next(i for i, split in result["split_by_intent"].items() if split == "test")
            for row in labels:
                if row["query_intent"] == dev_id:
                    row["grade"] = 2 if row["document_intent"] == test_id else 0
            write_rows(local, labels)
            local.with_suffix(".manifest.json").write_text(
                json.dumps(
                    {
                        "material_sha256": sha256(material),
                        "intent_judgments_sha256": sha256(local),
                    }
                )
            )
            with self.assertRaisesRegex(ValueError, "positive in its reserved split"):
                build(material, training, root / "bad-split", judgments=local)
            self.assertFalse((root / "bad-split").exists())


if __name__ == "__main__":
    unittest.main()
