import tempfile
import unittest
from pathlib import Path

from domains import build
from prepare import read_rows, verify_manifest, write_rows


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
                    "bl-bn", "bn-bl", "en-bn", "bn-en", "bl-en", "en-bl", "en-en", "bn-bn", "bl-bl"
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


if __name__ == "__main__":
    unittest.main()
