import csv
import json
import tempfile
import unittest
import zipfile
from pathlib import Path

from alignment import build
from prepare import read_rows, sha256, write_rows


class AlignmentTests(unittest.TestCase):
    def test_source_checksums_and_saved_groups_control_pair_eligibility(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            data = Path(directory)
            (data / "raw/banglatlit").mkdir(parents=True)
            (data / "raw/tatoeba").mkdir()
            (data / "processed").mkdir()
            raw_pairs = [
                ("one", "ami valo achi", "আমি ভালো আছি"),
                ("two", "tumi kemon acho", "তুমি কেমন আছো"),
                ("three", "notun ekta boi", "নতুন একটা বই"),
                ("held", "ami ekhane nei", "আমি এখানে নেই"),
            ]
            train = []
            held = []
            for group, roman, bengali in raw_pairs:
                destination = held if group == "held" else train
                for text, bucket in ((roman, "banglish"), (bengali, "bangla")):
                    destination.append(
                        {
                            "group": group,
                            "text": text,
                            "bucket": bucket,
                            "source": "banglatlit",
                            "attribution": "fixture",
                            "split": "validation" if group == "held" else "train",
                        }
                    )
            for text, bucket in (("this is good", "english"), ("এটা ভালো হয়েছে", "bangla")):
                train.append(
                    {
                        "group": "english",
                        "text": text,
                        "bucket": bucket,
                        "source": "tatoeba-ben",
                        "attribution": "fixture",
                        "split": "train",
                    }
                )
            write_rows(data / "processed/train.jsonl", train)
            write_rows(data / "processed/validation.jsonl", held)
            paths = []
            for split in ("train", "validation"):
                path = data / f"raw/banglatlit/{split}.csv"
                paths.append(path)
                with path.open("w", encoding="utf-8", newline="") as stream:
                    writer = csv.writer(stream)
                    writer.writerow(["id", "text_transliterated", "text_bengali"])
                    if split == "train":
                        writer.writerows(raw_pairs)
            archive = data / "raw/tatoeba/ben-eng.zip"
            paths.append(archive)
            with zipfile.ZipFile(archive, "w") as stream:
                stream.writestr("ben.txt", "this is good\tএটা ভালো হয়েছে\tfixture\n")
            lock = [
                {
                    "path": str(p.relative_to(data / "raw")),
                    "bytes": p.stat().st_size,
                    "sha256": sha256(p),
                }
                for p in paths
            ]
            (data / "sources.lock.json").write_text(json.dumps(lock))
            result = build(data, data / "alignment", count=4)
            self.assertEqual(result["pairs"], 4)
            self.assertFalse(result["frozen_test_opened"])
            groups = {r["group"] for r in read_rows(data / "alignment/pairs.jsonl")}
            self.assertNotIn("held", groups)
            self.assertIn("english", groups)
            self.assertEqual(len(read_rows(data / "alignment/adaptation.jsonl")), 8)
            paths[0].write_text("changed")
            with self.assertRaisesRegex(ValueError, "checksum"):
                build(data, data / "bad", count=4)
            self.assertFalse((data / "bad").exists())


if __name__ == "__main__":
    unittest.main()
