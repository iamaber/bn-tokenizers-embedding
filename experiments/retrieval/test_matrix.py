import copy
import json
import tempfile
import unittest
from pathlib import Path

from prepare import sha256
from run_matrix import run


class MatrixResumeTests(unittest.TestCase):
    def test_resume_rejects_wrong_original_arm_base_and_changed_checkpoint(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            base = root / "revision"
            base.mkdir()
            (base / "model.safetensors").write_bytes(b"base")
            alignment = root / "alignment"
            alignment.mkdir()
            (alignment / "manifest.json").write_text("{}")
            protocol = root / "protocol.json"
            protocol.write_text(
                json.dumps(
                    {
                        "alignment_directory": str(alignment),
                        "alignment_manifest_sha256": sha256(alignment / "manifest.json"),
                        "base_revision": base.name,
                        "training_seeds": [17],
                        "arms": ["original"],
                        "batch_size": 16,
                        "maximum_training_tokens": 128,
                        "equal_data": {"representation_steps": 64, "contrastive_steps": 128},
                    }
                )
            )
            checkpoint = root / "encoders/equal-data/original-seed17"
            checkpoint.mkdir(parents=True)
            (checkpoint / "model.safetensors").write_bytes(b"checkpoint")
            metadata = checkpoint / "encoder.json"
            metadata.write_text(json.dumps({"custom_tokenizer": None}))
            valid = {
                "seed": 17,
                "budget_kind": "equal-data",
                "batch_size": 16,
                "maximum_tokens": 128,
                "alignment_manifest_sha256": sha256(alignment / "manifest.json"),
                "tokenizer_initialization": {
                    "method": "retain original tokenizer and input embeddings"
                },
                "base_model_files": {"model.safetensors": sha256(base / "model.safetensors")},
                "stages": {"representation": {"steps": 64}, "contrastive": {"steps": 128}},
                "checkpoint_files": {"model.safetensors": sha256(checkpoint / "model.safetensors")},
                "total_training_seconds": 100,
                "training_sequence_sha256": "sequence",
            }
            report = checkpoint / "training-report.json"
            report.write_text(json.dumps(valid))
            self.assertEqual(len(run(protocol, base, root / "encoders", "equal-data")["runs"]), 1)
            metadata.write_text(json.dumps({"custom_tokenizer": "go-tokenizer.json"}))
            with self.assertRaisesRegex(ValueError, "identity"):
                run(protocol, base, root / "encoders", "equal-data")
            metadata.write_text(json.dumps({"custom_tokenizer": None}))
            for key, value, message in (
                ("base_model_files", {}, "base model"),
                ("tokenizer_initialization", {"method": "other"}, "initialization"),
            ):
                bad = copy.deepcopy(valid)
                bad[key] = value
                report.write_text(json.dumps(bad))
                with self.assertRaisesRegex(ValueError, message):
                    run(protocol, base, root / "encoders", "equal-data")
            report.write_text(json.dumps(valid))
            (checkpoint / "model.safetensors").write_bytes(b"changed")
            with self.assertRaisesRegex(ValueError, "checksum"):
                run(protocol, base, root / "encoders", "equal-data")


if __name__ == "__main__":
    unittest.main()
