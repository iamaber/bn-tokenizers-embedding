"""Reject relabeled or changed checkpoints before fresh model evaluation."""

import json
import tempfile
import unittest
from pathlib import Path

from original_comparison import run, save_frozen, verify_short
from prepare import sha256


class OriginalComparisonTests(unittest.TestCase):
    def test_inventory_name_cannot_relabel_a_different_training_run(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            protocol = root / "protocol.json"
            protocol.write_text("{}")
            criteria = root / "criteria.json"
            criteria.write_text(json.dumps({"required_budgets": ["equal-data"],
                                             "required_training_seeds": [17]}))
            checkpoint = root / "checkpoint"
            checkpoint.mkdir()
            report = checkpoint / "training-report.json"
            report.write_text(json.dumps({"protocol_sha256": sha256(protocol),
                                           "arm": "bpe-context", "seed": 999,
                                           "budget": "equal-data"}))
            inventory = root / "inventory.json"
            inventory.write_text(json.dumps({"protocol_sha256": sha256(protocol), "runs": [
                {"name": "equal-data/original-seed17", "directory": str(checkpoint),
                 "training_report_sha256": sha256(report)}]}))
            output = root / "evaluation"
            with self.assertRaisesRegex(ValueError, "checkpoint does not match"):
                run(root / "unread-benchmark", inventory, root / "base", criteria, output, protocol)
            self.assertFalse(output.exists())

    def test_changed_short_checkpoint_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            base, checkpoint = root / "base", root / "checkpoint"
            base.mkdir()
            checkpoint.mkdir()
            (checkpoint / "encoder.json").write_text('{"custom_tokenizer": null}')
            weights = checkpoint / "model.safetensors"
            weights.write_bytes(b"original")
            report = {"seed": 17, "budget_kind": "equal-data", "base_model_files": {},
                      "tokenizer_initialization": {"method": "retain original tokenizer and input embeddings"},
                      "checkpoint_files": {p.name: sha256(p) for p in checkpoint.iterdir()}}
            (checkpoint / "training-report.json").write_text(json.dumps(report))
            verify_short(checkpoint, base, "equal-data", 17)
            weights.write_bytes(b"replacement")
            with self.assertRaisesRegex(ValueError, "checkpoint changed"):
                verify_short(checkpoint, base, "equal-data", 17)

    def test_frozen_selection_cannot_be_replaced(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "selection.json"
            save_frozen(path, {"selection": "unchanged"})
            with self.assertRaisesRegex(ValueError, "frozen artifact differs"):
                save_frozen(path, {"selection": "long-original"})
            self.assertEqual(json.loads(path.read_text()), {"selection": "unchanged"})


if __name__ == "__main__":
    unittest.main()
