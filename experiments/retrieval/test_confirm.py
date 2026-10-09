import json
import tempfile
import unittest
from pathlib import Path

from confirm import confirm
from prepare import sha256


class ConfirmationTests(unittest.TestCase):
    def test_changed_frozen_selection_is_rejected_before_analysis(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            development = root / "development.json"
            development.write_text("{}")
            selection = root / "selection.json"
            selection.write_text(
                json.dumps(
                    {
                        "selection": "unchanged",
                        "selection_frozen_before_reserved_test": True,
                        "evaluation_suite": str(development),
                        "evaluation_suite_sha256": sha256(development),
                    }
                )
            )
            suite = root / "test.json"
            suite.write_text(json.dumps({"frozen_selection_sha256": sha256(selection)}))
            decision = json.loads(selection.read_text())
            decision["selection"] = "unigram-32000"
            selection.write_text(json.dumps(decision))
            with self.assertRaisesRegex(ValueError, "unchanged frozen development"):
                confirm(suite, selection, root / "output.json")
            self.assertFalse((root / "output.json").exists())


if __name__ == "__main__":
    unittest.main()
