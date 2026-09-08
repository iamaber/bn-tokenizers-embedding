"""Integration tests exercise the actual installed Go library, without mocks."""

import json
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from bn_tokenizers_embedding import Tokenizer, normalize


class TokenizerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.directory = tempfile.TemporaryDirectory()
        cls.path = Path(cls.directory.name) / "model.json"
        cls.path.write_text(
            json.dumps(
                {
                    "version": 1,
                    "algorithm": "unigram",
                    "normalization": "nfc-whitespace-v1",
                    "pieces": [{"text": "আমি", "score": -1}, {"text": "hello", "score": -2}],
                }
            ),
            encoding="utf-8",
        )

    @classmethod
    def tearDownClass(cls) -> None:
        cls.directory.cleanup()

    def test_round_trip_and_batch(self) -> None:
        texts = [
            "",
            " আমি\tভালো আছি। ",
            "ami aaj valo asi",
            "meetingটা 🙂",
            "ক্\u200dষ ক্\u200cষ",
            "e\u0301",
            "<unk> ▁ <0x20>",
            "a\0b",
        ]
        with Tokenizer(self.path) as tok:
            self.assertEqual(tok.vocab_size, 262)
            batch = tok.encode_batch(texts)
            self.assertEqual(batch, [tok.encode(text) for text in texts])
            self.assertEqual([tok.decode(ids) for ids in batch], [normalize(s) for s in texts])
            self.assertEqual(tok.encode_batch([]), [])
            self.assertEqual(tok.decode([]), "")

    def test_normalization_contract(self) -> None:
        self.assertEqual(normalize("  e\u0301\tHello\n"), "é Hello")

    def test_close_and_errors(self) -> None:
        tok = Tokenizer(self.path)
        with self.assertRaises(ValueError):
            tok.decode([0])
        with self.assertRaises(ValueError):
            tok.decode([259])  # Invalid standalone UTF-8 byte.
        with self.assertRaises(TypeError):
            tok.decode([True])
        with self.assertRaises(TypeError):
            tok.encode_batch("not a batch")
        with self.assertRaises(UnicodeEncodeError):
            tok.encode("\ud800")
        tok.close()
        tok.close()
        self.assertTrue(tok.closed)
        with self.assertRaises(RuntimeError):
            tok.encode("closed")
        with self.assertRaises(ValueError):
            Tokenizer(self.path.parent / "missing.json")

    def test_concurrent_encoding(self) -> None:
        with Tokenizer(self.path) as tok:
            text = "আমি hello ami valo 🙂"
            expected = tok.encode(text)
            with ThreadPoolExecutor(max_workers=8) as pool:
                self.assertEqual(list(pool.map(tok.encode, [text] * 200)), [expected] * 200)


if __name__ == "__main__":
    unittest.main()
