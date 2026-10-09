import importlib.util
import tempfile
import unittest
from pathlib import Path

BASE = Path(
    "models/hf-cache/models--intfloat--multilingual-e5-small/snapshots/"
    "614241f622f53c4eeff9890bdc4f31cfecc418b3"
)
HAS_RUNTIME = all(
    importlib.util.find_spec(name) for name in ("torch", "transformers", "bn_tokenizers_embedding")
)


@unittest.skipUnless(HAS_RUNTIME and BASE.exists(), "requires local pinned research environment")
class EncoderTests(unittest.TestCase):
    def test_control_ids_truncation_mask_and_checkpoint_restore(self) -> None:
        import torch
        from encoder import Encoder

        torch.set_num_threads(4)
        encoder = Encoder(BASE, Path("models/unigram-32000.json"), max_length=8)
        self.addCleanup(encoder.close)
        inputs, counts = encoder.inputs(["আমি ভালো আছি " * 20, "hi"])
        self.assertEqual(int(inputs["input_ids"][0, 0]), 2)
        self.assertEqual(int(inputs["input_ids"][0, -1]), 3)
        self.assertEqual(counts["truncated_texts"], 1)
        self.assertEqual(tuple(inputs["input_ids"].shape), (2, 8))
        self.assertTrue(torch.all(inputs["input_ids"][inputs["attention_mask"] == 0] == 0))
        self.assertEqual(encoder.model.get_input_embeddings().padding_idx, 0)
        vectors, _ = encoder.encode(["query: ami valo achi"])
        self.assertTrue(torch.allclose(vectors.norm(dim=1), torch.ones(1)))
        with tempfile.TemporaryDirectory() as directory:
            saved = Path(directory) / "encoder"
            encoder.save(saved)
            restored = Encoder(saved)
            self.addCleanup(restored.close)
            self.assertEqual(restored.max_length, 8)
            again, _ = restored.encode(["query: ami valo achi"])
            self.assertTrue(torch.allclose(vectors, again, atol=1e-6))

    def test_original_padding_matches_tokenizer_so_bos_can_train(self) -> None:
        from encoder import Encoder

        original = Encoder(BASE)
        self.addCleanup(original.close)
        self.assertEqual(
            original.model.get_input_embeddings().padding_idx, original.tokenizer.pad_token_id
        )
        self.assertNotEqual(
            original.model.get_input_embeddings().padding_idx, original.tokenizer.bos_token_id
        )


if __name__ == "__main__":
    unittest.main()
