import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch
from context_transfer import apply, sparsemax
from prepare import sha256


class SparseTransferTests(unittest.TestCase):
    def test_projection_is_sparse_normalized_and_translation_invariant(self):
        scores = torch.tensor([[3., 1., 0.], [.6, .5, .1], [0., 0., 0.]])
        weights = sparsemax(scores)
        self.assertTrue(torch.allclose(weights.sum(1), torch.ones(3)))
        self.assertTrue(torch.equal(weights[0], torch.tensor([1., 0., 0.])))
        self.assertTrue(torch.allclose(weights[2], torch.ones(3) / 3))
        self.assertTrue(torch.all(weights >= 0))
        self.assertTrue(torch.allclose(weights, sparsemax(scores + 100), atol=1e-5))

    def test_invalid_scores_fail(self):
        for scores in (torch.zeros(2, 0), torch.ones(3), torch.tensor([[float('nan')]])):
            with self.assertRaises(ValueError):
                sparsemax(scores)

    def test_artifact_application_checks_inputs_before_changing_weights(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            base, artifact = root / 'base', root / 'artifact'
            base.mkdir(); artifact.mkdir()
            (base / 'config.json').write_text('{}')
            tokenizer = root / 'tokenizer.json'
            tokenizer.write_text('{}')
            weights = torch.tensor([[0., 0., 0.], [1., 2., 3.]])
            torch.save(weights, artifact / 'embeddings.pt')
            manifest = {
                'tokenizer_sha256': sha256(tokenizer),
                'base_files': {'config.json': sha256(base / 'config.json')},
                'embedding_sha256': sha256(artifact / 'embeddings.pt')
            }
            (artifact / 'manifest.json').write_text(json.dumps(manifest))
            table = torch.nn.Embedding(2, 3, padding_idx=0)
            encoder = SimpleNamespace(custom_path=tokenizer,
                                      model=SimpleNamespace(get_input_embeddings=lambda: table))
            apply(encoder, artifact, base)
            self.assertTrue(torch.equal(table.weight, weights))
            for changed in (tokenizer, base / 'config.json', artifact / 'embeddings.pt'):
                original = changed.read_bytes()
                changed.write_bytes(original + b' ')
                with self.assertRaises(ValueError):
                    apply(encoder, artifact, base)
                self.assertTrue(torch.equal(table.weight, weights))
                changed.write_bytes(original)


if __name__ == "__main__":
    unittest.main()
