import json
import random
import tempfile
import unittest
from pathlib import Path

from hard_negatives import clean_alignment, freeze
from prepare import read_rows, sha256, write_rows
from train_extended import balanced_batch, verified_rows


class HardNegativeTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.alignment, self.packet = self.root / 'alignment', self.root / 'packet'
        self.alignment.mkdir(); self.packet.mkdir()
        self.pairs = [{
            'group': str(i), 'query': f'question {i}', 'document': f'answer {i}',
            'query_language': 'english' if i < 20 else 'banglish', 'source': 'fixture'
        } for i in range(40)]
        write_rows(self.alignment / 'pairs.jsonl', self.pairs)
        (self.alignment / 'manifest.json').write_text(json.dumps({
            'files': {'pairs.jsonl': sha256(self.alignment / 'pairs.jsonl')}
        }))
        packet = [dict(row, negative_group=str((i + 1) % 40), negative=f'answer {(i + 1) % 40}')
                  for i, row in enumerate(self.pairs)]
        write_rows(self.packet / 'packet.jsonl', packet)
        (self.packet / 'manifest.json').write_text(json.dumps({
            'alignment': str(self.alignment),
            'alignment_manifest_sha256': sha256(self.alignment / 'manifest.json'),
            'packet_sha256': sha256(self.packet / 'packet.jsonl')
        }))
        self.labels = self.root / 'labels.jsonl'
        self.decisions = [{
            'group': row['group'], 'negative_group': row['negative_group'],
            'pair_correct': True, 'negative_grade': 0, 'rationale': 'Different requested fact',
            'reviewer_id': 'test', 'reviewer_origin': 'agent'
        } for row in packet]
        write_rows(self.labels, self.decisions)

    def tearDown(self):
        self.temp.cleanup()

    def test_partial_matches_excluded_and_known_pair_error_removed_from_new_alignment(self):
        self.decisions[0]['pair_correct'] = False
        self.decisions[1]['negative_grade'] = 1
        write_rows(self.labels, self.decisions)
        bank = self.root / 'bank'
        result = freeze(self.packet, self.labels, bank)
        self.assertEqual(result['accepted'], 38)
        clean = self.root / 'clean'
        report = clean_alignment(self.alignment, bank, self.labels, clean)
        self.assertEqual(report['known_pair_errors_removed'], 1)
        self.assertEqual(len(read_rows(clean / 'pairs.jsonl')), 39)
        self.assertEqual(read_rows(self.alignment / 'pairs.jsonl'), self.pairs)

    def test_incomplete_boolean_grade_or_text_override_rejected_before_output(self):
        for rows in (self.decisions[:-1], [dict(self.decisions[0], negative_grade=False), *self.decisions[1:]],
                     [dict(self.decisions[0], query='changed query'), *self.decisions[1:]]):
            write_rows(self.labels, rows)
            with self.assertRaises(ValueError):
                freeze(self.packet, self.labels, self.root / 'bad-bank')
            self.assertFalse((self.root / 'bad-bank').exists())

    def test_balanced_sampling_and_input_checksum(self):
        batch = balanced_batch(self.pairs, random.Random(17), 16)
        self.assertEqual(sum(row['query_language'] == 'english' for row in batch), 8)
        self.assertEqual(len({row['group'] for row in batch}), 16)
        for size in (1, 3, 100):
            with self.assertRaises(ValueError):
                balanced_batch(self.pairs, random.Random(17), size)
        digest = sha256(self.alignment / 'manifest.json')
        self.assertEqual(verified_rows(self.alignment, digest), self.pairs)
        write_rows(self.alignment / 'pairs.jsonl', self.pairs[:-1])
        with self.assertRaisesRegex(ValueError, 'pairs changed'):
            verified_rows(self.alignment, digest)


if __name__ == '__main__':
    unittest.main()
