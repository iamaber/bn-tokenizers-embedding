import unittest

from deployment import distribution


class DeploymentTests(unittest.TestCase):
    def test_sample_tails_and_units(self):
        result = distribution([i / 1000 for i in range(1, 101)])
        self.assertAlmostEqual(result['p50_ms'], 50.5)
        self.assertAlmostEqual(result['p95_ms'], 95)
        self.assertAlmostEqual(result['p99_ms'], 99)
        self.assertEqual(result['samples'], 100)

    def test_invalid_samples_fail(self):
        for samples in ([], [0.], [-1.], [float('inf')], [float('nan')]):
            with self.assertRaises(ValueError):
                distribution(samples)


if __name__ == '__main__':
    unittest.main()
