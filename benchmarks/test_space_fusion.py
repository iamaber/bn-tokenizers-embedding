import unittest

from space_fusion import expand, fuse


class SpaceFusionTests(unittest.TestCase):
    def test_exact_id_recovery_with_unmapped_and_adjacent_spaces(self) -> None:
        mapping = {260: 16000, 261: 16001}
        reverse = {new: old for old, new in mapping.items()}
        for ids in [[], [36], [36, 260], [260, 36, 261], [36, 36, 260], [36, 999]]:
            with self.subTest(ids=ids):
                self.assertEqual(expand(fuse(ids, mapping), reverse), ids)
        self.assertEqual(fuse([260, 36, 261], mapping), [260, 16001])


if __name__ == "__main__":
    unittest.main()
