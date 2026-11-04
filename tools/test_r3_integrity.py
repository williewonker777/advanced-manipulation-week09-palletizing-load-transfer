from pathlib import Path
import tempfile
import unittest
from tools.r3_integrity import check_hashes, sha


class TestR3Integrity(unittest.TestCase):
    def test_hash_checker_rejects_missing_or_changed_files(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            path = root / 'sample.txt'
            path.write_text('original')
            expected = {'sample.txt': sha(path)}
            self.assertTrue(all(check_hashes(root, expected).values()))
            path.write_text('changed')
            self.assertFalse(check_hashes(root, expected)['sample.txt'])
            path.unlink()
            self.assertFalse(check_hashes(root, expected)['sample.txt'])
