import hashlib
from pathlib import Path
import tempfile
import unittest
from tools.audit_public import audit, text_errors


class PublicAuditTests(unittest.TestCase):
    def test_sensitive_content(self):
        self.assertTrue(text_errors('/' + 'home/user/work'))
        self.assertTrue(text_errors('TASK-' + '363'))
        self.assertTrue(text_errors('person' + '@example.com'))
        self.assertFalse(text_errors('python evaluate.py --ctrl my_servo --detector my_detector'))

    def test_checksum_coverage_and_exclusion(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'artifacts').mkdir()
            (root / 'README.md').write_text('Our code only\n')
            digest = hashlib.sha256((root / 'README.md').read_bytes()).hexdigest()
            manifest = root / 'artifacts/SHA256SUMS'
            manifest.write_text(f'{digest}  README.md\n')
            self.assertEqual(audit(root)['status'], 'pass')
            (root / 'evaluate.py').write_text('kit file')
            self.assertEqual(audit(root)['status'], 'FAIL')


if __name__ == '__main__':
    unittest.main()
