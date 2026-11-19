"""Exercise the offline export using the finalized, kit-free R26 release."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import zipfile

from tools import r26_challenge_bundle as bundle


class IndependentBundleTests(unittest.TestCase):
    def test_fresh_output_never_overwrites_or_writes_into_project(self):
        with tempfile.TemporaryDirectory() as tmp:
            existing = Path(tmp) / 'existing.zip'
            existing.write_bytes(b'keep')
            with self.assertRaises(ValueError):
                bundle.build(existing)
            self.assertEqual(existing.read_bytes(), b'keep')
        with self.assertRaises(ValueError):
            bundle.build(bundle.ROOT / 'should-not-be-written.zip')

    def test_source_pin_cannot_admit_kit_or_traversal(self):
        for name in ('sim/plant.py', 'model/f.xml', 'run.py', '../elsewhere'):
            with self.subTest(name=name), tempfile.TemporaryDirectory() as tmp:
                frozen = {'files': {name: '0' * 64}, 'entrypoint': 'test'}
                out = Path(tmp) / 'bad.zip'
                with patch.object(bundle.support, 'read_release', return_value=frozen):
                    with self.assertRaises(ValueError):
                        bundle.build(out)
                self.assertFalse(out.exists())

    def test_deterministic_and_every_archived_member_has_a_hash(self):
        with tempfile.TemporaryDirectory() as tmp:
            first, second = (Path(tmp) / name for name in ('first.zip', 'second.zip'))
            self.assertEqual(bundle.build(first), bundle.build(second))
            self.assertEqual(first.read_bytes(), second.read_bytes())
            with zipfile.ZipFile(first) as archive:
                prefix = bundle.PREFIX + '/'
                receipt = json.loads(archive.read(prefix + 'bundle.json'))
                self.assertIsNone(archive.testzip())
                self.assertFalse(receipt['result_included'])
                self.assertFalse(receipt['course_kit_included'])
                self.assertFalse(receipt['default_promoted'])
                pins = receipt['members_sha256']
                self.assertEqual(set(archive.namelist()),
                                 {prefix + name for name in [*pins, 'bundle.json']})
                for name, sha in pins.items():
                    self.assertEqual(hashlib.sha256(archive.read(prefix + name)).hexdigest(), sha, name)
                self.assertTrue(set(bundle.PATHS).issubset(pins))
                self.assertIn('tools/r26_challenge_replay.py', pins)
                self.assertNotIn('results/r26/independent_ledger.json', pins)


if __name__ == '__main__':
    unittest.main()
