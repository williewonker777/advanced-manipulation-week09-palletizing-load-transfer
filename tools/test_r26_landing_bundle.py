"""The box-8 example stays outside the course-free study bundle."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
import zipfile

from tools import r26_landing_bundle as bundle


class LandingBundleTests(unittest.TestCase):
    def test_course_program_cannot_be_read_as_research_member(self):
        for n in ('sim/runner.py','model/a.xml','evaluate.py','run.py','../outside','collision_detector.py'):
            with self.subTest(n=n), self.assertRaises(ValueError):bundle.read_member(n)

    def test_existing_output_is_never_overwritten(self):
        with tempfile.TemporaryDirectory() as t:
            p=Path(t)/'existing';p.write_bytes(b'sentinel')
            with self.assertRaises(ValueError):bundle.build(p)
            self.assertEqual(p.read_bytes(),b'sentinel')

    def test_reproducible_members_and_tool_separation(self):
        with tempfile.TemporaryDirectory() as t:
            a=Path(t)/'a.zip';b=Path(t)/'b.zip'
            self.assertEqual(bundle.build(a),bundle.build(b))
            with zipfile.ZipFile(a)as z:
                self.assertIsNone(z.testzip())
                prefix=bundle.ROOT_NAME+'/'
                receipt=json.loads(z.read(prefix+'receipt.json'))
                self.assertFalse(receipt['kit_or_original_image_included'])
                self.assertFalse(receipt['default_promoted'])
                self.assertEqual(set(z.namelist()),{prefix+n for n in receipt['files_sha256']}|{prefix+'receipt.json'})
                for n,h in receipt['files_sha256'].items():
                    self.assertEqual(hashlib.sha256(z.read(prefix+n)).hexdigest(),h,n)
                self.assertIn('tools/r26_landing_replay.py',receipt['files_sha256'])
                self.assertNotIn('result.png',receipt['files_sha256'])


if __name__=='__main__':unittest.main()
