import tempfile
from pathlib import Path
import unittest
import zipfile

from tools import r27_force_replay as api


class ForceReplayTests(unittest.TestCase):
    def test_course_contents_and_new_blind_source_are_absent(self):
        _,reg,src,names=api.closure()
        self.assertFalse(reg['kit_included']);self.assertFalse(names['default_promoted'])
        self.assertTrue(all(api.legal(f) for f in src))
        self.assertFalse(any('r27_loaded_holdout' in f for f in src))
        for n in ('../outside','/tmp/private','sim/runner.py','model/a.xml','run.py','a/../../x','.git/HEAD','course.stl'):
            with self.subTest(n=n):self.assertFalse(api.legal(n))

    def test_refuses_existing_or_in_tree_output(self):
        with tempfile.TemporaryDirectory()as tmp:
            f=Path(tmp)/'existing';f.write_bytes(b'keep')
            with self.assertRaises(ValueError):api.build(f)
            self.assertEqual(f.read_bytes(),b'keep')
        with self.assertRaises(ValueError):api.build(api.ROOT/'r27-should-not-be-created.zip')

    def test_rebuild_includes_anchors_but_no_new_candidate_results(self):
        with tempfile.TemporaryDirectory()as tmp:
            a,b=(Path(tmp)/n for n in ('a.zip','b.zip'))
            self.assertEqual(api.build(a),api.build(b))
            with zipfile.ZipFile(a)as z:
                self.assertIsNone(z.testzip());names=z.namelist()
                self.assertIn(api.PREFIX+'/experiments/r27_portable/reference.json',names)
                self.assertNotIn(api.PREFIX+'/result.json',names)
