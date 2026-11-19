import json
from pathlib import Path
import tempfile
import unittest
import zipfile

from tools import r27_g4_replay as replay


class ResultComparisonTest(unittest.TestCase):
    def result(self):
        return {
            'episodes':58,
            'records':{
                'c1':{'cells':{'control':{'0.0':['1.0']},'collisions':[]},'stdout_sha256':'a'},
                'r27':{'cells':{'control':{'0.0':['1.0']},'collisions':[]},'stdout_sha256':'b'},
            },
        }

    def test_exact_result_is_exact(self):
        value=self.result();comparison=replay.compare_result(value,value,b'x',b'x')
        self.assertTrue(comparison['byte_exact'])
        self.assertTrue(comparison['semantic_exact'])
        self.assertTrue(comparison['all_non_time_cells_exact'])

    def test_only_opaque_evaluator_stdout_hash_may_differ(self):
        expected=self.result();actual=json.loads(json.dumps(expected))
        actual['records']['c1']['stdout_sha256']='changed-c1'
        actual['records']['r27']['stdout_sha256']='changed-r27'
        comparison=replay.compare_result(actual,expected,b'a',b'b')
        self.assertFalse(comparison['byte_exact'])
        self.assertTrue(comparison['semantic_exact'])
        self.assertTrue(comparison['all_non_time_cells_exact'])
        self.assertTrue(comparison['stdout_hash_only_difference'])

    def test_scientific_or_cell_difference_is_never_stripped(self):
        expected=self.result();actual=json.loads(json.dumps(expected))
        actual['records']['r27']['cells']['control']['0.0'][0]='2.0'
        comparison=replay.compare_result(actual,expected)
        self.assertFalse(comparison['semantic_exact'])
        self.assertFalse(comparison['all_non_time_cells_exact'])
        self.assertEqual(comparison['scientific_difference_paths'],
                         ['/records/r27/cells/control/0.0/0'])


class BundleTest(unittest.TestCase):
    def test_zip_is_deterministic_and_kit_free(self):
        with tempfile.TemporaryDirectory() as folder:
            first=Path(folder)/'first.zip';second=Path(folder)/'second.zip'
            a=replay.bundle(first);b=replay.bundle(second)
            self.assertEqual(a,b)
            self.assertEqual(first.read_bytes(),second.read_bytes())
            with zipfile.ZipFile(first) as archive:
                names=archive.namelist()
                self.assertTrue(all(name.startswith(replay.PREFIX+'/') for name in names))
                self.assertFalse(any('/sim/' in name or '/model/' in name for name in names))
                bundle=json.loads(archive.read(replay.PREFIX+'/bundle.json'))
                self.assertFalse(bundle['kit_included'])
                self.assertFalse(bundle['new_results_included'])


if __name__=='__main__':
    unittest.main()
