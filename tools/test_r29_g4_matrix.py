import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
import zipfile

from tools import r29_g4_matrix as m


class G4MatrixTest(unittest.TestCase):
    def test_nothing_but_runner_location_can_be_ignored(self):
        j=dict(rows=[dict(delay=34,detected=True)],runner_sha256='a',primary_protection_passed=False)
        self.assertEqual(m.scientific(j),m.scientific(dict(j,runner_sha256='b')))
        for other in (dict(j,primary_protection_passed=True),dict(j,rows=[dict(delay=33,detected=True)])):
            self.assertNotEqual(m.scientific(j),m.scientific(other))
        with self.assertRaises(ValueError):m.scientific({})

    def test_source_list_excludes_original_and_unopened_inputs(self):
        for name in ('sim/runner.py','/etc/passwd','../x','x/evaluate.py','x/result.npz','model/x.stl',
                     'experiments/r27_loaded_holdout/plan.json','x/__pycache__/x.pyc','results/r29/g4-prospective-locked.json','old/result.json'):
            with self.assertRaises(ValueError):m.file(name)

    def test_rebuild_is_deterministic_and_enrolled_arms_only(self):
        with TemporaryDirectory()as tmp:
            a,b=(Path(tmp)/name for name in ('a.zip','b.zip'))
            self.assertEqual(m.bundle(a),m.bundle(b))
            with self.assertRaises(ValueError): m.bundle(a)
            with zipfile.ZipFile(a)as z:
                p=m.PREFIX+'/'
                files=set(z.namelist())
                self.assertIn(p+'LICENSE',files);self.assertIn(p+'THIRD_PARTY_NOTICES',files)
                plan=json.loads(z.read(p+'experiments/r29_g4_matrix_replay/plan.json'))
                self.assertEqual(set(plan['arms']),{'c1','v3','memory','controller'})
                self.assertFalse(any(n.startswith(p+'sim/') or n.startswith(p+'model/') or n.endswith('result.json') for n in files))
                self.assertFalse(json.loads(z.read(p+'bundle.json'))['prior_results_included'])
