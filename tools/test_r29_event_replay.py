"""Integrity and scope checks for the R29 event replay; physical equivalence uses the licensed kit."""
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
import zipfile
from tools import r29_event_replay as replay


class R29ReplayTest(unittest.TestCase):
    def test_authored_deterministic_archive(self):
        with TemporaryDirectory() as tmp:
            a,b=(Path(tmp)/n for n in ('a.zip','b.zip'))
            self.assertEqual(replay.bundle(a),replay.bundle(b))
            self.assertEqual(a.read_bytes(),b.read_bytes())
            with zipfile.ZipFile(a) as z:
                names=z.namelist();prefix=replay.PREFIX+'/'
                historical=set(replay.base.closure()[1]['files'])
                self.assertEqual(len(names),len(set(names)))
                self.assertTrue(all(n.startswith(prefix) for n in names))
                for name in names:
                    rest=name[len(prefix):]; parts=Path(rest).parts
                    self.assertFalse({'..','sim','model','r27_loaded_holdout','r28_landing_prospect','__pycache__'}&set(parts))
                    if 'results' in parts:
                        self.assertIn(rest,historical)
                        self.assertNotIn('r29',parts)
                for n in ('LICENSE','THIRD_PARTY_NOTICES','experiments/r29_event_replay/runner.py'):
                    self.assertIn(prefix+n,names)
                manifest=json.loads(z.read(prefix+'bundle.json'))
                self.assertFalse(manifest['candidate_results_included'])
                self.assertFalse(manifest['kit_included'])
                self.assertFalse(manifest['unopened_conditions_included'])
            with self.assertRaises(ValueError):replay.bundle(a)

    def test_do_not_enroll_a_later_arm_in_frozen_prospect(self):
        with self.assertRaisesRegex(ValueError,'locked'):
            replay.compare(SimpleNamespace(scope='prospective-detached',arm='composed'))

    def test_reject_source_tree_and_symlink_outputs(self):
        with self.assertRaises(ValueError):replay.bundle(replay.ROOT/'forbidden-archive.zip')
        with TemporaryDirectory()as td:
            path=Path(td);(path/'link').symlink_to(path,target_is_directory=True)
            with self.assertRaises(ValueError):replay.bundle(path/'link'/'refuse.zip')
