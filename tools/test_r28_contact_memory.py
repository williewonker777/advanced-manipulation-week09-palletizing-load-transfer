import tempfile
from pathlib import Path
import unittest
import zipfile

from tools import r28_contact_memory as replay


class MemoryBundleTests(unittest.TestCase):
    def test_full_kit_free_dependency_closure(self):
        sources,overlay=replay.sources()
        self.assertIn('experiments/r28_contact_memory/official.py',overlay)
        self.assertIn('experiments/r28_finite_contact/pulse_diagnostic.py',overlay)
        self.assertFalse(any(set(Path(n).parts)&{'sim','model','r27_loaded_holdout','fresh'} for n in sources))
        self.assertFalse(any(Path(n).name in {'evaluate.py','collision_detector.py','result.json'} for n in sources))

    def test_deterministic_external_archive_and_no_overwrite(self):
        with tempfile.TemporaryDirectory()as tmp:
            first,second=(Path(tmp)/n for n in ('A.zip','B.zip'))
            self.assertEqual(replay.bundle(first),replay.bundle(second))
            with self.assertRaises(ValueError):replay.bundle(first)
            with zipfile.ZipFile(first)as z:
                self.assertIsNone(z.testzip())
                self.assertIn(replay.PREFIX+'/README.md',z.namelist())
        with self.assertRaises(ValueError):replay.bundle(replay.ROOT/'never-in-tree.zip')
