"""Checks for deterministic, closed R30 replay and non-destructive output behavior."""
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
import zipfile

from tools import r30_event_replay as replay


class DisclosedReplay(unittest.TestCase):
    def test_no_unrun_inputs_or_original_material_in_archive(self):
        with TemporaryDirectory() as td:
            first,second=(Path(td)/n for n in ('one.zip','two.zip'))
            self.assertEqual(replay.bundle(first),replay.bundle(second))
            self.assertEqual(first.read_bytes(),second.read_bytes())
            with zipfile.ZipFile(first) as z:
                prefix=replay.PREFIX+'/'
                names=[n.removeprefix(prefix) for n in z.namelist()]
                self.assertEqual(len(names),len(set(names)))
                self.assertTrue(all(n.startswith(prefix) for n in z.namelist()))
                raw,sources,_=replay.load()
                self.assertEqual(set(names),set(sources)|{replay.CLOSURE,'README.md','bundle.json'})
                self.assertEqual(z.read(prefix+replay.CLOSURE),raw)
                contract=json.loads(z.read(prefix+'bundle.json'))
                for name,digest in contract['members'].items():
                    self.assertEqual(replay.shared.digest(z.read(prefix+name)),digest)
                self.assertFalse(any({'sim','model','r27_loaded_holdout','r28_landing_prospect',
                                      'r30_torque_qualification','r30_yaw_qualification','__pycache__'}
                                     & set(Path(n).parts) for n in names))
                self.assertFalse(any(Path(n).name in replay.previous.base.STOCK for n in names))
                self.assertTrue({'LICENSE','THIRD_PARTY_NOTICES'}.issubset(names))
                self.assertFalse(contract['kit_included'] or contract['unopened_conditions_included'])
            with self.assertRaises(ValueError):
                replay.bundle(first)

    def test_only_published_fixed_physical_scopes_and_arms(self):
        self.assertEqual(replay.SCOPES,{'signed-stock':50,'paired-directions':60,'place':25})
        self.assertEqual(replay.ARMS,('c3','composed','rising','geometry','headroom'))
        raw,blobs,overlay=replay.load()
        self.assertEqual(set(json.loads(raw)['files']),replay.MEMBERS)
        self.assertTrue(set(overlay)<=blobs.keys())

    def test_reject_unsafe_destination(self):
        with self.assertRaises(ValueError):
            replay.bundle(replay.ROOT/'reserved-r30.zip')
        with TemporaryDirectory() as td:
            d=Path(td); (d/'alias').symlink_to(d,target_is_directory=True)
            with self.assertRaises(ValueError):
                replay.bundle(d/'alias'/'not-created.zip')


if __name__=='__main__':
    unittest.main()
