"""Public membership, actual matched twins and safe output behavior."""
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
import zipfile

from tools import r31_event_replay as replay


class ReplayContract(unittest.TestCase):
    def test_only_closed_members_roundtrip_and_repeat(self):
        with TemporaryDirectory() as td:
            a,b=(Path(td)/n for n in ('a.zip','b.zip'))
            self.assertEqual(replay.bundle(a), replay.bundle(b))
            self.assertEqual(a.read_bytes(), b.read_bytes())
            raw, src, _ = replay.load()
            with zipfile.ZipFile(a) as z:
                names={s.removeprefix(replay.PREFIX+'/') for s in z.namelist()}
                self.assertEqual(names, set(src)|{replay.CLOSURE,'README.md','bundle.json'})
                self.assertEqual(z.read(replay.PREFIX+'/'+replay.CLOSURE), raw)
                self.assertFalse(any('qualification' in Path(n).parts or Path(n).parts[0] in {'sim','model'} or
                                     n.startswith(('experiments/r31_distinct','experiments/r31_fresh','results/r31/'))
                                     for n in names))

    def test_non_destructive_output(self):
        with TemporaryDirectory() as td:
            p=Path(td)/'used.zip'
            p.write_bytes(b'keep')
            with self.assertRaises(ValueError): replay.bundle(p)
            self.assertEqual(p.read_bytes(), b'keep')

    def test_twins_are_true_matching_references(self):
        from experiments.r31_event_replay import runner
        for scope,ids in runner.CASES.items():
            rows,refs=runner.select(runner.check.build_cases('my_torque'),ids)
            seen={c['case_id'] for c in rows}
            self.assertEqual({c['case_id'] for c in rows if c['kind']=='collision'}, set(ids))
            for c in rows:
                ref=refs[runner.check.old.reference_key(c)]
                self.assertIn(ref['case_id'], seen, scope)
                self.assertIsNone(ref['collision'])
                self.assertIsNone(ref['pick_offset'])
                self.assertEqual((ref['mass'],ref['com']),(c['mass'],c['com']))


    def test_widening_scope_preserves_explicit_comparators_and_failures(self):
        from experiments.r31_event_replay import runner
        score = dict(strict=True, delay_ms=3, duplicate=0, raw_duplicate=0, false=[], delayed_false=[], masked=False,
                     tool=dict(detected=True, prealarm=False), stuck=False)
        base = dict(score=score, active_samples=4, first_tool_rise=100)
        self.assertFalse(runner.audit(base, base))
        new = dict(score={**score, 'delay_ms':9, 'duplicate':1, 'raw_duplicate':1},
                   active_samples=10, first_tool_rise=106)
        self.assertEqual(set(runner.audit(new,base)), {'no_new_duplicate','latency_plus_5_ms',
                         'duration_plus_5_samples','tool_time_plus_5_ms'})
        self.assertEqual(replay.SCOPES['full-d4'],180)
