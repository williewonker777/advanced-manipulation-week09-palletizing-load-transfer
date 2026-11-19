"""Check provenance and refusal to publish an incomplete or promoted R18 replay."""
import copy
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from tools import r18_recovery_audit as audit


class AuditTests(unittest.TestCase):
    def test_original_reference_code_and_normal_model_are_pinned(self):
        blobs,_=audit.prior.snapshots()
        for name in audit.ADDITIONS:blobs[name]=(audit.ROOT/name).read_bytes()
        audit.verify_source(blobs)
        for name in ('experiments/r18_single_pass/reference_recovery.py',audit.prior.MODEL):
            broken=dict(blobs);broken[name]+=b' '
            with self.assertRaises(ValueError):audit.verify_source(broken)

    def test_paired_scorer_is_authored_and_identical_to_the_frozen_assessor(self):
        rel='experiments/r18_paired_impulses/scoring.py'
        self.assertIn(rel,audit.PAIR_FILES)
        self.assertEqual(audit.sha((audit.ROOT/rel).read_bytes()),'9b8a9a1f1e11418d4d412d63c8e23e8d95db3bdd70e3d8bef6c2b7d5aadc4068')
        self.assertFalse(any(Path(x).name in ('run.py','evaluate.py') for x in audit.PAIR_FILES))

    def test_incomplete_negative_or_promoted_result_fails(self):
        names=('genuine_reference_signature','all_prefixes_and_realized_forces','all_D_physics_identical',
               'fast_exact_residual_alarm_and_state','preserve_tool')
        base={'schema':'r18-known-sentinel-live.v1','reproduced':True,'promotion_eligible':False,'gates':dict.fromkeys(names,True)}
        self.assertTrue(audit.valid_report(base))
        for patch in ({'promotion_eligible':True},{'gates':{}},{'reproduced':False}):
            self.assertFalse(audit.valid_report({**base,**patch}))
        bad=copy.deepcopy(base);bad['gates'][names[0]]=False;self.assertFalse(audit.valid_report(bad))

    def test_outside_cwd_help(self):
        with tempfile.TemporaryDirectory() as folder:
            p=subprocess.run([sys.executable,'-I','-B',str(Path(audit.__file__).resolve()),'--help'],
                             cwd=folder,text=True,capture_output=True)
            self.assertEqual(p.returncode,0,p.stderr);self.assertIn('--height-calibration',p.stdout)


class ProgressiveReports(unittest.TestCase):
    def test_legitimate_failed_gate_does_not_unlock_next_stage(self):
        with tempfile.TemporaryDirectory() as temp:
            out=Path(temp);n=37
            rows=[{'id':str(i),'gates':{'original':True},'extra_gates':{'time':i!=4},'passed':i!=4}
                  for i in range(n)]
            doc={'stage':'known-normal','rows':rows,'extra_evidence':[],'passed':False}
            stage={'stage':'known-normal','count':n,'passed':False,'passed_count':n-1}
            summary={'schema':'r18-integrated-outcome.v1','qualified_for_deployment':False,'stages':[stage]}
            (out/'known-normal.json').write_text(json.dumps(doc));(out/'summary.json').write_text(json.dumps(summary))
            self.assertFalse(audit.load_progress(out)['all_stages_passed'])
            stage['passed']=True
            (out/'summary.json').write_text(json.dumps(summary))
            with self.assertRaises(ValueError):audit.load_progress(out)
            stage['passed']=False
            rows[4]['id']=rows[3]['id'];(out/'known-normal.json').write_text(json.dumps(doc))
            (out/'summary.json').write_text(json.dumps(summary))
            with self.assertRaises(ValueError):audit.load_progress(out)
