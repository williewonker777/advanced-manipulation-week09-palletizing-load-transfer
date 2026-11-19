"""Public contract tests for the stock R15 payload audit."""
import contextlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from tools import r15_payload_audit as audit


class PayloadAuditTest(unittest.TestCase):
    def setUp(self):
        self.r14=json.loads((audit.ROOT/'experiments/r14_airborne/plan.json').read_text())
        self.r15=json.loads((audit.ROOT/'experiments/r15_com_model/plan.json').read_text())

    def test_frozen_plan_shape_and_payload_validation(self):
        audit.validate_plans(self.r14,self.r15)
        bad=json.loads(json.dumps(self.r15));bad['variants'][1]='zero'
        with self.assertRaises(ValueError):audit.validate_plans(self.r14,bad)
        bad=json.loads(json.dumps(self.r14));bad['fit'][0]['mass']=float('nan')
        with self.assertRaises(ValueError):audit.validate_plans(bad,self.r15)

    def test_authored_offsets_distinguish_none_and_contact_enabled_zero(self):
        case=self.r15['test'][0];x,y,z=case['shift']
        self.assertIsNone(audit.offset(case,'none'))
        self.assertEqual(audit.offset(case,'explicit_zero'),(0.,0.,0.))
        self.assertEqual(audit.offset(case,'xy'),(x,y,0.))
        self.assertEqual(audit.offset(case,'up'),(x,y,z))
        self.assertEqual(audit.offset(case,'down'),(x,y,-z))

    def test_metrics_count_abstentions_errors_and_direction(self):
        rows=[]
        for _ in range(8):
            rows.append({'true_xy_mm':[0.,0.],'point_error_mm':[.1,0.],
                         'action_error_mm':[0.,0.],'accept':False,'covered':True,
                         'model_action_xy_mm':[0.,0.]})
        for vector in ([3.,4.],[-3.,4.],[4.,-3.])*4:
            rows.append({'true_xy_mm':list(vector),'point_error_mm':[.1,-.1],
                         'action_error_mm':[.1,-.1],'accept':True,'covered':True,
                         'model_action_xy_mm':list(vector)})
        result=audit.measures(rows)
        self.assertTrue(result['qualified'])
        self.assertEqual(result['accepted'],12)
        self.assertEqual(result['abstained'],8)
        self.assertEqual(result['big_signed'],12)
        rows[-1]['model_action_xy_mm']=[-4.,3.]
        self.assertFalse(audit.measures(rows)['qualified'])

    def test_physics_digest_covers_all_retained_stock_signals(self):
        log={key:np.arange(6,dtype=float) for key in
             ('q','tau','f_tool','f_table','r','detect')}
        first=audit.physics_digest(log)
        self.assertEqual(set(first),set(log))
        log['q']=log['q'].copy();log['q'][0]=99.
        second=audit.physics_digest(log)
        self.assertNotEqual(first['q'],second['q'])
        self.assertEqual(first['tau'],second['tau'])

    def test_public_source_has_no_private_fixture_or_raw_output_contract(self):
        source=Path(audit.__file__).read_text()
        self.assertNotIn('r5_fixture',source)
        self.assertNotIn('savez',source)
        self.assertIn('predictions_sealed_before_truth',source)
        self.assertIn('score_capsule',source)

    def test_source_snapshot_is_immutable_after_working_file_mutation(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            for key,relative in audit.SOURCE_PATHS.items():
                path=root/relative;path.parent.mkdir(parents=True,exist_ok=True)
                path.write_bytes((key+'-original').encode())
            with patch.object(audit,'ROOT',root):
                blobs,pins=audit.source_inputs()
            target=root/audit.SOURCE_PATHS['sensing']
            target.write_bytes(b'mutated-after-snapshot')
            self.assertEqual(blobs['sensing'],b'sensing-original')
            self.assertEqual(pins['sensing'],audit.sha(b'sensing-original'))
            self.assertNotEqual(pins['sensing'],audit.sha(target.read_bytes()))

    def test_help_runs_from_unrelated_cwd_and_paths_fail_closed(self):
        with tempfile.TemporaryDirectory() as folder:
            result=subprocess.run([sys.executable,'-B',str(Path(audit.__file__).resolve()),'--help'],
                                  cwd=folder,env={**os.environ,'PYTHONDONTWRITEBYTECODE':'1'},
                                  text=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
            self.assertEqual(result.returncode,0,result.stderr)
            self.assertIn('--kit',result.stdout)
            existing=Path(folder)/'existing';existing.mkdir()
            argv=['r15_payload_audit.py','--kit',folder,'--out',str(existing)]
            with patch.object(sys,'argv',argv),contextlib.redirect_stderr(io.StringIO()), \
                    self.assertRaises(SystemExit):
                audit.main()


if __name__=='__main__':unittest.main()
