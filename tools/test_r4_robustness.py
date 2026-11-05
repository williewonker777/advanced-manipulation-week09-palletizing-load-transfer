"""Simulation-free contract tests for R4 robustness orchestration."""
from pathlib import Path
import copy
import contextlib
import hashlib
import io
import unittest
import json
import numpy as np
from unittest import mock
from tools import r4_robustness as B

class RobustnessPureTests(unittest.TestCase):
    def test_cli_prints_numpy_boolean_without_repeating_completed_work(self):
        stream = io.StringIO()
        with mock.patch.object(B, 'replay_d4', return_value={'passed': np.bool_(True)}) as run, contextlib.redirect_stdout(stream):
            B.main(['d4'])
        run.assert_called_once()
        self.assertIs(json.loads(stream.getvalue())['passed'], True)
    def test_noise_design_exact_registered_matrix(self):
        rows=B.noise_design(); self.assertEqual(len(rows),9)
        self.assertEqual([r['seed'] for r in rows],[371401]*3+[371402]*3+[371403]*3)
        self.assertEqual([r['mass'] for r in rows],[0.,4.3,8.]*3)
        self.assertTrue(all(r['collision'] is None and r['perturb_family']=='new_noise'
                            and 'kind' not in r for r in rows))
    def test_merge_preserves_inputs_metadata_outliers_and_unique_ids(self):
        old=[{'case_id':i,'stage':'perturb','official':False,'timing':{'outliers':[i]}} for i in range(25)]
        new=[{'case_id':i,'stage':'r4_noise','official':False,'timing':{'outliers':[100+i]}} for i in range(9)]
        a,b=copy.deepcopy(old),copy.deepcopy(new); rows=B._merged_rows(old,new)
        self.assertEqual(old,a);self.assertEqual(new,b);self.assertEqual([r['case_id'] for r in rows],list(range(34)))
        self.assertEqual(rows[25]['source_case_id'],0);self.assertEqual(rows[25]['source_stage'],'r4_noise')
        self.assertEqual(rows[25]['timing']['outliers'],[100]);self.assertTrue(all(r['stage']=='r4_perturb' and r['official'] is False for r in rows))
    def test_first25_excludes_old_new_noise_tail(self):
        fs=['noise']*9+['friction']*4+['stiffness']*4+['corner']*2+['collision']*6
        rows=[{'case_id':i,'perturb_family':f,'collision':[3,[1,0,0]] if f=='collision' else None} for i,f in enumerate(fs)]
        rows += [{'case_id':25+i,'perturb_family':'new_noise','collision':None} for i in range(9)]
        self.assertEqual(len(B._physical_rows({'results':rows})),25)
    def test_bench_firewall_requires_development_review(self):
        with mock.patch.object(B.S,'require_equivalence',return_value={'x':'y'}), mock.patch.object(Path,'is_file',return_value=False):
            with self.assertRaisesRegex(RuntimeError,'development review'): B.require_bench_ready()
    def test_historical_proof_is_synthetic_and_rejects_unlisted_drift(self):
        names=('tools/r3_resources.py','tools/r3_study.py')
        old={name:('old-'+name).encode() for name in names}
        new={name:('new-'+name).encode() for name in names}
        pairs={name:(hashlib.sha256(old[name]).hexdigest(),
                     hashlib.sha256(new[name]).hexdigest()) for name in names}
        diff=b'synthetic reviewed diff'; scope=b'synthetic scope'; s6=b'synthetic s6'
        scope_doc={'architectures':{mode:{'historical_R1_reported_field_parity':True,
                                        'mismatches':[]} for mode in ('servo','torque')}}
        s6_doc={'scientific_evidence':{'source_commit':'synthetic-current',
                    'source_current_HEAD_Git_verified':True,
                    'final_three_files_R1_byte_identical':True}}
        def git_bytes(commit,name):
            return old[name] if commit=='synthetic-old' else new[name]
        def digest(path):
            path=Path(path)
            if path.name=='R3_D4_SCOPE_PARITY.json': return hashlib.sha256(scope).hexdigest()
            if path.name=='R3_S6_VALIDATION.json': return hashlib.sha256(s6).hexdigest()
            return pairs[path.relative_to(B.ROOT).as_posix()][1]
        def read(path):
            return scope_doc if Path(path).name=='R3_D4_SCOPE_PARITY.json' else s6_doc
        patches=(mock.patch.object(B,'HISTORICAL_D4_ORCHESTRATION',pairs),
                 mock.patch.object(B,'HISTORICAL_COMMIT','synthetic-old'),
                 mock.patch.object(B,'CURRENT_R3_COMMIT','synthetic-current'),
                 mock.patch.object(B,'D4_TOOL_DIFF_SHA256',hashlib.sha256(diff).hexdigest()),
                 mock.patch.object(B,'D4_SCOPE_SHA256',hashlib.sha256(scope).hexdigest()),
                 mock.patch.object(B,'R3_S6_SHA256',hashlib.sha256(s6).hexdigest()),
                 mock.patch.object(B,'_git_bytes',side_effect=git_bytes),
                 mock.patch.object(B,'_digest',side_effect=digest),
                 mock.patch.object(B.subprocess,'check_output',return_value=diff),
                 mock.patch.object(B.S,'read',side_effect=read))
        with patches[0],patches[1],patches[2],patches[3],patches[4],patches[5], \
             patches[6],patches[7],patches[8],patches[9]:
            out=B._verify_d4_historical_noninterference()
            self.assertEqual(out['exact_diff_sha256'],hashlib.sha256(diff).hexdigest())
            self.assertIn('instrumentation/guard',out['comparison_reason'])
            doc={'metadata':{'source_sha256':{names[0]:pairs[names[0]][0]}}}
            checked=B._verify_source_hashes(doc,True)
            self.assertEqual(checked[names[0]]['status'],'pinned_historical_orchestration_only')
            with self.assertRaisesRegex(RuntimeError,'differs'):
                B._verify_source_hashes(doc,False)
    def test_cli_firewall_requires_private_paths(self):
        for action in ('perturb-parity','perturb-collect','perturb-replay'):
            with contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit): B.main([action])
if __name__=='__main__':unittest.main()
