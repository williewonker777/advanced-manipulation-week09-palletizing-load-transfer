import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from tools import r19_event_audit as audit


def fixture(stages):
    known=[{'id':'old'+str(i)} for i in range(6)]
    plan={'new':[{'id':'new'+str(i)} for i in range(6)]}
    previous={'fresh_paired_cases':known[:4],'diagnostic_sequences':known[4:]}
    doc={'schema':'r19-close-separation.v1','stages':{},'episodes':0,'plan_sha256':audit.FROZEN_PLAN,
         'new_forces_opened':0,'promotion_eligible':False,'all_stages_passed':False}
    for name,passed in stages:
        rows=[]
        for i,c in enumerate(known if name=='known' else plan['new']):
            ok=passed or i!=0
            normal={'gates':{'example':ok},'events':dict.fromkeys(audit.ARMS)}
            o={'id':c['id'],'episodes':5 if name=='normal' else 20,'normal':normal,'passed':ok}
            if name!='normal':
                arm={'levels':dict.fromkeys(('none','first','second','both')),
                     'slots':dict.fromkeys(('isolated_first','isolated_second','both_first','both_second'))}
                o.update(arms=dict.fromkeys(audit.ARMS,arm),assessment_signatures=dict.fromkeys(audit.ARMS),
                         gates={'example':ok,'sealed_reference_all_arms':True})
            doc['episodes']+=o['episodes'];rows.append(o)
        doc['stages'][name]={'rows':rows,'passed':passed}
    if 'new' in doc['stages']:
        doc['new_forces_opened']=6
        doc['all_stages_passed']=doc['stages']['new']['passed']
    return doc,plan,previous


class PublicEventAuditTest(unittest.TestCase):
    def test_plan_is_the_preexposure_plan_and_uses_disjoint_grid(self):
        plan=audit.ROOT/'experiments/r19_close_pairs/plan.json'
        self.assertEqual(audit.sha(plan.read_bytes()),audit.FROZEN_PLAN)
        cases=json.loads(plan.read_text())['new']
        self.assertEqual([r['gap_ms'] for r in cases],[12,24,40,80,160,24])
        self.assertEqual(len({r['mass'] for r in cases}),6)

    def test_scientific_first_gate_failure_is_valid_but_does_not_open_fresh(self):
        d,p,r=fixture([('known',False)])
        self.assertIs(audit.checked_report(d,p,r),d)
        d['new_forces_opened']=6
        with self.assertRaises(ValueError):audit.checked_report(d,p,r)

    def test_guarded_stages_and_per_case_evidence(self):
        for stages in ([('known',True)], [('known',False),('normal',True)], [('new',False)]):
            with self.subTest(stages=stages),self.assertRaises(ValueError):
                audit.checked_report(*fixture(stages))
        a,p,r=fixture([('known',True),('normal',True),('new',False)])
        self.assertIs(audit.checked_report(a,p,r),a)
        for change in (lambda t:t['stages']['new']['rows'].pop(),
                       lambda t:t['stages']['known']['rows'][0]['gates'].update(sealed_reference_all_arms=False),
                       lambda t:t['stages']['new']['rows'][0]['arms']['event']['levels'].pop('first'),
                       lambda t:t['stages']['normal']['rows'][0].update(passed=1),
                       lambda t:t.update(episodes=269)):
            d=copy.deepcopy(a);change(d)
            with self.assertRaises(ValueError):audit.checked_report(d,p,r)

    def test_snapshot_refuses_changed_physical_event_source_paths(self):
        blobs,pins=audit.snapshots()
        self.assertNotIn('sim/runner.py',blobs)
        self.assertTrue(all(Path(n).name not in ('run.py','evaluate.py') for n in blobs))
        self.assertEqual(pins['experiments/r19_close_pairs/plan.json'],audit.FROZEN_PLAN)
        with tempfile.TemporaryDirectory() as folder, mock.patch.object(audit,'ROOT',Path(folder)):
            with self.assertRaises(ValueError):audit.snapshots()
