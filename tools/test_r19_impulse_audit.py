import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from tools import r19_impulse_audit as impl


def synthetic(stage_flags):
    old=['d'+str(i) for i in range(12)]
    plan={'new':[{'id':f'u{i}'} for i in range(6)]}
    result={'schema':'r19-impulse-gate-result.v1','stages':{},'episodes':0,'novel_forces_opened':0,
            'promotion_eligible':False,'aggregate_success':False,'published_reference_sha256':impl.STOCK_SHA}
    for name,passed in stage_flags:
        rows=[]
        for i,cid in enumerate(old if name=='known' else [r['id'] for r in plan['new']]):
            ok=i!=0 or passed;n={'known':8,'normal':5,'novel':20}[name]
            row={'id':cid,'episodes':n,'passed':ok,'gates':{'criterion':ok}}
            if name=='known':
                row['gates'].update(published_R18_score_exact=True,published_R18_character_exact=True)
            if name=='novel':
                row['arms']={'candidate':{'r18_gates':{
                    'no_added_unmatched_rise':True,'no_programmed_normal_event_overlap':i!=5}}}
            rows.append(row);result['episodes']+=n
        result['stages'][name]={'rows':rows,'passed':passed}
    if 'novel' in result['stages']:
        result['novel_forces_opened']=6;result['aggregate_success']=result['stages']['novel']['passed']
    return result,old,plan


class IndependentAuditTest(unittest.TestCase):
    def test_pinned_algorithms_and_cpu_neutral_scoring(self):
        for n,pin in impl.PIN_ALGORITHM.items():
            self.assertEqual(impl.setup.sha((impl.ROOT/n).read_bytes()),pin)
        src=(impl.ROOT/impl.MODULE).read_text()
        self.assertNotIn('sched_setaffinity',src)
        self.assertEqual(json.loads((impl.ROOT/impl.PLAN).read_bytes())['published_reference']['sha256'],impl.STOCK_SHA)

    def test_full_original_twin_gates_still_required(self):
        d,old,p=synthetic([('known',True),('normal',True),('novel',True)])
        cross=impl.verify(d,old,p)
        self.assertEqual([False]+[True]*5,list(reversed([row['passed'] for row in cross])))
        self.assertEqual(cross[-1]['failed_original_gates'],['no_programmed_normal_event_overlap'])
        for edit in (lambda x:x.update(episodes=245),lambda x:x['stages']['normal']['rows'].pop(),
                     lambda x:x['stages']['known']['rows'][1]['gates'].update(published_R18_score_exact=False)):
            bad=copy.deepcopy(d);edit(bad)
            with self.assertRaises(ValueError):impl.verify(bad,old,p)

    def test_legitimate_early_stop_is_distinct_from_missing_results(self):
        for flag in (True,False):
            args=synthetic([('known',flag)])
            if flag:
                with self.assertRaises(ValueError):impl.verify(*args)
            else:self.assertEqual(impl.verify(*args),[])
        with self.assertRaises(ValueError):impl.verify(*synthetic([('known',False),('normal',True)]))

    def test_small_or_altered_v2_report_rejected_before_simulation(self):
        with tempfile.TemporaryDirectory() as t:
            base=Path(t);c=base/'kit';c.mkdir();(c/'sim').mkdir();(c/'model').mkdir();(c/'evaluate.py').touch()
            h=base/'cal.json';h.write_text('{}');f=base/'v2.json';f.write_text('{}');out=base/'out.json'
            with mock.patch.object(impl.setup,'snapshots') as lookup,self.assertRaises(SystemExit):
                impl.main(['--kit',str(c),'--height-calibration',str(h),'--v2-audit',str(f),'--out',str(out)])
            lookup.assert_not_called();self.assertFalse(out.exists())
