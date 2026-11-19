import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from tools import r12_handoff_check as check, r12_handoff_demo as demo
from tools.r11_placement import score

ROOT=Path(__file__).resolve().parents[1]


def event(outcome='reference',delay=None,alarms=()):
    return dict(outcome=outcome,delay_ms=delay,false_alarms=len(alarms),false_alarm_times=list(alarms),
                tool_detected=False,tool_present=False,tool_prealarm=False,prealarm=False)


def normal(mean,ev):
    return dict(status='ok',support={'mean_mg':mean,'min_ge_025':True,'mean_ge_050':mean>=.5,
                                    'peak_le_110':True},sat_pct=0.,event=ev,cost={'rms':1.,'peak':1.})


class HandoffAssessment(unittest.TestCase):
    def test_sustained_support_not_peak_or_contact(self):
        a,b=normal(.10,event()),normal(.54,event())
        self.assertTrue(check.normal_checks(a,b,score)['pass'])
        b['support']['min_ge_025']=False
        self.assertFalse(check.normal_checks(a,b,score)['pass'])

    def test_cost_false_alarm_and_tool_gate_independent_of_support(self):
        a,b=normal(.14,event()),normal(.8,event(alarms=[3.903]))
        b['cost']['peak']=1.051
        a['event'].update(tool_present=True,tool_detected=True)
        gates=check.normal_checks(a,b,score)['gates']
        self.assertTrue(gates['g4']);self.assertFalse(gates['cost'] or gates['alarms'] or gates['tool'])

    def test_candidate_cannot_hide_unobserved_or_diverged_input(self):
        b=normal(.9,event())
        for status in ('diverged','placing footprint unverified'):
            self.assertIsNone(check.normal_checks({'status':status},b,score)['g4'])
            self.assertFalse(check.impact_checks({'status':status},b)['pass'])

    def test_nonzero_cost_from_zero_never_becomes_a_pass(self):
        a,b=normal(.1,event()),normal(.7,event())
        a['cost']['rms']=0.
        self.assertFalse(check.normal_checks(a,b,score)['gates']['cost'])
        b['cost']['rms']=0.
        self.assertTrue(check.normal_checks(a,b,score)['gates']['cost'])

    def test_same_number_of_alarms_is_not_the_same_normal_motion(self):
        self.assertTrue(check.no_new_false_events(event(alarms=[1.,2.]),event(alarms=[2.019])))
        self.assertFalse(check.no_new_false_events(event(alarms=[1.,2.]),event(alarms=[1.001,1.002])))
        self.assertFalse(check.no_new_false_events(event(alarms=[1.,2.]),event(alarms=[2.3])))

    def test_indistinguishable_is_not_a_new_detection_and_delay_is_bounded(self):
        a={'event':event('detected',40.),'normal_prefix_equal':True,'status':'ok'}
        b={'event':event('indistinguishable',15.),'normal_prefix_equal':True,'status':'ok'}
        self.assertFalse(check.impact_checks(a,b)['pass'])
        b['event']=event('detected',45.)
        self.assertTrue(check.impact_checks(a,b)['pass'])
        b['event']=event('detected',46.)
        self.assertFalse(check.impact_checks(a,b)['pass'])
        a['event']=event('missed')
        self.assertTrue(check.impact_checks(a,b)['pass'])
        b['normal_prefix_equal']=False
        self.assertFalse(check.impact_checks(a,b)['pass'])

    def test_builder_keeps_default_files_immutable_and_requires_new_path(self):
        with tempfile.TemporaryDirectory() as d:
            target=Path(d)/'part';demo.write_demo(target)
            a=demo.assemble()
            self.assertEqual(sorted(a),sorted(p.name for p in target.iterdir()))
            self.assertEqual(a['my_servo.py'],(ROOT/'r1_servo.py').read_text())
            self.assertEqual(a['my_detector.py'],(ROOT/'experiments/r10_scratch/CANDIDATE.py').read_text())
            self.assertNotIn('import r1_torque',a['my_torque.py'])
            with self.assertRaises(ValueError):demo.write_demo(target)
            link=Path(d)/'symlink';link.symlink_to(Path(d)/'absent')
            with self.assertRaises(ValueError):demo.write_demo(link)
            original=Path.read_bytes
            def altered(p):
                return original(p)+b'\n' if p.name=='r1_torque.py' else original(p)
            with patch.object(Path,'read_bytes',altered), self.assertRaisesRegex(ValueError,'source differs'):
                demo.write_demo(Path(d)/'bad')
            self.assertFalse((Path(d)/'bad').exists())

    def test_outcomes_retain_failures_and_differential_false_alarm_reduction(self):
        data=json.loads((ROOT/'results/r12/paired_outcomes.json').read_text())
        rows=data['physics']
        self.assertEqual(sum(r['g4']['all_four_g4'] for r in rows if r['kind']=='normal'),3)
        self.assertEqual(sum(r['g4']['all_four_g4'] for r in rows if r['kind']=='offset'),1)
        self.assertGreater(max(r['support']['counter']['peak_mg'] for r in rows),2.7)
        self.assertEqual([sum(r['false_alarms'][k] for r in rows) for k in ('R1','hybrid','counter')],[0,4,0])
        self.assertEqual([r['counter_outcome'] for r in data['impacts']],['detected']*3)
