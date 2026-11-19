import copy
import unittest

import numpy as np
from experiments.r18_paired_impulses import runner
from experiments.r19_close_pairs import assay


CASE = {'id':'test','impacts':[[4.,[0,0,20]],[4.16,[28,0,0]]]}


def twins():
    t=np.arange(6000)*.001
    ret={}
    for tw in assay.TWINS:
        one, two=CASE['impacts']
        a=runner.expected_signed(t,[one if tw in ('first','both') else None])
        b=runner.expected_signed(t,[two if tw in ('second','both') else None])
        det=np.zeros(len(t),bool)
        if tw in ('first','both'):det[4021:4090]=True
        if tw in ('second','both'):det[4191:4251]=True
        vals={n:np.zeros((6000,6)) for n in ('q','qpos','q_d','tau','r')}
        ret[tw]=dict(vals,t=t,e_mm=t*0.,f_table=t*0.,f_tool=t*0.,detect=det,
                     f_ext=np.array([np.linalg.norm(row) for row in a]),f_ext_xyz_native=a,
                     f_ext_xyz_second=b,f_ext_xyz_committed=a+b,f_ext_combined_norm=np.linalg.norm(a+b,axis=1),
                     r18_diverged=False)
    return ret


def score(logs):
    return assay.characterize(CASE,logs,20,320)[0]


class SeparationTest(unittest.TestCase):
    def test_edge_runs_include_terminal(self):
        self.assertEqual(assay.intervals(np.array([1,1,0,1,0,1],bool)),[[0,2],[3,4],[5,6]])
        self.assertEqual(assay.intervals(np.zeros(1,bool)),[])
        with self.assertRaises(ValueError):assay.intervals([1,0])

    def test_ablation_and_independence(self):
        logs=twins(); a=score(logs)
        self.assertTrue(assay.one_per_pulse(a))
        self.assertTrue(all(a['r18_gates'].values()))
        self.assertEqual(a['slots']['both_second']['credited_delays_ms'],[31])
        self.assertIn(a['actual_quiescent_force_samples'],(40,41))
        logs['second']['detect'][:]=False
        self.assertFalse(assay.one_per_pulse(score(logs)))

    def test_high_across_second_with_late_rerise_is_not_two_hits(self):
        tr=twins()
        for k in ('first','both'):tr[k]['detect'][4021:4168]=True
        # The background twin is quiet around 4191 but not at the second onset.
        a=score(tr)
        self.assertEqual(a['slots']['both_second']['attributable_delays_ms'],[31])
        self.assertEqual(a['slots']['both_second']['credited_delays_ms'],[])
        self.assertFalse(assay.one_per_pulse(a))

    def test_background_retrigger_at_both_times_not_credited(self):
        tr=twins();tr['first']['detect'][4192:4194]=True
        a=score(tr)
        self.assertEqual(a['slots']['both_second']['raw_delays_ms'],[31])
        self.assertEqual(a['slots']['both_second']['attributable_delays_ms'],[])

    def test_known_tail_and_inpulse_duplication_are_separate_failures(self):
        tr=twins(); parent=score(tr)
        tr['both']['detect'][4211:4217]=False
        trial=score(tr)
        self.assertFalse(assay.one_per_pulse(trial))
        self.assertTrue(assay.all_comparisons([assay.compare(trial,parent)]))
        latched=copy.deepcopy(parent)
        latched['levels']['first']['high_at_isolated_window_end']=True
        self.assertFalse(assay.all_comparisons([assay.compare(latched,parent)]))

    def test_force_gap_checks_actual_flange(self):
        tr=twins()['both'];tr['f_ext_xyz_committed'][4130,0]=.2
        with self.assertRaises(ValueError):assay.off_samples(tr)
        tr['f_ext_xyz_second'][4050,1]=2.
        with self.assertRaises(ValueError):assay.off_samples(tr)
