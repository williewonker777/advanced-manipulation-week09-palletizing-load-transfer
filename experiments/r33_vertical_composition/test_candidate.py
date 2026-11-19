from types import SimpleNamespace
import unittest
import numpy as np
from .candidate import AxialCarry,FirstLift,Travel
from experiments.r33_pick_vertical.state import expected

class CompositionTest(unittest.TestCase):
    def test_return_to_band_does_not_reopen_pickup(self):
        r=FirstLift()
        for h in (.012,.027,.036,.045,.041,.034):
            center=expected(h,r.medians)
            self.assertFalse(r.classify(h,np.r_[0.,0.,center[0]],center[1],False,18.,4))
        self.assertTrue(r.over)
        r.reset_lift()
        self.assertFalse(r.over or r.departed)
    def test_carry_evidence_requires_two_live_vector_estimates(self):
        r=AxialCarry()
        frame=SimpleNamespace(r27_projection=SimpleNamespace(last_force=None),_shared=SimpleNamespace(last_table_allowed=False))
        self.assertFalse(r.feed(frame,True,False))
    def test_mixed_direction_and_native_load_leave_no_extra_notice(self):
        def test(vector,native):
            packet=SimpleNamespace(last_table_allowed=False,last_residual_wrench=np.zeros(6),force_threshold=np.array([20,20,18.4]),out_of_height_fz_threshold=27.,persistence=4)
            d=SimpleNamespace(r27_projection=SimpleNamespace(last_force=np.zeros(3)),_shared=packet,r27_gate=SimpleNamespace(windows=SimpleNamespace(windows=[SimpleNamespace(present=True)])))
            route=AxialCarry();out=[]
            for k in range(215):
                v=np.array(vector,float)*(k>=90)
                d.r27_projection.last_force=v;packet.last_residual_wrench[:3]=v
                out.append(route.feed(d,True,native and k>92))
            return sum(out)
        self.assertEqual(test((6,6,6),False),0)
        self.assertEqual(test((0,0,10),True),0)
        self.assertEqual(test((0,0,35),False),0)
        self.assertEqual(test((0,0,8),False),4)
    def test_travel_must_complete_observable_lift_and_departure(self):
        ramp=[.002*k for k in range(31)]+[.060]*15
        tr=Travel();q=np.zeros(6)
        xyz=np.array([.6,0.,.1])
        d=SimpleNamespace(nom=SimpleNamespace(tcp_position=lambda q:xyz.copy()),_shared=SimpleNamespace(q_prev=q,height_band=.035))
        self.assertFalse(tr.feed(d,True,False))
        for h in ramp:
            xyz[2]=.1+h
            self.assertFalse(tr.feed(d,True,False))
        q[0]=.02;self.assertFalse(tr.feed(d,True,True))
        self.assertTrue(tr.feed(d,True,False))
        self.assertFalse(tr.feed(d,False,False))
