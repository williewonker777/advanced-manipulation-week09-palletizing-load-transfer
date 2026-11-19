import unittest
import numpy as np
from experiments.r31_held_identity.candidate import NativeAcquisitions,Source


class PhysicalOwnership(unittest.TestCase):
    def setUp(self):
        self.g=object();self.zero=np.zeros((2,3))
        self.pose=np.array([[2.,0.,0.],[2.,0.,0.]])
        self.s=Source.start(self.g,self.zero,self.pose,1.)
        self.n=NativeAcquisitions()

    def feed(self,routes,levels=None,gen=True,**kw):
        return self.n.feed(routes,kw.get('complete',True),kw.get('grip',True),self.g if gen else object(),
                           self.pose if levels is None else levels,self.s,kw.get('yaw',2.),[1,0])

    def test_only_known_directional_source_is_owned(self):
        self.assertTrue(self.feed({'yaw'}));self.assertTrue(self.feed({'yaw','normal','r8_base'}))
        self.assertFalse(self.feed({'yaw','light'}))
        self.assertFalse(self.feed({'light'}))
        self.assertFalse(self.feed(set()))
        self.assertFalse(self.feed({'directional'}))

    def test_opposing_yaw_or_new_generation_remains_reported(self):
        self.assertFalse(self.feed({'yaw'},yaw=-1.));self.assertFalse(self.feed({'yaw'}))
        self.assertFalse(self.feed(set()));self.assertTrue(self.feed({'yaw'}))
        self.assertFalse(self.feed({'yaw'},gen=False))

    def test_second_force_in_new_plane_even_if_raw_route_stays_live(self):
        self.assertTrue(self.feed({'normal','r8_base'}))
        self.assertFalse(self.feed({'normal','r8_base'},levels=np.array([[0,0,4],[0,0,4]])))
        self.assertFalse(self.feed({'normal','r8_base'}))  # do not retroactively hide it

    def test_no_supply_for_missing_or_disagreeing_witness(self):
        self.assertIsNone(Source.start(None,self.zero,self.pose,1.))
        self.assertIsNone(Source.start(self.g,self.zero,self.zero,1.))
        self.assertFalse(self.s.agrees(self.g,self.pose,{'yaw'},-2.,[1,0]))
        self.assertFalse(self.s.agrees(self.g,self.pose,{'pickup_hold'},2.,[1,0]))
        self.assertFalse(self.feed({'normal'},levels=[[2,0,0],[-2,0,0]]))

    def test_no_fixed_deadtime_or_linger_on_ambiguous_route(self):
        self.assertTrue(self.feed({'normal'}));self.assertFalse(self.feed({'normal'},complete=False))
        self.assertFalse(self.feed({'normal'}));self.assertFalse(self.feed(set()))
        self.assertTrue(self.feed({'normal'}))
