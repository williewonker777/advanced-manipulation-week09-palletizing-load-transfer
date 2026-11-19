import unittest
import numpy as np
from experiments.r31_held_identity.candidate import Source
from experiments.r32_measured_level.candidate import Excursion


class MeasuredContinuation(unittest.TestCase):
    def setUp(self):
        self.g=object();self.z=np.zeros((2,3));self.p=np.array([[4.,0.,0.],[3.,0.,0.]])
        self.t=Source.start(self.g,self.z,self.p,0.);self.s=Excursion();self.s.begin(self.t)

    def test_holds_while_load_persists_then_cannot_reopen(self):
        for _ in range(150):self.assertTrue(self.s.feed(True,True,self.g,2*self.p,2*self.p,False))
        self.assertFalse(self.s.feed(True,True,self.g,self.p*.5,2*self.p,False))
        self.assertFalse(self.s.feed(True,True,self.g,2*self.p,2*self.p,True))

    def test_unrelated_direction_release_and_measurement_incompleteness(self):
        t=np.array([[0,0,9],[0,0,11]])
        self.assertFalse(self.s.feed(True,True,self.g,t,self.p,True))
        for grip,ok,gen,instant in ((False,True,self.g,self.p),(True,False,self.g,self.p),
                                     (True,True,object(),self.p),(True,True,self.g,None)):
            self.s.begin(self.t)
            self.assertFalse(self.s.feed(grip,ok,gen,instant,self.p,False))

    def test_lone_fast_sensor_is_not_joint_discharge(self):
        level=self.p.copy();fast=self.p.copy();fast[0]=0
        self.assertTrue(self.s.feed(True,True,self.g,fast,level,False))

    def test_native_claim_does_not_override_incompatible_spatial_data(self):
        self.assertFalse(self.s.feed(True,True,object(),self.p,-self.p,True))
