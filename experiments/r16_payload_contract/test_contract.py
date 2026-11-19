import unittest
import numpy as np

import my_detector
import r1_torque
from sim.nominal import Nominal
from experiments.r16_payload_contract import controller as policy


class Tests(unittest.TestCase):
    def init_ctrl(self, kind, mass, com=(0.,0.,0.)):
        c=kind({},{});c.box_mass,c.box_com=mass,com
        c.nom,c.nom_box=Nominal(),Nominal();c.nom_box.set_payload(mass,com)
        c.reset();return c

    def test_no_empty_standin_for_invalid_mass_or_com(self):
        for mass,com in ((float('nan'),(0,0,0)),(-.001,(0,0,0)),(0.,[0.]),(3.,[0.,np.inf,0.])):
            with self.assertRaises(ValueError):
                policy.has_payload(mass,com)
        self.assertFalse(policy.has_payload(0.,[0.,0.,0.]))
        self.assertTrue(policy.has_payload(.01,[0.,0.,0.]))

    def test_exact_default_for_empty_without_any_research_calibration(self):
        for cls in (policy.Controller,policy.Baseline):
            c=self.init_ctrl(cls,0.)
            self.assertIs(type(c._active),r1_torque.Controller)
            self.assertIs(c.compute.__func__,r1_torque.Controller.compute)
            d=policy.Detector();policy.attach(c,d);d.ctrl=c;d.reset()
            self.assertIs(type(d._active),my_detector.Detector)
            self.assertIs(d._active.ctrl,c._active)
            self.assertIs(d.update.__func__,my_detector.Detector.update)

    def test_loaded_retains_original_independent_policies(self):
        for cls,impl in ((policy.Controller,policy.Balanced),(policy.Baseline,policy.Early)):
            c=self.init_ctrl(cls,6.)
            self.assertIs(type(c._active),impl)
            d=policy.Detector();policy.attach(c,d);d.ctrl=c;d.reset()
            self.assertIs(type(d._active),policy.Coupled)
            self.assertIs(d._active.ctrl,c._active)

if __name__=='__main__':unittest.main()
