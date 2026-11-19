import unittest
from unittest.mock import patch

from experiments.r20_official import policy


class Stub:
    def reset(self):
        self.update = lambda *args: (None, False)


class DispatchTest(unittest.TestCase):
    def wrapper(self, mass, registered=True, active=True):
        c = policy.Controller.__new__(policy.Controller) if registered else object()
        if registered:
            c._active = object() if active else None
            c.loaded_policy = 2.86 <= mass <= 8.
        d = policy.Detector()
        d.ctrl, d.box_mass, d.box_com = c, mass, (0, 0, 0)
        d.nom, d.nom_box = object(), object()
        return d

    def test_only_registered_supported_torque_gets_impulse(self):
        for mass, registered, uses_impulse in ((2.85, True, False), (2.86, True, True),
                                               (8., True, True), (8.1, True, False),
                                               (8., False, False), (0., True, False)):
            with self.subTest(mass=mass, registered=registered):
                d = self.wrapper(mass, registered)
                fallback, impulse = Stub(), Stub()
                with patch.object(policy.my_detector, 'Detector', return_value=fallback),                      patch.object(policy, 'Impulse', return_value=impulse):
                    d.reset()
                chosen = impulse if uses_impulse else fallback
                self.assertIs(d._active, chosen)
                self.assertIs(chosen.ctrl, d.ctrl._active if registered else d.ctrl)
                self.assertIs(chosen.nom_box, d.nom_box)
                self.assertIs(d.update, chosen.update)

    def test_unset_or_inconsistent_actual_controller_cannot_lend_calibration(self):
        with self.assertRaises(ValueError):
            self.wrapper(5., active=False).reset()
        d = self.wrapper(5.)
        d.ctrl.loaded_policy = False
        with self.assertRaises(ValueError):
            d.reset()


if __name__ == '__main__':
    unittest.main()
