"""Independent boundary checks for the late, horizontal-only research candidate."""
import unittest
from unittest.mock import Mock, patch

import numpy as np

from experiments.r14_online.stream import Stream
from experiments.r15_com_model import controller, sensing


class HorizontalDecisionTests(unittest.TestCase):
    def decision(self, prior, *, guard=0., mass=5.):
        # Zero information about any additional moment: every fit, including
        # each deletion, must keep the encoder-derived prior and subtract the
        # *normal* mass-specific horizontal COM, not the geometric box centre.
        system = (np.zeros((5, 3, 3)), np.zeros((5, 3)), np.ones(3), None)
        model = {
            'schema': 'r15-xy-calibration.v1',
            'dynamics': {'center': 5., 'range': [3., 8.]},
            'normal_guard_mm': guard,
            'baseline_xy': [[1., -2.], [.5, -.25]],
        }
        features = {'prior_com_mm': prior, 'mass': mass}
        with patch.object(sensing, '_system', return_value=system):
            return sensing.decide(features, model)

    def test_signed_flange_axes_and_normal_com_at_supplied_mass(self):
        row = self.decision([7.5, -10.25, 14.], mass=6.)
        np.testing.assert_allclose(row['relative_xy_mm'], [6., -8.])
        np.testing.assert_allclose(row['model_action_xy_mm'], [6., -8.])
        self.assertTrue(row['accept'])

    def test_inconclusive_vertical_or_out_of_domain_has_zero_action(self):
        for row in (self.decision([1., -2., 18.]),
                    self.decision([1.75, -2., 0.]),
                    self.decision([7., -10., 0.], guard=2.),
                    self.decision([7., -10., 0.], mass=8.01)):
            self.assertFalse(row['accept'])
            self.assertEqual(row['model_action_xy_mm'], [0., 0.])

    def test_cap_preserves_signed_direction(self):
        row = self.decision([13., -18., 0.])
        self.assertTrue(row['accept'])
        np.testing.assert_allclose(row['model_action_xy_mm'], [9., -12.])

    def test_invalid_calibration_guard_raises(self):
        for guard in (float('nan'), -1.):
            with self.assertRaises(ValueError):
                self.decision([1., -2., 0.], guard=guard)


class CausalControllerTests(unittest.TestCase):
    def make_controller(self, row):
        c = controller.Controller.__new__(controller.Controller)
        c.xy_decided, c.xy_result, c.xy_updates = False, None, 0
        c.xy_decided_at_s, c._last_weight = None, 0.
        c.xy_offset_m = np.zeros(3)
        c._provided_com = np.array([.02, -.01, .04])
        c.box_mass, c.xy_calibration = 5., {}
        c.xy_decide_fn, c.registration_stream = Mock(return_value=row), Mock()
        c.nom_box = Mock()
        return c

    def test_stream_contract_decide_once_and_blend_for_next_sample(self):
        incomplete = Stream.__new__(Stream)
        incomplete.count = 3549
        with self.assertRaisesRegex(ValueError, '3.55'):
            incomplete.finish()
        self.assertEqual(round(controller.SENSE / .001) + 1, 3550)
        c = self.make_controller({'accept': True, 'model_action_xy_mm': [-3., 4.]})
        sentinel = object()
        with patch.object(controller.HeightController, 'compute', return_value=sentinel):
            for t in (3.527, 3.548):
                self.assertIs(c.compute(t, *([None] * 7)), sentinel)
                c.registration_stream.finish.assert_not_called()
            c.compute(3.549, *([None] * 7))
            c.nom_box.set_payload.assert_not_called()
            c.compute(3.550, *([None] * 7))
            c.compute(3.710, *([None] * 7))
            c.compute(3.800, *([None] * 7))
        c.registration_stream.finish.assert_called_once()
        c.xy_decide_fn.assert_called_once()
        self.assertEqual(c.xy_decided_at_s, 3.549)
        self.assertEqual(c.xy_updates, 2)
        mass, com = c.nom_box.set_payload.call_args.args
        self.assertEqual(mass, 5.)
        np.testing.assert_allclose(com, [.017, -.006, .04], atol=1e-14)
        np.testing.assert_array_equal(c._provided_com, [.02, -.01, .04])

    def test_zero_action_never_touches_nominal_payload(self):
        c = self.make_controller({'accept': False, 'model_action_xy_mm': [0., 0.]})
        with patch.object(controller.HeightController, 'compute', return_value=0):
            for t in (3.549, 3.550, 3.709, 3.8):
                c.compute(t, *([None] * 7))
        c.nom_box.set_payload.assert_not_called()

    def test_bad_decisions_fail_closed_before_writing_payload(self):
        for accept, xy in [(False, [1., 0.]), (True, [0., 0., 1.]),
                           (True, [16., 0.]), (True, [0., np.nan])]:
            c = self.make_controller({'accept': accept, 'model_action_xy_mm': xy})
            with patch.object(controller.HeightController, 'compute', return_value=0):
                with self.assertRaises(ValueError):
                    c.compute(3.549, *([None] * 7))
            c.nom_box.set_payload.assert_not_called()


if __name__ == '__main__':
    unittest.main()
