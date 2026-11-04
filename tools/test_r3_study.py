"""Pure registration/cache/classification checks; never advance a plant."""
import importlib
import json
import unittest
from unittest.mock import patch

import numpy as np

from tools import r3_study as R
from tools.r3_build import compiled_path


class R3StudyTests(unittest.TestCase):
    def test_portable_protocol_still_requires_the_registered_sha(self):
        with patch('subprocess.check_output', side_effect=AssertionError('unexpected Git lookup')) as lookup:
            self.assertEqual(R.protocol()['schema'], 'R3_PROTOCOL.v1')
            lookup.assert_not_called()
        with patch.object(R, 'PROTOCOL') as changed:
            changed.read_bytes.return_value = b'{}\n'
            with self.assertRaisesRegex(RuntimeError, 'protocol bytes changed'):
                R.protocol()

    def test_registered_gates_are_verbatim(self):
        p = R.protocol()
        old = json.loads((R.D/'PROTOCOL_R2.json').read_text())
        for key in ('G4','safety_gates','cost_gates','timing','stability','normal_detector_policy'):
            self.assertEqual(p[key], old[key])
        self.assertEqual(p['C3_selection'],old['methods']['C3']['selection'])
        self.assertEqual(p['C4_selection'],old['methods']['C4']['selection'])

    def test_self_contained_defaults_bind_each_registered_spec(self):
        for spec in R.protocol()['candidate_grid']:
            path = compiled_path(spec)
            module = importlib.import_module('experiments.r3_generated.'+path.stem)
            self.assertEqual(module.R3_DEFAULT_PARAMETERS,spec)
            controller, detector = R.candidate_classes(spec['architecture'],spec)
            self.assertIs(controller,module.Controller)
            self.assertIs(detector,R.r1_detector.Detector)

    def test_unregistered_parameter_or_other_architecture_rejected(self):
        spec = dict(R.protocol()['candidate_grid'][0])
        spec['fade_start_s'] += .01
        with self.assertRaises(ValueError):R.candidate_classes('servo',spec)
        with self.assertRaises(ValueError):R.candidate_classes('torque',R.baseline_spec('servo'))

    def test_reference_matching_keeps_analysis_noise_distinct(self):
        refs=[]
        for seed in (1,2):
            refs.append(dict(mass=2.,com=[0.,0.,0.],perturb={'seed':seed},
                             kind='reference',collision=None,_detect=np.array([seed],int)))
        collision=dict(mass=2.,com=[0.,0.,0.],perturb={'seed':1},kind='collision',
                       collision=[2.,[1.,0.,0.]],_detect=np.array([0]))
        used=[]
        def classify(row,reference,t):
            used.append(reference.copy());return {'collision_outcome':'fixture'}
        with patch.object(R.sweep,'classify_collision',side_effect=classify):
            R.classify_group(refs+[collision])
        np.testing.assert_array_equal(used[0],[1])


if __name__=='__main__':unittest.main()
