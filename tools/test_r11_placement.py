"""Independent unit and CLI contract tests for tools.r11_placement."""
from __future__ import annotations

import contextlib
import hashlib
import io
import json
import os
from pathlib import Path
from types import SimpleNamespace
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from tools import r11_placement as placement


class FaceClearanceTest(unittest.TestCase):
    def test_wxyz_identity_and_opposite_quaternion_match(self):
        positions = np.array([[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]])
        quaternions = np.array([[1.0, 0.0, 0.0, 0.0],
                                [-1.0, 0.0, 0.0, 0.0]])
        heights, tilt = placement.face_clearance(
            positions, quaternions, [0.1, 0.2, 0.3], 0.25)
        np.testing.assert_allclose(heights, 0.45)
        np.testing.assert_allclose(tilt, 0.0)
        np.testing.assert_array_equal(heights[0], heights[1])

    def test_wxyz_tilted_face_uses_all_four_corners(self):
        angle = np.deg2rad(30.0)
        quaternion = np.array([[np.cos(angle/2), np.sin(angle/2), 0.0, 0.0]])
        heights, tilt = placement.face_clearance(
            [[0.0, 0.0, 0.30]], quaternion, [0.10, 0.20, 0.05], 0.10)
        expected = np.sort([
            0.20 + sy*0.20*np.sin(angle) - 0.05*np.cos(angle)
            for sy in (-1.0, 1.0) for _ in (-1.0, 1.0)
        ])
        np.testing.assert_allclose(heights[0], expected, atol=1e-14)
        self.assertAlmostEqual(tilt[0], 30.0)
        self.assertLess(heights[0, 0], 0.1)
        self.assertGreater(heights[0, -1], 0.1)

    def test_opposite_tilted_quaternion_is_identical(self):
        quaternion = np.array([0.8, -0.2, 0.3, -0.458257569495584])
        quaternion /= np.linalg.norm(quaternion)
        positions = np.repeat([[0.1, -0.2, 0.5]], 2, axis=0)
        heights, tilt = placement.face_clearance(
            positions, np.vstack((quaternion, -quaternion)), [0.1, 0.2, 0.3], 0.0)
        np.testing.assert_allclose(heights[0], heights[1], atol=1e-14)
        np.testing.assert_allclose(tilt[0], tilt[1], atol=1e-14)

    def test_one_clear_corner_does_not_imply_all_four_clear(self):
        angle = np.deg2rad(30.0)
        quaternion = [[np.cos(angle/2), np.sin(angle/2), 0.0, 0.0]]
        heights, _ = placement.face_clearance(
            [[0.0, 0.0, 0.20]], quaternion, [0.10, 0.20, 0.05], 0.10)
        self.assertLess(heights[0, 0], 0.0)
        self.assertGreater(heights[0, -1], 0.0)
        self.assertFalse(np.all(heights[0] > 0.0))

    def test_rejects_bad_shapes_nonunit_and_nonfinite_inputs(self):
        valid = (np.zeros((1, 3)), np.array([[1.0, 0.0, 0.0, 0.0]]),
                 np.ones(3), 0.0)
        bad = [
            (np.zeros(3), valid[1], valid[2], valid[3]),
            (valid[0], np.zeros((1, 3)), valid[2], valid[3]),
            (valid[0], np.array([[2.0, 0.0, 0.0, 0.0]]), valid[2], valid[3]),
            (valid[0], valid[1], [1.0, 0.0, 1.0], valid[3]),
            (np.array([[0.0, 0.0, np.nan]]), valid[1], valid[2], valid[3]),
        ]
        for args in bad:
            with self.subTest(args=args), self.assertRaises(ValueError):
                placement.face_clearance(*args)


class SupportWindowTest(unittest.TestCase):
    DT = 0.001
    RELEASE = 3.9
    MASS = 2.0

    def grid(self):
        return np.arange(6000)*self.DT

    def test_exact_3p9_final_and_peak_window_endpoints(self):
        t = self.grid()
        force = np.zeros_like(t)
        mg = 9.81*self.MASS
        force[3880:3900] = 0.50*mg
        force[3599] = 2.0*mg       # before [-300 ms, ...): excluded
        force[3600] = 1.10*mg      # left endpoint: included
        force[3900] = 0.80*mg      # release sample: peak only
        force[4099] = 1.05*mg      # last included post-release sample
        force[4100] = 2.0*mg       # +200 ms endpoint: excluded
        result = placement.support(t, force, self.MASS, self.RELEASE, self.DT)
        self.assertEqual(result['samples'], 20)
        self.assertAlmostEqual(result['mean_mg'], 0.50)
        self.assertAlmostEqual(result['min_mg'], 0.50)
        self.assertAlmostEqual(result['peak_mg'], 1.10)
        self.assertTrue(result['mean_ge_050'])
        self.assertTrue(result['min_ge_025'])
        self.assertTrue(result['peak_le_110'])

    def test_support_masks_accept_exact_boundaries(self):
        t = self.grid()
        force = np.zeros_like(t)
        mg = 9.81*self.MASS
        force[3880:3900] = 0.50*mg
        force[3880] = 0.25*mg
        force[3881] = 0.75*mg
        force[3600] = 1.10*mg
        result = placement.support(t, force, self.MASS, self.RELEASE, self.DT)
        self.assertAlmostEqual(result['mean_mg'], 0.50)
        self.assertAlmostEqual(result['min_mg'], 0.25)
        self.assertAlmostEqual(result['peak_mg'], 1.10)
        self.assertTrue(result['mean_ge_050'])
        self.assertTrue(result['min_ge_025'])
        self.assertTrue(result['peak_le_110'])

    def test_support_masks_reject_values_just_outside_boundaries(self):
        t = self.grid()
        mg = 9.81*self.MASS
        force = np.zeros_like(t)
        force[3880:3900] = np.nextafter(0.50, 0.0)*mg
        below = placement.support(t, force, self.MASS, self.RELEASE, self.DT)
        self.assertFalse(below['mean_ge_050'])
        force[3880:3900] = 0.50*mg
        force[3880] = np.nextafter(0.25, 0.0)*mg
        low_min = placement.support(t, force, self.MASS, self.RELEASE, self.DT)
        self.assertFalse(low_min['min_ge_025'])
        force[3600] = np.nextafter(1.10, np.inf)*mg
        high_peak = placement.support(t, force, self.MASS, self.RELEASE, self.DT)
        self.assertFalse(high_peak['peak_le_110'])

    def test_rejects_negative_force_bad_shape_grid_and_release(self):
        t = self.grid()
        cases = [
            (t, np.full_like(t, -1e-6), self.MASS, self.RELEASE, self.DT),
            (t, np.zeros((len(t), 1)), self.MASS, self.RELEASE, self.DT),
            (t[:-1], np.zeros_like(t), self.MASS, self.RELEASE, self.DT),
            (t + 0.001, np.zeros_like(t), self.MASS, self.RELEASE, self.DT),
            (t, np.zeros_like(t), 0.0, self.RELEASE, self.DT),
            (t, np.zeros_like(t), self.MASS, 3.9005, self.DT),
        ]
        for args in cases:
            with self.subTest(args=(np.shape(args[0]), np.shape(args[1]), args[2:])), \
                    self.assertRaises(ValueError):
                placement.support(*args)


class ScoreTest(unittest.TestCase):
    @staticmethod
    def row(mean, minimum=True, peak=True):
        return {'support': {'mean_mg': mean, 'mean_ge_050': mean >= 0.50,
                            'min_ge_025': minimum, 'peak_le_110': peak}}

    def test_exact_gain_boundary_passes_all_masks(self):
        result = placement.score(self.row(0.25), self.row(0.50))
        self.assertEqual(result['mean_gain_mg'], 0.25)
        self.assertTrue(result['all_four_g4'])

    def test_each_gain_mask_is_required(self):
        baseline = self.row(0.25)
        pairs = [
            (baseline, self.row(np.nextafter(0.50, 0.0))),
            (baseline, self.row(0.50, minimum=False)),
            (baseline, self.row(0.50, peak=False)),
            (self.row(np.nextafter(0.25, np.inf)), self.row(0.50)),
        ]
        for reference, candidate in pairs:
            with self.subTest(reference=reference, candidate=candidate):
                self.assertFalse(placement.score(reference, candidate)['all_four_g4'])

    def test_missing_support_or_fields_are_rejected(self):
        with self.assertRaises(KeyError):
            placement.score({}, self.row(0.50))
        with self.assertRaises(KeyError):
            placement.score(self.row(0.25), {'support': {'mean_mg': 0.50}})


class CliProtectionTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.kit = self.root/'kit'
        self.candidate = self.root/'candidate'
        self.output_parent = self.root/'output'
        for path in (self.kit, self.candidate, self.output_parent):
            path.mkdir()
        for name in ('servo', 'torque', 'detector'):
            (self.candidate/f'my_{name}.py').write_text(f'# {name}\n')

    def tearDown(self):
        self.temp.cleanup()

    def invoke(self, output, candidate=None):
        argv = ['r11_placement.py', '--kit', str(self.kit), '--candidate-dir',
                str(candidate or self.candidate), '--out', str(output)]
        with patch.object(sys, 'argv', argv), contextlib.redirect_stderr(io.StringIO()):
            placement.main()

    def test_rejects_existing_output(self):
        output = self.output_parent/'exists.json'
        output.write_text('preserve')
        with self.assertRaises(SystemExit):
            self.invoke(output)
        self.assertEqual(output.read_text(), 'preserve')

    def test_rejects_output_inside_kit_or_missing_parent(self):
        for output in (self.kit/'result.json', self.root/'missing'/'result.json'):
            with self.subTest(output=output), self.assertRaises(SystemExit):
                self.invoke(output)
            self.assertFalse(output.exists())

    def test_rejects_symlinked_candidate_source(self):
        target = self.root/'outside.py'
        target.write_text('# detector\n')
        (self.candidate/'my_detector.py').unlink()
        (self.candidate/'my_detector.py').symlink_to(target)
        output = self.output_parent/'result.json'
        with self.assertRaises(SystemExit):
            self.invoke(output)
        self.assertFalse(output.exists())

    def simulation_mocks(self, rows=None):
        plan = json.loads(placement.PLAN.read_text())
        if rows is None:
            rows = [{'id': case['id'], 'support': {
                'mean_mg': 0.5, 'min_mg': 0.5, 'peak_mg': 1.0,
                'samples': 20, 'mean_ge_050': True, 'min_ge_025': True,
                'peak_le_110': True}} for case in plan['cases']]

        def prepare(_kit, overlay, _prefix, mode, **_kwargs):
            (overlay/f'{mode}_controller.py').write_text('# archived\n')
            (overlay/'collision_detector.py').write_text('# archived\n')
            return 'archived-controller', 'archived-detector'

        def run(*_args, **kwargs):
            self.assertNotIn('PYTHONPATH', kwargs['env'])
            self.assertEqual(kwargs['env']['PYTHONDONTWRITEBYTECODE'], '1')
            return SimpleNamespace(stdout=json.dumps(rows))
        return prepare, run

    def test_valid_cli_writes_new_output_without_modifying_inputs(self):
        output = self.output_parent/'result.json'
        before = {path: hashlib.sha256(path.read_bytes()).hexdigest()
                  for path in self.candidate.iterdir()}
        prepare, run = self.simulation_mocks()
        with patch('tools.validate_submission.prepare_overlay', side_effect=prepare), \
                patch.object(placement.subprocess, 'run', side_effect=run):
            self.invoke(output)
        result = json.loads(output.read_text())
        self.assertEqual(result['schema'], 'r11-placement.v1')
        self.assertEqual(set(result['modes']), {'servo', 'torque'})
        after = {path: hashlib.sha256(path.read_bytes()).hexdigest()
                 for path in self.candidate.iterdir()}
        self.assertEqual(before, after)

    def test_missing_child_rows_do_not_create_output(self):
        output = self.output_parent/'result.json'
        prepare, run = self.simulation_mocks(rows=[])
        with patch('tools.validate_submission.prepare_overlay', side_effect=prepare), \
                patch.object(placement.subprocess, 'run', side_effect=run), \
                self.assertRaises(ValueError):
            self.invoke(output)
        self.assertFalse(output.exists())


class MeasuredSummaryTest(unittest.TestCase):
    def test_measured_summary_when_supplied(self):
        supplied = os.environ.get('R11_PLACEMENT_SUMMARY')
        path = Path(supplied) if supplied else placement.ROOT/'results/r11/r10_placement.json'
        result = json.loads(path.read_text())
        plan = json.loads(placement.PLAN.read_text())
        ids = [case['id'] for case in plan['cases']]
        self.assertEqual(result['schema'], 'r11-placement.v1')
        self.assertEqual(result['plan_sha256'], placement.digest(placement.PLAN))
        self.assertEqual(result['analysis_sha256'], placement.digest(Path(placement.__file__)))
        for mode in ('servo', 'torque'):
            measured = result['modes'][mode]
            baseline = measured['archived_r1']['rows']
            candidate = measured['candidate']['rows']
            self.assertEqual([row['id'] for row in baseline], ids)
            self.assertEqual([row['id'] for row in candidate], ids)
            expected = [dict(id=a['id'], **placement.score(a, b))
                        for a, b in zip(baseline, candidate, strict=True)]
            self.assertEqual(measured['comparison'], expected)
            for row in baseline + candidate:
                self.assertEqual(row['support']['samples'], 20)
                self.assertIn('all_four_corners_below_plane_samples',
                              row['last20_geometry'])





class UnqualifiedPlacementTest(unittest.TestCase):
    def test_divergence_is_a_failed_candidate_not_a_zero_force_measurement(self):
        out = placement.score({'status':'ok'}, {'status':'diverged'})
        self.assertFalse(out['all_four_g4'])
        self.assertIsNone(out['mean_gain_mg'])
        self.assertEqual(out['reason'], 'candidate diverged')

    def test_missing_footprint_or_reference_never_scores_as_a_win(self):
        for b,c in [('ok','footprint_unverified'),('footprint_unverified','ok'),('diverged','ok')]:
            with self.subTest(b=b,c=c):
                out = placement.score({'status':b}, {'status':c})
                self.assertIsNone(out['all_four_g4'])
                self.assertIsNone(out['mean_gain_mg'])


if __name__ == '__main__':
    unittest.main()
