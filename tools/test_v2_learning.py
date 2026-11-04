"""Dependency-free regressions for normal-only numeric inference."""
import ast
import importlib.util
import json
from pathlib import Path
import sys
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.dont_write_bytecode = True
from tools.v2_learned_core import NormalResidualModel, normal_features
from tools.production_audit import audit_text


class NormalLearningTests(unittest.TestCase):
    def model(self):
        return NormalResidualModel(np.zeros(77), np.ones(77), np.zeros((77, 2)), 3.)

    def test_feature_shape_finite_and_no_lookup(self):
        for t in (0., 1.1, 1.49, 2.2, 3.81, 4.4):
            x = normal_features(t, 3.45, (.01,-.02,.03))
            self.assertEqual(x.shape, (77,))
            self.assertTrue(np.isfinite(x).all())
        source = (ROOT/'tools/v2_learned_core.py').read_text()
        self.assertEqual(audit_text(source)['errors'], [])
        self.assertNotIn('pickle', source)

    def test_direction_neutral_score(self):
        for force in ((7., 0), (-7.,0), (0,7.), (0,-7.)):
            model = self.model()
            flags = [model.update(2.2, 3.45, (0.,0.,0.), np.r_[force,np.zeros(4)], True)
                     for _ in range(50)]
            self.assertTrue(flags[-1])
            self.assertAlmostEqual(model.ema, 7*(1-.9**50))

    def test_gate_resets_and_nan_falls_back(self):
        model = self.model()
        for _ in range(40):
            model.update(2.2, 3.45, (0.,0.,0.), np.ones(6)*10, True)
        self.assertTrue(model.detected)
        self.assertFalse(model.update(2.2, 3.45, (0.,0.,0.), np.zeros(6), False))
        self.assertEqual(model.ema, 0.)
        self.assertFalse(model.update(2.2, 3.45, (0.,0.,0.), np.ones(6)*np.nan, True))

    def test_predictor_has_only_reference_mass_com_features(self):
        tree = ast.parse((ROOT/'tools/v2_learned_core.py').read_text())
        f = next(n for n in tree.body if isinstance(n, ast.FunctionDef))
        self.assertEqual([a.arg for a in f.args.args], ['t','mass','com'])
        self.assertNotIn('open', [n.id for n in ast.walk(f) if isinstance(n,ast.Name)])

    def test_training_code_has_no_mujoco_or_oracle(self):
        source = (ROOT/'tools/v2_train.py').read_text()
        tree = ast.parse(source)
        imports = [n.names[0].name for n in ast.walk(tree) if isinstance(n, ast.Import)]
        self.assertNotIn('mujoco', imports)
        for name in ('f_table','qpos','true_torque','JTS_SEED','JTS_OFFSET','JTS_DRIFT'):
            self.assertNotIn(name, source)

    def test_embedded_core_and_allowed_imports_when_built(self):
        path = ROOT/'v2_detector_candidate.py'
        if not path.exists():
            self.skipTest('weights not fitted yet')
        self.assertEqual(audit_text(path.read_text())['errors'], [])
        tree = ast.parse(path.read_text())
        core = ast.parse((ROOT/'tools/v2_learned_core.py').read_text())
        for kind in (ast.FunctionDef, ast.ClassDef):
            for original in (n for n in core.body if isinstance(n, kind)):
                embedded = next(n for n in tree.body if isinstance(n, kind) and n.name == original.name)
                self.assertEqual(ast.dump(original, include_attributes=False),
                                 ast.dump(embedded, include_attributes=False))


if __name__ == '__main__':
    unittest.main()
