"""The R8 assembly is reproducible, standalone and exact-mode dispatched."""
import ast
import importlib.util
from pathlib import Path
import unittest
from types import SimpleNamespace
import numpy as np
from tools import r8_build
from sim.nominal import Nominal


class R8BuildTests(unittest.TestCase):
    def test_reproducible(self):
        self.assertEqual(r8_build.render(), (r8_build.FINAL / 'C1.py').read_bytes())

    def test_runtime_imports(self):
        tree = ast.parse((r8_build.FINAL / 'C1.py').read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                self.assertFalse((node.module or '').startswith(('experiments', 'tools')))
            if isinstance(node, ast.Import):
                self.assertFalse(any(n.name.startswith(('experiments', 'tools')) for n in node.names))

    def test_exact_mode_and_model_bytes(self):
        spec = importlib.util.spec_from_file_location('r8_candidate_test', r8_build.FINAL / 'C1.py')
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        for mode in ('servo', 'torque', 'ser', None):
            det = module.Detector()
            det.nom, det.nom_box = Nominal(), Nominal()
            det.box_mass, det.box_com = 0., (0., 0., 0.)
            det.ctrl = SimpleNamespace(MODE=mode)
            det.reset()
            self.assertEqual(det._r8_member is not None, mode == 'servo')
        for kind in ('empty', 'positive'):
            model = module._r8_model(kind == 'empty')
            with np.load(r8_build.FINAL / f'servo_{kind}.npz') as z:
                for key in z.files:
                    self.assertEqual(model[key].tobytes(), z[key].tobytes())


if __name__ == '__main__':
    unittest.main()
