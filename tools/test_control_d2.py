import ast
import importlib
from pathlib import Path
import sys
import unittest
import numpy as np

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
from check_integrity import check_integrity

class ControlD2Tests(unittest.TestCase):
    def test_production_import_and_oracle_audit(self):
        allowed = {"numpy", "mujoco", "sim"}
        forbidden = {"plant", "runner", "table_force", "true_torque", "JTS_OFFSET", "JTS_DRIFT", "JTS_SEED", "my_detector"}
        for name in ("my_servo.py", "my_torque.py"):
            text = (ROOT / name).read_text(encoding="utf-8")
            tree = ast.parse(text)
            imports = set()
            for node in ast.walk(tree):
                if isinstance(node, ast.Import): imports.update(x.name.split('.')[0] for x in node.names)
                if isinstance(node, ast.ImportFrom): imports.add((node.module or '').split('.')[0])
            self.assertLessEqual(imports, allowed)
            for token in forbidden:
                self.assertNotIn(token, text)

    def test_shapes_finite_and_bounded_b2(self):
        check_integrity(ROOT)
        for module, shape in (("my_servo", (4, 6)), ("my_torque", (6,))):
            C = importlib.import_module(module).Controller()
            from sim.nominal import Nominal
            from sim.drive import Drive
            C.nom, C.nom_box = Nominal(), Nominal(); C.nom_box.set_payload(8, (0.02,-0.01,0.03))
            C.box_mass, C.box_com, C.drive = 8.0, (0.02,-0.01,0.03), Drive(); C.reset()
            from sim.trajectory import reference
            q, qd, qdd = reference(1.42)
            out = np.asarray(C.compute(1.42, q, qd, np.zeros(6), 1, q, qd, qdd))
            self.assertEqual(out.shape, shape); self.assertTrue(np.isfinite(out).all())
            if module == "my_torque":
                self.assertLess(C.k_tau, 0.0); self.assertGreater(C.k_tau, -1.0)
                self.assertGreater(1.0 / (1.0 - C.k_tau), 0.5)

if __name__ == '__main__': unittest.main()
