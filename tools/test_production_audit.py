import unittest

from tools.production_audit import audit_text


class ProductionAuditTests(unittest.TestCase):
    def test_public_geometry_dynamics_and_optional_controller(self):
        text = '''import numpy as np
import mujoco
from sim import config as C, trajectory as T
class Detector:
    def update(self):
        ctrl = getattr(self, "ctrl", None)
        return C.K, C.BOX_SIZE, T.T_GRASP, getattr(ctrl, "load_fraction", 1.0)
'''
        self.assertEqual(audit_text(text)["status"], "pass")

    def test_reject_private_config_io_and_cross_file_import(self):
        for source in ("from sim import plant", "import my_servo", "from scipy import signal",
                       "from sim.config import JTS_OFFSET", "x = C.JTS_SEED",
                       "x = getattr(C, 'JTS_DRIFT')", "x = C.SUCTION_SOLREF",
                       "x = plant.true_torque()", "open('log')", "eval('1')",
                       "if t > 1.42: pass", "if t > 3.86: pass"):
            with self.subTest(source=source):
                self.assertEqual(audit_text(source)["status"], "FAIL")

    def test_sim_configuration_and_trajectory_are_read_only(self):
        prefix = "from sim import config as C, trajectory as T\n"
        for source in ("C.K[0] = 1", "T.T_GRASP = 2", "T.WAYPOINTS.append((9, [0]*6))"):
            with self.subTest(source=source):
                self.assertEqual(audit_text(prefix + source)["status"], "FAIL")


if __name__ == "__main__":
    unittest.main()
