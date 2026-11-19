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

    def test_only_the_in_memory_model_loader_and_nominal_model_are_accepted(self):
        prefix = ("import numpy as np\nimport io as stream\nimport base64 as codec\n"
                  "from collections import deque\nfrom sim.nominal import Nominal\n")
        ok = "np.load(stream.BytesIO(codec.b85decode(payload)), allow_pickle=False)"
        self.assertEqual(audit_text(prefix + ok)["errors"], [])
        for bad in ("stream.open('private')", "np.load('external.npz', allow_pickle=False)",
                    "np.load(stream.BytesIO(b), allow_pickle=True)", "from sim.nominal import model",
                    "from collections import ChainMap", "from numpy import load as reader\nreader('trace.npz')"):
            with self.subTest(bad=bad):
                self.assertEqual(audit_text(prefix + bad)["status"], "FAIL")

    def test_bounded_nominal_model_copy_is_accepted(self):
        source = '''
import numpy as np
import mujoco
from copy import copy
from sim.nominal import Nominal
def _clone_nominal(source, share_model=False):
    nominal = Nominal.__new__(Nominal)
    nominal.m = source.m if share_model else copy(source.m)
    nominal.d = mujoco.MjData(nominal.m)
    nominal._M = np.zeros_like(source._M)
    return nominal
class Membership:
    def __init__(self):
        self.empty = Nominal()
        self.full = _clone_nominal(self.empty)
        self.geometry = _clone_nominal(self.empty, share_model=True)
        self.scratch = _clone_nominal(self.full)
'''
        self.assertEqual(audit_text(source)["errors"], [])

    def test_copy_import_and_target_misuse_are_rejected(self):
        helper = '''
import numpy as np
import mujoco
from copy import copy
from sim.nominal import Nominal
def _clone_nominal(source, share_model=False):
    nominal = Nominal.__new__(Nominal)
    nominal.m = source.m if share_model else copy(source.m)
    nominal.d = mujoco.MjData(nominal.m)
    nominal._M = np.zeros_like(source._M)
    return nominal
'''
        bad_sources = (
            "import copy\n",
            "from copy import deepcopy\n",
            "from copy import copy as clone\n",
            helper.replace("copy(source.m)", "copy(source.d)"),
            helper.replace("nominal.m = source.m if share_model else copy(source.m)",
                           "nominal.other = copy(source.m)"),
            helper.replace("nominal = Nominal.__new__(Nominal)", "nominal = source"),
            helper.replace("nominal.d = mujoco.MjData(nominal.m)", "nominal.d = source.d"),
            helper + "leak = copy\n",
            helper + "bad = copy(Nominal().m)\n",
            helper + "bad = _clone_nominal(self.nom)\n",
            helper + "bad = _clone_nominal(self.empty, share_model=flag)\n",
            helper + "copy = lambda value: value\n",
            helper + "del copy\n",
            helper + "def copy(value):\n    return value\n",
            helper + "def misuse(copy):\n    return copy(source.m)\n",
            helper.replace("def _clone_nominal", "@np.vectorize\ndef _clone_nominal"),
            helper.replace("return nominal", "nominal = source\n    return nominal"),
            helper.replace("return nominal", "return nominal\n    return nominal"),
            helper + '''
class Membership:
    def reset(self):
        self.empty = Nominal()
        self.full = _clone_nominal(self.empty)
''',
            helper + '''
class Membership:
    def __init__(self):
        self.empty = object()
        self.full = _clone_nominal(self.empty)
''',
            helper + '''
class Membership:
    def __init__(self):
        self.empty = Nominal()
        self.full = _clone_nominal(self.full, share_model=True)
''',
        )
        for source in bad_sources:
            with self.subTest(source=source[-80:]):
                self.assertEqual(audit_text(source)["status"], "FAIL")

    def test_sim_configuration_and_trajectory_are_read_only(self):
        prefix = "from sim import config as C, trajectory as T\n"
        for source in ("C.K[0] = 1", "T.T_GRASP = 2", "T.WAYPOINTS.append((9, [0]*6))"):
            with self.subTest(source=source):
                self.assertEqual(audit_text(prefix + source)["status"], "FAIL")


if __name__ == "__main__":
    unittest.main()
