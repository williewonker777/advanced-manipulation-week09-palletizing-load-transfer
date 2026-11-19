import ast
import inspect
import textwrap
import unittest

from experiments.r26_yaw_acquisition.candidate import Detector as C3
from experiments.r29_detached_release.candidate import Detector as Released
from experiments.r29_native_event_link.candidate import Detector as Held
from experiments.r30_detached_lifetime.candidate import Detector as Lifetime
from experiments.r31_torque_control_candidate.candidate import Detector


class CandidateTest(unittest.TestCase):
    def test_cooperative_mro_is_disjoint_and_reaches_c3_once(self):
        mro = Detector.__mro__
        self.assertLess(mro.index(Lifetime), mro.index(Released))
        self.assertLess(mro.index(Released), mro.index(Held))
        self.assertLess(mro.index(Held), mro.index(C3))
        self.assertEqual(sum(cls is C3 for cls in mro), 1)

    def test_candidate_adds_no_update_or_case_switch(self):
        tree = ast.parse(inspect.getsource(Detector))
        functions = [node.name for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)]
        self.assertNotIn("update", functions)
        self.assertNotIn("reset", functions)

    def test_existing_chain_has_one_cooperative_and_one_physical_update(self):
        released = ast.parse(textwrap.dedent(inspect.getsource(Released.update)))
        held = ast.parse(textwrap.dedent(inspect.getsource(Held.update)))
        released_super = [n for n in ast.walk(released) if isinstance(n, ast.Call)
                          and isinstance(n.func, ast.Attribute) and n.func.attr == "update"]
        held_physical = [n for n in ast.walk(held) if isinstance(n, ast.Call)
                         and isinstance(n.func, ast.Attribute) and n.func.attr == "update"]
        self.assertEqual(len(released_super), 2)  # super().update and endpoint.update
        self.assertEqual(len(held_physical), 1)  # manifold.Base.update only
        self.assertIn("super().update", inspect.getsource(Released.update))
        self.assertIn("manifold.Base.update", inspect.getsource(Held.update))


if __name__ == "__main__":
    unittest.main()
