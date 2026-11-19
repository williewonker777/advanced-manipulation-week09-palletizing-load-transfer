import contextlib
import hashlib
import io
import json
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np

from experiments.r26_holdout import evaluator, runner, verify


ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent


class ProtocolTests(unittest.TestCase):
    def test_dry_accounting_runtime_and_worker_cap(self):
        dry = evaluator.dry_validate()
        self.assertEqual(dry["episodes"], 156)
        self.assertEqual(dry["mujoco"], "3.17.0")
        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                evaluator.main(["--lock", "x", "--out", "y", "--jobs", "3"])

    def test_long_close_five_twins_and_isolated_second_are_registered(self):
        plan = json.loads((HERE / "plan.json").read_bytes())
        durations = {row["impact"][1] for row in plan["single"]}
        durations.update(impact[1] for row in plan["paired"]
                         for impact in [row["long"], *row["impacts"]])
        self.assertGreaterEqual(len(durations), 10)
        self.assertEqual(plan["required_episodes"], {"per_arm": 78, "both_arms": 156})
        self.assertTrue(all(len(row["impacts"]) == 2 and "long" in row
                            for row in plan["paired"]))
        self.assertIn("isolated second", plan["qualification"]["positive_qualification"])

    def test_signed_force_arithmetic_is_exact(self):
        t = .001 * np.arange(6000)
        impacts = ((1.11, 49, (4., -5., 2.)), (1.20, 73, (-2., 6., -3.)))
        target = runner.expected(t, impacts)
        self.assertTrue(np.array_equal(target["committed"],
                                       target["first"] + target["second"]))
        self.assertTrue(np.array_equal(target["scalar"],
                                       np.linalg.norm(target["first"], axis=1)))
        self.assertGreater(np.count_nonzero(target["first"]), 0)

    def test_real_contact_is_protected_not_named_normal_or_spurious(self):
        plan = json.loads((HERE / "plan.json").read_bytes())
        probes = [row for row in plan["negative"] if row["contact_probe"]]
        self.assertTrue(probes)
        self.assertTrue(all(row["role"] == "real_tool_contact_control" for row in probes))
        self.assertIn("protected", plan["qualification"]["real_contact"])

    def test_protocol_pins_are_current(self):
        protocol = json.loads((HERE / "protocol.json").read_bytes())
        actual = {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
                  for name in protocol["source_pins"]}
        self.assertEqual(actual, protocol["source_pins"])

    def test_new_actuator_and_place_surface_mechanics_materialize(self):
        from sim import trajectory

        plan = json.loads((HERE / "plan.json").read_bytes())
        q0 = trajectory.reference(0.)[0]
        make = lambda name: runner.instrumented((None, None), plan["variants"][name])(
            q0, 3.1, (0.014, -0.018, 0.009))
        stock = make("stock")
        actuator = make("actuator_deadzone")
        surface = make("place_surface_shift")
        self.assertFalse(np.array_equal(stock.r26_gain, actuator.r26_gain))
        self.assertGreater(actuator.r26_deadzone, 0.)
        place = stock.m.geom("table_place").id
        self.assertNotEqual(stock.m.geom_friction[place, 0],
                            surface.m.geom_friction[place, 0])
        self.assertFalse(np.array_equal(stock.m.dof_damping, surface.m.dof_damping))

    def test_reused_r25_scoring_interfaces_exist_and_are_callable(self):
        self.assertTrue(callable(evaluator.prior.event))
        self.assertTrue(callable(evaluator.prior.preserved))
        self.assertIs(evaluator.tool_floor, evaluator.prior.prior.prior.tool_floor)
        self.assertIs(evaluator.tool_gates, evaluator.prior.prior.prior.tool_gates)

    def test_runtime_and_verifier_fail_closed(self):
        with patch.object(evaluator.mujoco, "__version__", "3.11.0"):
            with self.assertRaisesRegex(RuntimeError, "3.17.0"):
                evaluator.require_runtime()
        with self.assertRaisesRegex(ValueError, "incomplete"):
            verify.validate({"schema": "r26-independent-long-close-result.v1"},
                            "0" * 64)


if __name__ == "__main__":
    unittest.main()
