import contextlib
import hashlib
import io
import json
from pathlib import Path
import unittest

import numpy as np

from experiments.r25_holdout import evaluator, runner, verify


ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent


class ProtocolTests(unittest.TestCase):
    def test_dry_accounting_and_worker_cap(self):
        self.assertEqual(evaluator.dry_validate()["episodes"], 198)
        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                evaluator.main(["--lock", "x", "--out", "y", "--jobs", "3"])

    def test_variable_profiles_and_five_twins_are_registered(self):
        plan = json.loads((HERE / "plan.json").read_bytes())
        durations = {row["impact"][1] for row in plan["single"]}
        durations.update(impact[1] for row in plan["paired"] for impact in row["impacts"])
        self.assertGreaterEqual(len(durations), 6)
        self.assertEqual(plan["required_episodes"], {"per_arm": 99, "both_arms": 198})
        self.assertTrue(all("long" in row for row in plan["paired"]))

    def test_signed_force_arithmetic_is_exact(self):
        t = .001 * np.arange(6000)
        impacts = ((1.2, 70, (5., -4., 2.)), (1.31, 50, (-3., 6., -1.)))
        target = runner.expected(t, impacts)
        self.assertTrue(np.array_equal(target["committed"],
                                       target["first"] + target["second"]))
        self.assertTrue(np.array_equal(target["scalar"],
                                       np.linalg.norm(target["first"], axis=1)))
        self.assertGreater(np.count_nonzero(target["first"]), 0)

    def test_real_contact_is_not_named_normal_or_spurious(self):
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

    def test_actuator_and_contact_shifts_materialize_on_authentic_plant(self):
        from sim import trajectory

        plan = json.loads((HERE / "plan.json").read_bytes())
        q0 = trajectory.reference(0.)[0]
        make = lambda name: runner.instrumented((None, None), plan["variants"][name])(
            q0, 3.1, (0.014, -0.018, 0.009))
        stock, actuator, contact = make("stock"), make("actuator_shift"), make("contact_shift")
        self.assertFalse(np.array_equal(stock.r25_gain, actuator.r25_gain))
        tables = [stock.m.geom(name).id for name in ("table_pick", "table_place")]
        self.assertFalse(np.array_equal(stock.m.geom_friction[tables, 0],
                                        contact.m.geom_friction[tables, 0]))

    def test_verifier_fails_closed(self):
        with self.assertRaisesRegex(ValueError, "incomplete"):
            verify.validate({"schema": "r25-independent-variable-pulse-result.v1"},
                            "0" * 64)


if __name__ == "__main__":
    unittest.main()
