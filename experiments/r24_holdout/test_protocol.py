import contextlib
import hashlib
import io
import json
from pathlib import Path
import unittest

import numpy as np

from experiments.r24_holdout import evaluator, lock, verify


ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent


class SealedProtocolTests(unittest.TestCase):
    def test_exact_accounting_and_worker_cap(self):
        plan = json.loads((HERE / "plan.json").read_bytes())
        self.assertEqual({kind: len(plan[kind])
                          for kind in ("negative", "single", "paired")},
                         {"negative": 4, "single": 5, "paired": 4})
        per_arm = len(plan["variants"]) * (4 + 2 * 5 + 4 * 4)
        self.assertEqual(per_arm, 120)
        self.assertEqual(plan["required_episodes"], {"per_arm": 120, "both_arms": 240})
        self.assertTrue(100 <= plan["required_episodes"]["both_arms"] <= 300)
        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                evaluator.main(["--lock", "missing", "--out", "fresh", "--jobs", "3"])

    def test_dry_protocol_is_executable_without_candidate(self):
        record = evaluator.dry_validate()
        self.assertEqual(record["episodes"], 240)
        self.assertEqual(record["cases"], 13)
        self.assertEqual(record["variants"], 4)
        self.assertEqual(record["worker_cap"], 2)

    def test_ids_and_case_tuples_are_new(self):
        def walk(value):
            if isinstance(value, dict):
                yield value
                for item in value.values():
                    yield from walk(item)
            elif isinstance(value, list):
                for item in value:
                    yield from walk(item)

        def signature(case):
            if "mass" not in case or "com" not in case:
                return None
            force = case.get("impacts", case.get("impact", case.get("collision")))
            if force is None and "time" in case and "force" in case:
                force = [case["time"], case["force"]]
            return (float(case["mass"]), tuple(case["com"]),
                    None if case.get("offset") is None else tuple(case["offset"]),
                    json.dumps(force, sort_keys=True))

        old, text = set(), []
        for path in (ROOT / "experiments").glob("r2[0-3]*/**/*plan*.json"):
            text.append(path.read_text(errors="ignore"))
            for row in walk(json.loads(path.read_bytes())):
                value = signature(row)
                if value:
                    old.add(value)
        plan = json.loads((HERE / "plan.json").read_bytes())
        current = {signature(row) for kind in ("negative", "single", "paired")
                   for row in plan[kind]}
        self.assertFalse(current & old)
        joined = "\n".join(text)
        for kind in ("negative", "single", "paired"):
            for row in plan[kind]:
                self.assertNotIn(row["id"], joined)

    def test_impulses_have_support_and_pairs_are_separate(self):
        plan = json.loads((HERE / "plan.json").read_bytes())
        t = .001 * np.arange(6000)
        for case in plan["single"]:
            start, _ = evaluator.certified.actual_start(t, case["impact"])
            self.assertGreater(start, 0)
        for case in plan["paired"]:
            active = []
            for impact in case["impacts"]:
                force = np.stack([evaluator.certified.pulse(value, impact) for value in t])
                active.append(np.flatnonzero(np.any(force != 0., axis=1)))
            self.assertTrue(all(len(row) for row in active))
            self.assertLess(active[0][-1], active[1][0])

    def test_protocol_pins_current_exact_r23_free_closure(self):
        protocol = json.loads((HERE / "protocol.json").read_bytes())
        actual = {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
                  for name in protocol["source_pins"]}
        self.assertEqual(actual, protocol["source_pins"])
        self.assertEqual(protocol["baseline_entrypoint"],
                         "experiments.r23_detached.free:Detector")
        self.assertEqual(protocol["source_pins"]["results/r23/independent.lock.json"],
                         "bcfdeace274ae7f24a11d2d8d63824304c312b7a0dd447c55953995fe7ea22d8")

    def test_three_nonstock_mechanics_materialize_independently(self):
        from sim import trajectory

        plan = json.loads((HERE / "plan.json").read_bytes())
        q0 = trajectory.reference(0.)[0]
        make = lambda name: evaluator.local_plant(None, plan["variants"][name])(
            q0, 3.2, (0.013, -0.019, 0.009))
        stock, actuator = make("stock"), make("actuator_delay_gain")
        damping, contact = make("damping_coupled"), make("contact_armature")
        self.assertEqual(actuator.r24_delay_steps, 1)
        self.assertFalse(np.array_equal(actuator.r24_gain, stock.r24_gain))
        self.assertFalse(np.array_equal(damping.m.dof_damping[:6],
                                        stock.m.dof_damping[:6]))
        tables = [stock.m.geom(name).id for name in ("table_pick", "table_place")]
        self.assertFalse(np.array_equal(contact.m.geom_friction[tables, 0],
                                        stock.m.geom_friction[tables, 0]))
        self.assertFalse(np.array_equal(contact.m.dof_armature[:6],
                                        stock.m.dof_armature[:6]))

    def test_absolute_gates_and_cost_are_explicit(self):
        plan = json.loads((HERE / "plan.json").read_bytes())
        gates = "\n".join(plan["gates"])
        self.assertIn("zero false rise", gates)
        self.assertIn("zero false active sample", gates)
        self.assertIn("zero duplicate", gates)
        self.assertIn("alarm-sample cost", gates)
        self.assertFalse(plan["qualification"]["safety_qualification"])

    def test_lock_rejects_holdout_as_candidate(self):
        protocol = json.loads((HERE / "protocol.json").read_bytes())
        with self.assertRaisesRegex(ValueError, "non-holdout"):
            lock.transitive_closure("experiments.r24_holdout.evaluator:Detector",
                                    set(protocol["source_pins"]))

    def test_verifier_rejects_missing_rows(self):
        with self.assertRaisesRegex(ValueError, "incomplete"):
            verify.validate({"schema": "r24-independent-actuator-transfer-result.v1"},
                            "0" * 64)


if __name__ == "__main__":
    unittest.main()
