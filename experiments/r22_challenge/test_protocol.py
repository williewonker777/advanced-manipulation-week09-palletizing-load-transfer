import hashlib
import json
from pathlib import Path
import unittest

import numpy as np

from experiments.r22_challenge import evaluator, lock, verify
from sim import trajectory


ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent


class SealedProtocolTests(unittest.TestCase):
    def test_plan_has_exact_unique_accounting(self):
        plan = json.loads((HERE / "plan.json").read_bytes())
        cases = [(kind, row["id"]) for kind in ("negative", "single", "paired")
                 for row in plan[kind]]
        self.assertEqual(len(cases), len(set(cases)))
        self.assertEqual({k: len(plan[k]) for k in ("negative", "single", "paired")},
                         {"negative": 6, "single": 8, "paired": 8})
        self.assertEqual(plan["required_episodes"], {"per_arm": 162, "all_three_arms": 486})
        self.assertEqual(set(plan["variants"]), {"stock", "contact_suction", "actuator_gain"})

    def test_new_ids_are_not_in_any_r21_plan(self):
        plan = json.loads((HERE / "plan.json").read_bytes())
        old = "\n".join(path.read_text(errors="ignore") for path in
                        (ROOT / "experiments").glob("r21*/**/*plan*.json"))
        for kind in ("negative", "single", "paired"):
            for case in plan[kind]:
                self.assertNotIn(case["id"], old)

    def test_case_tuples_are_not_copied_from_r21_plans(self):
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

        old = set()
        for path in (ROOT / "experiments").glob("r21*/**/*plan*.json"):
            for case in walk(json.loads(path.read_bytes())):
                value = signature(case)
                if value:
                    old.add(value)
        plan = json.loads((HERE / "plan.json").read_bytes())
        current = {signature(case) for kind in ("negative", "single", "paired") for case in plan[kind]}
        self.assertFalse(current & old)

    def test_every_impulse_has_real_support_and_pairs_have_zero_gap(self):
        plan = json.loads((HERE / "plan.json").read_bytes())
        t = .001 * np.arange(6000)
        for case in plan["single"]:
            start, record = evaluator.actual_start(t, case["impact"])
            self.assertGreater(start, 0)
            self.assertIn(record["nominal_sample_zero"], (True, False))
        for case in plan["paired"]:
            arrays = [np.stack([evaluator.pulse(value, impact) for value in t])
                      for impact in case["impacts"]]
            active = [np.flatnonzero(np.any(value != 0., axis=1)) for value in arrays]
            self.assertTrue(all(len(value) for value in active))
            self.assertLess(active[0][-1], active[1][0])
            self.assertTrue(np.all(arrays[0][active[0][-1] + 1:active[1][0]] == 0.))

    def test_reference_matching_is_injective_per_candidate_edge(self):
        trace = {"detect": np.zeros(6000, bool)}
        ref = {"detect": np.zeros(6000, bool)}
        trace["detect"][[100, 150]] = True
        ref["detect"][101] = True
        self.assertTrue(evaluator.match_reference(100, [ref], 20))
        self.assertFalse(evaluator.match_reference(150, [ref], 20))

    def test_event_requires_one_unmasked_rising_edge(self):
        log = {"detect": np.zeros(6000, bool)}
        ref = {"detect": np.zeros(6000, bool)}
        log["detect"][120:135] = True
        record = evaluator.event_record(log, [ref], 100, 300, 20, 20)
        self.assertTrue(record["exactly_one"])
        log["detect"][180:185] = True
        self.assertFalse(evaluator.event_record(log, [ref], 100, 300, 20, 20)["exactly_one"])

    def test_protocol_source_pins_are_current(self):
        protocol = json.loads((HERE / "protocol.json").read_bytes())
        actual = {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
                  for name in protocol["source_pins"]}
        self.assertEqual(actual, protocol["source_pins"])

    def test_mechanical_variants_are_local_and_materialized(self):
        plan = json.loads((HERE / "plan.json").read_bytes())
        q0 = trajectory.reference(0.)[0]
        stock_type = evaluator.local_plant(None, plan["variants"]["stock"])
        contact_type = evaluator.local_plant(None, plan["variants"]["contact_suction"])
        gain_type = evaluator.local_plant(None, plan["variants"]["actuator_gain"])
        stock = stock_type(q0, 2.3, (0.01, -0.02, 0.005))
        changed = contact_type(q0, 2.3, (0.01, -0.02, 0.005))
        gain = gain_type(q0, 2.3, (0.01, -0.02, 0.005))
        tables = [stock.m.geom(name).id for name in ("table_pick", "table_place")]
        self.assertFalse(np.array_equal(stock.m.geom_friction[tables, 0],
                                        changed.m.geom_friction[tables, 0]))
        self.assertFalse(np.array_equal(stock.m.geom_solref[tables, 0],
                                        changed.m.geom_solref[tables, 0]))
        self.assertFalse(np.array_equal(stock.m.eq_solref[:, 0], changed.m.eq_solref[:, 0]))
        self.assertTrue(np.array_equal(stock.r22_gain, np.ones(6)))
        self.assertTrue(np.array_equal(gain.r22_gain,
                                       np.asarray(plan["variants"]["actuator_gain"]["joint_gain"])))

    def test_lock_rejects_unpinned_r22_import(self):
        with self.assertRaisesRegex(ValueError, "unpinned"):
            lock.validate_python_dependencies({"experiments/r22_challenge/test_protocol.py"})

    def test_verifier_rejects_missing_rows(self):
        with self.assertRaisesRegex(ValueError, "incomplete"):
            verify.validate({"schema": "r22-independent-protection-result.v1"}, "0" * 64)


if __name__ == "__main__":
    unittest.main()
