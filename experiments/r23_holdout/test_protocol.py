import contextlib
import hashlib
import io
import json
from pathlib import Path
import unittest

import numpy as np

from experiments.r23_holdout import evaluator, lock, verify


ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent


class SealedProtocolTests(unittest.TestCase):
    def test_plan_has_exact_unique_accounting(self):
        plan = json.loads((HERE / "plan.json").read_bytes())
        cases = [(kind, row["id"]) for kind in ("negative", "single", "paired")
                 for row in plan[kind]]
        self.assertEqual(len(cases), len(set(cases)))
        self.assertEqual({kind: len(plan[kind])
                          for kind in ("negative", "single", "paired")},
                         {"negative": 4, "single": 6, "paired": 5})
        self.assertEqual(plan["required_episodes"], {"per_arm": 108, "both_arms": 216})
        calculated = 3 * (4 + 2 * 6 + 4 * 5)
        self.assertEqual(calculated, plan["required_episodes"]["per_arm"])
        self.assertEqual(2 * calculated, plan["required_episodes"]["both_arms"])
        self.assertEqual(set(plan["variants"]),
                         {"stock", "contact_surface", "actuator_skew"})

    def test_scope_includes_transfer_post_detachment_and_borderline_diagonals(self):
        plan = json.loads((HERE / "plan.json").read_bytes())
        phases = {row["phase"] for kind in ("single", "paired") for row in plan[kind]}
        self.assertTrue({"transfer", "post_detachment", "cross_phase"} <= phases)
        diagonal = []
        for kind in ("single", "paired"):
            for row in plan[kind]:
                impacts = [row["impact"]] if kind == "single" else row["impacts"]
                diagonal.extend(force for _, force in impacts if force[0] and force[1])
        self.assertGreaterEqual(len(diagonal), 6)
        self.assertTrue(any(7.5 < np.linalg.norm(force) < 10.0 for force in diagonal))

    def test_case_tuples_and_ids_are_not_copied_from_pre_r23_plans(self):
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

        old_signatures, old_text = set(), []
        for path in (ROOT / "experiments").glob("r2[0-2]*/**/*plan*.json"):
            old_text.append(path.read_text(errors="ignore"))
            for case in walk(json.loads(path.read_bytes())):
                value = signature(case)
                if value:
                    old_signatures.add(value)
        plan = json.loads((HERE / "plan.json").read_bytes())
        current = {signature(case) for kind in ("negative", "single", "paired")
                   for case in plan[kind]}
        self.assertFalse(current & old_signatures)
        joined = "\n".join(old_text)
        for kind in ("negative", "single", "paired"):
            for case in plan[kind]:
                self.assertNotIn(case["id"], joined)

    def test_every_impulse_has_real_support_and_pairs_have_a_zero_gap(self):
        plan = json.loads((HERE / "plan.json").read_bytes())
        t = .001 * np.arange(6000)
        for case in plan["single"]:
            start, record = evaluator.certified.actual_start(t, case["impact"])
            self.assertGreater(start, 0)
            self.assertIn(record["nominal_sample_zero"], (True, False))
        for case in plan["paired"]:
            arrays = [np.stack([evaluator.certified.pulse(value, impact) for value in t])
                      for impact in case["impacts"]]
            active = [np.flatnonzero(np.any(value != 0., axis=1)) for value in arrays]
            self.assertTrue(all(len(value) for value in active))
            self.assertLess(active[0][-1], active[1][0])
            self.assertTrue(np.all(arrays[0][active[0][-1] + 1:active[1][0]] == 0.))

    def test_protocol_source_pins_are_current(self):
        protocol = json.loads((HERE / "protocol.json").read_bytes())
        actual = {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
                  for name in protocol["source_pins"]}
        self.assertEqual(actual, protocol["source_pins"])
        self.assertEqual(protocol["baseline_entrypoint"],
                         "experiments.r22_light.candidate:Detector")

    def test_worker_count_is_directly_capped_below_three(self):
        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                evaluator.main(["--lock", "missing", "--out", "fresh", "--jobs", "3"])

    def test_mechanical_variants_are_local_and_materialized(self):
        from sim import trajectory

        plan = json.loads((HERE / "plan.json").read_bytes())
        q0 = trajectory.reference(0.)[0]
        stock = evaluator.local_plant(None, plan["variants"]["stock"])(
            q0, 2.9, (0.012, -0.018, 0.007))
        contact = evaluator.local_plant(None, plan["variants"]["contact_surface"])(
            q0, 2.9, (0.012, -0.018, 0.007))
        skew = evaluator.local_plant(None, plan["variants"]["actuator_skew"])(
            q0, 2.9, (0.012, -0.018, 0.007))
        tables = [stock.m.geom(name).id for name in ("table_pick", "table_place")]
        self.assertFalse(np.array_equal(stock.m.geom_friction[tables, 0],
                                        contact.m.geom_friction[tables, 0]))
        self.assertFalse(np.array_equal(stock.m.geom_solref[tables, 0],
                                        contact.m.geom_solref[tables, 0]))
        self.assertTrue(np.array_equal(stock.r23_gain, np.ones(6)))
        self.assertTrue(np.array_equal(skew.r23_gain,
                                       np.asarray(plan["variants"]["actuator_skew"]["joint_gain"])))

    def test_lock_requires_new_non_holdout_r23_entrypoint(self):
        protocol = json.loads((HERE / "protocol.json").read_bytes())
        with self.assertRaisesRegex(ValueError, "non-holdout"):
            lock.transitive_closure("experiments.r23_holdout.evaluator:Detector",
                                    set(protocol["source_pins"]))

    def test_verifier_rejects_missing_rows(self):
        with self.assertRaisesRegex(ValueError, "incomplete"):
            verify.validate({"schema": "r23-independent-transfer-detachment-result.v1"},
                            "0" * 64)

    def test_safety_claim_is_explicitly_false(self):
        plan = json.loads((HERE / "plan.json").read_bytes())
        protocol = json.loads((HERE / "protocol.json").read_bytes())
        self.assertFalse(plan["qualification"]["safety_qualification"])
        self.assertIn("always false", protocol["safety_scope"])


if __name__ == "__main__":
    unittest.main()
