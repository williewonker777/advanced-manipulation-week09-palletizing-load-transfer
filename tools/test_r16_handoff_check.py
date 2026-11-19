import json
import hashlib
from pathlib import Path
import unittest
import numpy as np

from tools.r16_handoff_check import (LAW_CONFIG, SCOPE, dependencies, make_auth,
                                     validate_law_plan, validate_upstream_wrapper)
from experiments.r16_handoff_check.qualification import KEYS, exact_pair, saturation_both


ROOT = Path(__file__).resolve().parents[1]


class Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.plan = json.loads((ROOT / LAW_CONFIG["release"]["plan"]).read_text())

    def auth(self, law="release"):
        config = LAW_CONFIG[law]
        plan = ROOT / config["plan"]
        candidate_plan = ROOT / config["candidate_plan"]
        pins = {config["controller"]: "c" * 64,
                config["candidate_plan"]: hashlib.sha256(candidate_plan.read_bytes()).hexdigest(),
                config["plan"]: hashlib.sha256(plan.read_bytes()).hexdigest()}
        pins.update({path: hashlib.sha256((ROOT / path).read_bytes()).hexdigest()
                     for path in config.get("extra_dependencies", ())})
        return make_auth(law, pins, b"h", b"e")

    def test_plan_is_frozen_complete_and_distinct(self):
        plan = self.plan
        self.assertEqual(plan["status"], "FROZEN_BEFORE_R16_CANDIDATE_OUTCOMES")
        self.assertEqual((len(plan["exposed_cases"]), len(plan["fresh_cases"]),
                          len(plan["impact_cases"])), (22, 6, 7))
        self.assertEqual(len({case["id"] for key in ("exposed_cases", "fresh_cases", "impact_cases")
                              for case in plan[key]}), 35)
        old_masses = {3.45, 4.90, 6.55, 7.75, 7.10, 6.65}
        self.assertTrue(all(case["mass"] not in old_masses for case in plan["fresh_cases"]))
        self.assertEqual({case["claim"] for case in plan["impact_cases"]},
                         {"new", "exposed_sentinel"})
        other = json.loads((ROOT / LAW_CONFIG["normal-phase"]["plan"]).read_text())
        balanced = json.loads((ROOT / LAW_CONFIG["balanced"]["plan"]).read_text())
        self.assertEqual(plan["law"], "release")
        self.assertEqual(other["law"], "normal-phase")
        for key in ("exposed_cases", "fresh_cases", "impact_cases", "thresholds"):
            self.assertEqual(plan[key], other[key])
            self.assertEqual(plan[key], balanced[key])
        self.assertNotEqual(plan["candidate_module"], other["candidate_module"])
        self.assertEqual(balanced["candidate_module"], "experiments.r16_balanced.controller")
        self.assertIs(validate_law_plan(plan, "release"), plan)
        for mutation in (lambda d: d["fresh_cases"].pop(),
                         lambda d: d["fresh_cases"][0].update(id=d["exposed_cases"][0]["id"]),
                         lambda d: d.update(thresholds={}),
                         lambda d: d.update(extra=True)):
            malformed = json.loads(json.dumps(plan)); mutation(malformed)
            with self.assertRaises(ValueError):
                validate_law_plan(malformed, "release")
        contract = json.loads((ROOT / LAW_CONFIG["payload-contract"]["plan"]).read_text())
        self.assertEqual((len(contract["exposed_cases"]), len(contract["fresh_cases"]),
                          len(contract["impact_cases"])), (28, 6, 7))
        self.assertEqual(contract["impact_cases"], plan["impact_cases"])
        self.assertEqual({case["offset"][2] if case["offset"] is not None else None
                          for case in contract["fresh_cases"]}, {None, 0.0, 0.0032, -0.0035, 0.0037})
        self.assertIs(validate_law_plan(contract, "payload-contract"), contract)

    def test_fresh_has_light_mid_heavy_zero_up_down(self):
        cases = self.plan["fresh_cases"]
        self.assertEqual(sorted({case["mass"] for case in cases}), [3.72, 5.38, 7.42])
        z = [case["offset"][2] for case in cases]
        self.assertIn(0.0, z); self.assertTrue(any(value > 0 for value in z)); self.assertTrue(any(value < 0 for value in z))

    def test_impacts_cover_declared_families_and_sentinel(self):
        rows = self.plan["impact_cases"]
        sentinel = next(row for row in rows if row["claim"] == "exposed_sentinel")
        self.assertEqual((sentinel["mass"], sentinel["collision"]),
                         (7.58, [3.905, [8.0, -5.0, -15.0]]))
        times = [row["collision"][0] for row in rows if row["claim"] == "new"]
        self.assertTrue(any(time < 3.9 for time in times)); self.assertTrue(any(time > 4.0 for time in times))
        vectors = [row["collision"][1] for row in rows]
        self.assertTrue(any(v[2] > 0 and v[0] == v[1] == 0 for v in vectors))
        self.assertTrue(any(v[2] < 0 and v[0] == v[1] == 0 for v in vectors))

    def test_upstream_fails_closed_for_incomplete_failed_or_wrong_auth(self):
        auth = self.auth()
        rows = [{"id": case["id"], "passed": True, "gates": {"all": True}}
                for case in self.plan["exposed_cases"]]
        base = {"schema": "r16-portable-handoff-check.v1",
                "scope": SCOPE, "law": "release",
                "pins": auth["source_pins"],
                "result": {"schema": "r16-progressive-stage.v1", "stage": "exposed",
                           "complete": True, "passed": True, "auth": auth, "rows": rows}}
        self.assertEqual(validate_upstream_wrapper(base, "exposed", auth)["stage"], "exposed")
        for mutation in (lambda d: d["result"].update(complete=False),
                         lambda d: d["result"].update(passed=False),
                         lambda d: d["result"].update(stage="fresh"),
                         lambda d: d["result"].update(auth={}),
                         lambda d: d["result"]["rows"].pop(),
                         lambda d: d["result"]["rows"][0].update(passed=False),
                         lambda d: d["result"]["rows"][0].update(gates={}),
                         lambda d: d["result"]["rows"][0].update(id=d["result"]["rows"][1]["id"]),
                         lambda d: d.update(scope="wrong"),
                         lambda d: d.update(pins={}),
                         lambda d: d.update(schema="wrong")):
            document = json.loads(json.dumps(base)); mutation(document)
            with self.assertRaises(ValueError):
                validate_upstream_wrapper(document, "exposed", auth)

    def test_cross_law_chain_and_stage_mismatch_are_rejected(self):
        release = self.auth("release")
        rows = [{"id": case["id"], "passed": True, "gates": {"all": True}}
                for case in self.plan["exposed_cases"]]
        document = {"schema": "r16-portable-handoff-check.v1", "scope": SCOPE,
                    "law": "release", "pins": release["source_pins"],
                    "result": {"schema": "r16-progressive-stage.v1", "stage": "exposed",
                               "complete": True, "passed": True, "auth": release, "rows": rows}}
        with self.assertRaises(ValueError):
            validate_upstream_wrapper(document, "fresh", release)
        with self.assertRaises(ValueError):
            validate_upstream_wrapper(document, "exposed", self.auth("normal-phase"))
        with self.assertRaises(ValueError):
            validate_upstream_wrapper(document, "exposed", self.auth("balanced"))

    def test_public_dependency_surface_has_no_run_basename(self):
        for law in LAW_CONFIG:
            values = dependencies(law)
            self.assertEqual(len(values), len(set(values)))
            self.assertFalse(any(Path(value).name == "run.py" for value in values))
            self.assertIn("experiments/r16_handoff_check/qualification.py", values)
        balanced = dependencies("balanced")
        for required in ("experiments/r16_balanced/coefficients.json",
                         "experiments/r16_release/controller.py",
                         "experiments/r16_release/plan.json",
                         "experiments/r16_normal_phase/controller.py",
                         "experiments/r16_normal_phase/plan.json"):
            self.assertIn(required, balanced)
        auth = self.auth("balanced")
        self.assertEqual(set(auth["supporting_pins"]), set(LAW_CONFIG["balanced"]["extra_dependencies"]))

    def test_contract_identity_and_min_saturation_fail_on_manufactured_mismatch(self):
        log = {key: np.zeros((2, 6)) if key in ("q", "q_d", "tau", "r")
               else np.zeros(2, dtype=bool) if key == "detect" else np.zeros(2)
               for key in KEYS}
        record = {"status": "ok", "cost": {"x": 1.0}, "saturation_pct": 0.0,
                  "false_alarm_times": [], "ctrl_p99_us": 10.0, "registration": None}
        self.assertTrue(exact_pair(record, log, dict(record),
                                   {key: value.copy() for key, value in log.items()})["passed"])
        changed = {key: value.copy() for key, value in log.items()}; changed["tau"][0, 0] = 1.0
        self.assertFalse(exact_pair(record, log, dict(record), changed)["passed"])
        self.assertFalse(saturation_both({"saturation_pct": .1},
                                         {"saturation_pct": 0.0},
                                         {"saturation_pct": .2}))
        self.assertTrue(saturation_both({"saturation_pct": .1},
                                        {"saturation_pct": .1},
                                        {"saturation_pct": .2}))

    def test_thresholds_are_predeclared(self):
        thresholds = self.plan["thresholds"]
        self.assertEqual(set(thresholds), {"g4", "cost", "false_events", "tool",
                                           "saturation", "runtime", "impact"})
        self.assertIn("1.05", thresholds["cost"])
        self.assertIn("20ms", thresholds["false_events"])
        self.assertIn("5ms", thresholds["impact"])


if __name__ == "__main__":
    unittest.main()
