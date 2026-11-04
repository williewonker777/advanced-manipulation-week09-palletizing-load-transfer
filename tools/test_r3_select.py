#!/usr/bin/env python3
"""Standard-library unit tests for conservative R3 selection postprocessing."""
import copy
import json
from pathlib import Path
import tempfile
import unittest

from tools import r3_select as S


ROOT = Path(__file__).resolve().parents[1]


def row(case_id, architecture="torque", variant="R1", mass=4.0,
        scale=1.0, **updates):
    result = {
        "variant": variant, "architecture": architecture, "case_id": case_id,
        "mass": mass, "com": [0.01, -0.01, 0.025], "kind": "reference",
        "collision": None, "pick_offset": None,
        "rms_mm": 10.0 * scale, "max_mm": 10.0 * scale,
        "torque_step_pick_nm": 10.0 * scale,
        "torque_step_place_nm": 10.0 * scale,
        "table_force_place_max_n": 10.0 * scale,
        "saturation_pct": 0.0, "diverged": False, "false_alarms": 0,
        "timing": {"combined": {"wall": {"p99_us": 700.0, "max_us": 900.0}},
                   "outliers": []},
        "g4": {"table_mean_over_mg": 0.6, "table_min_over_mg": 0.3,
               "mean_increase_over_v1": 0.3, "place_peak_over_mg": 1.0},
        "normal_residual": {"joint_RMS_Nm": 1.0},
        "full_cycle_step": {"max_nm": 10.0, "time_s": 3.9,
                            "joint_1based": 2},
    }
    result.update(updates)
    return result


def document(rows):
    return {"metadata": {"completed": True, "source_sha256": {
        f"experiments/r3_generated/{row['variant']}_{row['architecture']}.py": "a" * 64
        for row in rows if row['variant'] != 'R1'}}, "results": rows}


def stability_row(candidate, architecture, source_kind="forced",
                  radius=0.99, relative=0.0005, time=2.5, mass=0.85):
    emitted_kind = {"forced": "forced_r1", "actual": "actual_candidate"}.get(
        source_kind, source_kind)
    return {
        "candidate": candidate, "architecture": architecture,
        "mass_kg": mass, "time_s": time, "sample_index": int(round(time * 1000)),
        "source_kind": emitted_kind,
        "module_sha256": "a" * 64,
        "fd_convergence": {"pass": True},
        "closed_loop": {
            "protocol_radius_pass_lt_1_plus_1e_minus_6": True,
            "spectral_radius": radius, "poles": [[0.9, 0.0]],
            "jacobian_step_diagnostics": {"A_relative_step_change": relative},
        },
        "mechanical_fd": {
            "step_input": {"step_converged": True},
            "previous_applied_input_for_jts": {"step_converged": True},
        },
    }


class R3SelectTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.protocol = json.loads(
            (ROOT / "results/v2_protocol/PROTOCOL_R3.json").read_text())
        cls.r2_protocol = json.loads(
            (ROOT / "results/v2_protocol/PROTOCOL_R2.json").read_text())

    def test_registered_protocol_keeps_exact_r2_gates(self):
        result = S.verify_protocol_gates(self.protocol, self.r2_protocol)
        self.assertEqual(result["status"], "PASS")
        self.assertTrue(all(result["exact_R2_gate_match"].values()))

    def test_stability_requires_full_fd_and_matching_validation_source(self):
        value = stability_row('B_mass', 'torque')
        evidence = {'analyses': [value]}
        result = S.review_stability(evidence, 'B_mass', 'torque', 'forced',
                                    expected_source_sha256='a' * 64)
        self.assertEqual(result['status'], 'PASS')
        result = S.review_stability(evidence, 'B_mass', 'torque', 'forced',
                                    expected_source_sha256='b' * 64)
        self.assertEqual(result['status'], 'FAIL')
        value.pop('fd_convergence')
        result = S.review_stability(evidence, 'B_mass', 'torque', 'forced',
                                    expected_source_sha256='a' * 64)
        self.assertEqual(result['status'], 'FAIL')

    def test_per_case_regression_rejects_even_if_global_worst_improves(self):
        baseline = document([
            row(0, mass=1.0, scale=10.0),
            row(1, mass=2.0, scale=1.0),
        ])
        candidate = document([
            row(0, variant="B_phase", mass=1.0, scale=5.0),
            row(1, variant="B_phase", mass=2.0, scale=1.06),
        ])
        # Candidate global maximum is half the baseline global maximum, but
        # the second matched case violates the mandatory 1.05 ratio.
        costs = S.review_costs(S.pair_results(baseline, candidate))
        self.assertEqual(costs["status"], "FAIL")
        self.assertAlmostEqual(costs["worst_ratio"], 1.06)

    def test_zero_baseline_allows_only_zero(self):
        base = row(0)
        candidate = row(0, variant="B_phase")
        base["rms_mm"] = 0.0
        candidate["rms_mm"] = 0.0
        self.assertEqual(S.review_costs([(base, candidate)])["status"], "PASS")
        candidate["rms_mm"] = 1e-9
        self.assertEqual(S.review_costs([(base, candidate)])["status"], "FAIL")

    def test_timing_800_is_strict_and_1000_max_is_record_only(self):
        under = row(0)
        under["timing"]["combined"]["wall"].update(p99_us=799.999,
                                                       max_us=1200.0)
        result = S.review_timing([under])
        self.assertEqual(result["status"], "PASS")
        self.assertFalse(result["rows"][0]["max_1000_goal_met"])
        equal = row(1)
        equal["timing"]["combined"]["wall"]["p99_us"] = 800.0
        self.assertEqual(S.review_timing([equal])["status"], "FAIL")

    def test_missing_or_bad_forced_stability_fails_closed(self):
        missing = S.review_stability(None, "B_phase", "torque", "forced")
        self.assertEqual(missing["status"], "FAIL")
        bad = {"analyses": [stability_row("B_phase", "torque", relative=0.002)]}
        self.assertEqual(S.review_stability(
            bad, "B_phase", "torque", "forced")["status"], "FAIL")
        wrong_label = {"analyses": [stability_row("B_mass", "torque")]}
        self.assertEqual(S.review_stability(
            wrong_label, "B_phase", "torque", "forced")["status"], "FAIL")

    def test_actual_stability_is_pending_not_pass(self):
        result = S.review_stability(None, "B_phase", "torque", "actual")
        self.assertEqual(result["status"], "pending")

    def test_phase_balance_and_perturbation_keys_are_part_of_match(self):
        baseline = row(0, kind="collision", collision=[1.453, 11.0, "+x"],
                       collision_phase="lift", perturb_seed=9301,
                       perturb_family="noise", perturb_params={"sigma": 1.0})
        candidate = copy.deepcopy(baseline)
        candidate["variant"] = "B_phase"
        candidate["collision_phase"] = "place"
        with self.assertRaisesRegex(ValueError, "matched conditions differ"):
            S.pair_results(document([baseline]), document([candidate]))
        candidate["collision_phase"] = "lift"
        candidate["perturb_seed"] = 9302
        with self.assertRaisesRegex(ValueError, "matched conditions differ"):
            S.pair_results(document([baseline]), document([candidate]))

    def test_priority_selection_and_actual_point_readiness(self):
        baselines = {
            "servo": document([row(i, "servo", mass=condition["mass"],
                                   com=list(condition["com"]))
                               for i, condition in enumerate(self.protocol["validation"])]),
            "torque": document([row(i, "torque", mass=condition["mass"],
                                    com=list(condition["com"]))
                                for i, condition in enumerate(self.protocol["validation"])]),
        }
        candidates = {}
        forced_rows = []
        for spec in self.protocol["candidate_grid"]:
            mode = spec["architecture"]
            scale = 0.85 if mode == "torque" else 0.95
            candidates[spec["id"]] = document([
                row(i, mode, spec["id"], mass=condition["mass"],
                    com=list(condition["com"]), scale=scale,
                    specification=spec)
                for i, condition in enumerate(self.protocol["validation"])
            ])
            for mass, time in S._protocol_forced_points(self.protocol, mode):
                forced_rows.append(stability_row(spec["id"], mode,
                                                 mass=mass, time=time))
        review = S.build_validation_review(
            self.protocol, self.r2_protocol, baselines, candidates,
            {"analyses": forced_rows})
        challengers = S.build_challengers(review)
        self.assertEqual(challengers["architectures"]["servo"]["selected"],
                         "A_handoff_long")
        self.assertEqual(challengers["architectures"]["torque"]["selected"],
                         "B_mass_phase")
        self.assertFalse(challengers["architectures"]["servo"]["development_ready"])
        self.assertEqual(challengers["architectures"]["servo"]["actual_stability_status"],
                         "pending")

    def test_diagnostic_challenger_is_irreversibly_ineligible(self):
        review = {"candidates": [
            {"candidate": "A_handoff_long", "architecture": "servo",
             "validation_status": "rejected", "validation_eligible": False,
             "cost_score": 1.06},
            {"candidate": "A_handoff_short", "architecture": "servo",
             "validation_status": "rejected", "validation_eligible": False,
             "cost_score": 1.08},
        ]}
        challengers = S.build_challengers(review)
        selected = challengers["architectures"]["servo"]
        self.assertEqual(selected["selected"], "A_handoff_long")
        self.assertTrue(selected["irreversibly_ineligible_for_adoption"])
        stages = {name: {"status": "PASS"}
                  for name in ("development", "d4", "perturbations", "holdout")}
        final = S.finalize_architectures(challengers, {"servo": stages})
        self.assertEqual(final["architectures"]["servo"]["selection"], "R1")

    def test_final_modes_are_independent_and_no_runner_up_promotion(self):
        challengers = {"architectures": {
            "servo": {"selected": "A_handoff_long", "validation_eligible": True,
                      "actual_stability_status": "PASS",
                      "irreversibly_ineligible_for_adoption": False},
            "torque": {"selected": "B_mass_phase", "validation_eligible": True,
                       "actual_stability_status": "PASS",
                       "irreversibly_ineligible_for_adoption": False},
        }}
        passed = {name: {"status": "PASS"}
                  for name in ("development", "d4", "perturbations", "holdout")}
        torque = copy.deepcopy(passed)
        torque["development"] = {"status": "FAIL"}
        final = S.finalize_architectures(
            challengers, {"servo": passed, "torque": torque})
        self.assertEqual(final["architectures"]["servo"]["selection"],
                         "A_handoff_long")
        self.assertEqual(final["architectures"]["torque"]["selection"], "R1")
        self.assertNotEqual(final["architectures"]["torque"]["selection"],
                            "B_phase")

    def test_development_rejects_lost_detection_and_extra_false_alarm(self):
        base_collision = row(0, kind="collision", collision=[1.453, 11.0, "+x"],
                             collision_outcome="detected")
        candidate_collision = copy.deepcopy(base_collision)
        candidate_collision.update(variant="B_phase", collision_outcome="missed")
        base_reference = row(1)
        candidate_reference = row(1, variant="B_phase", false_alarms=1)
        result = S.review_development(
            document([base_collision, base_reference]),
            document([candidate_collision, candidate_reference]))
        self.assertEqual(result["status"], "FAIL")
        self.assertTrue(any(item["lost_R1_detection"] for item in result["detection"]))
        self.assertTrue(any(item["reference_false_alarm"] for item in result["detection"]))

    def test_development_and_holdout_reject_divergence_or_saturation_regression(self):
        baseline = document([row(0)])
        diverged = document([row(0, variant="B_phase", diverged=True)])
        saturated = document([row(0, variant="B_phase", saturation_pct=0.01)])
        self.assertEqual(S.review_development(baseline, diverged)["status"], "FAIL")
        self.assertEqual(S.review_holdout(baseline, saturated)["status"], "FAIL")

    def test_perturbations_include_detection_false_alarm_and_safety_guards(self):
        baseline_row = row(0, architecture="servo", perturb_seed=9301,
                           perturb_family="noise",
                           perturb_params={"sigma": 1.0})
        candidate_row = row(0, architecture="servo", variant="A_handoff_long",
                            scale=0.85, perturb_seed=9301,
                            perturb_family="noise", perturb_params={"sigma": 1.0},
                            false_alarms=1, saturation_pct=0.01)
        result = S.review_perturbations(
            document([baseline_row]), document([candidate_row]), "servo")
        self.assertEqual(result["status"], "FAIL")
        self.assertEqual(result["detection"]["status"], "FAIL")
        self.assertEqual(result["safety"]["status"], "FAIL")

    def test_actual_stability_requires_exact_capture_points(self):
        analyses = [stability_row("B_phase", "torque", "actual",
                                  mass=5.47, time=2.5)]
        capture = {"candidate": "B_phase", "architecture": "torque",
                   "source_kind": "actual_candidate", "mass_kg": 5.47,
                   "sample_indices": [2500]}
        good = {"analyses": analyses, "captures": [capture]}
        self.assertEqual(S.review_stability(
            good, "B_phase", "torque", "actual")["status"], "PASS")
        capture["sample_indices"].append(3720)
        bad = S.review_stability(good, "B_phase", "torque", "actual")
        self.assertEqual(bad["status"], "FAIL")
        self.assertEqual(bad["coverage"]["missing"], [(5.47, 3720)])

    def test_real_forced_stability_preserves_all_21_radius_failures(self):
        artifact = json.loads(
            (ROOT / "results/v2_protocol/R3_FORCED_STABILITY.json").read_text())
        failures = 0
        radius_failures = 0
        for spec in self.protocol["candidate_grid"]:
            result = S.review_stability(
                artifact, spec["id"], spec["architecture"], "forced",
                protocol=self.protocol)
            self.assertEqual(result["status"], "FAIL")
            self.assertTrue(result["coverage"]["pass"])
            failures += sum(not row["pass"] for row in result["rows"])
            radius_failures += sum(
                not row["closed_loop"]["protocol_radius_pass_lt_1_plus_1e_minus_6"]
                for row in artifact["analyses"] if row["candidate"] == spec["id"])
        self.assertEqual(radius_failures, 21)
        self.assertGreaterEqual(failures, radius_failures)

    def test_d4_enforces_counts_costs_and_timing(self):
        baseline_rows = []
        candidate_rows = []
        for index in range(180):
            outcome = "detected" if index < 111 else "missed"
            collision = [1.453 + index * 1e-6, 11.0, "+x"]
            baseline_rows.append(row(
                index, kind="collision", collision=collision,
                collision_outcome=outcome))
            candidate_rows.append(row(
                index, variant="B_phase", kind="collision",
                collision=collision, collision_outcome=outcome))
        baseline_rows.append(row(180))
        candidate_rows.append(row(180, variant="B_phase"))
        result = S.review_d4(document(baseline_rows), document(candidate_rows),
                             "torque", expected_rows=181)
        self.assertEqual(result["status"], "PASS")
        candidate_rows[0]["diverged"] = True
        result = S.review_d4(document(baseline_rows), document(candidate_rows),
                             "torque", expected_rows=181)
        self.assertEqual(result["status"], "FAIL")
        self.assertEqual(result["safety"]["status"], "FAIL")
        candidate_rows[0]["diverged"] = False
        # The servo aggregate ceiling cannot excuse moving saturation onto
        # a previously unsaturated R1 case.
        for item in baseline_rows + candidate_rows:
            item["architecture"] = "servo"
        baseline_rows[1]["saturation_pct"] = .1
        candidate_rows[0]["saturation_pct"] = .1
        result = S.review_d4(document(baseline_rows), document(candidate_rows),
                             "servo", expected_rows=181)
        self.assertEqual(result["status"], "FAIL")
        self.assertEqual(result["safety"]["status"], "FAIL")
        candidate_rows[0]["saturation_pct"] = 0.
        candidate_rows[1]["saturation_pct"] = .1
        result = S.review_d4(document(baseline_rows), document(candidate_rows),
                             "servo", expected_rows=181)
        self.assertEqual(result["status"], "PASS")
        candidate_rows[-1]["torque_step_place_nm"] = 10.6
        result = S.review_d4(document(baseline_rows), document(candidate_rows),
                             "servo", expected_rows=181)
        self.assertEqual(result["status"], "FAIL")
        self.assertEqual(result["costs"]["status"], "FAIL")

    def test_redesign_requires_prior_reason_spec_and_new_seed(self):
        spec = {"id": "B_redesign", "architecture": "torque"}
        baseline = document([
            row(i, mass=condition["mass"], com=list(condition["com"]))
            for i, condition in enumerate(self.protocol["redesign_validation"])
        ])
        candidate = document([
            row(i, variant="B_redesign", mass=condition["mass"],
                com=list(condition["com"]), scale=0.85,
                specification=spec)
            for i, condition in enumerate(self.protocol["redesign_validation"])
        ])
        forced = {"analyses": [
            stability_row("B_redesign", "torque", mass=mass, time=time)
            for mass, time in S._protocol_forced_points(self.protocol, "torque")
        ]}
        bad = S.review_redesign_candidate(
            self.protocol, baseline, candidate, forced,
            {"redesign_number": 1, "reason": "", "spec": spec,
             "seed": 369812})
        self.assertEqual(bad["validation_status"], "FAIL")
        good = S.review_redesign_candidate(
            self.protocol, baseline, candidate, forced,
            {"redesign_number": 1, "reason": "registered physical cause",
             "spec": spec, "seed": 369812})
        self.assertTrue(good["validation_eligible"])
        self.assertFalse(good["automatic_promotion"])
        self.assertTrue(good["root_decision_required"])

    def test_cli_writes_only_when_explicitly_requested(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            protocol_path = root / "PROTOCOL_R3.json"
            protocol_path.write_bytes(
                (ROOT / "results/v2_protocol/PROTOCOL_R3.json").read_bytes())
            r2_path = root / "PROTOCOL_R2.json"
            r2_path.write_bytes(
                (ROOT / "results/v2_protocol/PROTOCOL_R2.json").read_bytes())
            forced_path = root / "R3_FORCED_STABILITY.json"
            forced_path.write_text(json.dumps({"analyses": []}))
            code = S.main(["--protocol", str(protocol_path), "--r2-protocol",
                           str(r2_path), "--results-dir", str(root),
                           "--forced-stability", str(forced_path), "--write"])
            self.assertEqual(code, 0)
            self.assertTrue((root / "R3_VALIDATION_REVIEW.json").is_file())
            self.assertTrue((root / "R3_CHALLENGERS.json").is_file())
            review = json.loads((root / "R3_VALIDATION_REVIEW.json").read_text())
            self.assertEqual(review["scope"],
                             "validation eligibility/provisional selection; not final adoption")


if __name__ == "__main__":
    unittest.main()
