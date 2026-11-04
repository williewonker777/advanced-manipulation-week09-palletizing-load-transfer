import importlib.util
from pathlib import Path
import tempfile
import unittest


SPEC = importlib.util.spec_from_file_location("v2_report", Path(__file__).with_name("v2_report.py"))
v2_report = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(v2_report)


def collision(outcome="detected", false_alarms=0, variant="v2", architecture="servo"):
    return {"collision": [3.0, [0, 10, 0]], "collision_outcome": outcome,
            "false_alarms": false_alarms, "variant": variant, "architecture": architecture}


class AggregationTests(unittest.TestCase):
    def test_outcome_counts_keep_all_bad_cases(self):
        rows = [
            {"collision": None, "false_alarms": 3},
            collision("detected", 2), collision("missed", 4),
            collision("alarm_before_collision", 5), collision("indistinguishable", 6),
        ]
        got = v2_report.outcome_counts(rows)
        self.assertEqual(got["no_collision_false_alarms"], 3)
        self.assertEqual(got["collision_extra_false_alarms"], 17)
        self.assertEqual(got["misses"], 1)
        self.assertEqual(got["masked"], 2)
        self.assertEqual(got["collisions"], 4)

    def test_offsets_are_not_counted_as_pure_references(self):
        rows = [{"kind": "reference", "collision": None, "false_alarms": 2},
                {"kind": "offset", "collision": None, "pick_offset": [0.01, 0, 0],
                 "false_alarms": 3, "tool_contact_peak_n": 4.0,
                 "tool_contact_delay_ms": None}]
        got = v2_report.outcome_counts(rows)
        self.assertEqual(got["references"], 1)
        self.assertEqual(got["no_collision_false_alarms"], 2)
        self.assertEqual(got["offset_episodes"], 1)
        self.assertEqual(got["offset_false_alarm_rises"], 3)
        self.assertEqual(got["offset_tool_contact_misses"], 1)

    def test_grouped_counts_accepts_perturbation_version_label(self):
        got = v2_report.grouped_counts([
            {"version": "v2", "architecture": "servo", "kind": "reference",
             "collision": None, "false_alarms": 0}])
        self.assertIn("v2/servo", got)

    def test_unclassified_collision_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "recognized outcome"):
            v2_report.outcome_counts([collision("conveniently-dropped")])

    def test_d4_goal_requires_more_detections_without_extra_false_alarms(self):
        counts = {
            "v1/servo": {"detected": 3, "collision_extra_false_alarms": 2,
                          "no_collision_false_alarms": 0},
            "v2/servo": {"detected": 4, "collision_extra_false_alarms": 2,
                          "no_collision_false_alarms": 0},
            "v1/torque": {"detected": 5, "collision_extra_false_alarms": 1,
                           "no_collision_false_alarms": 0},
            "L3/torque": {"detected": 6, "collision_extra_false_alarms": 0,
                           "no_collision_false_alarms": 0},
        }
        self.assertTrue(v2_report.d4_goal(counts)["G3"])
        counts["v2/servo"]["collision_extra_false_alarms"] = 3
        self.assertFalse(v2_report.d4_goal(counts)["G3"])


class FormulaTests(unittest.TestCase):
    def test_official_goals_use_transport_8kg_and_strict_delay(self):
        def combo(delay10, delay15):
            return {"collisions": [{"mass": 8.0, "time": 3.0, "force10": delay10,
                                     "force15": delay15}], "control": {"8.0": {}}}
        official = {"combinations": {
            "servo_v2_v2": combo("42", "68"), "torque_v2_v2": combo("놓침", "69")}}
        got = v2_report.official_goals(official)
        self.assertFalse(got["G1"])
        self.assertFalse(got["G2"])
        self.assertTrue(got["per_architecture"]["servo"]["G1_8kg_15N_le_68ms"])

    def test_subthreshold_cpu_is_labeled_as_row_aggregate(self):
        rows = [{"variant": "v2", "architecture": "servo", "case_id": 1,
                 "timing": {"combined": {"wall": {"max_us": 999.0},
                                           "thread_cpu": {"max_us": 600.0}},
                            "outliers": []}}]
        got = v2_report.timing_extremes(rows)
        self.assertTrue(got["strict_below_1ms"])
        self.assertEqual(got["retained_outlier_records"], 0)
        self.assertIn("row aggregate", got["cpu_at_worst_wall"]["semantics"])

    def test_stress_timing_failure_cannot_be_hidden_by_serial_timing(self):
        def row(maximum):
            return {"timing": {"combined": {"wall": {"max_us": maximum},
                                              "thread_cpu": {"max_us": maximum - 10}},
                               "outliers": []}}
        got = v2_report.timing_extremes([("timing", row(700)), ("d4", row(1400))])
        self.assertFalse(got["strict_below_1ms"])
        self.assertEqual(got["observed_wall_max_us"], 1400)
        self.assertEqual(got["worst_dataset"], "d4")

    def test_v1_only_outlier_does_not_fail_final_v2_g5(self):
        def row(version, architecture, maximum, cpu, outliers=()):
            return {"version": version, "architecture": architecture, "case_id": 1,
                    "timing": {"combined": {
                        "wall": {"median_us": 100.0, "p99_us": maximum - 20, "max_us": maximum},
                        "thread_cpu": {"median_us": 90.0, "p99_us": cpu - 10, "max_us": cpu}},
                        "outliers": list(outliers)}}
        v1_event = {"cycle": 3, "combined": {"wall_ns": 1_400_000, "cpu_ns": 300_000,
                                              "voluntary_context_switches": 1,
                                              "involuntary_context_switches": 0}}
        named = [("d4", row("v1", "servo", 1400.0, 500.0, [v1_event])),
                 ("d4", row("v1", "torque", 800.0, 700.0)),
                 ("holdout", row("v2", "servo", 900.0, 850.0)),
                 ("perturbations", row("v2", "torque", 950.0, 880.0))]
        groups = v2_report.timing_group_summaries(named)
        goal = v2_report.final_timing_goal(groups)
        self.assertTrue(goal["G5"])
        self.assertEqual(groups["v1/servo"]["observed_wall_max_us"], 1400.0)
        self.assertEqual(goal["per_architecture"]["servo"]["observed_wall_max_us"], 900.0)
        self.assertEqual(goal["per_architecture"]["torque"]["worst_dataset"], "perturbations")

    def test_exact_worst_wall_event_uses_same_cycle_cpu_and_context_switches(self):
        event = {"cycle": 7, "combined": {"wall_ns": 1_500_000, "cpu_ns": 410_000,
                                          "voluntary_context_switches": 1,
                                          "involuntary_context_switches": 2}}
        row = {"timing": {"combined": {"wall": {"max_us": 1500.0},
                                         "thread_cpu": {"max_us": 999.0}},
                          "outliers": [event]}}
        got = v2_report.timing_extremes([("holdout", row)])
        detail = got["cpu_at_worst_wall"]
        self.assertEqual(detail["value_us"], 410.0)
        self.assertEqual(detail["voluntary_context_switches"], 1)
        self.assertEqual(detail["involuntary_context_switches"], 2)
        self.assertIn("exact worst-wall cycle", detail["semantics"])

    def test_baseline_parser_reads_control_rows_only(self):
        text = "8.0kg | 6.14 | 14.00 | 0.0 | 57.5 / 73.8 | 101 N | x | 5 / 2 / 0 회 | 정상\n" \
               "8.0kg 3.0s | 놓침 | 68 | 26 | 17 | 8회 | 15 N\n"
        self.assertEqual(v2_report.baseline_cells(text)[8.0][:2], ["6.14", "14.00"])

    def test_full_evaluator_summary_has_all_requested_table_fields(self):
        text = ("8kg | 6.14 | 14.00 | 0.0 | 57.5 / 73.8 | 101 N | 6 | 5 / 2 / 0 회 | 정상\n"
                "8kg 3.00s | 놓침 | 68 | 26 | 17 | 8회 | 15 N\n")
        got = v2_report.evaluator_summary(text)
        self.assertEqual(got, {"rms_max": "6.14/14.00", "pick_place": "57.5/73.8",
                               "table_n": "101", "false_alarms": "5/2/0",
                               "transport_15n": "68", "transport_10n": "놓침",
                               "minimum": "15 N"})

    def test_g4_uses_matched_8kg_last20_formulas(self):
        rows = []
        for architecture in ("servo", "torque"):
            rows += [
                {"variant": "v1", "architecture": architecture, "mass": 8.0,
                 "kind": "reference", "g4": {"table_mean_over_mg": .20,
                                                "table_min_over_mg": .1,
                                                "place_peak_over_mg": 1.0}},
                {"variant": "L3", "architecture": architecture, "mass": 8.0,
                 "kind": "reference", "g4": {"table_mean_over_mg": .55,
                                                "table_min_over_mg": .3,
                                                "place_peak_over_mg": 1.05}},
            ]
        got = v2_report.g4_goal(rows)
        self.assertTrue(got["G4"])
        self.assertAlmostEqual(got["per_architecture"]["servo"]["mean_increase_over_mg"], .35)

    def test_physical_parity_compares_matched_v1_and_final_labels(self):
        common = {"architecture": "servo", "case_id": 4, "mass": 3.7,
                  "com": [0.01, 0, 0.02], "kind": "reference", "phase": "none",
                  "direction": "none", "force_n": 0, "pick_offset": None,
                  "rms_mm": 1.2, "max_mm": 3.4, "saturation_pct": 0,
                  "diverged": False, "torque_step_pick_nm": 9.0,
                  "torque_step_place_nm": 12.0, "table_force_place_max_n": 40.0,
                  "tool_contact_peak_n": 0.0, "tool_contact_first_s": None}
        rows = [{**common, "version": "v1"}, {**common, "version": "v2"}]
        got = v2_report.matched_physical_comparison(rows, references_only=True)
        self.assertEqual(got["matched_pairs"], 1)
        self.assertTrue(got["unchanged"])

    def test_deck_has_16_explicitly_mapped_numeric_slides(self):
        count = {"detected": 7, "misses": 2, "masked": 1,
                 "collision_extra_false_alarms": 3, "no_collision_false_alarms": 0}
        g4_item = {"v1": {"table_mean_over_mg": .1},
                   "v2": {"table_mean_over_mg": .6, "table_min_over_mg": .3,
                          "place_peak_over_mg": 1.0},
                   "mean_increase_over_mg": .5, "pass": True}
        cpu = {"value_us": 400.0, "voluntary_context_switches": 1,
               "involuntary_context_switches": 0, "semantics": "exact"}
        assessment = {
            "final_variant": "L3",
            "supplemental_counts": {
                "fresh_development": {"v2/servo": count},
                "d4": {"v2/servo": count},
                "sealed_holdout": {"v2/servo": count}},
            "g4": {"per_architecture": {"servo": g4_item, "torque": g4_item}},
            "timing": {
                "final_v2_goal": {"per_architecture": {
                    "servo": {"observed_wall_max_us": 800.0, "cpu_at_worst_wall": cpu,
                              "strict_below_1ms": True},
                    "torque": {"observed_wall_max_us": 900.0, "cpu_at_worst_wall": cpu,
                               "strict_below_1ms": True}}},
                "all_versions_global_comparison": {
                    "observed_wall_max_us": 1400.0, "worst_dataset": "d4",
                    "cpu_at_worst_wall": cpu}},
            "goals": {"G1": True, "G2": False, "G3": True, "G4": True,
                      "G5": True, "G6": "unchanged", "G7": "not solved"},
        }
        figures = {name: Path(f"{name}.png") for name in
                   ("official", "control", "d4", "holdout", "timing", "goals", "torque_residual")}
        slides = v2_report.build_deck_slides(
            assessment, {"A": ["A numeric"], "B": ["B numeric"]}, figures)
        self.assertEqual(len(slides), 16)
        self.assertEqual(slides[6]["image"], figures["official"])
        self.assertIsNone(slides[8]["image"])
        self.assertEqual(slides[9]["image"], figures["d4"])
        self.assertEqual(slides[12]["image"], figures["timing"])
        self.assertEqual(slides[15]["image"], figures["torque_residual"])
        self.assertIn("detected 7", " ".join(slides[8]["lines"]))
        self.assertIn("0.600", " ".join(slides[11]["lines"]))
        self.assertIn("800.0 us", " ".join(slides[12]["lines"]))
        self.assertIn("entire state box", " ".join(slides[5]["lines"]))


class ValidationTests(unittest.TestCase):
    def documents(self):
        docs = {key: {} for key in v2_report.REQUIRED_JSON}
        docs["official"] = {"official": True, "source": "unchanged evaluate.py",
                            "combinations": {str(i): {} for i in range(8)}}
        for key, count in v2_report.EXPECTED_ROWS.items():
            docs[key] = {"results": [{} for _ in range(count)]}
        docs["holdout"]["metadata"] = {"official": False}
        docs["perturbations"]["nominal_parity"] = {"pass": True}
        return docs

    def test_exact_final_row_counts_are_required(self):
        docs = self.documents()
        docs["d4"]["results"].pop()
        with self.assertRaisesRegex(ValueError, "1540"):
            v2_report.validate_inputs(docs)

    def test_perturbation_nominal_parity_is_required(self):
        docs = self.documents()
        docs["perturbations"].pop("nominal_parity")
        with self.assertRaisesRegex(ValueError, "nominal_parity"):
            v2_report.validate_inputs(docs)

    def test_missing_final_file_fails_instead_of_making_placeholders(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(FileNotFoundError, "required final input"):
                v2_report.load_final_inputs(Path(directory))


if __name__ == "__main__":
    unittest.main()
