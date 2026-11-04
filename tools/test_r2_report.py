import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest


SPEC = importlib.util.spec_from_file_location("r2_report", Path(__file__).with_name("r2_report.py"))
r2_report = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(r2_report)


class ManifestFixture:
    def __init__(self):
        self.temp = tempfile.TemporaryDirectory(dir=r2_report.ROOT / "results")
        self.directory = Path(self.temp.name)
        self.relative = self.directory.relative_to(r2_report.ROOT)

    def close(self):
        self.temp.cleanup()

    def write(self, name, value):
        path = self.directory / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")
        return {
            "path": path.relative_to(r2_report.ROOT).as_posix(),
            "provenance": "fresh",
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }

    def manifest(self):
        generic = {"schema": "fixture.v1"}
        validation = {
            "official_used_for_selection": False,
            "holdout_opened": False,
            "results": [{"id": "C1-a"}],
        }
        development_review = {
            "official_used_for_selection": False, "holdout_opened": False,
            "no_threshold_or_gain_tuning_after_collision": True,
            "observer_results": [], "control_eligible": [],
        }
        common_reference = {
            "official": False, "provenance": "immutable fixture, no new episode",
            "R2_source_alias": "R1 exact bytes",
            "results": [{"variant": variant, "architecture": architecture}
                        for variant in ("base", "v1", "R1")
                        for architecture in ("servo", "torque")],
        }
        parity = {"official": False, "pairs": 156, "all_physical_pairs_exact": True,
                  "comparison": "physical arrays only; alarm not claimed equal",
                  "keys": ["tau", "q"],
                  "results": [{"exact": True} for _ in range(156)]}
        lie_comparison = {"official": False, "selection_unchanged": True,
                          "results": [{} for _ in range(4)]}
        environment = {"original_entire_tree_unchanged": True,
                       "official_venv_packages_unchanged": True,
                       "packages_added": [], "packages_removed": []}
        stability = {
            "summary": {"total_cases": 300, "strict_stable_cases": 190,
                        "protocol_radius_pass_cases": 222},
            "all_cases": [[f"C{index % 6 + 1}", .9] for index in range(300)],
            "case_columns": ["candidate", "spectral_radius"],
            "uncertainty": "finite local fixture",
        }
        full_stability = {"analyses": [{"poles": []} for _ in range(300)],
                          "summary": stability["summary"],
                          "scope_warning": "finite local fixture"}
        prne = {
            "summary": {"all_protocol_gates_pass": True, "maximum_error": 1e-8},
            "qualification": "finite sampled identities",
            "cases": [{"id": index} for index in range(9)],
        }
        r1_stability = {
            "summary": {"total_cases": 12, "strict_stable_cases": 11},
            "analyses": [{"spectral_radius": 1.001} for _ in range(12)],
            "global_scope_warning": "not a global guarantee",
        }
        allocation = {
            "rows": [
                {"method": "full columns", "thread_median_us": 825,
                 "thread_p99_us": 1522, "allocation_peak_bytes": 11000,
                 "answer": [1.0, 2.0]},
                {"method": "reverse VJP", "thread_median_us": 178,
                 "thread_p99_us": 206, "allocation_peak_bytes": 10736,
                 "answer": [1.0, 2.0]},
            ],
            "global_GC_settings_changed": False, "plant_episodes": 0,
            "warning": "helper microbenchmark only",
        }
        adoption = {"decision": "retain_r1", "holdout_opened": False,
                    "official_used_for_selection": False,
                    "post_holdout_or_official_tuning_allowed": False,
                    "redesigns_used": 0}
        freeze = {"holdout_opened": False, "official_used_for_selection": False,
                  "post_holdout_and_official_tuning_allowed": False}
        protocol = {"remote_operations_allowed": 0}
        artifacts = {
            "p0": self.write("R2_P0_BASELINE.json", generic),
            "protocol": self.write("PROTOCOL_R2.json", protocol),
            "validation_review": self.write("R2_VALIDATION_REVIEW.json", validation),
            "prne_math": self.write("R2_PRNE_MATH.json", prne),
            "prne_normal": self.write("R2_PRNE_NORMAL.json", generic),
            "stability_full": self.write("R2_CANDIDATE_STABILITY.json", full_stability),
            "stability_compact": self.write("R2_CANDIDATE_STABILITY_COMPACT.json", stability),
            "r1_stability": self.write("R2_R1_STABILITY.json", r1_stability),
            "allocation_comparison": self.write("R2_ALLOCATION_COMPARISON.json", allocation),
            "common_reference": self.write("R2_COMMON_REFERENCE.json", common_reference),
            "control_detector_parity": self.write("R2_CONTROL_DETECTOR_PARITY.json", parity),
            "lie_comparison": self.write("R2_LIE_COMPARISON.json", lie_comparison),
            "environment_integrity": self.write("R2_ENVIRONMENT_INTEGRITY.json", environment),
            "development": [self.write("R2_DEVELOPMENT_R1.json", generic),
                            self.write("R2_DEVELOPMENT_C5.json", generic)],
            "development_review": self.write("R2_DEVELOPMENT_REVIEW.json", development_review),
            "d4": self.write("R2_D4_base.json", {"results": [{} for _ in range(770)]}),
            "d4_reused": self.write("S5_D4.json", {"results": [{} for _ in range(1540)]}),
            "perturbations": self.write("R2_PERTURBATIONS.json", {
                "metadata": {"nominal_parity": [{"passed": True} for _ in range(8)]},
                "results": [{} for _ in range(272)],
            }),
            "holdout": self.write("R2_HOLDOUT.json", {
                "metadata": {"one_batch": True}, "results": [{} for _ in range(152)]}),
            "holdout_opened": self.write("R2_HOLDOUT_OPENED.json", {"batch_attempt": 1}),
            "preholdout_review": self.write("R2_PREHOLDOUT_REVIEW.json", {
                "holdout_opened": False, "official_exposed": False,
                "source_alias_R2_R1": True}),
            "holdout_review": self.write("R2_HOLDOUT_REVIEW.json", {
                "one_batch": True, "selection_unchanged": True,
                "final_R2_alias": "R1", "no_tuning_after_holdout": True}),
            "freeze": self.write("R2_FREEZE.json", freeze),
            "adoption": self.write("R2_ADOPTION.json", adoption),
            "official": self.write("R2_OFFICIAL_REGRESSION.json", {
                "official": True, "results": {"servo": {}, "torque": {}},
                "regression_errors": [], "R1_compatibility_pass": True,
                "holdout_sha256": "fixture", "tuning_after_exposure": False}),
            "official_started": self.write("R2_OFFICIAL_STARTED.json", {
                "tuning_or_repair_after_official": False}),
            "timing_review": self.write("R2_TIMING_REVIEW.json", {
                "official": False, "strict_goal_met": False,
                "global_GC_settings_changed": False,
                "results": [
                    {"architecture": "servo", "rows": 102, "cycles": 612000,
                     "largest_row_p99_us": 300.0, "maximum_wall_us": 5000.0,
                     "same_cycle_thread_CPU_us": 4990.0, "overrun_cycles": 1},
                    {"architecture": "torque", "rows": 102, "cycles": 612000,
                     "largest_row_p99_us": 350.0, "maximum_wall_us": 6000.0,
                     "same_cycle_thread_CPU_us": 5990.0, "overrun_cycles": 5},
                ]}),
        }
        artifacts["d4_reused"]["provenance"] = "reused_exact_source"
        artifacts["common_reference"]["provenance"] = "reused_exact_source"
        metrics = []
        for index in range(1, 7):
            metrics.append({
                "id": f"T{index}", "name": f"fixture {index}", "unit": "mm",
                "condition": "matched fixture", "baseline": 3.0, "v1": 2.0,
                "r1": 1.0, "r2": .9, "r2_status": "met",
                "source_keys": ["development"],
            })
        candidates = []
        for index in range(1, 7):
            candidates.append({
                "id": f"C{index}-fixture", "family": f"C{index}",
                "method": f"method {index}", "decision": "reject",
                "gate_results": ["strict gate failed"],
                "measured_effect": "negative fixture", "cost": "fixture cost",
                "source_keys": ["validation_review"],
            })
        method_documents = []
        for name in ("docs/R2_METHODS.md", "docs/R2_ROBUST_DERIVATION.md",
                     "docs/LITERATURE.md"):
            path = r2_report.ROOT / name
            method_documents.append({"path": name,
                                     "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
        return {
            "schema": "R2_REPORT_MANIFEST.v1", "status": "frozen",
            "remote_operations": 0,
            "publication": {"state": "local_staging_only", "push_approved": False},
            "artifacts": artifacts, "metrics": metrics,
            "candidate_decisions": candidates,
            "lie_group_usage": [{
                "method": "pRNE", "section": "Chapter 7",
                "formula": "Mdot = C + C^T",
                "plant_adaptation": "elastic joint, 1 ms ZOH, one-cycle delay",
                "euclidean_comparison": "no benefit", "decision": "reject",
                "source_keys": ["prne_math", "development"],
            }],
            "misses_and_outliers": {"summary": ["all retained"],
                                    "unmet": ["fixture miss"],
                                    "outliers_retained": ["fixture outlier"],
                                    "rows": [1]},
            "verification": {"executed": ["fixture validation"],
                             "not_executed": ["hardware"],
                             "resources": ["CPU only"]},
            "method_documents": method_documents,
            "literature_updates": [{"title": "primary fixture",
                                    "url": "https://example.org/paper",
                                    "verified": True}],
            "changed_paths": ["tools/r2_report.py"],
            "limitations": ["one holdout mass overlaps one challenger training mass",
                            "isolated matched-ridge comparison has zero effect"],
            "conclusion": "R1 유지",
        }


class R2ReportTests(unittest.TestCase):
    def setUp(self):
        self.fixture = ManifestFixture()
        self.manifest = self.fixture.manifest()

    def tearDown(self):
        self.fixture.close()

    def test_complete_fixture_validates_and_retains_all_compact_cases(self):
        context = r2_report.validate_manifest(self.manifest)
        self.assertEqual(len(context["stability"]["all_cases"]), 300)
        self.assertEqual([row["id"] for row in context["metrics"]],
                         ["T1", "T2", "T3", "T4", "T5", "T6"])

    def test_missing_final_artifact_is_rejected(self):
        manifest = copy.deepcopy(self.manifest)
        del manifest["artifacts"]["holdout"]
        with self.assertRaisesRegex(ValueError, "artifact key mismatch"):
            r2_report.validate_manifest(manifest)

    def test_tampered_artifact_hash_is_rejected(self):
        manifest = copy.deepcopy(self.manifest)
        manifest["artifacts"]["freeze"]["sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "hash mismatch"):
            r2_report.validate_manifest(manifest)

    def test_official_selection_leak_is_rejected(self):
        entry = self.manifest["artifacts"]["validation_review"]
        path = r2_report.ROOT / entry["path"]
        value = json.loads(path.read_text())
        value["official_used_for_selection"] = True
        path.write_text(json.dumps(value))
        entry["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
        with self.assertRaisesRegex(ValueError, "must not be used"):
            r2_report.validate_manifest(self.manifest)

    def test_all_candidate_families_are_required(self):
        manifest = copy.deepcopy(self.manifest)
        manifest["candidate_decisions"] = manifest["candidate_decisions"][:-1]
        with self.assertRaisesRegex(ValueError, "omit registered families"):
            r2_report.validate_manifest(manifest)

    def test_lie_row_must_disclose_elastic_zoh_and_delay_adaptation(self):
        manifest = copy.deepcopy(self.manifest)
        manifest["lie_group_usage"][0]["plant_adaptation"] = "rigid theorem"
        with self.assertRaisesRegex(ValueError, "must state elastic"):
            r2_report.validate_manifest(manifest)

    def test_public_text_has_six_metrics_and_provenance(self):
        context = r2_report.validate_manifest(self.manifest)
        rendered = r2_report.render_results(context)
        for metric in ("T1", "T2", "T3", "T4", "T5", "T6"):
            self.assertIn(metric, rendered)
        self.assertIn("fresh", rendered)
        self.assertIn("원격 작업: 0회", rendered)

    def test_deck_has_sixteen_slides(self):
        context = r2_report.validate_manifest(self.manifest)
        slides = r2_report._slide_specs(context)
        self.assertEqual(len(slides), 16)
        self.assertIn("push 미승인", " ".join(slides[-1]["lines"]))

    def test_fixture_pdf_is_generated_without_touching_r1_pdf(self):
        context = r2_report.validate_manifest(self.manifest)
        r1_path = r2_report.ROOT / "results" / "presentation_R1.pdf"
        before = hashlib.sha256(r1_path.read_bytes()).hexdigest() if r1_path.exists() else None
        pdf = self.fixture.directory / "fixture.pdf"
        r2_report.make_pdf(pdf, context)
        self.assertTrue(pdf.read_bytes().startswith(b"%PDF"))
        after = hashlib.sha256(r1_path.read_bytes()).hexdigest() if r1_path.exists() else None
        self.assertEqual(before, after)

    def test_private_reference_is_blocked(self):
        with self.assertRaisesRegex(ValueError, "prohibited private references"):
            private = "copied from https://" + "notion" + ".com/private"
            r2_report._privacy_check(private, "fixture")


if __name__ == "__main__":
    unittest.main()
