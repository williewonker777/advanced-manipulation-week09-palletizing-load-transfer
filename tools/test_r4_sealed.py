import importlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from tools import r4_sealed as F


class SealedStageTests(unittest.TestCase):
    def test_import_has_no_rng_or_marker_side_effect(self):
        with patch("numpy.random.default_rng", side_effect=AssertionError("RNG at import")):
            importlib.reload(F)
        self.assertFalse(hasattr(F, "_holdout_design"))

    def test_holdout_marker_is_exclusive_and_precedes_rng(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            marker = root / "opened.json"
            seen = []
            class DummyRng:
                pass
            def factory(seed):
                seen.append(marker.is_file())
                return DummyRng()
            value = {"schema": "test", "seed": 7}
            F._exclusive_json(marker, value, "already")
            with patch.object(F, "_rng_factory", factory):
                F._rng_factory(7)
            self.assertEqual(seen, [True])
            with self.assertRaisesRegex(RuntimeError, "already"):
                F._exclusive_json(marker, value, "already")

    def test_synthetic_open_path_writes_marker_before_rng_factory(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            marker = root / "opened.json"
            private = root / "private"
            observed = []
            def factory(seed):
                observed.append(marker.is_file())
                return object()
            def stop_after_rng(protocol, rng):
                self.assertTrue(marker.is_file())
                raise RuntimeError("stop-before-physics")
            freeze = {"local_commit": "a" * 40, "chosen": {"servo": "R1", "torque": "R1"}}
            protocol = {"holdout": {"seed": 17}}
            with (patch.object(F, "HOLDOUT_MARKER", marker),
                  patch.object(F, "HOLDOUT", root / "result.json"),
                  patch.object(F, "HOLDOUT_REVIEW", root / "review.json"),
                  patch.object(F.R, "require_private", return_value=private),
                  patch.object(F, "check_integrity"),
                  patch.object(F, "verify_freeze", return_value=freeze),
                  patch.object(F, "_effective_protocol", return_value=protocol),
                  patch.object(F, "_rng_factory", factory),
                  patch.object(F, "_build_holdout_design", side_effect=stop_after_rng),
                  patch.object(F, "FREEZE", root / "freeze.json"),
                  patch.object(F, "PROTOCOL", root / "protocol.json"),
                  patch.object(F, "RESEAL", root / "reseal.json")):
                (root / "freeze.json").write_text("freeze")
                (root / "protocol.json").write_text("protocol")
                (root / "reseal.json").write_text("reseal")
                with self.assertRaisesRegex(RuntimeError, "stop-before-physics"):
                    F.open_holdout(private)
            self.assertEqual(observed, [True])

    def test_balanced_design_is_novel_and_deterministic(self):
        protocol = {"holdout": {
            "exclude_masses_kg": [1.0, 2.0], "mass_round_digits": 3,
            "mass_ranges_kg": [[.1, .5], [1.1, 1.5], [2.1, 2.5], [3.1, 3.5]],
            "com_ranges_m": [[-.02, -.02, .01], [.02, .02, .04]],
            "phase_time_ranges_s": {"pick": [1.1, 1.2], "transport": [2.1, 2.2],
                                    "place": [3.1, 3.2]},
            "force_range_N": [6.0, 12.0],
            "directions": list(F.DIRECTIONS),
            "rows_per_source_architecture": {"references": 4, "collisions": 30, "offsets": 4},
            "pick_offsets_m": [[.01, 0, 0], [-.01, 0, 0], [0, .01, 0], [0, -.01, 0]],
        }}
        class MidpointRng:
            def uniform(self, low, high, size=None):
                low, high = F.np.asarray(low), F.np.asarray(high)
                value = (low + high) / 2
                return value if value.ndim else float(value)
        first = F._build_holdout_design(protocol, MidpointRng())
        second = F._build_holdout_design(protocol, MidpointRng())
        self.assertEqual(first, second)
        self.assertEqual(len(first), 38)
        counts = {kind: sum(row["kind"] == kind for row in first)
                  for kind in ("reference", "collision", "offset")}
        self.assertEqual(counts, {"reference": 4, "collision": 30, "offset": 4})
        excluded = set(protocol["holdout"]["exclude_masses_kg"])
        masses = [row["condition"]["mass"] for row in first[:4]]
        self.assertFalse(excluded.intersection(masses))
        collisions = [row for row in first if row["kind"] == "collision"]
        self.assertEqual({row["phase"] for row in collisions}, {"pick", "transport", "place"})
        self.assertEqual({row["direction"] for row in collisions}, set(F.DIRECTIONS))

    def test_registered_holdout_seeds_are_never_instantiated_by_tests(self):
        # This uses a synthetic dummy seed and rejects any other value.  The
        # registered addendum is intentionally not read in this unit test.
        seen = []
        def factory(seed):
            if seed != 17:
                raise AssertionError("non-synthetic holdout seed accessed")
            seen.append(seed)
            return object()
        with patch.object(F, "_rng_factory", factory):
            F._rng_factory(17)
        self.assertEqual(seen, [17])

    def test_effective_protocol_uses_reseal_and_excludes_revalidation_masses(self):
        original = {"holdout": {"seed": 11, "exclude_masses_kg": [1.0, 2.0]}}
        redesign = {"new_holdout_exclusion_masses": [3.0, 4.0, 5.0, 6.0]}
        with (patch.object(F.S, "protocol", return_value=original),
              patch.object(F, "_reseal", return_value={"replacement_seed": 17}),
              patch.object(F, "_redesign_protocol", return_value=redesign)):
            effective = F._effective_protocol()
        self.assertEqual(effective["holdout"], {
            "seed": 17, "exclude_masses_kg": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]})
        self.assertEqual(original["holdout"], {
            "seed": 11, "exclude_masses_kg": [1.0, 2.0]})

    def test_failure_falls_back_without_promoting_runner_up(self):
        self.assertEqual(F._effective_after_holdout("CUSUM_ACTIVE_K05", {"passed": False}),
                         ("R1", "FAIL"))
        self.assertEqual(F._effective_after_holdout("CUSUM_ACTIVE_K05", {"passed": True}),
                         ("CUSUM_ACTIVE_K05", "PASS"))
        self.assertEqual(F._effective_after_holdout("R1", None), ("R1", "PRESERVED"))
        self.assertEqual(F._official_effective("CUSUM_ACTIVE_K05", "CUSUM_ACTIVE_K05", False),
                         "R1")
        with self.assertRaisesRegex(ValueError, "alternative promotion"):
            F._official_effective("CUSUM_ACTIVE_K05", "W40_ACTIVE", True)

    def test_top_level_holdout_map_cannot_stage_a_runner_up(self):
        freeze = {"chosen": {"servo": "CUSUM_ACTIVE_K05", "torque": "R1"}}
        valid = {"architectures": {
            "servo": {"status": "FAIL", "effective_selection": "R1"},
            "torque": {"status": "PRESERVED", "effective_selection": "R1"}},
            "effective_selection": {"servo": "R1", "torque": "R1"}}
        self.assertEqual(F._validated_effective_map(freeze, valid),
                         {"servo": "R1", "torque": "R1"})
        mutated = json.loads(json.dumps(valid))
        mutated["effective_selection"]["servo"] = "W40_ACTIVE"
        with self.assertRaisesRegex(RuntimeError, "top-level holdout effective map"):
            F._validated_effective_map(freeze, mutated)
        mutated = json.loads(json.dumps(valid))
        mutated["architectures"]["servo"]["effective_selection"] = "W40_ACTIVE"
        with self.assertRaisesRegex(RuntimeError, "runner-up promotion"):
            F._validated_effective_map(freeze, mutated)

    def test_stage_auxiliary_fields_cannot_be_mutated(self):
        expected = {"normal_audit_pass": True, "full_loop_parity_pass": True,
                    "timing_p99_us": 123.25, "timing_max_us": 456.5,
                    "timing_pass": True, "literal_store_free": True}
        F._require_auxiliary_fields(dict(expected), expected)
        for name, replacement in (("normal_audit_pass", False),
                                  ("full_loop_parity_pass", False),
                                  ("timing_p99_us", 1.0),
                                  ("timing_max_us", 2.0),
                                  ("timing_pass", False),
                                  ("literal_store_free", False)):
            mutated = dict(expected); mutated[name] = replacement
            with self.assertRaisesRegex(RuntimeError, name):
                F._require_auxiliary_fields(mutated, expected)

    def test_reused_equivalence_rows_are_source_exact_and_stay_ineligible(self):
        old = [{"architecture": mode, "variant": variant, "case_id": case,
                "passed": True, "physical_arrays_bit_equal": {"q": True}}
               for mode in ("servo", "torque")
               for variant in ("CUSUM_ACTIVE_K05", "CUSUM_ACTIVE_K10", "CUSUM_PHASE_K05")
               for case in (0, 1)]
        reused = [{**row, "source_reused": True,
                   "source_certificate_sha256": "a" * 64,
                   "literal_store_free": False, "adoption_eligible": False,
                   "reuse_reason": "unchanged diagnostic source"} for row in old]
        self.assertTrue(F._exact_reused_equivalence(old, reused, "a" * 64))
        for field, value in (("literal_store_free", True),
                             ("adoption_eligible", True), ("passed", False)):
            mutated = json.loads(json.dumps(reused))
            mutated[0][field] = value
            self.assertFalse(F._exact_reused_equivalence(old, mutated, "a" * 64), field)

    def test_w40_reconstruction_is_one_exact_registered_source_edit(self):
        built = F._read(F.D / "R4_BUILT.json")
        entry = next(row for row in built["generated"]
                     if row["path"].endswith("W40_ACTIVE.py"))
        original = (F.X.B.R1.read_text(encoding="utf-8") + "\n\n"
                    + F.X.B.KERNEL.read_text(encoding="utf-8")
                    + F.X.B._wrapper(entry["parameters"]))
        revised = F.X.revised_source(original, entry["parameters"])
        self.assertEqual(original.count(F.X.OLD_GATE), 1)
        self.assertEqual(revised.count(F.X.OLD_GATE), 0)
        self.assertEqual(revised.count(F.X.NEW_GATE), 1)
        self.assertTrue(F.X.literal_audit(revised)["passed"])
        self.assertNotEqual(revised + "\n# unregistered change\n", revised)

    def test_selected_detector_assembly_uses_only_repaired_wrapper(self):
        r1 = {"servo": "R1", "torque": "R1"}
        both = {"servo": "W40_ACTIVE", "torque": "W40_ACTIVE"}
        self.assertEqual(F._selected_detector_bytes(r1),
                         (F.ROOT / "r1_detector.py").read_bytes())
        self.assertEqual(F._selected_detector_bytes(both),
                         (F.ROOT / "experiments/r4_generated/W40_ACTIVE.py").read_bytes())
        for mixed, none_mode in (({"servo": "W40_ACTIVE", "torque": "R1"}, "torque"),
                                 ({"servo": "R1", "torque": "W40_ACTIVE"}, "servo")):
            mixed_text = F._selected_detector_bytes(mixed).decode()
            self.assertNotIn(F.X.OLD_GATE, mixed_text)
            self.assertEqual(mixed_text.count(F.X.NEW_GATE), 1)
            self.assertTrue(F.X.literal_audit(mixed_text)["passed"])
            self.assertIn(f"'{none_mode}': None", mixed_text)
            self.assertEqual(F._expected_detector_bytes(mixed), mixed_text.encode())

    def test_mixed_official_stage_uses_selected_detector_bytes(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "r1_servo.py").write_bytes(b"servo")
            (root / "r1_torque.py").write_bytes(b"torque")
            effective = {"servo": "W40_ACTIVE", "torque": "R1"}
            freeze = {"chosen": dict(effective)}
            review = {"effective_selection": dict(effective)}
            with (patch.object(F, "ROOT", root),
                  patch.object(F, "_validated_effective_map", return_value=effective),
                  patch.object(F, "_selected_detector_bytes",
                               return_value=b"qualified-read-only-detector") as selected):
                prefix, record = F._stage_effective(root / "private", freeze, review)
            selected.assert_called_once_with(effective)
            self.assertEqual(Path(str(prefix) + "_detector.py").read_bytes(),
                             b"qualified-read-only-detector")
            self.assertEqual(record["effective_selection"], effective)

    def test_freeze_verifier_rejects_git_byte_mismatch(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            file = root / "source.py"
            file.write_text("current")
            digest = F.hashlib.sha256(file.read_bytes()).hexdigest()
            with (patch.object(F, "ROOT", root),
                  patch.object(F, "_git_bytes", return_value=b"committed")):
                with self.assertRaisesRegex(RuntimeError, "current/committed source mismatch"):
                    F._verify_bindings({"source.py": digest}, "c" * 40)
            with (patch.object(F, "ROOT", root),
                  patch.object(F, "_git_bytes", return_value=b"current")):
                F._verify_bindings({"source.py": digest}, "c" * 40)

    def test_official_payload_retains_literal_evaluator_precision(self):
        records = {"servo": {"control_and_collisions": {
            "control": {"8.0": ["1.17", "3.91", "0.0", "14.6"]},
            "collisions": [{"cells": ["62", "45", "3", "1", "0회", "10 N"]}]}},
            "torque": {"control_and_collisions": {
                "control": {"8.0": ["3.23", "8.44", "0.0", "13.1"]},
                "collisions": []}}}
        # JSON serialization must not normalize evaluator-origin strings.
        round_trip = json.loads(json.dumps(records))
        self.assertEqual(round_trip["servo"]["control_and_collisions"]["control"]["8.0"][:2],
                         ["1.17", "3.91"])
        self.assertEqual(round_trip["torque"]["control_and_collisions"]["control"]["8.0"][:2],
                         ["3.23", "8.44"])

    def test_science_freeze_excludes_publication_processors(self):
        names = F.source_paths()
        self.assertIn("tools/r4_sealed.py", names)
        self.assertIn("tools/test_r4_sealed.py", names)
        self.assertIn("tools/r4_redesign.py", names)
        self.assertIn("results/v2_protocol/PROTOCOL_R4_REDESIGN_W40.json", names)
        self.assertNotIn("tools/r4_report.py", names)
        self.assertNotIn("tools/r4_public.py", names)
        self.assertNotIn("tools/test_r4_report.py", names)
        self.assertFalse(any(name.startswith("docs/") for name in names))

    def test_runtime_import_closure_contains_every_local_dependency(self):
        closure = F.runtime_import_closure()
        self.assertIn("tools/v2_learned_core.py", closure)
        for relative in closure:
            self.assertLessEqual(F._local_imports(relative), closure, relative)

    def test_registered_ranking_is_recomputed_not_trusted(self):
        def row(variant, gain, eligible=True, timing=10.0):
            return {"architecture": "servo", "variant": variant,
                    "adoption_eligible": eligible, "timing_p99_us": timing,
                    "review": {"additional_detected": gain}}
        reviews = {
            "DEVELOPMENT": {"results": [row("A", 2), row("B", 3)]},
            "D4": {"results": [row("A", 9), row("B", 1)]},
            "PERTURB": {"results": [row("A", 0), row("B", 0)]},
        }
        self.assertEqual(F._ranked_choice(reviews, "servo", ["A", "B"]), "B")
        reviews["PERTURB"]["results"][1]["adoption_eligible"] = False
        self.assertEqual(F._ranked_choice(reviews, "servo", ["A", "B"]), "A")


if __name__ == "__main__":
    unittest.main()
