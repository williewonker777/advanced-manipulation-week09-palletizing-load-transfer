"""Standard-library-only tests for R3 official gates; no episodes."""
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from tools import r3_official_regression as O


ROOT = Path(__file__).resolve().parents[1]


def records_from_exposed(original):
    records = {}
    for mode in ("servo", "torque"):
        old = original["combinations"][mode + "_v2_v2"]
        control = {}
        for mass, row in old["control"].items():
            control[mass] = [str(row[key]) for key in (
                "rms_mm", "max_mm", "saturation_pct", "pick_nm",
                "place_nm", "table_n")]
            control[mass] += [str(value) for value in row["false_alarms"]]
            control[mass].append(row["status"])
        collisions = [
            {
                "mass": row["mass"],
                "time": row["time"],
                "cells": [row[key] for key in (
                    "force10", "force15", "force40", "force100",
                    "false_max", "minimum")],
            }
            for row in old["collisions"]
        ]
        records[mode] = {
            "kit_integrity": {"status": "ok"},
            "self_contained_default_names": True,
            "control_and_collisions": {
                "control": control,
                "collisions": collisions,
            },
        }
    return records


def synthetic_holdout(selection, freeze_hash="f" * 64,
                      protocol_hash="p" * 64, marker_hash="m" * 64):
    specs = list(selection)
    for mode in ("servo", "torque"):
        baseline = {"id": "R1", "family": "R1", "architecture": mode}
        if baseline not in specs:
            specs.append(baseline)
    rows = []
    for spec in specs:
        for case_id in range(38):
            if case_id < 4:
                kind, collision, offset = "reference", None, None
            elif case_id < 34:
                kind, collision, offset = (
                    "collision", [2.0 + case_id / 1000.0, [1.0, 0.0, 0.0]], None)
            else:
                kind, collision, offset = "offset", None, [0.01, 0.0, 0.01]
            row = {
                "variant": spec["id"], "specification": spec,
                "architecture": spec["architecture"], "case_id": case_id,
                "mass": float(case_id % 4 + 1), "com": [0.01, -0.01, 0.02],
                "collision": collision, "pick_offset": offset,
                "official": False, "stage": "holdout", "kind": kind,
                "phase": "transport" if kind == "collision" else "none",
                "direction": "+x" if kind == "collision" else "none",
                "force_n": 10.0 if kind == "collision" else 0.0,
                "rms_mm": 1.0, "max_mm": 2.0, "saturation_pct": 0.0,
                "diverged": False, "torque_step_pick_nm": 3.0,
                "torque_step_place_nm": 4.0, "table_force_place_max_n": 5.0,
                "false_alarms": 0, "g4": {},
                "timing": {"combined": {"wall": {"p99_us": 1.0, "max_us": 2.0}},
                           "outliers": []},
                "private_raw_basename": (
                    f"raw_{spec['architecture']}_{spec['id']}_{case_id}.npz"),
                "private_raw_sha256": "a" * 64,
            }
            if kind == "collision":
                row.update(collision_outcome="detected", tool_contact_delay_ms=None)
            rows.append(row)
    return {"schema": "R3_HOLDOUT.v1", "metadata": {
        "one_batch": True, "chosen_specs": selection, "evaluated_specs": specs,
        "freeze_sha256": freeze_hash, "protocol_sha256": protocol_hash,
        "opened_marker_sha256": marker_hash, "local_commit": "c" * 40,
        "cases_per_source_architecture": 38,
        "source_sha256": {f"{s['architecture']}:{s['id']}": {} for s in specs},
        "orchestration_sha256": {}}, "results": rows}


class TestR3OfficialRegression(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.protocol = json.loads(
            (ROOT / "results/v2_protocol/PROTOCOL_R3.json").read_text()
        )
        cls.original = json.loads(
            (ROOT / "results/v2_protocol/S4_OFFICIAL.json").read_text()
        )
        cls.selection = [
            {"id": "A_handoff_long", "family": "R3_A", "architecture": "servo"},
            {"id": "B_mass_phase", "family": "R3_B", "architecture": "torque"},
        ]

    def test_exposed_r1_cells_pass_without_exact_equality_requirement(self):
        records = records_from_exposed(self.original)
        for mode in ("servo", "torque"):
            for mass, cells in records[mode]["control_and_collisions"]["control"].items():
                baseline = self.original["combinations"][mode + "_v2_v2"]["control"][mass]
                for key, index in O.CONTROL_METRICS:
                    cells[index] = str(float(baseline[key]) * 1.05)
            for row in records[mode]["control_and_collisions"]["collisions"]:
                for index in range(4):
                    delay = O._delay(row["cells"][index])
                    if delay is not None:
                        row["cells"][index] = str(delay + 5.0)
        result = O.assess(records, self.original, self.protocol, self.selection)
        self.assertTrue(result["all_pass"])
        self.assertEqual(result["architectures"]["servo"]["status"], "PASS")
        self.assertEqual(result["architectures"]["torque"]["status"], "PASS")

    def test_cost_over_105_fails_only_affected_architecture(self):
        records = records_from_exposed(self.original)
        records["servo"]["control_and_collisions"]["control"]["4.0"][0] = "999"
        result = O.assess(records, self.original, self.protocol, self.selection)
        servo = result["architectures"]["servo"]
        torque = result["architectures"]["torque"]
        self.assertEqual(servo["status"], "FAIL")
        self.assertEqual(servo["effective_selection"]["id"], "R1")
        self.assertEqual(torque["status"], "PASS")
        self.assertEqual(torque["effective_selection"], self.selection[1])

    def test_wrapper_source_hash_mismatch_fails_only_affected_mode(self):
        records = records_from_exposed(self.original)
        records["servo"]["source_sha256"] = "wrong"
        records["torque"]["source_sha256"] = "torque-hash"
        result = O.assess(
            records,
            self.original,
            self.protocol,
            self.selection,
            {"servo": "servo-hash", "torque": "torque-hash"},
        )
        self.assertEqual(result["architectures"]["servo"]["status"], "FAIL")
        self.assertEqual(result["architectures"]["torque"]["status"], "PASS")

    def test_lost_detection_extra_fa_and_fixed_caps_fail(self):
        records = records_from_exposed(self.original)
        servo_rows = records["servo"]["control_and_collisions"]["collisions"]
        transport = next(row for row in servo_rows
                         if row["mass"] == 8.0 and row["time"] == 3.0)
        transport["cells"][0] = "68"
        transport["cells"][1] = "놓침"
        transport["cells"][4] = "1회"
        result = O.assess(records, self.original, self.protocol, self.selection)
        failures = [failure for row in result["architectures"]["servo"]["collisions"]
                    for failure in row["failures"]]
        self.assertIn("10N_delay_above_fixed_cap", failures)
        self.assertIn("lost_R1_15N_detection", failures)
        self.assertIn("extra_collision_false_alarm", failures)

    def test_reference_false_alarm_saturation_and_divergence_fail(self):
        records = records_from_exposed(self.original)
        cells = records["torque"]["control_and_collisions"]["control"]["8.0"]
        cells[2], cells[6], cells[9] = "0.1", "1", "발산"
        result = O.assess(records, self.original, self.protocol, self.selection)
        failures = result["architectures"]["torque"]["control"][-1]["failures"]
        self.assertIn("nonzero_saturation", failures)
        self.assertIn("reference_false_alarm", failures)
        self.assertIn("divergence", failures)

    def test_zero_r1_cost_permits_only_zero(self):
        records = records_from_exposed(self.original)
        records["servo"]["control_and_collisions"]["control"]["0.0"][5] = "0.01"
        result = O.assess(records, self.original, self.protocol, self.selection)
        failures = result["architectures"]["servo"]["control"][0]["failures"]
        self.assertIn("table_n_ratio_gt_1.05", failures)

    def test_holdout_requires_selected_and_deduplicated_r1_groups(self):
        freeze = {"chosen_specs": self.selection, "local_commit": "c" * 40}
        document = synthetic_holdout(self.selection)
        with (patch.object(O.R, "runtime_hashes", return_value={}),
              patch.object(O.E, "orchestration_hashes", return_value={})):
            complete = O._holdout_complete(
                document, freeze, "f" * 64, "p" * 64, "m" * 64)
            self.assertEqual(set(complete["reviews"]), {"servo", "torque"})
            malformed = copy.deepcopy(document)
            malformed["results"][0].pop("rms_mm")
            with self.assertRaises(RuntimeError):
                O._holdout_complete(
                    malformed, freeze, "f" * 64, "p" * 64, "m" * 64)

        r1 = [{"id": "R1", "family": "R1", "architecture": mode}
              for mode in ("servo", "torque")]
        document = synthetic_holdout(r1)
        freeze = {"chosen_specs": r1, "local_commit": "c" * 40}
        with (patch.object(O.R, "runtime_hashes", return_value={}),
              patch.object(O.E, "orchestration_hashes", return_value={})):
            O._holdout_complete(document, freeze, "f" * 64, "p" * 64, "m" * 64)

    def test_preflight_verifies_freeze_before_completed_holdout(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            freeze_path, holdout_path = root / "freeze.json", root / "holdout.json"
            protocol_path, marker_path = root / "protocol.json", root / "opened.json"
            review_path = root / "review.json"
            protocol_path.write_text("{}")
            protocol_hash = hashlib.sha256(protocol_path.read_bytes()).hexdigest()
            freeze = {"chosen_specs": self.selection, "local_commit": "c" * 40}
            freeze_path.write_text(json.dumps(freeze))
            freeze_hash = hashlib.sha256(freeze_path.read_bytes()).hexdigest()
            holdout_path.write_text("{}")
            marker_path.write_text(json.dumps({"schema": "R3_HOLDOUT_OPENED.v1",
                "batch_attempt": 1, "chosen_specs": self.selection,
                "local_commit": "c" * 40, "protocol_sha256": protocol_hash}))
            marker_hash = hashlib.sha256(marker_path.read_bytes()).hexdigest()
            holdout_hash = hashlib.sha256(holdout_path.read_bytes()).hexdigest()
            recomputed = {mode: {"status": "PASS"} for mode in ("servo", "torque")}
            review = {"schema": "R3_HOLDOUT_REVIEW.v1",
                "holdout_sha256": holdout_hash, "freeze_sha256": freeze_hash,
                "protocol_sha256": protocol_hash, "opened_marker_sha256": marker_hash,
                "no_tuning_permitted": True, "architectures": {
                    mode: {"status": "PASS", "review": recomputed[mode],
                           "effective_selection": self.selection[index]}
                    for index, mode in enumerate(("servo", "torque"))}}
            review_path.write_text(json.dumps(review))
            with (patch.object(O.E, "verify_freeze") as verify,
                  patch.object(O, "OFFICIAL_FROZEN_PATHS", ()),
                  patch.object(O, "PROTOCOL", protocol_path),
                  patch.object(O, "_holdout_complete", return_value={
                      "specs": self.selection, "reviews": recomputed})):
                result = O.preflight(
                    freeze_path, holdout_path, review_path, marker_path)
            verify.assert_called_once_with(self.selection, freeze_path)
            self.assertEqual(result["freeze_sha256"], freeze_hash)
            self.assertEqual(result["effective_selection"], self.selection)

    def test_failed_holdout_review_forces_r1_and_cannot_be_revived(self):
        recomputed = {"servo": {"status": "FAIL", "reason": "fixture"},
                      "torque": {"status": "PASS"}}
        review = {"schema": "R3_HOLDOUT_REVIEW.v1", "no_tuning_permitted": True,
                  "architectures": {
                      "servo": {"status": "FAIL", "review": recomputed["servo"],
                                "effective_selection": {
                                    "id": "R1", "family": "R1",
                                    "architecture": "servo"}},
                      "torque": {"status": "PASS", "review": recomputed["torque"],
                                 "effective_selection": self.selection[1]}}}
        effective = O._review_effective_selection(
            review, self.selection, recomputed)
        self.assertEqual(effective[0]["id"], "R1")
        records = records_from_exposed(self.original)
        result = O.assess(records, self.original, self.protocol, effective)
        self.assertEqual(
            result["architectures"]["servo"]["effective_selection"]["id"], "R1")
        bad = copy.deepcopy(review)
        bad["architectures"]["servo"]["effective_selection"] = self.selection[0]
        with self.assertRaises(RuntimeError):
            O._review_effective_selection(bad, self.selection, recomputed)

    def test_exclusive_marker_is_fsynced_and_blocks_second_open(self):
        prepared = {"freeze_sha256": "f" * 64, "holdout_sha256": "h" * 64,
                    "selection": self.selection, "evaluated_specs": self.selection,
                    "effective_selection": self.selection,
                    "holdout_review_sha256": "r" * 64,
                    "protocol_sha256": "p" * 64,
                    "opened_marker_sha256": "o" * 64,
                    "freeze": {"local_commit": "c" * 40}}
        resolved = {mode: {"sha256": mode * 8} for mode in ("servo", "torque")}
        with tempfile.TemporaryDirectory() as name:
            marker = Path(name) / "opened.json"
            value = O._exclusive_marker(marker, prepared, resolved)
            self.assertEqual(value["batch_attempt"], 1)
            self.assertEqual(value["expected_episodes"], 58)
            with self.assertRaises(RuntimeError):
                O._exclusive_marker(marker, prepared, resolved)

    def test_command_and_environment_are_cpu0_gpuempty_and_isolated(self):
        command = O.official_command(Path("/private/r3"), O.OFFICIAL_DIR)
        self.assertEqual(command[:3], ["taskset", "-c", "0"])
        self.assertIn("tools/validate_submission.py", command)
        env = O.official_environment({"PYTHONPATH": "bad", "CUDA_VISIBLE_DEVICES": "0"})
        self.assertNotIn("PYTHONPATH", env)
        self.assertEqual(env["PYTHONDONTWRITEBYTECODE"], "1")
        self.assertEqual(env["OPENBLAS_NUM_THREADS"], "1")
        self.assertEqual(env["OMP_NUM_THREADS"], "1")
        self.assertEqual(env["CUDA_VISIBLE_DEVICES"], "")

    def test_stage_sources_uses_frozen_bytes_without_repository_edits(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            sources = {}
            for mode in ("servo", "torque"):
                path = root / (mode + ".py")
                path.write_text(mode)
                sources[mode] = {"source": path}
            detector = root / "detector.py"
            detector.write_text("detector")
            with patch.object(O.R, "require_private", side_effect=lambda value: Path(value)):
                prefix = O.stage_sources(root / "private", sources, detector)
            self.assertEqual(Path(str(prefix) + "_servo.py").read_text(), "servo")
            self.assertEqual(Path(str(prefix) + "_torque.py").read_text(), "torque")
            self.assertEqual(Path(str(prefix) + "_detector.py").read_text(), "detector")

    def test_shared_failure_falls_back_both_without_retest(self):
        prepared = {"selection": self.selection,
                    "effective_selection": self.selection}
        result = O._shared_failure("fixture", prepared)
        self.assertTrue(result["shared_unidentified_failure"])
        for mode in ("servo", "torque"):
            self.assertEqual(result["architectures"][mode]["effective_selection"]["id"],
                             "R1")
            self.assertFalse(result["architectures"][mode]["retest_permitted"])

    def test_launch_exception_after_marker_writes_shared_failure_no_retry(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            marker, result_path = root / "official_opened.json", root / "result.json"
            protocol = root / "protocol.json"
            protocol.write_text("{}")
            detector = root / "detector.py"
            detector.write_text("detector")
            resolved = {}
            for mode, spec in zip(("servo", "torque"), self.selection, strict=True):
                source = root / (mode + ".py")
                source.write_text(mode)
                resolved[mode] = {"specification": spec, "source": source,
                    "relative_source": mode + ".py",
                    "sha256": hashlib.sha256(source.read_bytes()).hexdigest()}
            prepared = {"selection": self.selection,
                "effective_selection": self.selection,
                "evaluated_specs": self.selection,
                "freeze_sha256": "f" * 64, "holdout_sha256": "h" * 64,
                "protocol_sha256": "p" * 64,
                "opened_marker_sha256": "o" * 64,
                "holdout_review_sha256": "r" * 64,
                "holdout_review": {"no_tuning_permitted": True},
                "freeze": {"local_commit": "c" * 40, "fallback_R1_sha256": {
                    "r1_servo.py": "s", "r1_torque.py": "t",
                    "r1_detector.py": "d"}}}
            with (patch.object(O, "MARKER", marker),
                  patch.object(O, "RESULT", result_path),
                  patch.object(O, "OFFICIAL_DIR", root / "official"),
                  patch.object(O, "PROTOCOL", protocol),
                  patch.object(O.R, "require_private", side_effect=lambda value: Path(value)),
                  patch.object(O, "check_integrity"),
                  patch.object(O, "preflight", return_value=prepared),
                  patch.object(O, "resolve_sources", return_value=(resolved, detector)),
                  patch.object(O, "stage_sources", return_value=root / "stage" / "r3"),
                  patch.object(O, "official_command", return_value=["fixture"]),
                  patch.object(O, "run_owned_group", side_effect=OSError("launch failed"))):
                output = O.run_official(root / "private")
            self.assertTrue(marker.is_file())
            self.assertTrue(result_path.is_file())
            self.assertTrue(output["shared_unidentified_failure"])
            self.assertIn("launch failed", output["reason"])
            for mode in ("servo", "torque"):
                self.assertEqual(output["architectures"][mode]["effective_selection"]["id"],
                                 "R1")

    def test_monitor_exception_terminates_only_owned_group(self):
        class Child:
            pid = 4321
            returncode = None

            def poll(self):
                return self.returncode

            def wait(self, timeout=None):
                del timeout
                self.returncode = -15
                return self.returncode

        child = Child()
        safe = {"mem_available_gib": 8.0, "foreign_heavy_python_pids": []}
        signals = []
        with tempfile.TemporaryDirectory() as name:
            with (patch.object(O.resources, "snapshot",
                               side_effect=[safe, OSError("monitor failed")]),
                  patch.object(O.subprocess, "Popen", return_value=child),
                  patch.object(O.os, "killpg",
                               side_effect=lambda pid, sig: signals.append((pid, sig)))):
                with self.assertRaisesRegex(OSError, "monitor failed"):
                    O.run_owned_group(["fixture"], Path(name) / "official.log")
        self.assertEqual(signals, [(4321, O.signal.SIGTERM)])


if __name__ == "__main__":
    unittest.main()
