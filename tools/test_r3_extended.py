"""Pure R3 extended-study builder/gate tests; no episode is executed."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from tools import r3_extended as E
from tools import r3_study as R


class FakeHarness:
    __file__ = "/private/fake_harness.py"

    @staticmethod
    def matrix_cases():
        common = {
            "family": "physical_matrix",
            "seed": 19,
            "mass": 4.3,
            "com": (0.02, -0.01, 0.03),
            "stiffness_scale": 0.9,
            "viscous_scale": 1.1,
            "coulomb_scale": 1.2,
        }
        return [
            dict(common, collision=None),
            dict(common, collision=(2.5, [1.0, 0.0, 0.0])),
        ]


class TestR3Extended(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.protocol = R.protocol()
        cls.selection = [R.baseline_spec("servo"), R.baseline_spec("torque")]

    def test_development_balance_and_sign_symmetry(self):
        for spec in self.selection:
            rows = E.development_cases(self.protocol, spec)
            self.assertEqual(len(rows), 26)
            collisions = rows[2:]
            self.assertEqual(
                {
                    direction: sum(
                        row["direction"] == direction for row in collisions
                    )
                    for direction in self.protocol["development"]["directions"]
                },
                {
                    direction: 3
                    for direction in self.protocol["development"]["directions"]
                },
            )
            self.assertEqual(
                {
                    force: sum(row["force_n"] == force for row in collisions)
                    for force in self.protocol["development"]["forces_N"]
                },
                {11.0: 12, 27.0: 12},
            )
            for positive, negative in (
                ("+x", "-x"),
                ("+y", "-y"),
                ("+z", "-z"),
                ("+xy", "-xy"),
            ):
                signature = lambda direction: [
                    (row["collision"][0], row["force_n"], row["mass"])
                    for row in collisions
                    if row["direction"] == direction
                ]
                self.assertEqual(signature(positive), signature(negative))

    def test_d4_is_exact_unchanged_385_row_design(self):
        rows = E.d4_cases(self.selection[0])
        legacy = E.d4_robustness.build_cases("unused", "unused")
        self.assertEqual(len(rows), 385)
        self.assertEqual(sum(row["kind"] == "reference" for row in rows), 189)
        self.assertEqual(sum(row["kind"] == "collision" for row in rows), 180)
        self.assertEqual(sum(row["kind"] == "offset" for row in rows), 16)
        for row, old in zip(rows, legacy, strict=True):
            self.assertEqual(row["case_id"], old["case_id"])
            self.assertEqual(row["mass"], old["mass"])
            self.assertEqual(row["com"], list(old["com"]))
            self.assertEqual(row["collision"], old["collision"])
            self.assertEqual(row["pick_offset"], old["pick_offset"])

    def test_selection_is_canonical_and_one_per_architecture(self):
        with tempfile.TemporaryDirectory() as name:
            path = Path(name) / "selection.json"
            path.write_text(json.dumps({"chosen_specs": self.selection}))
            self.assertEqual(E.load_selection(path), self.selection)
            altered = [dict(self.selection[0], extra=True), self.selection[1]]
            path.write_text(json.dumps({"chosen_specs": altered}))
            with self.assertRaises(ValueError):
                E.load_selection(path)
            path.write_text(json.dumps({"chosen_specs": [self.selection[0]]}))
            with self.assertRaises(ValueError):
                E.load_selection(path)

    def test_include_r1_adds_each_missing_baseline_once(self):
        specs = R.specifications()
        chosen = [
            next(spec for spec in specs if spec["architecture"] == mode)
            for mode in ("servo", "torque")
        ]
        expanded = E.selected_with_r1(chosen, True)
        self.assertEqual(len(expanded), 4)
        self.assertEqual(sum(spec["id"] == "R1" for spec in expanded), 2)
        self.assertEqual(E.selected_with_r1(self.selection, True), self.selection)

    def test_holdout_runs_matched_r1_and_deduplicates_exact_alias(self):
        candidate = next(spec for spec in R.specifications()
                         if spec["architecture"] == "servo")
        chosen = [candidate, R.baseline_spec("torque")]
        seen = []

        def fake_case(row, private_dir):
            seen.append((row["architecture"], row["variant"]))
            return row

        with (
            tempfile.TemporaryDirectory() as name,
            patch.object(E, "HOLDOUT_RESULT", Path(name) / "result.json"),
            patch.object(E, "open_holdout", return_value=(
                {"batch_attempt": 1}, [], {"local_commit": "c" * 40})),
            patch.object(E, "holdout_rows", side_effect=lambda design, spec: [
                {"architecture": spec["architecture"], "variant": spec["id"]}]),
            patch.object(E.R, "run_case", side_effect=fake_case),
            patch.object(E.R, "classify_group"),
            patch.object(E.R, "runtime_hashes", return_value={}),
            patch.object(E.R, "sha", return_value="a" * 64),
            patch.object(E.R, "write"),
            patch.object(E, "_verify_design_seal", return_value={}),
            patch.object(E, "orchestration_hashes", return_value={}),
        ):
            result = E.run_holdout(self.protocol, chosen, Path(name))
        self.assertEqual(seen, [("servo", candidate["id"]),
                                ("torque", "R1"), ("servo", "R1")])
        self.assertEqual(len(result["metadata"]["evaluated_specs"]), 3)

    def test_perturb_rows_bind_full_perturbation_and_new_noise(self):
        rows = E.perturb_cases(self.protocol, self.selection[0], FakeHarness)
        self.assertEqual(len(rows), 11)
        self.assertEqual(sum(row["perturb_family"] == "new_noise" for row in rows), 9)
        for row in rows:
            self.assertEqual(
                set(row["perturb"]),
                {"seed", "stiffness_scale", "viscous_scale", "coulomb_scale"},
            )
            for key, value in row["perturb"].items():
                self.assertEqual(row[key], value)
        reference = rows[0]
        collision = rows[1]
        self.assertEqual(reference["mass"], collision["mass"])
        self.assertEqual(reference["com"], collision["com"])
        self.assertEqual(reference["perturb"], collision["perturb"])

    def test_parity_compares_all_arrays_but_excludes_timing(self):
        first = {"rms_mm": 1.0, "ctrl_time_us": np.array([1.0, 2.0])}
        second = {"rms_mm": 1.0, "ctrl_time_us": np.array([9.0])}
        log = {
            "t": np.arange(3),
            "q": np.arange(18).reshape(3, 6),
            "physical_extra": np.arange(4),
            "t_ctrl": np.arange(3),
        }
        different_timing = dict(log, t_ctrl=np.arange(3) + 100)
        self.assertTrue(E._parity_result(first, second, log, different_timing)["passed"])
        changed = dict(log)
        changed["physical_extra"] = np.arange(4) + 1
        self.assertFalse(E._parity_result(first, second, log, changed)["passed"])

    def test_nominal_parity_uses_43kg_seed7_and_capacity_guard(self):
        calls = []
        result = {"rms_mm": 1.0, "ctrl_time_us": np.array([1.0])}
        log = {"t": np.arange(3), "q": np.zeros((3, 6))}

        class Controller:
            pass

        class Detector:
            pass

        class Harness:
            @staticmethod
            def run_perturbed(controller, **kwargs):
                calls.append((controller, kwargs))
                return dict(result), dict(log)

        with (
            patch.object(
                E.R,
                "candidate_classes",
                return_value=(Controller, Detector),
            ),
            patch.object(E.R.resources, "wait_for_capacity") as capacity,
            patch.object(E, "run_episode", return_value=(dict(result), dict(log))) as original,
        ):
            evidence = E.nominal_parity(self.selection[0], Harness)
        self.assertTrue(evidence["passed"])
        self.assertEqual(capacity.call_count, 2)
        self.assertEqual(original.call_args.kwargs["box_mass"], 4.3)
        self.assertIsInstance(original.call_args.kwargs["detector"], Detector)
        self.assertEqual(calls[0][1]["box_mass"], 4.3)
        self.assertEqual(
            calls[0][1]["perturb"],
            {
                "seed": 7,
                "stiffness_scale": 1.0,
                "viscous_scale": 1.0,
                "coulomb_scale": 1.0,
            },
        )

    def test_all_variant_parity_precedes_any_perturb_group(self):
        events = []

        def parity(spec, harness):
            del harness
            events.append(("parity", spec["architecture"]))
            return {
                "passed": True,
                "variant": spec["id"],
                "architecture": spec["architecture"],
            }

        def group(stage, spec, cases, private_dir, harness=None):
            del cases, private_dir, harness
            events.append((stage, spec["architecture"]))
            return {"results": []}

        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            harness_path = root / "harness.py"
            harness_path.write_text("# synthetic\n")
            with (
                patch.object(E, "PERTURB_RESULT", root / "aggregate.json"),
                patch.object(E, "PERTURB_PARITY", root / "parity.json"),
                patch.object(E, "D", root),
                patch.object(E.R, "require_private", side_effect=lambda value: Path(value)),
                patch.object(E, "_load_perturb_harness", return_value=FakeHarness),
                patch.object(E, "nominal_parity", side_effect=parity),
                patch.object(E, "perturb_cases", return_value=[]),
                patch.object(E.R, "run_group", side_effect=group),
                patch.object(E.R, "write"),
                patch.object(E.R, "sha", return_value="a" * 64),
            ):
                E.run_perturb(
                    self.protocol, self.selection, root, harness_path
                )
        self.assertEqual(events[:2], [("parity", "servo"), ("parity", "torque")])
        self.assertEqual(events[2:], [("perturb", "servo"), ("perturb", "torque")])

    def test_fake_holdout_design_is_fresh_balanced_and_sign_symmetric(self):
        # Deliberately not the sealed production seed.
        rows = E.build_holdout_design(
            self.protocol, np.random.default_rng(12345)
        )
        self.assertEqual(len(rows), 38)
        self.assertEqual(sum(row["kind"] == "reference" for row in rows), 4)
        self.assertEqual(sum(row["kind"] == "collision" for row in rows), 30)
        self.assertEqual(sum(row["kind"] == "offset" for row in rows), 4)
        references = [row for row in rows if row["kind"] == "reference"]
        excluded = set(self.protocol["holdout"]["exclude_masses_kg"])
        for row, bounds in zip(
            references, self.protocol["holdout"]["mass_ranges_kg"], strict=True
        ):
            mass = row["condition"]["mass"]
            self.assertGreaterEqual(mass, bounds[0])
            self.assertLessEqual(mass, bounds[1])
            self.assertNotIn(mass, excluded)
            self.assertEqual(mass, round(mass, 4))
        collisions = [row for row in rows if row["kind"] == "collision"]
        for positive, negative in zip(
            self.protocol["holdout"]["directions"][::2],
            self.protocol["holdout"]["directions"][1::2],
            strict=True,
        ):
            signature = lambda direction: [
                (
                    row["phase"],
                    row["collision"][0],
                    row["force_n"],
                    row["condition"],
                )
                for row in collisions
                if row["direction"] == direction
            ]
            self.assertEqual(signature(positive), signature(negative))

    def test_freeze_verification_uses_mocked_git_objects(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            protocol_path = root / "PROTOCOL_R3.json"
            protocol_path.write_text("{}\n")
            contents = {
                "r1_servo.py": b"servo",
                "r1_torque.py": b"torque",
                "r1_detector.py": b"detector",
            }
            for relative, data in contents.items():
                (root / relative).write_bytes(data)
            hashes = {
                relative: hashlib.sha256(data).hexdigest()
                for relative, data in contents.items()
            }
            fallback = dict(hashes)
            contents['gate.json'] = b'{"status":"PASS"}'
            proof = dict(schema='R3_PREFREEZE_DECISION.v1', chosen_specs=self.selection,
                         no_runner_up_or_diagnostic_revival=True,
                         architectures={spec['architecture']: dict(effective_selection=spec,
                             preholdout_adoption_eligible=False,
                             irreversibly_ineligible_for_adoption=True) for spec in self.selection},
                         source_evidence_sha256={'gate.json': hashlib.sha256(contents['gate.json']).hexdigest()})
            contents['prefreeze.json'] = json.dumps(proof).encode()
            for relative in ('gate.json', 'prefreeze.json'):
                (root / relative).write_bytes(contents[relative])
                hashes[relative] = hashlib.sha256(contents[relative]).hexdigest()
            freeze = {
                "schema": "R3_FREEZE.v1",
                "local_commit": "b" * 40,
                "protocol_sha256": hashlib.sha256(
                    protocol_path.read_bytes()
                ).hexdigest(),
                "source_files": hashes,
                "chosen_specs": self.selection,
                "fallback_R1_sha256": fallback,
                "selection_evidence": {"path": "prefreeze.json", "sha256": hashes['prefreeze.json']},
            }
            freeze_path = root / "fake_freeze.json"
            freeze_path.write_text(json.dumps(freeze))
            fake_protocol = {"fallback_R1_sha256": fallback}
            with (
                patch.object(E, "ROOT", root),
                patch.object(E, "PROTOCOL", protocol_path),
                patch.object(E, "_required_freeze_paths", return_value=set(hashes)),
                patch.object(
                    E,
                    "_canonical_specs",
                    return_value={
                        (spec["architecture"], spec["id"]): spec
                        for spec in self.selection
                    },
                ),
                patch.object(E, "_git_bytes", side_effect=lambda commit, relative: contents[relative]),
                patch.object(E.subprocess, "run"),
                patch.object(E.R, "protocol", return_value=fake_protocol),
            ):
                self.assertEqual(
                    E.verify_freeze(self.selection, freeze_path), freeze
                )

    def test_marker_precedes_fake_seed_and_resume_never_reseeds(self):
        fake_protocol = json.loads(json.dumps(self.protocol))
        fake_protocol["holdout"]["seed"] = 12345
        freeze = {"local_commit": "c" * 40}
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            private = root / "private"
            marker = root / "opened.json"
            protocol_path = root / "protocol.json"
            protocol_path.write_text(json.dumps(fake_protocol))
            real_default_rng = np.random.default_rng

            def checked_rng(seed):
                self.assertTrue(marker.is_file())
                self.assertEqual(seed, 12345)
                return real_default_rng(seed)

            with (
                patch.object(E, "PROTOCOL", protocol_path),
                patch.object(E, "verify_freeze", return_value=freeze),
                patch.object(E.R, "require_private", side_effect=lambda value: Path(value)),
                patch.object(E.np.random, "default_rng", side_effect=checked_rng),
            ):
                opened, design, actual_freeze = E.open_holdout(
                    fake_protocol,
                    self.selection,
                    private,
                    marker=marker,
                    freeze_path=root / "freeze.json",
                )
            self.assertEqual(actual_freeze, freeze)
            self.assertEqual(opened["batch_attempt"], 1)
            self.assertEqual(len(design), 38)
            self.assertTrue((private / "R3_HOLDOUT_DESIGN.json").is_file())

            with (
                patch.object(E, "PROTOCOL", protocol_path),
                patch.object(E, "verify_freeze", return_value=freeze),
                patch.object(E.R, "require_private", side_effect=lambda value: Path(value)),
                patch.object(E.np.random, "default_rng", side_effect=AssertionError("reseeded")),
            ):
                resumed = E.open_holdout(
                    fake_protocol,
                    self.selection,
                    private,
                    marker=marker,
                    freeze_path=root / "freeze.json",
                )
            self.assertEqual(resumed[1], design)
            design_path = private / 'R3_HOLDOUT_DESIGN.json'
            saved = json.loads(design_path.read_text())
            saved['design'][0]['condition']['mass'] += .0001
            design_path.write_text(json.dumps(saved))
            with (
                patch.object(E, 'PROTOCOL', protocol_path),
                patch.object(E, 'verify_freeze', return_value=freeze),
                patch.object(E.R, 'require_private', side_effect=lambda value: Path(value)),
                patch.object(E.np.random, 'default_rng', side_effect=AssertionError('reseeded')),
            ):
                with self.assertRaisesRegex(RuntimeError, 'sealed holdout design changed'):
                    E.open_holdout(fake_protocol, self.selection, private,
                                   marker=marker, freeze_path=root / 'freeze.json')

    def test_final_selection_refuses_diagnostic_revival(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            chosen = [next(s for s in R.specifications() if s['architecture'] == 'servo'),
                      R.baseline_spec('torque')]
            record = dict(schema='R3_PREFREEZE_DECISION.v1', chosen_specs=chosen,
                          no_runner_up_or_diagnostic_revival=True,
                          architectures={spec['architecture']: dict(effective_selection=spec,
                              preholdout_adoption_eligible=False,
                              irreversibly_ineligible_for_adoption=True) for spec in chosen})
            path = root / 'decision.json'
            path.write_text(json.dumps(record))
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            freeze = dict(selection_evidence=dict(path='decision.json', sha256=digest),
                          source_files={'decision.json': digest})
            with patch.object(E, 'ROOT', root):
                with self.assertRaisesRegex(RuntimeError, 'cannot be revived'):
                    E.verify_final_selection(chosen, freeze)

    def test_public_aggregate_rejects_private_or_absolute_paths(self):
        E._assert_portable({"result_file": "R3_RESULT.json"})
        with self.assertRaises(RuntimeError):
            E._assert_portable({"private_path": "secret.npz"})
        with self.assertRaises(RuntimeError):
            E._assert_portable({"value": "/tmp/private/result.npz"})

    def test_regular_orchestration_delegates_without_running_episodes(self):
        calls = []

        def run_group(stage, spec, cases, private_dir):
            calls.append((stage, spec["architecture"], len(cases), private_dir))
            return {"results": []}

        with patch.object(E.R, "run_group", side_effect=run_group):
            E.run_regular_stage(
                "development", self.protocol, self.selection, Path("private")
            )
        self.assertEqual(
            [(stage, mode, count) for stage, mode, count, _ in calls],
            [("development", "servo", 26), ("development", "torque", 26)],
        )


if __name__ == "__main__":
    unittest.main()
