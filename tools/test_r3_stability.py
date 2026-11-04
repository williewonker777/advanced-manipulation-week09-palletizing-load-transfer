import copy
from pathlib import Path
import tempfile
import unittest

import numpy as np

from sim import trajectory as T
from sim import config as C
from tools.r2_p0 import mirrored_servo_drive
from tools.r3_stability import (
    _resolve_class,
    actual_peak_indices,
    capture_snapshots,
    fd_convergence,
    qualify_mass_schedule,
    requested_indices,
    run_specs,
)


class FakeShared:
    R2_STABILITY_STATE = ("detector_updates",)

    def __init__(self):
        self.detector_updates = 0.0
        self.detected = False
        self.last_update_t = None


class FakeController:
    MODE = "torque"
    R2_STABILITY_STATE = ("compute_updates", "last_command")

    def __init__(self):
        self.shared_estimator = FakeShared()
        self.compute_updates = 0.0
        self.last_command = (np.zeros(6))

    def compute(self, t, th, thd, tau_s, grip, q_d, qd_d, qdd_d):
        # Deliberately depends on the detector having run at this timestamp.
        assert self.shared_estimator.last_update_t == float(t)
        self.compute_updates += 1.0
        self.last_command = np.asarray(th, float).copy()
        return self.last_command.copy()


class FakeDetector:
    def __init__(self, controller):
        self.controller = controller

    def update(self, t, th, thd, tau_m, tau_s, grip):
        shared = self.controller.shared_estimator
        shared.detector_updates += 1.0
        shared.detected = bool(grip)
        shared.last_update_t = float(t)
        return np.zeros(6), shared.detected


class FakeServoController(FakeController):
    MODE = "servo"

    def compute(self, t, th, thd, tau_s, grip, q_d, qd_d, qdd_d):
        assert self.shared_estimator.last_update_t == float(t)
        self.compute_updates += 1.0
        self.last_command = np.array([
            np.asarray(th, float), np.zeros(6), np.zeros(6), np.zeros(6)
        ])
        return self.last_command.copy()


def fake_controller_builder(_controller_class, architecture, mass, com):
    assert architecture == "torque"
    assert mass == 2.0
    assert tuple(com) == (0.01, -0.02, 0.03)
    return FakeController()


def fake_detector_builder(_detector_class, controller, mass, com):
    return FakeDetector(controller)


def fake_servo_builder(_controller_class, architecture, mass, com):
    assert architecture == "servo"
    return FakeServoController()


def make_raw(path: Path):
    time_s = 3.598 + 0.001 * np.arange(7)
    encoder = np.repeat(np.arange(len(time_s), dtype=float)[:, None], 6, axis=1)
    # One-cycle torque queue: tau[k] is command generated from encoder[k-1].
    tau = np.vstack((np.zeros((1, 6)), encoder[:-1]))
    np.savez_compressed(
        path,
        obs_t=time_s,
        encoder=encoder,
        encoder_velocity=np.zeros_like(encoder),
        jts=np.zeros_like(encoder),
        grip=np.array([False, False, True, True, True, False, False]),
        motor_torque=np.vstack((np.zeros((1, 6)), tau[:-1])),
        tau=tau,
        qpos=np.repeat(np.arange(len(time_s), dtype=float)[:, None], 16, axis=1),
        e_mm=np.array([0.0, 1.0, 2.0, 9.0, 3.0, 2.0, 1.0]),
    )
    return time_s, encoder, tau


WELD = {
    "source": "synthetic",
    "translation_in_flange_m": [0.0, 0.0, 0.0],
    "relative_quaternion_wxyz": [1.0, 0.0, 0.0, 0.0],
}


def test_actual_capture_is_causal_and_uses_old_queued_command(tmp_path):
    raw_path = tmp_path / "actual.npz"
    time_s, encoder, tau = make_raw(raw_path)
    original_bytes = raw_path.read_bytes()
    requested = [3.600]
    requested_copy = copy.deepcopy(requested)
    payload = capture_snapshots(
        raw_path, "torque", 2.0, (0.01, -0.02, 0.03), requested,
        object, source_kind="actual_candidate", candidate_label="synthetic",
        detector_class=object, controller_builder=fake_controller_builder,
        detector_builder=fake_detector_builder, weld_relation=WELD,
        place_window_s=(3.598, 3.605),
    )
    snap = payload["snapshots"][0]
    assert snap["sample_index"] == 2
    assert np.isclose(snap["time_s"], time_s[2], atol=0.0, rtol=0.0)
    # Snapshot is before both current detector and compute; the analyzer then
    # performs the shared update once through controller.compute.
    assert snap["before_compute"]["controller"]["compute_updates"] == 2.0
    assert snap["before_compute"]["shared_estimator"]["detector_updates"] == 2.0
    assert snap["before_compute"]["shared_estimator"]["detected"] is False
    # Current applied torque is the old queued command; current output queues next.
    assert np.array_equal(snap["before_compute"]["actuation"]["pending_command"], encoder[1])
    assert np.array_equal(snap["after_compute"]["actuation"]["raw_applied_torque"], tau[2])
    assert np.array_equal(snap["output"], encoder[2])
    assert snap["after_compute"]["actuation"]["torque_replay_max_abs_error"] == 0.0
    # qpos is post-step logged, so physical pre-step k is qpos[k-1], previous k-2.
    assert snap["before_compute"]["physical_before_step"]["qpos"] == [1.0] * 16
    assert snap["before_compute"]["physical_before_step"]["qpos_previous"] == [0.0] * 16
    assert payload["source_raw_name"] == "actual.npz"
    assert str(tmp_path) not in str(payload)
    assert raw_path.read_bytes() == original_bytes
    assert requested == requested_copy


def test_actual_capture_rejects_non_generating_controller(tmp_path):
    raw_path = tmp_path / "mismatch.npz"
    make_raw(raw_path)
    with np.load(raw_path, allow_pickle=False) as archive:
        raw = {name: archive[name] for name in archive.files}
    raw["tau"] = raw["tau"].copy()
    raw["tau"][3, 0] += 1e-8
    np.savez_compressed(raw_path, **raw)
    with unittest.TestCase().assertRaisesRegex(AssertionError, "actuation mismatch"):
        capture_snapshots(
            raw_path, "torque", 2.0, (0.01, -0.02, 0.03), [3.600],
            object, source_kind="actual_candidate", candidate_label="wrong",
            detector_class=object, controller_builder=fake_controller_builder,
            detector_builder=fake_detector_builder, weld_relation=WELD,
        )


def test_servo_drive_pi_and_one_cycle_pending_are_reconstructed(tmp_path):
    raw_path = tmp_path / "servo_actual.npz"
    time_s = 3.598 + 0.001 * np.arange(5)
    encoder = np.repeat((0.01 * np.arange(5))[:, None], 6, axis=1)
    thd = np.zeros_like(encoder)
    pending = np.array([T.reference(0.0)[0], np.zeros(6), np.zeros(6), np.zeros(6)])
    velocity = np.zeros(6)
    integral = np.zeros(6)
    tau, velocity_before, integral_before = [], [], []
    for index in range(len(time_s)):
        velocity_before.append(velocity.copy())
        integral_before.append(integral.copy())
        applied, velocity, integral = mirrored_servo_drive(
            pending, encoder[index], thd[index], velocity, integral
        )
        tau.append(np.clip(applied, -C.TAU_MAX, C.TAU_MAX))
        pending = np.array([encoder[index], np.zeros(6), np.zeros(6), np.zeros(6)])
    tau = np.asarray(tau)
    np.savez_compressed(
        raw_path, obs_t=time_s, encoder=encoder, encoder_velocity=thd,
        jts=np.zeros_like(encoder), grip=np.zeros(5, bool),
        motor_torque=np.vstack((np.zeros((1, 6)), tau[:-1])), tau=tau,
        qpos=np.zeros((5, 16)), e_mm=np.arange(5, dtype=float),
    )
    payload = capture_snapshots(
        raw_path, "servo", 2.0, (0.01, -0.02, 0.03), [3.601], object,
        source_kind="actual_candidate", candidate_label="servo",
        detector_class=object, controller_builder=fake_servo_builder,
        detector_builder=fake_detector_builder, weld_relation=WELD,
    )
    snap = payload["snapshots"][0]
    assert snap["sample_index"] == 3
    assert np.array_equal(
        snap["before_compute"]["actuation"]["pending_command"][0], encoder[2]
    )
    assert np.allclose(
        snap["before_compute"]["actuation"]["mirrored_drive_velocity_state"],
        velocity_before[3], atol=0.0, rtol=0.0,
    )
    assert np.allclose(
        snap["before_compute"]["actuation"]["mirrored_drive_integral_state"],
        integral_before[3], atol=0.0, rtol=0.0,
    )
    assert payload["actuation_replay"]["maximum_applied_torque_abs_error_nm"] == 0.0


def test_forced_r1_capture_allows_torque_difference_but_records_it(tmp_path):
    raw_path = tmp_path / "forced.npz"
    make_raw(raw_path)
    with np.load(raw_path, allow_pickle=False) as archive:
        raw = {name: archive[name] for name in archive.files}
    raw["tau"] = raw["tau"] + 20.0
    np.savez_compressed(raw_path, **raw)
    payload = capture_snapshots(
        raw_path, "torque", 2.0, (0.01, -0.02, 0.03), [3.600],
        object, source_kind="forced_r1", candidate_label="counterfactual",
        detector_class=object, controller_builder=fake_controller_builder,
        detector_builder=fake_detector_builder, weld_relation=WELD,
    )
    assert payload["actuation_replay"]["exact_match_required"] is False
    assert payload["actuation_replay"]["maximum_applied_torque_abs_error_nm"] == 20.0


def test_all_requested_times_and_actual_peak_indices_are_represented(tmp_path):
    raw_path = tmp_path / "peaks.npz"
    time_s, _, _ = make_raw(raw_path)
    with np.load(raw_path, allow_pickle=False) as archive:
        raw = {name: archive[name] for name in archive.files}
    peaks = actual_peak_indices(raw, place_window_s=(3.598, 3.604))
    assert peaks["actual_tcp_max"] == 3
    # The first queued command is zero, so the earliest nonzero step is sample 2.
    assert peaks["actual_place_torque_step"] == 2
    selected = requested_indices(
        time_s, [3.599, 3.600, 3.601], extra_indices=peaks
    )
    assert set(selected) == {1, 2, 3}
    assert selected[2]["peak_tags"] == ["actual_place_torque_step"]
    assert selected[3]["peak_tags"] == ["actual_tcp_max"]
    payload = capture_snapshots(
        raw_path, "torque", 2.0, (0.01, -0.02, 0.03),
        [3.599, 3.600, 3.601], object,
        source_kind="actual_candidate", candidate_label="peaks",
        detector_class=object, include_actual_peaks=True,
        controller_builder=fake_controller_builder,
        detector_builder=fake_detector_builder, weld_relation=WELD,
        place_window_s=(3.598, 3.605),
    )
    assert [row["sample_index"] for row in payload["snapshots"]] == [1, 2, 3]
    represented = {
        value for row in payload["snapshots"] for value in row["requested_times_s"]
    }
    assert represented == {3.599, 3.600, 3.601}
    tags = {tag for row in payload["snapshots"] for tag in row["peak_tags"]}
    assert tags == {"actual_tcp_max", "actual_place_torque_step"}


def test_mass_schedule_knots_midpoint_and_one_sided_branches():
    result = qualify_mass_schedule(3.15, 5.65)
    exact = result["endpoint_midpoint_exact"]
    assert exact == {"start": 0.0, "midpoint": 0.5, "full": 1.0}
    derivative = result["one_sided_derivative_per_kg"]
    assert derivative["start_left"] == 0.0
    assert derivative["full_right"] == 0.0
    assert derivative["start_right"] >= 0.0
    assert derivative["full_left"] >= 0.0
    assert result["weights"]["3.149000"] == 0.0
    assert result["weights"]["5.651000"] == 1.0


def test_torque_analysis_selector_excludes_only_inactive_integral():
    from experiments import r3_torque

    params = {
        "family": "R3_B", "architecture": "torque", "rate": 4.5,
        "damping_scale": 1.0, "schedule": "phase",
        "mass_start_kg": 3.15, "mass_full_kg": 5.65,
        "handoff_start_s": 3.6, "handoff_end_s": 3.84,
    }
    original = r3_torque.make_controller("torque", params)
    assert "r2_stability_state" not in original.__dict__
    _module, adapted = _resolve_class({
        "module": "experiments.r3_torque", "architecture": "torque",
        "params": params,
    })
    assert adapted is not original
    assert "compute" not in adapted.__dict__
    assert "r2_stability_state" not in original.__dict__
    controller = object.__new__(adapted)
    controller.box_mass = 7.7
    assert controller.r2_stability_state(3.839, True) == (
        "r3_reference_integral",
    )
    assert controller.r2_stability_state(3.84, True) == ()


def test_fd_convergence_requires_both_mechanics_and_closed_map():
    row = {
        "mechanical_fd": {
            "step_input": {
                "step_converged": True, "max_relative_step_change": 1e-6,
            },
            "previous_applied_input_for_jts": {
                "step_converged": True, "max_relative_step_change": 2e-6,
            },
        },
        "closed_loop": {"jacobian_step_diagnostics": {
            "A_relative_step_change": 5e-4, "B_relative_step_change": None,
        }},
    }
    assert fd_convergence(row)["pass"] is True
    row["mechanical_fd"]["previous_applied_input_for_jts"][
        "step_converged"
    ] = False
    assert fd_convergence(row)["pass"] is False
    row["mechanical_fd"]["previous_applied_input_for_jts"][
        "step_converged"
    ] = True
    row["closed_loop"]["jacobian_step_diagnostics"][
        "A_relative_step_change"
    ] = 1.1e-3
    assert fd_convergence(row)["pass"] is False


def test_resolve_class_accepts_plain_r1_module_without_factory():
    import r1_torque

    module, controller_class = _resolve_class({
        "id": "R1", "module": "r1_torque", "architecture": "torque",
        "params": {"id": "R1"},
    })
    assert module is r1_torque
    assert controller_class is r1_torque.Controller


class R3StabilityTests(unittest.TestCase):
    def test_completed_stability_output_is_not_overwritten(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'stability.json'
            path.write_text('original evidence\n')
            with self.assertRaisesRegex(RuntimeError, 'no overwrite'):
                run_specs([], [], path)
            self.assertEqual(path.read_text(), 'original evidence\n')

    """Stdlib discovery wrappers; helpers retain compact assertion bodies."""

    def _with_temp(self, function):
        with tempfile.TemporaryDirectory() as directory:
            function(Path(directory))

    def test_actual_capture_is_causal_and_uses_old_queued_command(self):
        self._with_temp(test_actual_capture_is_causal_and_uses_old_queued_command)

    def test_actual_capture_rejects_non_generating_controller(self):
        self._with_temp(test_actual_capture_rejects_non_generating_controller)

    def test_servo_drive_pi_and_one_cycle_pending_are_reconstructed(self):
        self._with_temp(test_servo_drive_pi_and_one_cycle_pending_are_reconstructed)

    def test_forced_r1_capture_allows_torque_difference_but_records_it(self):
        self._with_temp(test_forced_r1_capture_allows_torque_difference_but_records_it)

    def test_all_requested_times_and_actual_peak_indices_are_represented(self):
        self._with_temp(test_all_requested_times_and_actual_peak_indices_are_represented)

    def test_mass_schedule_knots_midpoint_and_one_sided_branches(self):
        test_mass_schedule_knots_midpoint_and_one_sided_branches()

    def test_torque_analysis_selector_excludes_only_inactive_integral(self):
        test_torque_analysis_selector_excludes_only_inactive_integral()

    def test_fd_convergence_requires_both_mechanics_and_closed_map(self):
        test_fd_convergence_requires_both_mechanics_and_closed_map()

    def test_resolve_class_accepts_plain_r1_module_without_factory(self):
        test_resolve_class_accepts_plain_r1_module_without_factory()


if __name__ == "__main__":
    unittest.main()
