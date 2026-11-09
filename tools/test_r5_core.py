"""Simulation-free behavioral tests for the fixed R5 normal monitors."""

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import unittest

import numpy as np

from experiments import r5_core as C
from sim import trajectory as T


ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "models/r5"
CONFIGS = ("SCALE_PHASE", "SCALE_STATE", "AR20_FULL", "AR50_FULL")
MODES = ("servo", "torque")


def payload(mode, config):
    return json.loads((MODEL_DIR / f"{mode}_{config}.json").read_text())


def episode(mode="servo", role="audit"):
    t = np.arange(6000, dtype=float) * .001
    refs = [T.reference(float(value)) for value in t]
    ref_q = np.vstack([row[0] for row in refs])
    ref_qd = np.vstack([row[1] for row in refs])
    ref_qdd = np.vstack([row[2] for row in refs])
    grip = (t >= T.T_GRASP) & (t < T.T_RELEASE)
    active = grip & (t >= 1.50) & (t < 3.80)
    phase_signal = np.column_stack((
        .35 * np.sin(2.1 * t), .27 * np.cos(1.7 * t),
        .02 * np.sin(.7 * t), np.zeros((6000, 3)),
    ))
    tracking = .002 * np.column_stack([np.sin(t * (index + 1)) for index in range(6)])
    return {
        "t": t, "th": ref_q + tracking, "thd": ref_qd.copy(),
        "tau_m": np.zeros((6000, 6)), "tau_s": np.zeros((6000, 6)),
        "grip": grip, "r_joint": .1 * tracking, "wrench": phase_signal,
        "active": active, "r1_detect": np.zeros(6000, dtype=bool),
        "ref_q": ref_q, "ref_qd": ref_qd, "ref_qdd": ref_qdd,
        "mass": 3.8125, "com": np.array([.011, -.017, .031]),
        "mode": mode, "role": role, "normal_only": role != "analysis_eval",
    }


def digest(value):
    h = hashlib.sha256()
    if isinstance(value, dict):
        for key in sorted(value):
            h.update(key.encode());h.update(digest(value[key]).encode())
    elif isinstance(value, np.ndarray):
        h.update(value.dtype.str.encode());h.update(str(value.shape).encode());h.update(value.tobytes())
    else:
        h.update(json.dumps(value, sort_keys=True, default=float).encode())
    return h.hexdigest()


class MonitorParityTests(unittest.TestCase):
    def test_online_and_batch_outputs_match_all_registered_models(self):
        for mode in MODES:
            ep = episode(mode)
            for config in CONFIGS:
                with self.subTest(mode=mode, config=config):
                    item = payload(mode, config)
                    batch = C.batch_trace(item, ep)
                    monitor = C.FixedMonitor(item)
                    own = []
                    score = []
                    for index, t in enumerate(ep["t"]):
                        own.append(monitor.update(t, ep["mass"], ep["com"], ep["th"][index],
                            ep["r_joint"][index], ep["wrench"][index], ep["active"][index], ep["grip"][index]))
                        score.append(monitor.last_score)
                    np.testing.assert_allclose(score, batch["score"], rtol=0., atol=1e-8)
                    np.testing.assert_array_equal(own, batch["own"])

    def test_batch_trace_does_not_mutate_episode_or_payload(self):
        item = payload("servo", "SCALE_STATE")
        ep = episode("servo")
        before = digest({"payload": item, "episode": ep})
        C.batch_trace(item, ep)
        self.assertEqual(before, digest({"payload": item, "episode": ep}))

    def test_batch_trace_rejects_forbidden_field(self):
        ep = episode("servo")
        ep["true_torque"] = np.zeros((6000, 6))
        with self.assertRaises(ValueError):
            C.batch_trace(payload("servo", "SCALE_STATE"), ep)

    def test_batch_trace_rejects_unknown_mode(self):
        ep = episode("servo")
        ep["mode"] = "unknown"
        with self.assertRaises(ValueError):
            C.batch_trace(payload("servo", "SCALE_STATE"), ep)

    def test_r1_alarm_is_preserved_by_batch_or(self):
        ep = episode("servo")
        ep["r1_detect"][123] = True
        result = C.batch_trace(payload("servo", "SCALE_STATE"), ep)
        self.assertTrue(result["detect"][123])


class CausalityTests(unittest.TestCase):
    def test_ar_prediction_uses_only_samples_at_least_twenty_ms_old(self):
        item = payload("servo", "AR20_FULL")
        baseline = episode("servo")
        changed = deepcopy(baseline)
        changed["wrench"][100:, 0] += 100.
        before = C.batch_trace(item, baseline)["prediction"]
        after = C.batch_trace(item, changed)["prediction"]
        np.testing.assert_array_equal(before[:120], after[:120])
        self.assertFalse(np.array_equal(before[120:], after[120:]))

    def test_ar_prediction_uses_only_samples_at_least_fifty_ms_old(self):
        item = payload("servo", "AR50_FULL")
        baseline = episode("servo")
        changed = deepcopy(baseline)
        changed["wrench"][100:, 1] -= 100.
        before = C.batch_trace(item, baseline)["prediction"]
        after = C.batch_trace(item, changed)["prediction"]
        np.testing.assert_array_equal(before[:150], after[:150])
        self.assertFalse(np.array_equal(before[150:], after[150:]))

    def test_ar_ten_sample_mean_starts_at_lag_plus_nine(self):
        item = payload("servo", "AR20_FULL")
        baseline = episode("servo")
        changed = deepcopy(baseline)
        changed["wrench"][0, 0] += 100.
        before = C.batch_trace(item, baseline)["prediction"]
        after = C.batch_trace(item, changed)["prediction"]
        np.testing.assert_array_equal(before[28], after[28])
        self.assertFalse(np.array_equal(before[29], after[29]))

    def test_phase_boundaries_are_right_open(self):
        boundaries = np.array([T.T_GRASP - .08, T.T_GRASP + .20,
                               T.T_RELEASE - .15, T.T_RELEASE + .10])
        samples = np.r_[boundaries[0] - 1e-9, boundaries]
        np.testing.assert_array_equal(C.phase_index(samples), [0, 1, 2, 3, 4])

    def test_unseen_fit_phases_disable_only_the_new_branch(self):
        ep = episode("servo")
        ep["active"][:] = True
        result = C.batch_trace(payload("servo", "SCALE_STATE"), ep)
        phases = C.phase_index(ep["t"])
        self.assertFalse(np.any(result["active"][(phases == 0) | (phases == 4)]))
        self.assertTrue(np.any(result["active"][phases == 2]))


class StateMachineTests(unittest.TestCase):
    def test_inactive_sample_immediately_clears_latched_alarm(self):
        item = payload("servo", "SCALE_PHASE")
        item["threshold"] = 0.
        monitor = C.FixedMonitor(item)
        t = 2.;mass = 3.8125;com = np.array([.011, -.017, .031])
        zero6 = np.zeros(6)
        for index in range(4):
            self.assertEqual(index == 3, monitor.update(t + index * .001, mass, com, zero6, zero6,
                np.full(6, 100.), True, True))
        self.assertFalse(monitor.update(t + .004, mass, com, zero6, zero6, np.full(6, 100.), False, True))
        self.assertFalse(monitor.last_own)
        self.assertEqual(0., monitor.last_ema)

    def test_equal_threshold_does_not_count_as_over(self):
        item = payload("servo", "SCALE_PHASE")
        ep = episode("servo")
        seed = C.batch_trace(item, ep)
        phase = C.phase_index(ep["t"])
        supported = np.flatnonzero(seed["active"] & (phase == 2))[:8]
        item["threshold"] = 0.
        monitor = C.FixedMonitor(item)
        for index in supported:
            prediction = seed["prediction"][index]
            wrench = np.zeros(6);wrench[:len(prediction)] = prediction
            self.assertFalse(monitor.update(ep["t"][index], ep["mass"], ep["com"], ep["th"][index],
                ep["r_joint"][index], wrench, True, True))

    def test_latched_alarm_clears_after_exactly_eight_below_samples(self):
        item = payload("servo", "SCALE_PHASE")
        ep = episode("servo")
        seed = C.batch_trace(item, ep)
        item["threshold"] = 1.
        monitor = C.FixedMonitor(item)
        indices = iter(range(2000, 2100))
        for _ in range(4):
            index = next(indices);wrench = np.zeros(6)
            wrench[:2] = seed["prediction"][index] + [20. * seed["scale"][index], 0.]
            monitor.update(ep["t"][index], ep["mass"], ep["com"], ep["th"][index],
                ep["r_joint"][index], wrench, True, True)
        self.assertTrue(monitor.last_own)
        below_states = []
        for index in indices:
            wrench = np.zeros(6);wrench[:2] = seed["prediction"][index]
            state = monitor.update(ep["t"][index], ep["mass"], ep["com"], ep["th"][index],
                ep["r_joint"][index], wrench, True, True)
            if monitor.last_ema <= item["threshold"]:
                below_states.append(state)
                if len(below_states) == 8:break
        self.assertEqual([True] * 7 + [False], below_states)


if __name__ == "__main__":
    unittest.main()
