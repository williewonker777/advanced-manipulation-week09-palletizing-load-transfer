"""Build the frozen R22 core or augmented detector as one default-name file."""
from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
R10 = ROOT / "my_detector.py"
R10_SHA256 = "09f9a8573814a31b335b36ec5a78e0dedf262daa9039954568abd6bedd8dbc67"
PINS = {
    "experiments/r22_combined/selection.json": "b74b3a1eac159f3836eea9b83b45b23872417573413cd416a1b3758abf6cedc0",
    "experiments/r22_combined/core.py": "bdcd186ea58db958e2d4de0a4bcfa9b2a32ea54d54b9860d8a29d249d5abd8ad",
    "experiments/r22_combined/candidate.py": "0d69976ee4f6529cdd8ae67d6321d1d2637a60b7b27adc1b2e95c4e48a37367b",
    "experiments/r22_normal/candidate.py": "658683de0a0db00d520ebbd4c7a28cbaa4beb11bd6e93dd8e71edded6472279d",
    "experiments/r22_release/state.py": "30cd330f75b7bd4078ba9d20ef0e902993626c14962466305136d0138c42e215",
    "experiments/r22_release/model.json": "fb02dd2121120ba90c64748272fe1c9f1873f174ac8ef107ba80e70b1b337340",
    "experiments/r22_tangent/state.py": "5ab5424fb795c2e2240b6cbed0f2dcdae30929d9ec543188cb570421288725a0",
    "experiments/r22_tangent/frozen.py": "dac541250c09e0e400414441e58458121d24f4e138d99da9769d7e2d90a0746a",
    "experiments/r22_tangent/model.json": "896ffbe13c95a24444f06541c55206a0718e6378da04fdc876c1907876fa0732",
    "experiments/r22_tangent/detector.py": "681c6a5d17a48efad81b4555bc2a0ff7a03f47556832ae752cc279ac0572bd4e",
    "experiments/r15_delayed_wrench/observer.py": "eff8f0b76d8c0358bb15b020be0db6e0b55d0e666a93ee2f46a1de6190ba2483",
    "experiments/r21_yaw_impulse/normal_bins.json": "5f22f640eb8824f764ffeec7ecbef7ba23df02a874547b93b8ddc72925e25c15",
    "experiments/r22_guarded/candidate.py": "23d8dc0b9747e38dd9a346872f27612b7ba67052aff4c6d604401aba387dc1f1",
    "experiments/r22_guarded/plan.json": "56ce7cdac16699a8db591195b89260ec24fc3991638a829368612ba1b27a546c",
    "experiments/r22_light/selection.json": "0650647478ce28542f66753f97242046e3c7123cb469de08908b40adf59affa3",
    "experiments/r22_light/plan.json": "e29341a31bd46a2aaa43b9ba2e4b0ee1b8b3a603f47a82226e9757b904bc69e6",
    "experiments/r22_light/witness.py": "430b2b2ebdda4c4cde9e9d1760a581f77628fca4ebe52e1211d80edc14153d7b",
    "experiments/r22_light/candidate.py": "1d6738b62af1a56e679544ed2a5f7695c920b5ecb6aa74d35c0b1952d141a5b1",
}


def sha(blob):
    if isinstance(blob, Path):
        blob = blob.read_bytes()
    return hashlib.sha256(blob).hexdigest()


def verify_inputs():
    if sha(R10) != R10_SHA256:
        raise ValueError("frozen R10 detector differs")
    actual = {name: sha(ROOT / name) for name in PINS}
    if actual != PINS:
        raise ValueError("frozen R22 integration input differs")


CORE = r'''
from collections import deque as _r22_deque

_R22Base = Detector
_R22_PHASES = ((1.35, 1.75, 18.3753578739),)
_R22_RELEASE_BOUNDS = np.asarray(%s, dtype=float)
_R22_AVERAGE = 12
_R22_SEPARATION = 32


def _r22_shared_contract(shared):
    return bool(
        getattr(shared, "API", None) == "icpbl.shared-estimator.v1"
        and getattr(shared, "SOURCE", None) == "jts"
        and callable(getattr(shared, "reset", None))
        and callable(getattr(shared, "update", None))
        and np.asarray(getattr(shared, "r_joint", ())).shape == (6,)
        and np.asarray(getattr(shared, "bias", ())).shape == (6,)
        and hasattr(shared, "mass_prev")
        and getattr(shared, "nom", None) is not None
        and getattr(shared, "nom_box", None) is not None
        and isinstance(getattr(shared, "box_com", None), tuple))


class _R22ReleaseBalance:
    def __init__(self, mass):
        self.dt = float(C.DT)
        self.elastic = np.asarray(C.ELASTIC, bool)
        stiffness, damping = C.K[self.elastic], C.D[self.elastic]
        self.decay = damping / (damping + stiffness * self.dt)
        self.in_gain = self.dt / (damping + stiffness * self.dt)
        self.phi = self.prev = self.mass_row = self.grip = None
        self.integral = 0.
        self.terms = _r22_deque(maxlen=_R22_AVERAGE)
        self.history = _r22_deque(maxlen=_R22_SEPARATION + 1)
        self.loaded = float(mass) > 0
        self.positions = _r22_deque(maxlen=_R22_AVERAGE + 1)

    def _balance(self, th, tau, mass_row, grip):
        measured = np.asarray(tau)[self.elastic]
        q = np.asarray(th, dtype=float).copy()
        if self.phi is None:
            self.phi = -measured / C.K[self.elastic]
        else:
            self.phi = self.decay * self.phi - self.in_gain * measured
        q[self.elastic] += self.phi
        row = np.asarray(mass_row, dtype=float)
        if self.prev is None or self.grip != bool(grip):
            self.terms.clear(); self.history.clear(); self.integral = 0.
            value = None
        else:
            momentum = float((self.mass_row + row) @ (q - self.prev)) / (2 * self.dt)
            self.integral += float(tau[0]) * self.dt
            self.terms.append(momentum - self.integral)
            value = None
            if len(self.terms) == _R22_AVERAGE:
                self.history.append(sum(self.terms) / _R22_AVERAGE)
                if len(self.history) > _R22_SEPARATION:
                    value = ((self.history[-1] - self.history[0]) /
                             (_R22_SEPARATION * self.dt))
        self.mass_row = row.copy(); self.grip = bool(grip); self.prev = q
        return value

    def feed(self, th, tau, mass_row, grip):
        switching = self.prev is not None and self.grip != bool(grip)
        if switching and not self.loaded:
            self.grip = bool(grip)
        elif switching and not grip and len(self.positions) == self.positions.maxlen:
            dq = (self.positions[-1] - self.positions[0]) / (_R22_AVERAGE * self.dt)
            self.integral += float((np.asarray(mass_row) - self.mass_row) @ dq)
            self.mass_row = np.asarray(mass_row).copy(); self.grip = False
        value = self._balance(th, tau, mass_row, grip)
        self.positions.append(self.prev)
        return value


class Detector(_R22Base):
    SOURCE = "r10+r22-core-research"

    @staticmethod
    def horizontal_threshold(t):
        for start, stop, value in _R22_PHASES:
            if start <= t < stop:
                return value
        return None

    def reset(self):
        ctrl = getattr(self, "ctrl", None)
        shared = getattr(ctrl, "shared_estimator", None)
        registered = bool(type(ctrl).__name__ == "Controller" and
                          type(ctrl).__module__ in ("my_torque", "torque_controller",
                                                    "my_servo", "servo_controller"))
        compatible = registered and _r22_shared_contract(shared)
        inherited = self.USE_SHARED
        if shared is not None and not compatible:
            self.USE_SHARED = False
        try:
            super().reset()
        finally:
            self.USE_SHARED = inherited
        self.r22_enabled = bool(compatible and getattr(ctrl, "MODE", None) == "torque"
                                and self._shared is shared)
        self.horizontal_samples = _r22_deque(maxlen=3)
        self.horizontal_raw = np.zeros(2); self.horizontal_filtered = np.zeros(2)
        self.horizontal_count = self.horizontal_clear = 0
        self.horizontal_detected = False; self.horizontal_score = 0.
        self.yaw = _R22ReleaseBalance(self.box_mass)
        self.yaw_bounds = _R22_RELEASE_BOUNDS.T.copy()
        self.yaw_value = None; self.yaw_count = self.yaw_quiet = 0
        self.yaw_active = False

    def update(self, t, th, thd, tau_m, tau_s, grip):
        residual, baseline = super().update(t, th, thd, tau_m, tau_s, grip)
        if not self.r22_enabled:
            return residual, bool(baseline)
        self.horizontal_raw = np.asarray(self._shared.last_residual_wrench[:2], float)
        self.horizontal_samples.append(self.horizontal_raw.copy())
        threshold = self.horizontal_threshold(float(t))
        if threshold is None or len(self.horizontal_samples) < 3:
            over = False; self.horizontal_score = 0.
        else:
            self.horizontal_filtered = np.median(np.asarray(self.horizontal_samples), axis=0)
            self.horizontal_score = float(np.max(np.abs(self.horizontal_filtered)) / threshold)
            over = bool(self.horizontal_score > 1.)
        self.horizontal_count = self.horizontal_count + 1 if over else 0
        self.horizontal_clear = 0 if over else self.horizontal_clear + 1
        if self.horizontal_count >= 1: self.horizontal_detected = True
        elif self.horizontal_clear >= 8: self.horizontal_detected = False
        packet = self._shared
        self.yaw_value = self.yaw.feed(th, tau_s - packet.bias, packet.mass_prev[0], grip)
        index = int(round(1000 * float(t))) // 100
        bounds = (self.yaw_bounds[index] if 0 <= index < 60
                  else np.asarray((3.23779270273884, 5.111942835466052)))
        if self.yaw_value is None or t < .08:
            self.yaw_count = self.yaw_quiet = 0; self.yaw_active = False
        else:
            above = abs(self.yaw_value) > bounds[int(bool(grip))]
            self.yaw_count = self.yaw_count + 1 if above else 0
            self.yaw_quiet = 0 if above else self.yaw_quiet + 1
            if self.yaw_count >= 3: self.yaw_active = True
            elif self.yaw_quiet >= 8: self.yaw_active = False
        return residual, bool(baseline or self.horizontal_detected or self.yaw_active)
'''


GUARDED = r'''
_R22GuardBase = Detector
_R22_NORMAL_BOUNDS = np.asarray(%s, dtype=float)


class _R22GuardedRelease(_R22ReleaseBalance):
    def __init__(self, mass):
        super().__init__(mass)
        self.pre_release_alarm = False
        self.release_continuity_selected = None
        self.use_reset_bank = False

    def feed(self, th, tau, mass_row, grip):
        switching = self.prev is not None and self.grip != bool(grip)
        releasing = switching and self.loaded and not bool(grip)
        picking = switching and self.loaded and bool(grip)
        if releasing and not self.pre_release_alarm:
            self.release_continuity_selected = False; self.use_reset_bank = True
            value = self._balance(th, tau, mass_row, grip)
            self.positions.append(self.prev)
            return value
        if releasing:
            self.release_continuity_selected = True; self.use_reset_bank = False
        elif picking:
            self.release_continuity_selected = None; self.use_reset_bank = False
        return super().feed(th, tau, mass_row, grip)


class Detector(_R22GuardBase):
    SOURCE = "r10+r22-guarded-research"

    def reset(self):
        super().reset()
        self.normal_yaw_bounds = _R22_NORMAL_BOUNDS.T.copy()
        self.continuity_yaw_bounds = self.yaw_bounds.copy()
        self.yaw = _R22GuardedRelease(self.box_mass)
        self.previous_combined_report = False

    def update(self, t, th, thd, tau_m, tau_s, grip):
        self.yaw_bounds = (self.normal_yaw_bounds if self.yaw.use_reset_bank
                           else self.continuity_yaw_bounds)
        self.yaw.pre_release_alarm = bool(self.previous_combined_report)
        residual, detected = super().update(t, th, thd, tau_m, tau_s, grip)
        self.previous_combined_report = bool(detected)
        return residual, detected
'''


AUGMENTED = r'''
import math as _r22_math

_R22Core = Detector
_R22_TANGENT_BOUNDS = %s
_R22_N = 31
_R22_STATE_LAG = 2 + _R22_N // 2
_R22_LAG = _R22_STATE_LAG + 1
_R22_X = np.linspace(-1., 1., _R22_N)
_R22_KERNEL = (1. - _R22_X * _R22_X) ** 2
_R22_KERNEL /= _R22_KERNEL.sum()


class _R22Sample:
    def __init__(self, time, effective_time, force_n):
        self.time, self.effective_time, self.force_n = time, effective_time, force_n


class _R22NormalForce:
    def __init__(self, empty, loaded):
        self.empty, self.loaded, self.dt = empty, loaded, float(C.DT)
        self.elastic = np.asarray(empty.elastic, bool)
        self.k = np.asarray(empty.K)[self.elastic]
        damping = np.asarray(C.D)[self.elastic]
        if (not np.isclose(self.dt, .001, rtol=0, atol=1e-12)
                or np.any(self.k <= 0) or np.any(damping <= 0) or C.JTS_CUTOFF <= 0):
            raise ValueError("invalid frozen tangent observer parameters")
        self.spring_memory = np.exp(-self.dt * self.k / damping)
        self.unfilter = 1. / np.expm1(2. * np.pi * C.JTS_CUTOFF * self.dt)
        self.prev_th = self.prev_sensor = self.deflection = None
        self.clock = self.prev_grip = None
        self.pos = _r22_deque(maxlen=3); self.residual = _r22_deque(maxlen=_R22_N)
        self.midpos = _r22_deque(maxlen=_R22_N); self.offsets = []; self.bias = None

    def update(self, t, encoder, jts, grip, *, project=True):
        th, raw = np.asarray(encoder, float), np.asarray(jts, float)
        if (not np.isfinite(t) or th.shape != (6,) or raw.shape != (6,)
                or not np.isfinite(th).all() or not np.isfinite(raw).all()):
            raise ValueError("finite time and two finite six-axis signals required")
        if self.clock is not None and abs(t - self.clock - self.dt) > 1e-8:
            raise ValueError("dropped or reordered tangent sample")
        self.clock = t
        if self.prev_grip != bool(grip):
            self.pos.clear(); self.residual.clear(); self.midpos.clear()
            self.prev_th = self.prev_sensor = self.deflection = None
        self.prev_grip = bool(grip)
        if self.prev_th is None:
            self.prev_th, self.prev_sensor = th.copy(), raw.copy(); return None
        aligned = raw + self.unfilter * (raw - self.prev_sensor)
        previous = self.prev_th.copy()
        self.prev_th, self.prev_sensor = th.copy(), raw.copy()
        equilibrium = aligned[self.elastic] / self.k
        self.deflection = (equilibrium if self.deflection is None else
                           equilibrium + self.spring_memory * (self.deflection - equilibrium))
        previous[self.elastic] -= self.deflection
        self.pos.append((previous, aligned))
        if len(self.pos) < 3: return None
        (qa, _), (q, tau), (qb, _) = self.pos
        velocity = (qb - qa) / (2. * self.dt)
        acceleration = ((qb - q) - (q - qa)) / self.dt**2
        model = self.loaded if grip else self.empty
        self.residual.append(model.inverse_dynamics(q, velocity, acceleration) - tau)
        self.midpos.append(q)
        if len(self.residual) != _R22_N: return None
        state_time = t - _R22_STATE_LAG * self.dt
        averaged = _R22_KERNEL @ np.asarray(self.residual)
        if self.bias is None:
            if not grip and .075 <= state_time < .425: self.offsets.append(averaged)
            elif state_time >= .425 and len(self.offsets) >= 300:
                self.bias = np.median(self.offsets, axis=0); self.offsets.clear()
        if not project or self.bias is None: return None
        model.d.qpos[:] = self.midpos[_R22_N // 2]
        mujoco.mj_kinematics(model.m, model.d); mujoco.mj_comPos(model.m, model.d)
        jp, jw = np.zeros((3, 6)), np.zeros((3, 6))
        mujoco.mj_jacSite(model.m, model.d, jp, jw, model.m.site("tcp").id)
        jacobian = np.vstack((jp, jw))
        if np.linalg.cond(jacobian) > 1e4: return None
        wrench = np.linalg.solve(jacobian.T, averaged - self.bias)
        return _R22Sample(float(t), float(t - _R22_LAG * self.dt), wrench[:3].copy())


def _r22_load_class(mass, grip):
    if not _r22_math.isfinite(mass) or not 0 <= mass <= 8: return None
    if not grip or mass == 0: return 0
    return 1 if mass <= 1 else 2 if mass <= 4 else 3


class _R22Evidence:
    def __init__(self):
        self.history = _r22_deque(maxlen=12); self.sx = self.sy = 0.
        self.above = self.quiet = 0; self.active = False
        self.score = None; self.group = None

    def feed(self, vector, group, bound):
        if vector is None or group is None or group != self.group:
            self.history.clear(); self.sx = self.sy = 0.
            self.above = self.quiet = 0; self.active = False; self.score = None
        self.group = group
        if vector is None or group is None: return False
        x, y = float(vector[0]), float(vector[1])
        if not _r22_math.isfinite(x) or not _r22_math.isfinite(y) or _r22_math.isnan(bound) or bound <= 0:
            raise ValueError("finite tangent signal and positive bound required")
        if len(self.history) == self.history.maxlen:
            a, b = self.history[0]; self.sx -= a; self.sy -= b
        self.history.append((x, y)); self.sx += x; self.sy += y
        if len(self.history) == self.history.maxlen:
            self.score = _r22_math.hypot(self.sx, self.sy) / len(self.history)
            self.above = self.above + 1 if self.score > bound else 0
            self.quiet = self.quiet + 1 if self.score < .8 * bound else 0
            if self.above >= 3: self.active = True
            elif self.quiet >= 8: self.active = False
        return self.active


class Detector(_R22Core):
    SOURCE = "r10+r22-augmented-research"

    def reset(self):
        super().reset()
        if self.r22_enabled:
            self.tangent = _R22Evidence()
            self.tangent_force = _R22NormalForce(self.nom, self.nom_box)

    def update(self, t, th, thd, tau_m, tau_s, grip):
        residual, core = super().update(t, th, thd, tau_m, tau_s, grip)
        if not self.r22_enabled: return residual, bool(core)
        sample = self.tangent_force.update(t, th, tau_s, grip)
        group = _r22_load_class(self.box_mass, grip)
        bound = _r22_math.inf if group is None else _R22_TANGENT_BOUNDS[group]
        flag = self.tangent.feed(None if sample is None else sample.force_n, group, bound)
        return residual, bool(core or flag)
'''


LIGHT = r'''
_R22LightBase = _R22Core
_R22_LIGHT_TARGET_N = 10.
_R22_LIGHT_GRAVITY = 9.81


def _r22_light_eligible(mass):
    return _r22_math.isfinite(mass) and 0 <= mass * _R22_LIGHT_GRAVITY <= _R22_LIGHT_TARGET_N


class _R22LightWitness:
    def __init__(self, mass, nominal, loaded):
        self.mass = mass; self.enabled = _r22_light_eligible(mass)
        self.nominal, self.loaded = nominal, loaded
        self.stop = min(t for t, _ in T.WAYPOINTS if t > T.T_GRASP)
        self.state = _R22Evidence(); self.calibrated = None
        self.observer = self._new() if self.enabled else None
        self.in_lift = False

    def _new(self):
        return _R22NormalForce(self.nominal, self.loaded)

    def feed(self, t, th, tau, grip):
        if not self.enabled: return False
        if self.calibrated is None:
            if self.observer is not None and t < .5 and not grip:
                self.observer.update(t, th, tau, False, project=False)
                if self.observer.bias is not None:
                    self.calibrated = self.observer.bias.copy(); self.observer = None
            return False
        open_window = bool(grip) and T.T_GRASP <= t < self.stop
        continue_event = self.in_lift and bool(grip) and self.state.active
        if not open_window and not continue_event:
            if self.in_lift:
                self.observer = None; self.state = _R22Evidence(); self.in_lift = False
            return False
        if not self.in_lift:
            self.observer = self._new(); self.observer.bias = self.calibrated.copy()
            self.in_lift = True
        sample = self.observer.update(t, th, tau, grip)
        group = _r22_load_class(self.mass, grip)
        return self.state.feed(None if sample is None else sample.force_n,
                               group, _R22_TANGENT_BOUNDS[group])


class Detector(_R22LightBase):
    SOURCE = "r10+r22-light-research"

    def reset(self):
        super().reset()
        if self.r22_enabled:
            self.light_witness = _R22LightWitness(self.box_mass, self.nom, self.nom_box)

    def update(self, t, th, thd, tau_m, tau_s, grip):
        residual, base = super().update(t, th, thd, tau_m, tau_s, grip)
        if not self.r22_enabled: return residual, bool(base)
        lifted = self.light_witness.feed(t, th, tau_s, grip)
        return residual, bool(base or lifted)
'''


def build(output, variant="core"):
    output = Path(output)
    if output.exists() or output.is_symlink():
        raise ValueError("new regular output path required")
    if variant not in ("core", "augmented", "guarded", "light"):
        raise ValueError("variant must be core, augmented, guarded or light")
    verify_inputs()
    release = json.loads((ROOT / "experiments/r22_release/model.json").read_bytes())
    tangent = json.loads((ROOT / "experiments/r22_tangent/model.json").read_bytes())
    normal = json.loads((ROOT / "experiments/r21_yaw_impulse/normal_bins.json").read_bytes())
    table, bounds, normal_table = release.get("threshold_Nm"), tangent.get("bounds_N"), normal.get("threshold_Nm")
    if (release.get("schema") != "r21-release-momentum-bounds.v1"
            or len(table) != 2 or any(len(row) != 60 for row in table)
            or tangent.get("schema") != "r22-tangent-normal-model.v1" or len(bounds) != 4
            or normal.get("schema") != "r21-yaw-normal-bins.v1"
            or len(normal_table) != 2 or any(len(row) != 60 for row in normal_table)):
        raise ValueError("invalid frozen embedded model")
    source = R10.read_text() + "\n" + (CORE % repr(table))
    if variant in ("guarded", "light"):
        source += "\n" + (GUARDED % repr(normal_table))
    if variant in ("augmented", "light"):
        source += "\n" + (AUGMENTED % repr(tuple(bounds)))
    if variant == "light":
        source += "\n" + LIGHT
    ast.parse(source)
    output.write_text(source)
    verify_inputs()
    payload = {"schema": "r22-standalone-build.v1", "variant": variant,
               "r10_detector_sha256": R10_SHA256, "frozen_sources_sha256": PINS,
               "standalone_sha256": sha(source.encode()), "lines": len(source.splitlines())}
    return payload


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=("core", "augmented", "guarded", "light"), default="core")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--receipt", type=Path)
    args = parser.parse_args()
    receipt = build(args.out, args.variant)
    if args.receipt:
        args.receipt.write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt, sort_keys=True))
