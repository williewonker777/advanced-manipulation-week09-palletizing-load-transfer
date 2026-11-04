"""Preregistered R2 C3/C4 elastic-joint controller candidates.

The candidates extend the frozen R1 controllers without reading simulator
state.  Their only runtime inputs are the documented encoder, sampled encoder
velocity, JTS, grip, reference, known payload and public nominal model.

The continuous-time formulas which motivated NRIC, DOB, APD and P-IDC do not
prove stability of this implementation.  Selection uses the separate elastic
plant, 1 ms ZOH, command-delay and JTS-delay analysis required by
``PROTOCOL_R2.json``.
"""
from __future__ import annotations

import importlib
from typing import Any, Mapping

import numpy as np

from sim import config as C
from sim import trajectory as T


MODES = ("servo", "torque")
C3_LAWS = ("NRIC", "DOB")
C4_LAWS = ("P-APD", "P-IDC")
GAMMAS = (1.0, 2.0, 4.0)
BANDWIDTHS = (4.5, 9.0, 18.0)
C4_RATES = (4.5, 9.0)
DAMPING_SCALES = (0.5, 1.0)

# P0 values are compiled constants, not runtime file input.  They are
# componentwise 25% of the measured torque headroom and maximum measured
# encoder-reference RMS over the four nonofficial P0 masses.
TORQUE_RESERVE = {
    "servo": np.array([76.08414215846011, 2.810339750862539,
                       10.241003511250316, 10.210734498401578,
                       6.547656127855314, 11.76677180580213]),
    "torque": np.array([76.11050185661823, 4.861909300407845,
                        10.756108095800712, 10.240250777351067,
                        7.2263397966225, 11.772429133142767]),
}
POSITION_SCALE = {
    "servo": np.array([7.223776658016757e-4, 1.2808896434489303e-3,
                       9.084281721631738e-4, 4.0372696203441874e-5,
                       6.356607448852051e-4, 4.073878867599031e-4]),
    "torque": np.array([1.0852359661170992e-3, 2.1129945173399014e-3,
                        1.2282315975258654e-3, 1.4115376149470194e-4,
                        1.8932633563804844e-3, 8.508155462461925e-4]),
}


def _vector(value: Any, name: str) -> np.ndarray:
    out = np.asarray(value, dtype=float)
    if out.shape != (6,) or not np.all(np.isfinite(out)):
        raise ValueError(f"{name} must be a finite six-vector")
    return out


def _validate(mode: str, params: Mapping[str, Any]) -> dict[str, Any]:
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}")
    p = dict(params)
    family = p.get("family")
    law = p.get("law")
    if family == "C3":
        if law not in C3_LAWS:
            raise ValueError(f"C3 law must be one of {C3_LAWS}")
        if law == "NRIC":
            gamma = float(p.get("gamma", np.nan))
            if gamma not in GAMMAS:
                raise ValueError(f"gamma must be one of {GAMMAS}")
            p["gamma"] = gamma
        else:
            bandwidth = float(p.get("bandwidth", np.nan))
            if bandwidth not in BANDWIDTHS:
                raise ValueError(f"bandwidth must be one of {BANDWIDTHS}")
            p["bandwidth"] = bandwidth
    elif family == "C4":
        if law not in C4_LAWS:
            raise ValueError(f"C4 law must be one of {C4_LAWS}")
        rate = float(p.get("rate", np.nan))
        damping = float(p.get("damping_scale", np.nan))
        if rate not in C4_RATES or damping not in DAMPING_SCALES:
            raise ValueError(
                f"C4 rate/damping_scale must be in {C4_RATES}/{DAMPING_SCALES}"
            )
        p.update(rate=rate, damping_scale=damping)
    else:
        raise ValueError("family must be C3 or C4")
    return p


def _smooth_friction(velocity: np.ndarray) -> np.ndarray:
    return C.VISCOUS * velocity + C.COULOMB * np.tanh(velocity / C.FRICTION_EPS)


def _nominal_derivative(state: np.ndarray, motor_torque: np.ndarray, nominal) -> np.ndarray:
    """Nine-coordinate elastic nominal dynamics used by NRIC.

    Coordinates are ``[theta(J1..3), q(J1..6)]``.  Rotor inertia for rigid
    J4--J6 is added to the link inertia; elastic rotor inertia remains in its
    own block.  This is a controller-owned nominal model, not Plant truth.
    """
    z, zd = state[:9], state[9:]
    theta, theta_d = z[:3], zd[:3]
    q, qd = z[3:], zd[3:]
    spring = C.K[:3] * (theta - q[:3]) + C.D[:3] * (theta_d - qd[:3])
    friction = _smooth_friction(np.r_[theta_d, qd[3:]])
    theta_dd = (motor_torque[:3] - friction[:3] - spring) / C.B[:3]
    link_force = np.zeros(6)
    link_force[:3] = spring
    link_force[3:] = motor_torque[3:] - friction[3:]
    mass = nominal.mass_matrix(q)
    mass[3:, 3:] += np.diag(C.B[3:])
    qdd = np.linalg.solve(mass, link_force - nominal.bias(q, qd))
    return np.r_[zd, theta_dd, qdd]


def implicit_midpoint_step(
    state: np.ndarray, motor_torque: np.ndarray, nominal, dt: float = C.DT
) -> np.ndarray:
    """Implicit-midpoint elastic update solved by bounded fixed-point iteration."""
    x = np.asarray(state, dtype=float)
    u = _vector(motor_torque, "motor_torque")
    if x.shape != (18,) or not np.all(np.isfinite(x)):
        raise ValueError("state must be a finite 18-vector")
    # The P0 bandwidth is at most 18 rad/s versus the lowest 173 rad/s elastic
    # natural frequency.  Eight bounded fixed-point corrections at 1 ms reduce
    # the midpoint residual below numerical relevance in the focused test and
    # avoid forward-Euler energy growth.
    next_state = x + dt * _nominal_derivative(x, u, nominal)
    for _ in range(8):
        midpoint = 0.5 * (x + next_state)
        candidate = x + dt * _nominal_derivative(midpoint, u, nominal)
        if np.max(np.abs(candidate - next_state)) <= 1e-12:
            next_state = candidate
            break
        next_state = candidate
    return next_state


def dob_observer_step(
    z: np.ndarray,
    velocity: np.ndarray,
    applied_torque: np.ndarray,
    transmitted_torque: np.ndarray,
    bandwidth: float,
    dt: float = C.DT,
) -> tuple[np.ndarray, np.ndarray]:
    """Derivative-free motor DOB exact filter update.

    For ``B vdot = u - tau_s - friction + d``, set ``p=L B v`` and
    ``d_hat=z+p``.  The state law ``zdot=-L(z+p+u-tau_s-friction)`` is
    integrated exactly while the sampled bracket is held.
    """
    z = _vector(z, "z")
    velocity = _vector(velocity, "velocity")
    applied = _vector(applied_torque, "applied_torque")
    transmitted = _vector(transmitted_torque, "transmitted_torque")
    L = float(bandwidth)
    if L <= 0.0 or dt <= 0.0:
        raise ValueError("bandwidth and dt must be positive")
    p = L * C.B * velocity
    bracket = p + applied - transmitted - _smooth_friction(velocity)
    decay = np.exp(-L * dt)
    z_next = decay * z - (1.0 - decay) * bracket
    return z_next, z_next + p


def _frozen(t: float, grip: bool, estimator: Any) -> bool:
    """Collision/contact gate fixed before development data are viewed."""
    near_pick = T.T_GRASP - 0.08 <= t <= T.T_GRASP + 0.08
    near_place = T.WAYPOINTS[3][0] <= t <= T.T_RELEASE + 0.10
    return bool(near_pick or near_place or getattr(estimator, "detected", False) or not grip)


def _drive_torque(command, th, thd, velocity_filter, integral):
    command = np.asarray(command, dtype=float)
    vf = velocity_filter + C.DRIVE_VEL_FILTER * (thd - velocity_filter)
    error = command[2] + C.DRIVE_KP * (command[0] - th) - vf
    integ = integral + C.DT * error
    torque = command[3] + C.DRIVE_KV * error + C.DRIVE_KI * integ
    return np.clip(torque, -C.TAU_MAX, C.TAU_MAX), vf, integ


def make_controller(mode: str, params: Mapping[str, Any]):
    """Return a runner-compatible preregistered C3 or C4 controller class."""
    p = _validate(mode, params)
    base = importlib.import_module(f"r1_{mode}").Controller
    actual_drive_state = (("r2_drive_velocity", "r2_drive_integral")
                   if mode == "servo" else ())
    nominal_drive_state = (("r2_nominal_drive_velocity",
                            "r2_nominal_drive_integral")
                           if mode == "servo" else
                           ("r2_nominal_motor_filter",
                            "r2_nominal_applied_previous"))
    dynamic = {
        ("C3", "NRIC"): ("r2_nominal_state", "r2_nric_integral",
                           "r2_nominal_pending",
                           "r2_last_grip") + nominal_drive_state,
        ("C3", "DOB"): ("r2_dob_z", "r2_pending",
                         "r2_applied_previous") + actual_drive_state,
        # The elastic lift reuses the exact R1 tracker.  Its tau_link_prev,
        # tau_dot and (torque mode) v_motor are already in the baseline
        # stability whitelist, so only genuinely new C4 memory is declared.
        ("C4", "P-APD"): (),
        ("C4", "P-IDC"): ("r2_reference_integral", "r2_last_grip"),
    }[(p["family"], p["law"])]

    class R2RobustController(base):
        R2_PARAMETERS = dict(p)
        R2_STABILITY_STATE = dynamic

        def reset(self):
            super().reset()
            self.r2_reserve = TORQUE_RESERVE[mode].copy()
            self.r2_position_scale = POSITION_SCALE[mode].copy()
            q0 = T.reference(0.0)[0]
            self.r2_pending = (np.array([q0, np.zeros(6), np.zeros(6),
                                         np.zeros(6)])
                               if mode == "servo" else np.zeros(6))
            self.r2_applied_previous = np.zeros(6)
            self.r2_drive_velocity = np.zeros(6)
            self.r2_drive_integral = np.zeros(6)
            self.r2_nominal_state = None
            self.r2_nominal_pending = self.r2_pending.copy()
            self.r2_nominal_applied_previous = np.zeros(6)
            self.r2_nominal_drive_velocity = np.zeros(6)
            self.r2_nominal_drive_integral = np.zeros(6)
            self.r2_nominal_motor_filter = np.zeros(6)
            self.r2_nric_integral = np.zeros(6)
            self.r2_dob_z = np.zeros(6)
            self.r2_reference_integral = np.zeros(6)
            self.r2_last_grip = False
            if p["family"] == "C4":
                # pRNE supplies C(q,v)u, not bias(q,u).  Import is delayed so
                # C3 remains independently testable while the C5 module is
                # developed in parallel.
                from experiments.r2_prne import PRNE
                self.r2_prne_empty = PRNE(self.nom)
                self.r2_prne_box = PRNE(self.nom_box)

        def _new_applied_torque(self, th, thd):
            if mode == "torque":
                return np.clip(self.r2_pending, -C.TAU_MAX, C.TAU_MAX)
            torque, self.r2_drive_velocity, self.r2_drive_integral = _drive_torque(
                self.r2_pending, th, thd,
                self.r2_drive_velocity, self.r2_drive_integral,
            )
            return torque

        def _nominal_applied_torque(self):
            """Apply the old uncompensated nominal queue to nominal state."""
            if mode == "torque":
                return np.clip(self.r2_nominal_pending, -C.TAU_MAX, C.TAU_MAX)
            state = self.r2_nominal_state
            z, zd = state[:9], state[9:]
            theta = np.r_[z[:3], z[6:9]]
            theta_d = np.r_[zd[:3], zd[6:9]]
            torque, self.r2_nominal_drive_velocity, self.r2_nominal_drive_integral = (
                _drive_torque(
                    self.r2_nominal_pending, theta, theta_d,
                    self.r2_nominal_drive_velocity,
                    self.r2_nominal_drive_integral,
                )
            )
            return torque

        def _nominal_jts(self):
            """Ideal public nominal JTS used only by the nominal outer loop."""
            state = self.r2_nominal_state
            z, zd = state[:9], state[9:]
            spring = (C.K[:3] * (z[:3] - z[3:6])
                      + C.D[:3] * (zd[:3] - zd[3:6]))
            derivative = _nominal_derivative(
                state, self.r2_nominal_applied_previous,
                self.nom_box if self.r2_last_grip else self.nom,
            )
            qdd = derivative[12:18]
            rigid_v = zd[6:9]
            rigid = (self.r2_nominal_applied_previous[3:]
                     - _smooth_friction(np.r_[np.zeros(3), rigid_v])[3:]
                     - C.B[3:] * qdd[3:])
            return np.r_[spring, rigid]

        def _uncompensated_nominal_command(
            self, q_ref, qd_ref, qdd_ref, load
        ):
            """Frozen R1 feedforward with feedback closed on nominal motors."""
            tau_empty = self.nom.inverse_dynamics(q_ref, qd_ref, qdd_ref)
            tau_full = self.nom_box.inverse_dynamics(q_ref, qd_ref, qdd_ref)
            tau_link = tau_empty + load * (tau_full - tau_empty)
            theta = q_ref + self.inv_k * tau_link
            if self.tau_link_prev is None:
                theta_d, theta_dd = qd_ref.copy(), qdd_ref.copy()
            else:
                raw = (tau_link - self.tau_link_prev) / C.DT
                old = self.tau_dot.copy()
                self.tau_dot += 0.04 * (raw - self.tau_dot)
                theta_d = qd_ref + self.inv_k * self.tau_dot
                theta_dd = qdd_ref + self.inv_k * (self.tau_dot - old) / C.DT
                theta_d = np.clip(theta_d, qd_ref - 0.35, qd_ref + 0.35)
                theta_dd = np.clip(theta_dd, -25.0, 25.0)
            self.tau_link_prev = tau_link.copy()
            friction = _smooth_friction(theta_d)
            feedforward = self.nom.B * theta_dd + tau_link + friction
            if mode == "servo":
                return np.array([theta, theta_d, theta_d, feedforward])

            state = self.r2_nominal_state
            z, zd = state[:9], state[9:]
            theta_nom = np.r_[z[:3], z[6:9]]
            velocity_nom = np.r_[zd[:3], zd[6:9]]
            self.r2_nominal_motor_filter += 0.2 * (
                velocity_nom - self.r2_nominal_motor_filter
            )
            pd = self.nom.B * (
                self.wn**2 * (theta - theta_nom)
                + 2.0 * self.wn * (theta_d - self.r2_nominal_motor_filter)
            )
            jts_nominal = self._nominal_jts()
            return feedforward + pd + self.k_tau * (jts_nominal - tau_link)

        def _actual_state(self, th, thd):
            packet = self.shared_estimator
            q = np.asarray(packet.q_prev, dtype=float)
            v = np.asarray(packet.v, dtype=float)
            z = np.r_[th[:3], q]
            zd = np.r_[thd[:3], v]
            return np.r_[z, zd], q, v

        def _nric(self, t, th, thd, tau_s, grip, q_d, qd_d, qdd_d):
            q_ref, qd_ref, qdd_ref, load = self._load_and_reference(
                t, th, thd, tau_s, grip, q_d, qd_d, qdd_d
            )
            actual_state, q, v = self._actual_state(th, thd)
            freeze = _frozen(t, bool(grip), self.shared_estimator)
            switched = self.r2_nominal_state is None or bool(grip) != self.r2_last_grip
            if switched:
                self.r2_nominal_state = actual_state.copy()
                self.r2_nric_integral.fill(0.0)
                self.r2_nominal_motor_filter = np.r_[thd[:3], v[3:]].copy()
                self.r2_nominal_applied_previous.fill(0.0)
            self.r2_last_grip = bool(grip)
            nominal_applied = self._nominal_applied_torque()
            nominal_command = self._uncompensated_nominal_command(
                q_ref, qd_ref, qdd_ref, load
            )
            error = self.r2_nominal_state[3:9] - q
            velocity_error = self.r2_nominal_state[12:18] - v
            omega = 9.0
            scale = np.maximum(self.r2_position_scale, C.ENCODER_RES)
            if freeze:
                self.r2_nric_integral *= 0.95
                correction = np.zeros(6)
            else:
                limit = 10.0 * scale / omega
                self.r2_nric_integral = np.clip(
                    self.r2_nric_integral + C.DT * error, -limit, limit
                )
                normalized = (
                    velocity_error / (omega * scale)
                    + error / scale
                    + self.r2_nric_integral / (scale / omega)
                )
                gain = 1.0 + 1.0 / p["gamma"] ** 2
                correction = self.r2_reserve * np.tanh(gain * normalized)

            output = np.asarray(nominal_command, dtype=float).copy()
            if mode == "servo":
                output[3] += correction
            else:
                output += correction
            nominal = self.nom_box if grip else self.nom
            self.r2_nominal_state = implicit_midpoint_step(
                self.r2_nominal_state, nominal_applied, nominal
            )
            self.r2_nominal_applied_previous = nominal_applied.copy()
            self.r2_nominal_pending = np.asarray(nominal_command, dtype=float).copy()
            self.r2_last_grip = bool(grip)
            return output

        def _dob(self, t, th, thd, tau_s, grip, base_output):
            freeze = _frozen(t, bool(grip), self.shared_estimator)
            if freeze:
                # Track p so release from a gated interval cannot expose a
                # stale disturbance estimate.
                self.r2_dob_z = -p["bandwidth"] * C.B * thd
                correction = np.zeros(6)
            else:
                self.r2_dob_z, disturbance = dob_observer_step(
                    self.r2_dob_z, thd, self.r2_applied_previous,
                    tau_s, p["bandwidth"],
                )
                correction = np.clip(-disturbance, -self.r2_reserve,
                                     self.r2_reserve)
            output = np.asarray(base_output, dtype=float).copy()
            if mode == "servo":
                output[3] += correction
            else:
                output += correction
            return output

        def _load_and_reference(self, t, th, thd, tau_s, grip, q_d, qd_d, qdd_d):
            if mode == "servo":
                q_ref, qd_ref, qdd_ref = self._reference(t, q_d, qd_d, qdd_d)
            else:
                q_ref, qd_ref, qdd_ref = self._reference(t)
            load = self._estimate_load(t, th, thd, tau_s, grip)
            if self.UNLOAD_START < t < T.T_RELEASE:
                s = np.clip((t - self.UNLOAD_START) /
                            (T.T_RELEASE - self.UNLOAD_START), 0.0, 1.0)
                unload = 1.0 - self.UNLOAD_AMOUNT * (
                    10.0 * s**3 - 15.0 * s**4 + 6.0 * s**5
                )
                load = min(load, unload)
            return q_ref, qd_ref, qdd_ref, float(load)

        def _candidate_link_torque(self, q, v, q_ref, qd_ref, qdd_ref, load, t, grip):
            e = q_ref - q
            ed = qd_ref - v
            rate = p["rate"]
            damping = p["damping_scale"]
            freeze = _frozen(t, bool(grip), self.shared_estimator)
            if p["law"] == "P-IDC":
                if bool(grip) != self.r2_last_grip:
                    self.r2_reference_integral.fill(0.0)
                if freeze:
                    self.r2_reference_integral *= 0.95
                else:
                    limit = 10.0 * self.r2_position_scale / rate
                    self.r2_reference_integral = np.clip(
                        self.r2_reference_integral + C.DT * e, -limit, limit
                    )
                self.r2_last_grip = bool(grip)

            def law(nom, prne):
                mass = nom.mass_matrix(q)
                gravity = nom.gravity(q)
                if p["law"] == "P-APD":
                    phi = rate
                    D = 2.0 * damping * rate * mass
                    K = D * phi
                    return (mass @ (qdd_ref + phi * ed)
                            + prne.action(q, v, qd_ref + phi * e)
                            + gravity + D @ ed + K @ e)
                kv, kp = 2.0 * damping * rate, rate**2
                qd_r = qd_ref + kv * e + kp * self.r2_reference_integral
                qdd_r = qdd_ref + kv * ed + kp * e
                ed_r = qd_r - v
                return (mass @ qdd_r + prne.action(q, v, qd_r) + gravity
                        + (damping * rate * mass) @ ed_r)

            empty = law(self.nom, self.r2_prne_empty)
            full = law(self.nom_box, self.r2_prne_box)
            candidate = empty + load * (full - empty)
            # Bound the candidate around the R1 model torque using the P0
            # componentwise reserve; this is anti-windup, not collision tuning.
            base_empty = self.nom.inverse_dynamics(q_ref, qd_ref, qdd_ref)
            base_full = self.nom_box.inverse_dynamics(q_ref, qd_ref, qdd_ref)
            baseline = base_empty + load * (base_full - base_empty)
            return baseline + np.clip(candidate - baseline,
                                      -self.r2_reserve, self.r2_reserve)

        def _elastic_output(self, q_ref, qd_ref, qdd_ref, tau_link, th, thd, tau_s):
            from experiments.r2_contact import lift_command
            return lift_command(
                self, q_ref, qd_ref, qdd_ref, tau_link, th, thd, tau_s
            )

        def compute(self, t, th, thd, tau_s, grip, q_d, qd_d, qdd_d):
            th = _vector(th, "th")
            thd = _vector(thd, "thd")
            tau_s = _vector(tau_s, "tau_s")
            if p["family"] == "C3":
                if p["law"] == "NRIC":
                    return self._nric(
                        t, th, thd, tau_s, grip, q_d, qd_d, qdd_d
                    )
                new_applied = self._new_applied_torque(th, thd)
                base_output = super().compute(
                    t, th, thd, tau_s, grip, q_d, qd_d, qdd_d
                )
                output = self._dob(t, th, thd, tau_s, grip, base_output)
                self.r2_applied_previous = new_applied.copy()
                self.r2_pending = np.asarray(output, dtype=float).copy()
                return output

            q_ref, qd_ref, qdd_ref, load = self._load_and_reference(
                t, th, thd, tau_s, grip, q_d, qd_d, qdd_d
            )
            q = np.asarray(self.shared_estimator.q_prev, dtype=float)
            v = np.asarray(self.shared_estimator.v, dtype=float)
            tau_link = self._candidate_link_torque(
                q, v, q_ref, qd_ref, qdd_ref, load, t, bool(grip)
            )
            return self._elastic_output(
                q_ref, qd_ref, qdd_ref, tau_link, th, thd, tau_s
            )

    R2RobustController.__name__ = f"R2{p['law'].replace('-', '')}{mode.title()}Controller"
    return R2RobustController


__all__ = [
    "make_controller", "implicit_midpoint_step", "dob_observer_step",
    "TORQUE_RESERVE", "POSITION_SCALE",
]
