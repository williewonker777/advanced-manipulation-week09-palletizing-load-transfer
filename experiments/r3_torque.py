"""Elastic-joint motor PD with bounded joint-torque-sensor feedback."""
import numpy as np
import mujoco
from sim import config as C
from sim import trajectory as T

MODE = "torque"


class SharedEstimator:
    """Timestamp-cached estimator packet shared by detector and controllers."""

    API = "icpbl.shared-estimator.v1"
    SOURCE = "jts"

    # Feature switches are used only by the owned ablation wrappers.
    USE_V = True
    USE_BIAS = True
    USE_WRENCH = True
    USE_CALIBRATED_THRESHOLDS = True
    USE_SUPPORT_LOAD = False
    SUPPORT_BLEND = 0.10
    LOAD_SOURCE = "sensor"

    def __init__(self, nom=None, nom_box=None, box_mass=0.0, box_com=(0.0, 0.0, 0.0)):
        # The simulator constructs Detector before attaching nominal models;
        # controller-owned shared cores pass all public model inputs here.
        self.box_mass = float(box_mass)
        self.box_com = tuple(box_com)
        if nom is not None and nom_box is not None:
            self.nom = nom
            self.nom_box = nom_box
            self.reset()

    def reset(self):
        self.dt = float(C.DT)
        self.observer_gain = 50.0
        self.r_joint = np.zeros(6)
        self.integral = None
        self.mass_prev = None
        self.q_prev = None
        self.v = np.zeros(6)
        self.motor_v = np.zeros(6)
        self.deflection_prev = None
        self.deflection_v = np.zeros(6)
        self.corrected_deflection_prev = None
        self.bias = np.zeros(6)
        self.bias_weight = 0.0
        self.last_grip = False
        self.alarm_count = 0
        self.fast_alarm_count = 0
        self.clear_count = 0
        self.detected = False
        self.fast_wrench_lp = None
        self.last_update_t = None
        self.sample_count = 0

        # Load estimation intentionally has separate state from detector V/B.
        # This preserves the historical controller algorithm while allowing a
        # single cached update to serve both consumers.
        self.load_q_prev = None
        self.load_v = np.zeros(6)
        self.load_sensor_bias = np.zeros(6)
        self.load_bias_ready = False
        self.lambda_ramp = 0.0
        self.lambda_direct = 0.0
        self.lambda_support = 0.0
        self.lambda_support_valid = False
        self.lambda_fused = 0.0
        self.load_fraction = 0.0

        self.elastic = np.asarray(self.nom.elastic, bool)
        self.inv_k = np.zeros(6)
        self.inv_k[self.elastic] = 1.0 / np.asarray(self.nom.K)[self.elastic]

        # The model's box origin is its centre.  In this model the flange +z
        # axis points toward the box, so its table-facing centre is +h/2.
        self.bottom_offset = np.array(
            [0.0, 0.0, C.BOX_POS[2] + 0.5 * C.BOX_SIZE[2]], dtype=float
        )
        self.half_x = 0.5 * float(C.BOX_SIZE[0])
        self.half_y = 0.5 * float(C.BOX_SIZE[1])
        self.friction_mu = 0.08
        self.torsion_radius = 0.02

        # Taught table heights come solely from trajectory reference poses.
        q_pick = T.reference(T.T_GRASP)[0]
        q_place = T.reference(T.T_RELEASE)[0]
        self.table_heights = np.array(
            [self._point_jacobian(q_pick)[1][2], self._point_jacobian(q_place)[1][2]]
        )
        self.height_band = 0.035

        # Collision-blind no-collision calibration covers 0--8 kg, a COM grid,
        # and both unchanged baseline controllers.  Horizontal force and moment
        # thresholds are direction-neutral; provenance is D3_FAST_CALIBRATION.json.
        self.force_threshold = np.array([20.377866929322302, 20.377866929322302,
                                         18.428782180318073])
        self.moment_threshold = np.array([6.38075916643147, 6.38075916643147,
                                          0.9579838574601636])
        self.out_of_height_fz_threshold = 27.105836764203143
        self.persistence = 4
        self.fast_force_threshold = np.array([11.141377260806653,
                                              11.141377260806653,
                                              27.842061547424592])
        self.fast_moment_threshold = np.array([4.885650078933188,
                                               4.885650078933188,
                                               0.4058946982950356])
        self.fast_positive_fz_threshold = 2.812605901696073
        self.fast_persistence = 8
        self.release_persistence = 8
        self.joint_threshold = np.array([10.0, 18.0, 8.0, 3.0, 3.0, 2.0])

    def _point_jacobian(self, q):
        """World 6D Jacobian, bottom-centre position, and box rotation."""
        self.nom.d.qpos[:] = q
        mujoco.mj_kinematics(self.nom.m, self.nom.d)
        mujoco.mj_comPos(self.nom.m, self.nom.d)
        body = self.nom.m.body("link_6").id
        rotation = self.nom.d.xmat[body].reshape(3, 3).copy()
        point = self.nom.d.xpos[body] + rotation @ self.bottom_offset
        jacp = np.zeros((3, 6))
        jacr = np.zeros((3, 6))
        mujoco.mj_jac(self.nom.m, self.nom.d, jacp, jacr, point, body)
        return np.vstack((jacp, jacr)), point.copy(), rotation

    def _estimate_state(self, t, th, thd, tau):
        deflection = -self.inv_k * tau
        q_measured = th + deflection
        if self.q_prev is None:
            self.q_prev = q_measured.copy()
            self.motor_v = thd.copy()
            self.deflection_prev = deflection.copy()
            self.corrected_deflection_prev = deflection.copy()
            return q_measured, np.zeros(6)

        if self.USE_V:
            # Causally lead the small elastic-deflection channel to compensate
            # the documented one-sample JTS delay/internal low-pass.  Motor
            # velocity remains the prompt independent channel; unlike a raw
            # 1 ms JTS finite difference, the lead derivative is filtered.
            self.motor_v = thd.copy()
            raw_deflection_v = (deflection - self.deflection_prev) / self.dt
            self.deflection_v += 0.2 * (raw_deflection_v - self.deflection_v)
            lead = 0.0010
            q = q_measured + lead * self.deflection_v
            corrected_deflection = deflection + lead * self.deflection_v
            correction_v = (corrected_deflection - self.corrected_deflection_prev) / self.dt
            target_v = thd + correction_v
            self.v += 0.2 * (target_v - self.v)
            self.corrected_deflection_prev = corrected_deflection.copy()
        else:
            q = q_measured
            raw_v = (q - self.q_prev) / self.dt
            self.v += 0.2 * (raw_v - self.v)
        self.q_prev = q.copy()
        self.deflection_prev = deflection.copy()
        return q, self.v.copy()

    def _initial_bias_update(self, t, q, thd, tau_s):
        """Estimate power-on bias during the initial taught near-rest segment."""
        if not self.USE_BIAS or t > 0.05:
            return
        _, qd_ref, _ = T.reference(t)
        if np.linalg.norm(qd_ref, ord=np.inf) > 0.035 or np.linalg.norm(thd, ord=np.inf) > 0.005:
            return
        sample = tau_s - self.nom.gravity(q)
        # The simulated rigid-joint sensor has not yet seen an applied motor
        # torque at power-on.  Calibrate elastic J1--J3 here; slow residual
        # tracking handles the small rigid-axis offsets after motion begins.
        sample = np.where(self.elastic, sample, 0.0)
        weight = 1.0
        self.bias = (self.bias_weight * self.bias + weight * sample) / (self.bias_weight + weight)
        self.bias_weight += weight

    def _track_bias(self, grip, suspicious):
        """Track slow drift only empty-handed; freeze on contact suspicion/alarm."""
        if not self.USE_BIAS or grip or suspicious or self.detected or self.integral is None:
            return
        # r_joint has the sign of an unmodelled input torque.  The deliberately
        # slow rate follows thermal drift but cannot learn away a short strike.
        self.bias -= 0.08 * self.r_joint * self.dt

    def _observer(self, q, v, tau, nom, grip):
        mass = nom.mass_matrix(q)
        momentum = mass @ v
        changed = bool(grip) != self.last_grip
        self.last_grip = bool(grip)
        if self.integral is None:
            self.integral = momentum.copy()
            self.mass_prev = mass.copy()
        mass_dot = (mass - self.mass_prev) / self.dt
        self.mass_prev = mass.copy()
        beta = nom.bias(q, v) - mass_dot @ v
        self.integral += (tau - beta + self.r_joint) * self.dt
        self.r_joint = self.observer_gain * (momentum - self.integral)
        # A model switch changes momentum algebraically.  Retaining integral is
        # intentional: the table-consistency projection explains the transfer.
        del changed

    def _table_allowed(self, grip, point_z):
        return (
            bool(grip)
            and self.box_mass > 0.0
            and bool(np.min(np.abs(self.table_heights - point_z)) <= self.height_band)
        )

    def _project_table(self, wrench, rotation):
        """Project into a conservative unilateral rectangular support set."""
        # The taught box has local +z downward.  Unilateral table support is
        # therefore expressed in WORLD +z, while COP bounds use box x/y.
        fz = max(float(wrench[2]), 0.0)
        tangential = wrench[:2].copy()
        limit = self.friction_mu * fz
        norm = float(np.linalg.norm(tangential))
        if norm > limit and norm > 0.0:
            tangential *= limit / norm
        support_force = np.array([tangential[0], tangential[1], fz])
        # Fit a COP in box coordinates, clamp it to the rectangular footprint,
        # then reconstruct r x F.  This is a physically feasible clipped
        # projection (not a unit-dependent Euclidean wrench projection).
        r_axes = rotation[:, :2]
        fx, fy, fz_support = support_force
        skew_force = np.array(
            [[0.0, -fz_support, fy], [fz_support, 0.0, -fx], [-fy, fx, 0.0]]
        )
        mapping = -skew_force @ r_axes
        cop = np.linalg.lstsq(mapping, wrench[3:], rcond=1e-8)[0]
        cop = np.clip(cop, [-self.half_x, -self.half_y], [self.half_x, self.half_y])
        lever = r_axes @ cop
        table_moment = np.cross(lever, support_force)
        torsion = np.clip(
            wrench[5] - table_moment[2], -self.torsion_radius * fz, self.torsion_radius * fz
        )
        table_moment += np.array([0.0, 0.0, torsion])
        return np.r_[support_force, table_moment]

    def _directional_alarm(self, residual, raw_wrench, table_allowed):
        force = residual[:3]
        moment = residual[3:]
        # Horizontal, negative-normal, COP/torsion violation, and a positive
        # normal force where table support is impossible are separate evidence.
        force_excess = np.array(
            [abs(force[0]), abs(force[1]), max(-raw_wrench[2], 0.0)]
        ) / self.force_threshold
        moment_excess = np.abs(moment) / self.moment_threshold
        out_height = 0.0 if table_allowed else max(raw_wrench[2], 0.0) / self.out_of_height_fz_threshold
        score = max(float(np.max(force_excess)), float(np.max(moment_excess)), float(out_height))
        return score > 1.0, score

    def _fast_directional_alarm(self, innovation):
        """Direction-neutral grip-off transient evidence."""
        force_excess = np.array(
            [abs(innovation[0]), abs(innovation[1]), max(-innovation[2], 0.0)]
        ) / self.fast_force_threshold
        moment_excess = np.abs(innovation[3:]) / self.fast_moment_threshold
        positive_fz = max(innovation[2], 0.0) / self.fast_positive_fz_threshold
        score = max(
            float(np.max(force_excess)), float(np.max(moment_excess)), float(positive_fz)
        )
        return score > 1.0, score

    def _update_load(self, t, th, tau_s, grip, table_allowed, table_wrench):
        """Update bounded public-signal payload-transfer estimates."""
        q = th - self.inv_k * tau_s
        qd = np.zeros(6) if self.load_q_prev is None else (q - self.load_q_prev) / self.dt
        self.load_q_prev = q.copy()
        self.load_v += 0.08 * (qd - self.load_v)
        qdd_ref = T.reference(t + self.dt)[2]
        tau_empty = self.nom.inverse_dynamics(q, self.load_v, qdd_ref)
        sensed = tau_s - tau_empty
        if not self.load_bias_ready:
            self.load_sensor_bias = sensed.copy()
            self.load_bias_ready = True
        elif t < T.T_GRASP - 0.08:
            self.load_sensor_bias += 0.002 * (sensed - self.load_sensor_bias)

        ramp = float(np.clip((t - T.T_GRASP) / 0.025, 0.0, 1.0)) if grip else 0.0
        if not grip:
            direct = 0.0
        else:
            payload = (
                self.nom_box.inverse_dynamics(q, self.load_v, qdd_ref)
                - self.nom.inverse_dynamics(q, self.load_v, qdd_ref)
            )
            centered = sensed - self.load_sensor_bias
            direct = float(
                np.clip(
                    float(centered @ payload) / (float(payload @ payload) + 1e-9),
                    0.0,
                    1.0,
                )
            )

        if t < T.T_GRASP + 0.20:
            band = 0.12
        elif t >= T.T_RELEASE - 0.06:
            band = 0.02
        else:
            band = 0.01
        bounded_direct = float(np.clip(direct, max(0.0, ramp - band), min(1.0, ramp + band)))

        if self.LOAD_SOURCE == "ramp":
            bounded_direct = ramp

        weight = max(float(self.box_mass) * 9.81, 1e-9)
        support_valid = bool(grip and self.box_mass > 0.0 and table_allowed)
        support = float(np.clip(1.0 - table_wrench[2] / weight, 0.0, 1.0)) if support_valid else bounded_direct
        # Optional support fusion is conservative: table support can delay but
        # never accelerate the bounded direct/ramp estimate.
        if self.USE_SUPPORT_LOAD and support_valid:
            alpha = float(np.clip(self.SUPPORT_BLEND, 0.0, 1.0))
            fused_target = bounded_direct + alpha * (min(bounded_direct, support) - bounded_direct)
        else:
            fused_target = bounded_direct
        self.load_fraction += 0.14 * (fused_target - self.load_fraction)
        if not grip and self.load_fraction < 1e-4:
            self.load_fraction = 0.0
        self.lambda_ramp = ramp
        self.lambda_direct = direct
        self.lambda_support = support
        self.lambda_support_valid = support_valid
        self.lambda_fused = float(self.load_fraction)

    def update(self, t, th, thd, tau_s, grip):
        t = float(t)
        if self.last_update_t is not None and t == self.last_update_t:
            return self
        th = np.asarray(th, float)
        thd = np.asarray(thd, float)
        tau_s = np.asarray(tau_s, float)

        # The state estimator needs the raw sensor reading because its elastic
        # deflection includes the physical transmitted torque; only the observer
        # input is zeroed.
        q_probe = th - self.inv_k * tau_s
        self._initial_bias_update(t, q_probe, thd, tau_s)
        tau = tau_s - self.bias if self.USE_BIAS else tau_s
        q, v = self._estimate_state(t, th, thd, tau_s)
        nom = self.nom_box if grip else self.nom
        self._observer(q, v, tau, nom, grip)

        if not self.USE_WRENCH:
            suspicious = bool(np.any(np.abs(self.r_joint) > 0.55 * self.joint_threshold))
            self._track_bias(bool(grip), suspicious)
            self.detected = bool(np.any(np.abs(self.r_joint) > self.joint_threshold))
            self._update_load(t, th, tau_s, bool(grip), False, np.zeros(6))
            self.sample_count += 1
            self.last_update_t = t
            return self

        jacobian, point, rotation = self._point_jacobian(q)
        try:
            wrench = np.linalg.solve(jacobian.T, self.r_joint)
        except np.linalg.LinAlgError:
            wrench = np.linalg.lstsq(jacobian.T, self.r_joint, rcond=1e-5)[0]
        allowed = self._table_allowed(grip, point[2])
        table = self._project_table(wrench, rotation) if allowed else np.zeros(6)
        residual = wrench - table
        over, score = self._directional_alarm(residual, wrench, allowed)
        if self.fast_wrench_lp is None:
            innovation = np.zeros(6)
            self.fast_wrench_lp = wrench.copy()
        else:
            innovation = wrench - self.fast_wrench_lp
            self.fast_wrench_lp += self.observer_gain * self.dt * innovation
        fast_over, fast_score = self._fast_directional_alarm(innovation)
        if grip:
            fast_over = False
            fast_score = 0.0
        self.last_wrench = wrench.copy()
        self.last_table_wrench = table.copy()
        self.last_table_allowed = allowed
        self.last_score = score
        self.last_fast_innovation = innovation.copy()
        self.last_fast_score = fast_score
        suspicious = max(score, fast_score) > 0.55
        self._track_bias(bool(grip), suspicious)

        if self.USE_CALIBRATED_THRESHOLDS:
            if over:
                self.alarm_count += 1
            else:
                self.alarm_count = 0
            if fast_over:
                self.fast_alarm_count += 1
            else:
                self.fast_alarm_count = 0
            if over or fast_over:
                self.clear_count = 0
            else:
                self.clear_count += 1
            if (
                self.alarm_count >= self.persistence
                or self.fast_alarm_count >= self.fast_persistence
            ):
                self.detected = True
            elif self.clear_count >= self.release_persistence:
                self.detected = False
        else:
            self.detected = over
        self.last_residual_wrench = residual.copy()
        self._update_load(t, th, tau_s, bool(grip), allowed, table)
        self.sample_count += 1
        self.last_update_t = t
        return self


class Controller:
    MODE = MODE
    USE_B1 = True
    USE_B2 = True
    USE_B3 = True
    LOAD_SOURCE = "sensor"
    USE_SUPPORT_LOAD = True
    UNLOAD_START = T.T_RELEASE - 0.06
    UNLOAD_AMOUNT = 0.10
    ENDPOINT_DELAY = -0.005

    def reset(self):
        self.dt = float(C.DT)
        self.elastic = np.asarray(self.nom.elastic, bool)
        self.inv_k = np.zeros(6)
        self.inv_k[self.elastic] = 1.0 / self.nom.K[self.elastic]
        self.v_motor = np.zeros(6)
        self.load_fraction = 0.0
        self.shared_estimator = SharedEstimator(self.nom, self.nom_box, self.box_mass, self.box_com)
        self.shared_estimator.USE_SUPPORT_LOAD = self.USE_SUPPORT_LOAD
        self.shared_estimator.LOAD_SOURCE = self.LOAD_SOURCE
        self.tau_link_prev = None
        self.tau_dot = np.zeros(6)
        # Motor-side feedback is deliberately below the elastic resonance.
        # B2 k_tau=-0.10 gives B_eff=B/(1-k_tau)=0.91 B, bounded away
        # from the empirically degraded large-negative-gain candidates.
        self.wn = 18.0
        self.k_tau = -0.10 if self.USE_B2 else 0.0

    def _reference(self, t):
        ta = t + self.dt
        if not self.USE_B3 or ta < T.T_RELEASE - 0.15 or ta >= T.T_RELEASE + 0.10:
            return T.reference(ta)
        if ta < T.T_RELEASE:
            duration = 0.15
            s = np.clip((ta - (T.T_RELEASE - duration)) / duration, 0.0, 1.0)
            sign = 1.0
        else:
            duration = 0.10
            s = np.clip((ta - T.T_RELEASE) / duration, 0.0, 1.0)
            sign = -1.0
        h = 10 * s**3 - 15 * s**4 + 6 * s**5
        hd = (30 * s**2 - 60 * s**3 + 30 * s**4) / duration
        hdd = (60 * s - 180 * s**2 + 120 * s**3) / duration**2
        delay = self.ENDPOINT_DELAY * (h if sign > 0 else 1.0 - h)
        delay_d, delay_dd = self.ENDPOINT_DELAY * sign * hd, self.ENDPOINT_DELAY * sign * hdd
        qr, qdr, qddr = T.reference(ta - delay)
        scale = 1.0 - delay_d
        return qr, qdr * scale, qddr * scale**2 - qdr * delay_dd

    def _estimate_load(self, t, th, thd, tau_s, grip):
        packet = self.shared_estimator.update(t, th, thd, tau_s, grip)
        self.load_fraction = packet.lambda_fused
        self.lambda_sensor = packet.lambda_direct
        self.lambda_trajectory = packet.lambda_ramp
        return self.load_fraction

    def compute(self, t, th, thd, tau_s, grip, q_d, qd_d, qdd_d):
        q_ref, qd_ref, qdd_ref = self._reference(t)
        use_load = self.USE_B3
        load = self._estimate_load(t, th, thd, tau_s, grip) if use_load else float(grip)
        if self.USE_B3 and self.UNLOAD_START < t < T.T_RELEASE:
            s = np.clip((t - self.UNLOAD_START) / (T.T_RELEASE - self.UNLOAD_START), 0.0, 1.0)
            unload = 1.0 - self.UNLOAD_AMOUNT * (10 * s**3 - 15 * s**4 + 6 * s**5)
            load = min(load, unload)
        tau_empty = self.nom.inverse_dynamics(q_ref, qd_ref, qdd_ref)
        tau_full = self.nom_box.inverse_dynamics(q_ref, qd_ref, qdd_ref)
        tau_link = tau_empty + load * (tau_full - tau_empty)

        if not self.USE_B1:
            self.v_motor += 0.2 * (thd - self.v_motor)
            a = qdd_ref + 40.0 * (qd_ref - self.v_motor) + 400.0 * (q_ref - th)
            tau = (self.nom.mass_matrix(th) + np.diag(self.nom.B)) @ a + self.nom.bias(th, self.v_motor)
            return tau + load * (tau_full - tau_empty)

        theta = q_ref + self.inv_k * tau_link
        self.v_motor += 0.2 * (thd - self.v_motor)
        if self.tau_link_prev is None:
            theta_dot = qd_ref.copy()
            theta_ddot = qdd_ref.copy()
        else:
            raw_tau_dot = (tau_link - self.tau_link_prev) / self.dt
            old_tau_dot = self.tau_dot.copy()
            self.tau_dot += 0.04 * (raw_tau_dot - self.tau_dot)
            theta_dot = qd_ref + self.inv_k * self.tau_dot
            theta_ddot = qdd_ref + self.inv_k * (self.tau_dot - old_tau_dot) / self.dt
            theta_dot = np.clip(theta_dot, qd_ref - 0.35, qd_ref + 0.35)
            theta_ddot = np.clip(theta_ddot, -25.0, 25.0)
        self.tau_link_prev = tau_link.copy()
        friction = C.VISCOUS * theta_dot + C.COULOMB * np.tanh(theta_dot / C.FRICTION_EPS)
        pd = self.nom.B * (self.wn**2 * (theta - th) + 2.0 * self.wn * (theta_dot - self.v_motor))
        torque_feedback = self.k_tau * (tau_s - tau_link)
        return self.nom.B * theta_ddot + tau_link + friction + pd + torque_feedback
_R1Controller = Controller


# Registered R3-B default.  The offline builder may replace this literal with
# another preregistered specification before a final submission is frozen.
R3_DEFAULT_PARAMETERS = {
    "id": "B_mass_phase",
    "family": "R3_B",
    "architecture": "torque",
    "rate": 4.5,
    "damping_scale": 1.0,
    "schedule": "mass_phase",
    "mass_start_kg": 3.15,
    "mass_full_kg": 5.65,
    "handoff_start_s": 3.60,
    "handoff_end_s": 3.84,
}

_R3_TORQUE_RESERVE = np.array([
    76.11050185661823, 4.861909300407845, 10.756108095800712,
    10.240250777351067, 7.2263397966225, 11.772429133142767,
])
_R3_POSITION_SCALE = np.array([
    1.0852359661170992e-3, 2.1129945173399014e-3,
    1.2282315975258654e-3, 1.4115376149470194e-4,
    1.8932633563804844e-3, 8.508155462461925e-4,
])


def _r3_skew(value, out=None):
    x, y, z = value
    if out is None:
        out = np.empty((3, 3))
    out[:] = ((0.0, -z, y), (z, 0.0, -x), (-y, x, 0.0))
    return out


def _r3_ad(twist, out):
    out.fill(0.0)
    _r3_skew(twist[:3], out[:3, :3])
    _r3_skew(twist[3:], out[3:, :3])
    out[3:, 3:] = out[:3, :3]
    return out


def _r3_motion_transform(parent_rotation, parent_position,
                         child_rotation, child_position, out):
    rotation = child_rotation.T @ parent_rotation
    offset = parent_rotation.T @ (child_position - parent_position)
    out.fill(0.0)
    out[:3, :3] = rotation
    out[3:, 3:] = rotation
    out[3:, :3] = -rotation @ _r3_skew(offset)
    return out


def _r3_body_inertia(model, body):
    mass = float(model.body_mass[body])
    centre = np.asarray(model.body_ipos[body], dtype=float)
    rotation_flat = np.empty(9)
    mujoco.mju_quat2Mat(rotation_flat, model.body_iquat[body])
    rotation = rotation_flat.reshape(3, 3)
    central = rotation @ np.diag(model.body_inertia[body]) @ rotation.T
    cross = _r3_skew(centre)
    out = np.zeros((6, 6))
    out[:3, :3] = central - mass * cross @ cross
    out[:3, 3:] = mass * cross
    out[3:, :3] = -mass * cross
    out[3:, 3:] = mass * np.eye(3)
    return out


class _R3BlendedPRNE:
    """Preallocated one-pass ``C(q, v)u`` with endpoint-linear inertia."""

    def __init__(self, empty, full):
        self.nominal = empty
        self.model = empty.m
        if self.model.nv != 6 or self.model.njnt != 6:
            raise ValueError("R3 pRNE requires the public six-hinge nominal model")
        self.bodies = tuple(int(self.model.jnt_bodyid[j]) for j in range(6))
        if any(int(self.model.jnt_type[j]) != mujoco.mjtJoint.mjJNT_HINGE
               for j in range(6)):
            raise ValueError("R3 pRNE supports hinge joints only")
        if any(int(self.model.body_parentid[self.bodies[j]]) !=
               (0 if j == 0 else self.bodies[j - 1]) for j in range(6)):
            raise ValueError("R3 pRNE requires a serial six-link chain")
        self.motion_subspaces = np.array([
            np.r_[self.model.jnt_axis[j],
                  -np.cross(self.model.jnt_axis[j], self.model.jnt_pos[j])]
            for j in range(6)
        ])
        self.inertia_empty = self._fold_fixed_inertias(empty)
        self.inertia_delta = self._fold_fixed_inertias(full) - self.inertia_empty
        # Workspaces are overwritten on every action.  They are scratch, not
        # controller dynamics, and action returns a copy to prevent aliasing.
        self.transforms = np.zeros((6, 6, 6))
        self.twists = np.zeros((6, 6))
        self.designs = np.zeros((6, 6))
        self.partials = np.zeros((6, 6))
        self.forces = np.zeros((6, 6))
        self.inertias = np.zeros((6, 6, 6))
        self.transported = np.zeros((6, 6))
        self.result = np.zeros(6)
        self._ad_work = np.zeros((6, 6))
        self._vector_work = np.zeros(6)
        self._vector_work2 = np.zeros(6)
        self._zero6 = np.zeros(6)
        self._zero3 = np.zeros(3)
        self._identity3 = np.eye(3)
        self._rotation_work = np.zeros((3, 3))
        self._offset_work = np.zeros(3)
        self._offset_work2 = np.zeros(3)
        self._skew_work = np.zeros((3, 3))

    def _fold_fixed_inertias(self, nominal):
        model = nominal.m
        data = mujoco.MjData(model)
        data.qpos[:] = 0.0
        mujoco.mj_forward(model, data)
        bodies = tuple(int(model.jnt_bodyid[j]) for j in range(6))
        inertias = np.array([_r3_body_inertia(model, body) for body in bodies])
        body_to_link = {body: index for index, body in enumerate(bodies)}
        transform = np.zeros((6, 6))
        for body in range(1, model.nbody):
            if body in body_to_link or model.body_jntnum[body]:
                continue
            ancestor = int(model.body_parentid[body])
            while ancestor not in body_to_link and ancestor != 0:
                ancestor = int(model.body_parentid[ancestor])
            if ancestor == 0:
                continue
            link = body_to_link[ancestor]
            _r3_motion_transform(
                data.xmat[ancestor].reshape(3, 3), data.xpos[ancestor],
                data.xmat[body].reshape(3, 3), data.xpos[body], transform)
            inertias[link] += transform.T @ _r3_body_inertia(model, body) @ transform
        return inertias

    @staticmethod
    def _vector(value, name):
        out = np.asarray(value, dtype=float)
        if out.shape != (6,) or not np.all(np.isfinite(out)):
            raise ValueError(name + " must be a finite six-vector")
        return out

    def action(self, q, velocity, design, load):
        q = self._vector(q, "q")
        velocity = self._vector(velocity, "velocity")
        design = self._vector(design, "design")
        load = float(load)
        if not np.isfinite(load) or load < 0.0 or load > 1.0:
            raise ValueError("load must be finite and in [0, 1]")

        # Exactly one kinematic model update and one forward recursion.
        self.nominal._update(q, self._zero6)
        data = self.nominal.d
        parent_rotation = self._identity3
        parent_position = self._zero3
        parent_twist = self._zero6
        parent_design = self._zero6
        parent_partial = self._zero6
        for index, body in enumerate(self.bodies):
            rotation = data.xmat[body].reshape(3, 3)
            position = data.xpos[body]
            transform = self.transforms[index]
            np.matmul(rotation.T, parent_rotation, out=self._rotation_work)
            np.subtract(position, parent_position, out=self._offset_work)
            np.matmul(parent_rotation.T, self._offset_work,
                      out=self._offset_work2)
            _r3_skew(self._offset_work2, self._skew_work)
            transform.fill(0.0)
            transform[:3, :3] = self._rotation_work
            transform[3:, 3:] = self._rotation_work
            np.matmul(self._rotation_work, self._skew_work,
                      out=transform[3:, :3])
            transform[3:, :3] *= -1.0
            np.matmul(transform, parent_twist, out=self.transported[index])
            self.twists[index] = self.transported[index]
            self.twists[index] += self.motion_subspaces[index] * velocity[index]
            np.matmul(transform, parent_design, out=self.designs[index])
            self.designs[index] += self.motion_subspaces[index] * design[index]
            np.matmul(transform, parent_partial, out=self.partials[index])
            _r3_ad(self.transported[index], self._ad_work)
            np.multiply(self.motion_subspaces[index], design[index],
                        out=self._vector_work2)
            np.matmul(self._ad_work, self._vector_work2,
                      out=self._vector_work)
            self.partials[index] += self._vector_work
            parent_twist = self.twists[index]
            parent_design = self.designs[index]
            parent_partial = self.partials[index]
            parent_rotation, parent_position = rotation, position

        for index in range(6):
            inertia = self.inertias[index]
            np.multiply(self.inertia_delta[index], load, out=inertia)
            inertia += self.inertia_empty[index]
            np.matmul(inertia, self.partials[index], out=self._vector_work)
            np.matmul(inertia, self.designs[index], out=self._vector_work2)
            _r3_ad(self.twists[index], self._ad_work)
            np.matmul(self._ad_work.T, self._vector_work2,
                      out=self.forces[index])
            self.forces[index] *= -1.0
            self.forces[index] += self._vector_work
        for index in range(5, -1, -1):
            self.result[index] = self.motion_subspaces[index] @ self.forces[index]
            if index:
                np.matmul(self.transforms[index].T, self.forces[index],
                          out=self._vector_work)
                self.forces[index - 1] += self._vector_work
        return self.result.copy()


def _r3_smoothstep(value):
    x = float(np.clip(value, 0.0, 1.0))
    return 10.0 * x**3 - 15.0 * x**4 + 6.0 * x**5


def _r3_validate(architecture, parameters):
    if architecture != "torque":
        raise ValueError("architecture must be torque")
    p = dict(parameters)
    if p.get("family", "R3_B") != "R3_B":
        raise ValueError("family must be R3_B")
    if p.get("architecture", architecture) != architecture:
        raise ValueError("parameter architecture must match architecture")
    schedule = p.get("schedule")
    if schedule not in {"phase", "mass", "mass_phase"}:
        raise ValueError("schedule must be phase, mass, or mass_phase")
    for name in ("rate", "damping_scale", "mass_start_kg", "mass_full_kg",
                 "handoff_start_s", "handoff_end_s"):
        value = float(p.get(name, np.nan))
        if not np.isfinite(value):
            raise ValueError(name + " must be finite")
        p[name] = value
    if p["rate"] <= 0.0 or p["damping_scale"] <= 0.0:
        raise ValueError("rate and damping_scale must be positive")
    if p["mass_full_kg"] <= p["mass_start_kg"]:
        raise ValueError("mass_full_kg must exceed mass_start_kg")
    if p["handoff_end_s"] <= p["handoff_start_s"]:
        raise ValueError("handoff_end_s must exceed handoff_start_s")
    p["family"] = "R3_B"
    p["architecture"] = architecture
    return p


def make_controller(architecture, params):
    """Return a self-contained configured R3 torque P-IDC controller class."""
    parameters = _r3_validate(architecture, params)

    class R3TorqueController(_R1Controller):
        R3_PARAMETERS = parameters
        R2_STABILITY_STATE = ("r3_reference_integral", "r3_last_grip")

        def reset(self):
            super().reset()
            self.r3_reference_integral = np.zeros(6)
            self.r3_last_grip = False
            self.r3_reserve = _R3_TORQUE_RESERVE.copy()
            self.r3_position_scale = _R3_POSITION_SCALE.copy()
            self.r3_prne = _R3BlendedPRNE(self.nom, self.nom_box)

        def _r3_alpha(self, t):
            mass_x = ((self.box_mass - parameters["mass_start_kg"])
                      / (parameters["mass_full_kg"]
                         - parameters["mass_start_kg"]))
            mass_alpha = _r3_smoothstep(mass_x)
            phase_x = ((t - parameters["handoff_start_s"])
                       / (parameters["handoff_end_s"]
                          - parameters["handoff_start_s"]))
            phase_alpha = 1.0 - _r3_smoothstep(phase_x)
            if parameters["schedule"] == "phase":
                return phase_alpha
            if parameters["schedule"] == "mass":
                return mass_alpha
            return mass_alpha * phase_alpha

        def _r3_pidc_link(self, q, velocity, q_ref, qd_ref, qdd_ref,
                          load, baseline, t, grip):
            error = q_ref - q
            velocity_error = qd_ref - velocity
            switched = bool(grip) != self.r3_last_grip
            if switched:
                self.r3_reference_integral.fill(0.0)
            near_pick = T.T_GRASP - 0.08 <= t <= T.T_GRASP + 0.08
            near_place = T.WAYPOINTS[3][0] <= t <= T.T_RELEASE + 0.10
            frozen = bool(near_pick or near_place
                          or self.shared_estimator.detected or not grip)
            if frozen:
                self.r3_reference_integral *= 0.95
            else:
                limit = 10.0 * self.r3_position_scale / parameters["rate"]
                self.r3_reference_integral = np.clip(
                    self.r3_reference_integral + self.dt * error,
                    -limit, limit)
            self.r3_last_grip = bool(grip)

            rate = parameters["rate"]
            damping = parameters["damping_scale"]
            qd_r = (qd_ref + 2.0 * damping * rate * error
                    + rate**2 * self.r3_reference_integral)
            qdd_r = qdd_ref + 2.0 * damping * rate * velocity_error + rate**2 * error
            acceleration = qdd_r + damping * rate * (qd_r - velocity)
            zeros = np.zeros(6)
            empty = self.nom.inverse_dynamics(q, zeros, acceleration)
            full = self.nom_box.inverse_dynamics(q, zeros, acceleration)
            candidate = (empty + load * (full - empty)
                         + self.r3_prne.action(q, velocity, qd_r, load))
            return baseline + np.clip(candidate - baseline,
                                      -self.r3_reserve, self.r3_reserve)

        def _r3_lift(self, q_ref, qd_ref, qdd_ref, tau_link, th, thd, tau_s):
            theta = q_ref + self.inv_k * tau_link
            self.v_motor += 0.2 * (thd - self.v_motor)
            if self.tau_link_prev is None:
                theta_dot = qd_ref.copy()
                theta_ddot = qdd_ref.copy()
            else:
                raw_tau_dot = (tau_link - self.tau_link_prev) / self.dt
                old_tau_dot = self.tau_dot.copy()
                self.tau_dot += 0.04 * (raw_tau_dot - self.tau_dot)
                theta_dot = qd_ref + self.inv_k * self.tau_dot
                theta_ddot = (qdd_ref + self.inv_k
                              * (self.tau_dot - old_tau_dot) / self.dt)
                theta_dot = np.clip(theta_dot, qd_ref - 0.35, qd_ref + 0.35)
                theta_ddot = np.clip(theta_ddot, -25.0, 25.0)
            self.tau_link_prev = tau_link.copy()
            friction = (C.VISCOUS * theta_dot
                        + C.COULOMB * np.tanh(theta_dot / C.FRICTION_EPS))
            pd = self.nom.B * (
                self.wn**2 * (theta - th)
                + 2.0 * self.wn * (theta_dot - self.v_motor))
            torque_feedback = self.k_tau * (tau_s - tau_link)
            return (self.nom.B * theta_ddot + tau_link + friction + pd
                    + torque_feedback)

        def compute(self, t, th, thd, tau_s, grip, q_d, qd_d, qdd_d):
            del q_d, qd_d, qdd_d
            q_ref, qd_ref, qdd_ref = self._reference(t)
            load = self._estimate_load(t, th, thd, tau_s, grip)
            if self.UNLOAD_START < t < T.T_RELEASE:
                x = ((t - self.UNLOAD_START)
                     / (T.T_RELEASE - self.UNLOAD_START))
                load = min(load, 1.0 - self.UNLOAD_AMOUNT * _r3_smoothstep(x))
            empty = self.nom.inverse_dynamics(q_ref, qd_ref, qdd_ref)
            full = self.nom_box.inverse_dynamics(q_ref, qd_ref, qdd_ref)
            baseline = empty + load * (full - empty)
            alpha = self._r3_alpha(float(t))
            if alpha == 0.0:
                tau_link = baseline
            else:
                q = np.asarray(self.shared_estimator.q_prev, dtype=float)
                velocity = np.asarray(self.shared_estimator.v, dtype=float)
                clipped = self._r3_pidc_link(
                    q, velocity, q_ref, qd_ref, qdd_ref, load,
                    baseline, float(t), bool(grip))
                tau_link = baseline + alpha * (clipped - baseline)
            return self._r3_lift(
                q_ref, qd_ref, qdd_ref, tau_link, th, thd, tau_s)

    R3TorqueController.__name__ = "R3TorqueController"
    return R3TorqueController


Controller = make_controller("torque", R3_DEFAULT_PARAMETERS)
Controller.R3_PARAMETERS = R3_DEFAULT_PARAMETERS


__all__ = [
    "SharedEstimator", "R3_DEFAULT_PARAMETERS", "make_controller", "Controller",
]
