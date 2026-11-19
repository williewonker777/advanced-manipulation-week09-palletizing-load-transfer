# Week09 final hand-in: unchanged R10 body below; R31 torque linkage, R32 servo linkage
# and the mode adapter follow.
# Original course simulator, models and score sheets are not distributed in this file.
"""Collision detector for load-transfer-aware palletizing.

The production path uses only the documented detector inputs and public models.
It deliberately does not read simulator state or configuration values describing
the JTS offset, drift, or random seed.

The returned residual preserves the simulator contract: six observer joint
torques in N m.  Internally, ``last_wrench`` and ``last_residual_wrench`` are
world-frame wrenches about the box bottom centre in N and N m.  During plausible
table contact a feasible table-consistent component is removed before
thresholding.  A vertical collision on the table-normal line is fundamentally
indistinguishable from table support with these signals alone; the detector
consequently does not claim to separate that case.
"""

import mujoco
import numpy as np
from copy import copy

from sim import config as C
from sim import trajectory as T


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



def normal_features(t, mass, com):
    q, qd, qdd = T.reference(float(t))
    previous_qdd = T.reference(max(float(t) - 0.001, 0.0))[2]
    jerk = (qdd - previous_qdd) / 0.001
    m = float(mass) / 8.0
    c = np.asarray(com, float) / 0.05
    return np.r_[1.0, np.sin(q), np.cos(q), qd / 0.8, qdd / 5.0,
                 qd * np.abs(qd), m, c, m * qd / 0.8, m * qdd / 5.0,
                 m * np.sin(q), (c[:, None] * np.sin(q)[None, :]).ravel(),
                 jerk / 50.0]


class NormalResidualModel:
    """Fixed numpy inference, bounded features, no adaptation or file access."""

    def __init__(self, mean, scale, weights, threshold):
        self.mean = np.asarray(mean, float)
        self.scale = np.asarray(scale, float)
        self.weights = np.asarray(weights, float)
        self.threshold = float(threshold)
        self.ema = 0.0
        self.count = 0
        self.clear = 0
        self.detected = False
        self.last_error = np.zeros(2)

    _reference_grid = None

    def predict_features(self, x):
        z = np.clip((x - self.mean) / self.scale, -8.0, 8.0)
        return z @ self.weights

    def update(self, t, mass, com, wrench, active):
        if not active:
            self.ema, self.count, self.clear, self.detected = 0.0, 0, 0, False
            return False
        grid = self._reference_grid
        k = None if grid is None else grid.index(t, mass, com)
        predicted = (self.predict_features(normal_features(t, mass, com)) if k is None
                     else grid.r1[k])
        self.last_error = np.asarray(wrench[:2], float) - predicted
        value = float(np.linalg.norm(self.last_error))
        if not np.isfinite(value):
            self.ema, self.count, self.clear, self.detected = 0.0, 0, 0, False
            return False
        self.ema += 0.1 * (value - self.ema)
        over = self.ema > self.threshold
        self.count = self.count + 1 if over else 0
        self.clear = 0 if over else self.clear + 1
        if self.count >= 4:
            self.detected = True
        elif self.clear >= 8:
            self.detected = False
        return bool(self.detected)


MODEL_PARAMETERS = {'servo': {'mean': [0.0, 0.11572670375487418, 0.38574042456434987, 0.983919475122151, 0.0, 0.9372780766510267, 0.11572670375487418, 0.7863772105471554, 0.9165817055548328, 0.047957464332617245, 1.0, 0.3417763751678762, 0.7863772105471554, 0.9400159437945879, 0.15816251071489998, -0.24604253870829085, 0.0, 0.08788002799339238, 0.9400159437945879, 4.86496242372324e-17, 0.17301245372256882, -0.07697772391314486, 0.0, -0.09603472980942399, 4.86496242372324e-17, 0.9375070423119342, 0.02676868518989958, -0.06271515370186044, 0.0, 0.005607011931503153, 0.9375070423119342, 0.5019015987603181, 0.009893415980032752, 0.020010582810498553, 0.4166135006424108, 0.4719663384468689, 0.07965326005666394, -0.12365510771073354, 0.0, 0.04400184765407005, 0.4719663384468689, -2.734264612542928e-18, 0.08675840746681697, -0.03859510121153015, 0.0, -0.04816330625528681, -2.734264612542928e-18, 0.05822126590872468, 0.1936145617304672, 0.49382959583155306, 0.0, 0.47042980061095374, 0.05822126590872468, 0.001225777924967238, 0.003822642406658235, 0.009733643495331738, 0.0, 0.009277828495378938, 0.001225777924967238, 0.0023077315756373958, 0.007718255337758821, 0.019688869415854283, 0.0, 0.01875499053116785, 0.0023077315756373958, 0.04825363435943686, 0.16070782574565742, 0.4099137962679053, 0.0, 0.3904851707539144, 0.04825363435943686, 9.919581303548586e-16, -0.05404561423943729, 0.023567164362335427, 0.0, 0.030478449877101518, 9.919581303548586e-16], 'scale': [1.0, 0.5897870271270552, 0.0968599365167439, 0.012038987386321575, 1e-06, 0.023658775313239417, 0.5897870271270552, 0.14272167110403539, 0.04123656868696441, 0.17163219649249153, 1e-06, 0.06433489533006023, 0.14272167110403539, 0.7623809933519893, 0.30589839706213867, 0.23800479287828993, 1e-06, 0.18753422186234567, 0.7623809933519893, 0.35306469093619813, 0.5111170565420915, 0.24961724011882422, 1e-06, 0.28308480088485233, 0.35306469093619813, 0.9855490574666339, 0.19097185044416906, 0.08042533234543066, 1e-06, 0.06304941492324134, 0.9855490574666339, 0.3067832530724128, 0.35246826284201327, 0.2825935731081902, 0.3238405631177272, 0.5332705123039666, 0.18546218312587934, 0.1587431252584119, 1e-06, 0.11328872104386163, 0.5332705123039666, 0.2077382707376466, 0.3053078488395893, 0.14871793824549662, 1e-06, 0.16911667770705452, 0.2077382707376466, 0.34870945379339896, 0.1313595885411736, 0.3019313331649242, 1e-06, 0.2878902275718613, 0.34870945379339896, 0.2119249727138117, 0.14018542522715152, 0.34682635720121235, 1e-06, 0.3304661574814163, 0.2119249727138117, 0.17025680008630262, 0.11241020591396325, 0.278070073339808, 1e-06, 0.2649547330625572, 0.17025680008630262, 0.31345503771437977, 0.13497129948335998, 0.3186962566795247, 1e-06, 0.30378595250904356, 0.31345503771437977, 0.12042732248164731, 1.150220848866899, 0.5150883893831165, 1e-06, 0.6451010706521326, 0.12042732248164731], 'weights': [[2.1415426087914837, -0.5972207110754633], [0.6541718026805936, 0.37464223114678313], [19.348145876377153, -12.226329716306475], [-12.773272745598641, 42.79557616132514], [0.0, 0.0], [-29.272410561861157, 33.67508535298402], [0.6541718057592105, 0.3746422274971113], [12.708080206344253, -9.138289144573434], [30.752231185062257, -18.189269748290307], [14.094269566203602, -6.299229077593697], [0.0, 0.0], [-29.935679284751053, 14.008474914278919], [12.70808020497769, -9.138289144061694], [21.549803952530894, 4.804733336682907], [-5.59692656783293, -2.1216286280885237], [29.644082136428388, 17.24485339002403], [0.0, 0.0], [-28.492627681282297, -18.42519689477563], [21.54980395005632, 4.804733332790375], [-1.4056167489786766, -2.0994318967580106], [-0.3823505000451137, -1.0615493655868213], [-4.400967856372608, -10.019913714278653], [0.0, 0.0], [4.57100951189915, 10.751969691206158], [-1.4056167486659719, -2.099431896704852], [-7.194055932798449, -7.230733484217605], [2.5521868033165473, 4.113711989381807], [-0.9846047009604553, -4.8177118375681385], [0.0, 0.0], [4.36203361260776, 7.982327766123036], [-7.194055932762588, -7.230733485553069], [-12.794012649109314, 4.921382680963286], [-4.089357665149339, 0.3696323964787152], [-0.3487126884999562, 0.02020948037404354], [-3.0415680627619737, 0.45120103600298184], [-2.1110919549078306, 4.513326678666471], [1.0111260641299797, -1.90406994031447], [-2.7463174295131716, 6.006935744204344], [0.0, 0.0], [2.192922295660957, -5.299969644444166], [-2.111091954230141, 4.513326677278052], [-0.19616187389580747, -0.08116058159022481], [-0.008972002385260235, -0.11551641548802843], [-0.3508962995801561, -0.936000681679578], [0.0, 0.0], [0.3247686592649537, 1.031643727669816], [-0.19616187393272488, -0.08116058153849005], [-0.9691791611444361, 1.5560369727010976], [4.081162034999742, -7.639385704749027], [8.655602478693131, -6.853794446098776], [0.0, 0.0], [0.8443704200088006, 8.291533516417191], [-0.9691791611780897, 1.5560369725330099], [-0.09158673261518863, 0.004735324953137031], [0.5998937463544323, -0.24866659234214286], [1.7160806672886764, -1.710750637466014], [0.0, 0.0], [1.8297626220960175, 1.5793081248142404], [-0.09158673275968907, 0.0047353252771468075], [-0.02603218742682271, -0.004394851495433097], [0.20869446773000747, 0.04643265497654859], [0.07680670493489536, 0.026630487807776607], [0.0, 0.0], [0.08039741184983933, -0.08995238222108744], [-0.026032187193709325, -0.004394851825151104], [-0.104488377014968, 0.003972805553853104], [0.48631952828681435, -0.19450377321619156], [1.6039601235133845, -1.4796287370381571], [0.0, 0.0], [1.015656820177521, 1.2071323426842382], [-0.10448837763433501, 0.003972806868650456], [1.8017218737059968, -2.72962459521185], [0.3797362552030761, -0.3596863536307485], [7.13331025272447, -7.986217601968725], [0.0, 0.0], [-6.372746904783257, 7.018011450321327], [1.8017218552212506, -2.7296246202343912]], 'threshold': 8.818292649779494}, 'torque': {'mean': [0.0, 0.11846134736949913, 0.38603033368335393, 0.9838879075654648, 0.0, 0.9374282528841942, 0.11846134736949913, 0.786345161095689, 0.9164407904727447, 0.04867014135256317, 1.0, 0.3414113336762904, 0.786345161095689, 0.9425809838898358, 0.1632530268550477, -0.24898378247142922, 0.0, 0.08573075561638306, 0.9425809838898358, 4.878237542896592e-17, 0.17075821436744001, -0.07586091938130096, 0.0, -0.09489729498613907, 4.878237542896592e-17, 0.9400652363180911, 0.030678341706983876, -0.06381044161816998, 0.0, 0.004618162384383577, 0.9400652363180911, 0.5015453270674028, 0.009717274312124116, 0.019986356401128242, 0.4165011748653416, 0.4732542023280247, 0.08266752619570635, -0.12536105859698085, 0.0, 0.04269353240127494, 0.4732542023280247, -2.7417256544630025e-18, 0.08527935844890018, -0.03786075576531893, 0.0, -0.04741860268358126, -2.7417256544630025e-18, 0.059823291920122014, 0.1936475754267334, 0.4934611984817107, 0.0, 0.47018693804811074, 0.059823291920122014, 0.0013663460427946737, 0.003769834577981256, 0.009559022358202824, 0.0, 0.009121995190418763, 0.0013663460427946737, 0.0023779968837002977, 0.007716382032582109, 0.019664263433600115, 0.0, 0.01873635650887966, 0.0023779968837002977, 0.04946527081661222, 0.1607931872080117, 0.4097894950818068, 0.0, 0.3904473898894535, 0.04946527081661222, 9.946649061217635e-16, -0.06722067933596845, 0.02997876365325317, 0.0, 0.03724191568271496, 9.946649061217635e-16], 'scale': [1.0, 0.5892230997931474, 0.09700701137496642, 0.012064313064604638, 1e-06, 0.023553741945714413, 0.5892230997931474, 0.14298368952766197, 0.04130979034200211, 0.17161076608838097, 1e-06, 0.06412326545188483, 0.14298368952766197, 0.7618352975258227, 0.29540898653013986, 0.23282325676431814, 1e-06, 0.1856902580674399, 0.7618352975258227, 0.35354607007483707, 0.5109654681538685, 0.24942784171526117, 1e-06, 0.2831837062298793, 0.35354607007483707, 0.9856736274693554, 0.18079074772013684, 0.07822756617864048, 1e-06, 0.06154420987113675, 0.9856736274693554, 0.3067940061702336, 0.352497499114425, 0.2825528712148625, 0.32388756189465784, 0.5334266009878934, 0.17807552100484247, 0.15585031087689769, 1e-06, 0.11151501348643574, 0.5334266009878934, 0.20802150741465572, 0.30465643642617224, 0.14832022121895258, 1e-06, 0.1689182298022465, 0.20802150741465572, 0.3482179234003986, 0.131510303797099, 0.30192940483151315, 1e-06, 0.2879668310568354, 0.3482179234003986, 0.21193366337235695, 0.1403047303719655, 0.34684440275075384, 1e-06, 0.3305432436333636, 0.21193366337235695, 0.1702209779793869, 0.11248970012314834, 0.2780207229933285, 1e-06, 0.2649618539424317, 0.1702209779793869, 0.313213607463804, 0.1351061312423909, 0.3187322087213883, 1e-06, 0.3038784296669911, 0.313213607463804, 0.12059151675610472, 1.1339963633196728, 0.5054714090470979, 1e-06, 0.6384465099008163, 0.12059151675610472], 'weights': [[1.9932328364277268, -0.9147903275901091], [0.3208517402464159, 0.640174089071607], [14.492161380547138, -9.235100122126159], [-14.530833037459587, 39.22491116871527], [0.0, 0.0], [-22.84106507426306, 27.643322170684886], [0.3208517417351296, 0.6401740874670223], [12.525742568042716, -7.892111552318201], [25.83694669821344, -14.931490817350992], [11.196869851823347, -4.57489930220624], [0.0, 0.0], [-25.960175353853785, 11.120828537939262], [12.525742573177832, -7.8921115522805785], [18.740407523705805, 4.10263065798091], [-4.231633519607574, -1.764747472948216], [24.802156447069994, 15.164667190644145], [0.0, 0.0], [-24.3656092506769, -16.206369524191725], [18.740407524114765, 4.10263065660549], [-1.3681653402511547, -1.9329054224914979], [-0.3297737188801834, -0.958501985488006], [-4.09207192748277, -9.38828833366304], [0.0, 0.0], [4.199322315010671, 9.998675229554486], [-1.368165341737726, -1.9329054209390586], [-5.47847224147668, -7.063638104285476], [1.0684590821244608, 4.055620228552184], [0.8117090308877934, -4.3449466717267216], [0.0, 0.0], [2.470121044826911, 7.500555378163836], [-5.478472239804629, -7.063638103825454], [-10.472167816250707, 1.7814266359362352], [-3.7770676930024356, -0.5609046308646535], [-0.3371189810345514, -0.4383905646201564], [-2.7039958130696076, 0.2202824937979239], [-1.001151535202005, 4.122728939399553], [0.5214641058736718, -1.6897849819368969], [-1.245410437209134, 5.38884105342867], [0.0, 0.0], [0.9078384012411518, -4.832920652177674], [-1.001151535251835, 4.122728939096936], [-0.17117953818850512, -0.07729518403961406], [-0.02188092896949199, -0.11112303800471768], [-0.526076380442855, -0.744235581812327], [0.0, 0.0], [0.5013901131937474, 0.853901533378803], [-0.1711795381800259, -0.07729518408067051], [-0.3872473725716493, 1.3672262703286435], [1.2213995354256963, -6.979804854946536], [5.314573611746667, -4.5716742741486875], [0.0, 0.0], [4.257570445528044, 8.636076020859088], [-0.3872473726120323, 1.3672262703294606], [-0.06087607527772601, 0.04745257125216117], [0.23978816060179362, -0.739584767709038], [0.8032951774738171, -1.6750948243981199], [0.0, 0.0], [2.7654573517461682, 2.9317322359737443], [-0.060876075365691035, 0.04745257151784892], [-0.01071593291927411, 0.005309020374209378], [0.05576328377378575, -0.13436193364812762], [-0.01235801675571768, -0.25383960989920035], [0.0, 0.0], [0.29964941471438183, 0.8198814665933098], [-0.010715932626266337, 0.005309019994946384], [-0.07147500965187555, 0.07484992454941064], [0.23101438742717825, -0.6112951298848894], [0.8348913819997539, -1.2607379443340507], [0.0, 0.0], [1.6741106151156278, 1.586508227481553], [-0.07147500967350941, 0.07484992544731177], [1.753276445870248, -2.5194534666257575], [0.3808679140710902, -0.3911797476675078], [6.492400012109273, -7.332451039806871], [0.0, 0.0], [-5.8166586444829855, 6.500060228248203], [1.7532764198705664, -2.519453482932055]], 'threshold': 8.044839263575923}}


def outside_table_height_band(packet, grip):
    """Geometric gate, independent of payload/contact-support eligibility."""
    if not grip:
        return False
    if getattr(packet, "USE_WRENCH", False) and packet.box_mass > 0:
        return not packet.last_table_allowed
    _, point, _ = packet._point_jacobian(packet.q_prev)
    return bool(np.min(np.abs(packet.table_heights - point[2])) > packet.height_band)


class Detector(SharedEstimator):
    """Runner-compatible adapter, optionally reusing a controller-owned core."""

    USE_SHARED = True
    _shared = None

    def reset(self):
        shared = getattr(getattr(self, "ctrl", None), "shared_estimator", None)
        if self.USE_SHARED and getattr(shared, "API", None) == self.API:
            self._shared = shared
        else:
            self._shared = self
            super().reset()
        # Recognized controller-specific normal distributions only. Baseline
        # controllers retain the original v1 detector without extrapolation.
        ctrl = getattr(self, "ctrl", None)
        mode = getattr(ctrl, "MODE", None)
        known = self._shared is not self and mode in MODEL_PARAMETERS
        self.normal_model = NormalResidualModel(**MODEL_PARAMETERS[mode]) if known else None

    def __getattr__(self, name):
        shared = self._shared
        if shared is not None and shared is not self:
            return getattr(shared, name)
        raise AttributeError(name)

    def update(self, t, th, thd, tau_m, tau_s, grip):
        del tau_m
        shared = self._shared
        packet = (
            SharedEstimator.update(self, t, th, thd, tau_s, grip)
            if shared is self
            else shared.update(t, th, thd, tau_s, grip)
        )
        extra = False
        if self.normal_model is not None:
            extra = self.normal_model.update(t, self.box_mass, self.box_com,
                packet.last_wrench, outside_table_height_band(packet, grip))
        return packet.r_joint.copy(), bool(packet.detected or extra)


"""Frozen NumPy runtime for the registered R5 monitors.

No fitting, threshold selection, file I/O, or shared-estimator mutation occurs
here.  A phase absent from normal fit data is fail-closed for the new branch;
the surrounding detector still retains the complete R1 decision.
"""

import numpy as np
from sim import trajectory as T


_PHASE_BOUNDARIES = np.array(
    [T.T_GRASP - 0.08, T.T_GRASP + 0.20,
     T.T_RELEASE - 0.15, T.T_RELEASE + 0.10], dtype=float)
_PHASE_BOUNDARIES.setflags(write=False)
PHASE_BOUNDARIES = _PHASE_BOUNDARIES
_FEATURE_CACHE = {}
_EPISODE_KEYS = {
    "t", "th", "thd", "tau_m", "tau_s", "r_joint", "wrench", "grip",
    "active", "r1_detect", "ref_q", "ref_qd", "ref_qdd", "mass", "com",
    "mode", "role", "normal_only",
}
_BATCH_ROLES = {"fit", "calibration", "audit", "analysis_eval"}


def phase_index(t):
    """Map scalar or array times to the five registered trajectory phases."""
    values = np.asarray(t, dtype=float)
    result = np.searchsorted(_PHASE_BOUNDARIES, values, side="right")
    return int(result) if values.ndim == 0 else result


_phase_index = phase_index


def _augmented_features(t, mass, com, grip, base=None):
    base = normal_features(t, mass, com) if base is None else np.asarray(base, dtype=float)
    phase = np.zeros(5, dtype=float)
    phase[_phase_index(t)] = 1.0
    grip_value = float(bool(grip))
    return np.r_[base, phase, grip_value, float(mass) * grip_value]


def _feature_grid(t, mass, com):
    """Pure cache for the exact offline 6000-sample public-reference grid."""
    t = np.asarray(t, dtype=float)
    if (t.shape != (6000,) or
            not np.allclose(t, np.arange(6000, dtype=float) * 0.001,
                            rtol=0.0, atol=1e-12)):
        raise ValueError("batch t must be the exact 6000-sample 1 ms grid")
    key = (float(mass), tuple(float(value) for value in np.asarray(com, dtype=float)))
    cached = _FEATURE_CACHE.get(key)
    if cached is None:
        cached = np.vstack([normal_features(value, key[0], key[1]) for value in t])
        cached.setflags(write=False)
        _FEATURE_CACHE[key] = cached
    return cached


def _array(payload, name, shape):
    value = np.asarray(payload[name], dtype=float).copy()
    if value.shape != shape or not np.all(np.isfinite(value)):
        raise ValueError("invalid frozen model array: " + name)
    value.setflags(write=False)
    return value


class FixedMonitor:
    """Fixed R5 monitor with state private to one detector instance."""

    def __init__(self, payload):
        if not isinstance(payload, dict):
            raise TypeError("payload must be a dictionary")
        mode = payload.get("mode")
        if mode not in MODEL_PARAMETERS:
            raise ValueError("unknown payload mode")
        model = payload.get("model")
        if (not isinstance(model, dict) or model.get("mode") != mode
                or model.get("format") != "r5.normal-model.v1"):
            raise ValueError("payload and frozen model mode differ")
        spec = model.get("spec")
        if not isinstance(spec, dict):
            raise ValueError("frozen model spec is missing")
        family = spec.get("family")
        if family not in {"SCALE_ACTIVE", "AR_FULL"}:
            raise ValueError("unsupported frozen model family")
        if int(spec.get("target_dims", -1)) != 2:
            raise ValueError("runtime candidates require two force dimensions")
        threshold = float(payload.get("threshold"))
        if not np.isfinite(threshold) or threshold < 0.0:
            raise ValueError("threshold must be finite and nonnegative")

        self.mode = mode
        self.spec = dict(spec)
        self.threshold = threshold
        self.phase_fit_counts = np.asarray(model.get("phase_fit_counts"), dtype=int).copy()
        if self.phase_fit_counts.shape != (5,) or np.any(self.phase_fit_counts < 0):
            raise ValueError("invalid phase fit counts")
        self.phase_fit_counts.setflags(write=False)

        r1 = MODEL_PARAMETERS[mode]
        self.r1_mean = np.asarray(r1["mean"], dtype=float).copy()
        self.r1_scale = np.asarray(r1["scale"], dtype=float).copy()
        self.r1_weights = np.asarray(r1["weights"], dtype=float).copy()
        for value in (self.r1_mean, self.r1_scale, self.r1_weights):
            value.setflags(write=False)

        if family == "SCALE_ACTIVE":
            expected_id = {"phase_mad": "scale_active_phase_mad",
                           "state_logscale": "scale_active_state_log"}.get(spec.get("scale"))
            if spec.get("gate") != "r1_active" or spec.get("id") != expected_id:
                raise ValueError("unknown scale model specification")
            self.location = _array(model, "location", (5, 2))
            self.phase_scale = None
            self.feature_mean = self.feature_scale = None
            self.logscale_weights = None
            self.weights = None
            if spec.get("scale") == "phase_mad":
                self.phase_scale = _array(model, "phase_scale", (5,))
                if np.any(self.phase_scale <= 0.0):
                    raise ValueError("phase scale must be positive")
            elif spec.get("scale") == "state_logscale":
                width = len(model.get("feature_mean", []))
                self.feature_mean = _array(model, "feature_mean", (width,))
                self.feature_scale = _array(model, "feature_scale", (width,))
                self.logscale_weights = _array(model, "logscale_weights", (width + 1,))
            else:
                raise ValueError("unknown scale model")
            self.lag = None
        else:
            lag = int(spec.get("lag_ms", -1))
            if (lag not in (20, 50) or spec.get("gate") != "t>=0.06"
                    or spec.get("id") != "ar_full_lag" + str(lag) + "ms"):
                raise ValueError("unknown AR lag")
            width = len(model.get("feature_mean", []))
            self.feature_mean = _array(model, "feature_mean", (width,))
            self.feature_scale = _array(model, "feature_scale", (width,))
            self.weights = _array(model, "weights", (width + 1, 2))
            self.phase_scale = _array(model, "phase_scale", (5,))
            if np.any(self.phase_scale <= 0.0):
                raise ValueError("phase scale must be positive")
            self.location = self.logscale_weights = None
            self.lag = lag
        self.reset()

    def reset(self):
        self.last_prediction = np.zeros(2, dtype=float)
        self.last_error_norm = 0.0
        self.last_scale = 1.0
        self.last_score = 0.0
        self.last_ema = 0.0
        self.last_active = False
        self.last_own = False
        self._count = 0
        self._clear = 0
        self._history = np.zeros((60, 14), dtype=float)
        self._history_write = 0
        self._history_count = 0

    def _r1_prediction(self, feature):
        normalized = np.clip((feature - self.r1_mean) / self.r1_scale, -8.0, 8.0)
        return normalized @ self.r1_weights

    def _ar_delayed(self):
        delayed = np.zeros(16, dtype=float)
        if self._history_count >= self.lag:
            delayed[:14] = self._history[(self._history_write - self.lag) % 60]
        if self._history_count >= self.lag + 9:
            indices = [(self._history_write - offset) % 60
                       for offset in range(self.lag + 9, self.lag - 1, -1)]
            delayed[14:] = np.mean(self._history[indices, :2], axis=0)
        return delayed

    def _push_history(self, wrench, r_joint, tracking_error):
        self._history[self._history_write] = np.r_[
            np.asarray(wrench, dtype=float)[:2],
            np.asarray(r_joint, dtype=float),
            np.asarray(tracking_error, dtype=float),
        ]
        self._history_write = (self._history_write + 1) % 60
        self._history_count = min(self._history_count + 1, 60)

    _reference_grid = None

    def _state_scale(self, augmented):
        normalized = np.clip(
            (augmented - self.feature_mean) / self.feature_scale, -8.0, 8.0)
        return float(np.clip(
            np.exp(np.r_[1.0, normalized] @ self.logscale_weights), 0.25, 1000.0))

    def _update(self, t, mass, com, th, r_joint, wrench, active, grip, feature):
        phase = _phase_index(t)
        supported = bool(self.phase_fit_counts[phase] > 0)
        base = np.asarray(feature, dtype=float)
        augmented = _augmented_features(t, mass, com, grip, base=base)
        wrench = np.asarray(wrench, dtype=float)

        if self.spec["family"] == "SCALE_ACTIVE":
            prediction = self._r1_prediction(base) + self.location[phase]
            branch_active = bool(active) and supported
            if self.spec["scale"] == "phase_mad":
                scale = float(self.phase_scale[phase])
            else:
                scale = self._state_scale(augmented)
        else:
            q_ref = T.reference(float(t))[0]
            raw = np.r_[augmented, self._ar_delayed()]
            normalized = np.clip(
                (raw - self.feature_mean) / self.feature_scale, -8.0, 8.0)
            prediction = np.r_[1.0, normalized] @ self.weights
            scale = float(self.phase_scale[phase])
            branch_active = bool(float(t) >= 0.06) and supported
            self._push_history(wrench, r_joint, np.asarray(th, dtype=float) - q_ref)

        if not supported:
            scale = 1.0
        return self._decision(wrench, prediction, scale, branch_active)

    def _decision(self, wrench, prediction, scale, branch_active):
        error_norm = float(np.linalg.norm(wrench[:2] - prediction))
        score = error_norm / scale
        if not branch_active or not np.isfinite(score):
            self.last_ema = 0.0
            self._count = self._clear = 0
            self.last_own = False
        else:
            self.last_ema += 0.1 * (score - self.last_ema)
            over = self.last_ema > self.threshold
            self._count = self._count + 1 if over else 0
            self._clear = 0 if over else self._clear + 1
            if self._count >= 4:
                self.last_own = True
            elif self._clear >= 8:
                self.last_own = False
        self.last_prediction = np.asarray(prediction, dtype=float).copy()
        self.last_error_norm = error_norm
        self.last_scale = scale
        self.last_score = score
        self.last_active = branch_active
        return bool(self.last_own)

    def update(self, t, mass, com, th, r_joint, wrench, active, grip):
        """Advance one 1 ms sample without mutating any caller-owned input."""
        grid = self._reference_grid
        k = None if grid is None else grid.index(t, mass, com)
        if k is not None:
            return self._decision(np.asarray(wrench), grid.r5[k], grid.scale[k, int(bool(grip))],
                                  bool(active and grid.supported[k]))
        feature = normal_features(float(t), float(mass), np.asarray(com, dtype=float))
        return self._update(t, mass, com, th, r_joint, wrench, active, grip, feature)


def _validate_batch_episode(episode):
    if not isinstance(episode, dict) or set(episode) != _EPISODE_KEYS:
        raise ValueError("episode keys are not allowed")
    if episode["role"] not in _BATCH_ROLES:
        raise ValueError("episode role is not allowed")
    if episode["mode"] not in MODEL_PARAMETERS:
        raise ValueError("episode mode is unknown")
    for name in ("th", "thd", "tau_m", "tau_s", "r_joint", "wrench",
                 "ref_q", "ref_qd", "ref_qdd"):
        value = np.asarray(episode[name])
        if value.shape != (6000, 6) or not np.all(np.isfinite(value)):
            raise ValueError("invalid episode signal: " + name)
    for name in ("grip", "active", "r1_detect"):
        value = np.asarray(episode[name])
        if value.shape != (6000,) or value.dtype != np.bool_:
            raise ValueError("invalid episode flag: " + name)
    com = np.asarray(episode["com"], dtype=float)
    if com.shape != (3,) or not np.all(np.isfinite(com)):
        raise ValueError("invalid episode COM")
    if not np.isfinite(float(episode["mass"])):
        raise ValueError("invalid episode mass")


def batch_trace(payload, episode):
    """Offline exact-grid trace using only the registered signal allowlist."""
    _validate_batch_episode(episode)
    monitor = FixedMonitor(payload)
    if monitor.mode != episode["mode"]:
        raise ValueError("episode and payload mode differ")
    t = np.asarray(episode["t"], dtype=float)
    features = _feature_grid(t, episode["mass"], episode["com"])
    prediction = np.empty((6000, 2), dtype=float)
    error_norm = np.empty(6000, dtype=float)
    scale = np.empty(6000, dtype=float)
    score = np.empty(6000, dtype=float)
    ema = np.empty(6000, dtype=float)
    active = np.empty(6000, dtype=bool)
    own = np.empty(6000, dtype=bool)
    for index in range(6000):
        own[index] = monitor._update(
            t[index], episode["mass"], episode["com"], episode["th"][index],
            episode["r_joint"][index], episode["wrench"][index],
            episode["active"][index], episode["grip"][index], features[index])
        prediction[index] = monitor.last_prediction
        error_norm[index] = monitor.last_error_norm
        scale[index] = monitor.last_scale
        score[index] = monitor.last_score
        ema[index] = monitor.last_ema
        active[index] = monitor.last_active
    return {
        "prediction": prediction, "error_norm": error_norm, "scale": scale,
        "score": score, "ema": ema, "active": active, "own": own,
        "detect": own | np.asarray(episode["r1_detect"], dtype=bool),
    }


# Frozen R5 normal-only payloads. No runtime fitting, adaptation, or file access.
R5_CONFIG_ID = 'SCALE_STATE'
R5_PAYLOADS = {'servo': {'mode': 'servo',
           'model': {'feature_mean': [1.0,
                                      0.11582705719556324,
                                      0.38574842755737465,
                                      0.983918622206109,
                                      0.0,
                                      0.9372841886700162,
                                      0.11582705719556324,
                                      0.7863799006258186,
                                      0.9165778350716829,
                                      0.04798223749701606,
                                      1.0,
                                      0.34176156857275614,
                                      0.7863799006258186,
                                      0.9401403103788221,
                                      0.15835934204940835,
                                      -0.24616309909317163,
                                      0.0,
                                      0.08780375704376045,
                                      0.9401403103788221,
                                      5.0841047043479107e-17,
                                      0.17295159220032216,
                                      -0.07694644768759501,
                                      0.0,
                                      -0.09600514451272825,
                                      5.0841047043479107e-17,
                                      0.9376310769619625,
                                      0.026922072282949804,
                                      -0.06276097752353821,
                                      0.0,
                                      0.005570376021577847,
                                      0.9376310769619625,
                                      0.5466430673423328,
                                      -0.024851707648990147,
                                      -0.05832870022113786,
                                      0.6148301082990882,
                                      0.5140040870688136,
                                      0.08669750896322964,
                                      -0.1346438128387721,
                                      0.0,
                                      0.047946303875542495,
                                      0.5140040870688136,
                                      7.430853534143351e-18,
                                      0.09450398950359812,
                                      -0.042042002182510696,
                                      0.0,
                                      -0.052461987321087276,
                                      7.430853534143351e-18,
                                      0.06338295448428581,
                                      0.21087199831809897,
                                      0.5378517328661443,
                                      0.0,
                                      0.5123639877057502,
                                      0.06338295448428581,
                                      -0.002866020297575259,
                                      -0.009586213505887179,
                                      -0.024452049657625242,
                                      0.0,
                                      -0.02329219420977964,
                                      -0.002866020297575259,
                                      -0.0067527267039860575,
                                      -0.02250100542619342,
                                      -0.057390546615697695,
                                      0.0,
                                      -0.054670126443404084,
                                      -0.0067527267039860575,
                                      0.07120372595267888,
                                      0.23716777631244476,
                                      0.604943059818602,
                                      0.0,
                                      0.5762701766942243,
                                      0.07120372595267888,
                                      9.925552056739757e-16,
                                      -0.054547699110651035,
                                      0.023817984184227727,
                                      0.0,
                                      0.03072971492642402,
                                      9.925552056739757e-16,
                                      0.0,
                                      0.015214802774575214,
                                      0.9752594076622125,
                                      0.009525789563212308,
                                      0.0,
                                      1.0,
                                      4.3731445387386625],
                     'feature_scale': [1.0,
                                       0.5897615018625307,
                                       0.09686404267882379,
                                       0.01203958408957876,
                                       1.0,
                                       0.023654324929096224,
                                       0.5897615018625307,
                                       0.14273092038508156,
                                       0.041238090744673,
                                       0.17163012024616578,
                                       1.0,
                                       0.06432614420294282,
                                       0.14273092038508156,
                                       0.7623547432486949,
                                       0.305436122733499,
                                       0.23778887297568316,
                                       1.0,
                                       0.1874273988355568,
                                       0.7623547432486949,
                                       0.35308804585782966,
                                       0.5111310747229529,
                                       0.249621398480506,
                                       1.0,
                                       0.28309676687690216,
                                       0.35308804585782966,
                                       0.9855552508088912,
                                       0.19051249826669375,
                                       0.08033150882867363,
                                       1.0,
                                       0.06297100614711033,
                                       0.9855552508088912,
                                       0.21372946374945456,
                                       0.28317557754469136,
                                       0.2937815638026445,
                                       0.1899395619973976,
                                       0.4905607216668072,
                                       0.1819355313881018,
                                       0.14897601109234374,
                                       1.0,
                                       0.11144516700178067,
                                       0.4905607216668072,
                                       0.20727083404850005,
                                       0.30226970018689936,
                                       0.147429758838698,
                                       1.0,
                                       0.16742709710398115,
                                       0.20727083404850005,
                                       0.3470193995685162,
                                       0.10016215097590288,
                                       0.2104096659811088,
                                       1.0,
                                       0.2008159720677017,
                                       0.3470193995685162,
                                       0.1708269940903556,
                                       0.11265009908198531,
                                       0.27864288153977684,
                                       1.0,
                                       0.26550044276484575,
                                       0.1708269940903556,
                                       0.1798924904520887,
                                       0.11698423700434915,
                                       0.2890792408958859,
                                       1.0,
                                       0.2754492384652756,
                                       0.1798924904520887,
                                       0.3801470279499432,
                                       0.09618888281435257,
                                       0.1870464852518945,
                                       1.0,
                                       0.17867561790294553,
                                       0.3801470279499432,
                                       0.12043528864408362,
                                       1.1494573523374774,
                                       0.5146565560563464,
                                       1.0,
                                       0.6447669879178761,
                                       0.12043528864408362,
                                       1.0,
                                       0.1224063419561812,
                                       0.15533349744517821,
                                       0.09713417985653859,
                                       1.0,
                                       1.0,
                                       1.7098357099956365],
                     'format': 'r5.normal-model.v1',
                     'location': [[0.0, 0.0],
                                  [0.06178592384515502, -0.12031131118627902],
                                  [0.1972623081111451, -0.009679893647076887],
                                  [0.042180471520655516, 0.40300893517771286],
                                  [0.0, 0.0]],
                     'logscale_weights': [0.572997685986319,
                                          0.0,
                                          -0.0435258501745018,
                                          0.027213026956915753,
                                          0.024356260623314616,
                                          0.0,
                                          -0.016404878950870148,
                                          -0.04352585017450182,
                                          0.042855988695010144,
                                          -0.025915277043815403,
                                          0.006781624756812429,
                                          0.0,
                                          0.01926696672747273,
                                          0.04285598869501506,
                                          -0.013678403060444438,
                                          0.0510942889440519,
                                          -0.01218470953260062,
                                          0.0,
                                          -0.06780573833150781,
                                          -0.013678403060446332,
                                          -0.06296554152033244,
                                          0.014523213205029709,
                                          0.05887832281432761,
                                          0.0,
                                          -0.07813778694489695,
                                          -0.06296554152033211,
                                          0.014478349198599746,
                                          -0.00511449192240977,
                                          0.022270369396043658,
                                          0.0,
                                          -0.04212243482967637,
                                          0.014478349198601257,
                                          0.004674962705634404,
                                          0.005156310437158087,
                                          0.002399027186360503,
                                          -0.00599793689054833,
                                          0.01893599750931345,
                                          -0.010810228469045866,
                                          0.001141195268474044,
                                          0.0,
                                          0.01612231369299125,
                                          0.018935997509313404,
                                          0.00831023893524699,
                                          0.01219922155759054,
                                          0.043868652935276524,
                                          0.0,
                                          -0.06065326426370966,
                                          0.008310238935246992,
                                          -0.01820431451645712,
                                          0.033303509109823506,
                                          0.0068428993865523625,
                                          0.0,
                                          0.0046924040866488685,
                                          -0.018204314516457123,
                                          0.0009668886084994883,
                                          -0.012273754705394276,
                                          -0.004050859852110439,
                                          0.0,
                                          0.005464542469414745,
                                          0.0009668886084995978,
                                          -0.0035343013042712372,
                                          0.02123798715061428,
                                          -0.007656391301572127,
                                          0.0,
                                          0.00038450568037147,
                                          -0.003534301304270999,
                                          -0.009079325333095379,
                                          0.03271800878270436,
                                          -0.009802560404723368,
                                          0.0,
                                          -0.003588943476673548,
                                          -0.009079325333095372,
                                          0.07484527852833116,
                                          0.02868998372174871,
                                          0.17466385839902496,
                                          0.0,
                                          -0.19056467663663576,
                                          0.07484527852833116,
                                          0.0,
                                          0.0064203904941409155,
                                          -0.012239200004840724,
                                          0.008996042979266122,
                                          0.0,
                                          0.0,
                                          0.004674962705614627],
                     'mode': 'servo',
                     'phase_fit_counts': [0, 805, 51600, 504, 0],
                     'spec': {'family': 'SCALE_ACTIVE',
                              'gate': 'r1_active',
                              'id': 'scale_active_state_log',
                              'ridge': 0.01,
                              'scale': 'state_logscale',
                              'target_dims': 2}},
           'threshold': 3.430453910703297},
 'torque': {'mode': 'torque',
            'model': {'feature_mean': [1.0,
                                       0.1186414174583133,
                                       0.3860326655725015,
                                       0.9838889456993728,
                                       0.0,
                                       0.9374418982218499,
                                       0.1186414174583133,
                                       0.7863762907171379,
                                       0.9164407741036129,
                                       0.048708311972246746,
                                       1.0,
                                       0.34137788784506873,
                                       0.7863762907171379,
                                       0.9430098521618469,
                                       0.16364651942983918,
                                       -0.2492718973700138,
                                       0.0,
                                       0.08562537794017082,
                                       0.9430098521618469,
                                       5.099622654931819e-17,
                                       0.1705816925823821,
                                       -0.0757615263292897,
                                       0.0,
                                       -0.0948201662530934,
                                       5.099622654931819e-17,
                                       0.9404929599412389,
                                       0.030973779232387717,
                                       -0.06391896662920131,
                                       0.0,
                                       0.004561622008775283,
                                       0.9404929599412389,
                                       0.5464504879334838,
                                       -0.024482956699788297,
                                       -0.0582678205808788,
                                       0.6147845529687639,
                                       0.515572955234774,
                                       0.08977800120109634,
                                       -0.13644157502553814,
                                       0.0,
                                       0.04666357382444195,
                                       0.515572955234774,
                                       7.453534345150347e-18,
                                       0.0930506013596276,
                                       -0.04131344344951326,
                                       0.0,
                                       -0.05173715791011421,
                                       7.453534345150347e-18,
                                       0.06501013193305236,
                                       0.21096057495689383,
                                       0.5376457473334915,
                                       0.0,
                                       0.5122767445607005,
                                       0.06501013193305236,
                                       -0.00302075162933563,
                                       -0.00944900360889152,
                                       -0.024089748664163354,
                                       0.0,
                                       -0.02296098806728052,
                                       -0.00302075162933563,
                                       -0.006905373247987668,
                                       -0.022489738911128907,
                                       -0.05732971237113348,
                                       0.0,
                                       -0.05462288993140164,
                                       -0.006905373247987668,
                                       0.07297199602516363,
                                       0.2373292619338919,
                                       0.6048795202502264,
                                       0.0,
                                       0.5763268792309137,
                                       0.07297199602516363,
                                       9.955847307386892e-16,
                                       -0.06812980880947211,
                                       0.03047465621726305,
                                       0.0,
                                       0.03765515259220974,
                                       9.955847307386892e-16,
                                       0.0,
                                       0.011867748540229013,
                                       0.9782361416546599,
                                       0.009896109805111094,
                                       0.0,
                                       1.0,
                                       4.371603903467871],
                      'feature_scale': [1.0,
                                        0.5891419158422206,
                                        0.09700094507889294,
                                        0.0120632679452381,
                                        1.0,
                                        0.023544248395196105,
                                        0.5891419158422206,
                                        0.1429977148741085,
                                        0.04130260676028841,
                                        0.1715940571285043,
                                        1.0,
                                        0.0641053314175164,
                                        0.1429977148741085,
                                        0.761743177608967,
                                        0.2940905165313096,
                                        0.23228243118200073,
                                        1.0,
                                        0.18526879733070062,
                                        0.761743177608967,
                                        0.3536264915195779,
                                        0.5108491940792804,
                                        0.24936915312821797,
                                        1.0,
                                        0.28313816222334587,
                                        0.3536264915195779,
                                        0.9856938058836557,
                                        0.1795286983345963,
                                        0.07800446213350144,
                                        1.0,
                                        0.06128787455910115,
                                        0.9856938058836557,
                                        0.21373709634022917,
                                        0.28317427357037855,
                                        0.29382607570279035,
                                        0.18992448710626225,
                                        0.49048494236316437,
                                        0.17457150779082478,
                                        0.14583263864174534,
                                        1.0,
                                        0.10976812638781089,
                                        0.49048494236316437,
                                        0.2075869140821053,
                                        0.3017627941757844,
                                        0.1471116670869127,
                                        1.0,
                                        0.16729145394154968,
                                        0.2075869140821053,
                                        0.3465505132358104,
                                        0.10026816956554299,
                                        0.21041034308776227,
                                        1.0,
                                        0.20086920570598996,
                                        0.3465505132358104,
                                        0.17077734942821676,
                                        0.11273183481375633,
                                        0.27863393054556274,
                                        1.0,
                                        0.2655430105274327,
                                        0.17077734942821676,
                                        0.179894647507914,
                                        0.11709398407899015,
                                        0.2891140601197419,
                                        1.0,
                                        0.275534427982339,
                                        0.179894647507914,
                                        0.3797410950899334,
                                        0.0962916243776523,
                                        0.18702541734031458,
                                        1.0,
                                        0.17869340641987114,
                                        0.3797410950899334,
                                        0.12061894781763126,
                                        1.1318085154278954,
                                        0.5043275285388172,
                                        1.0,
                                        0.6373913582246692,
                                        0.12061894781763126,
                                        1.0,
                                        0.1082908356455576,
                                        0.14591159246361346,
                                        0.09898574046713865,
                                        1.0,
                                        1.0,
                                        1.7098967707218333],
                      'format': 'r5.normal-model.v1',
                      'location': [[0.0, 0.0],
                                   [0.5461184954589826, -0.2592875812859312],
                                   [0.5234156443902716, 0.1082903660608486],
                                   [0.35406596970573423, 0.8119863836208956],
                                   [0.0, 0.0]],
                      'logscale_weights': [0.607552265166393,
                                           0.0,
                                           -0.006129384332607593,
                                           0.005258278063303338,
                                           0.0043842788559116394,
                                           0.0,
                                           0.0025139887346516928,
                                           -0.006129384332607616,
                                           0.013506352211343796,
                                           -0.007398463663610397,
                                           0.0037323437679885225,
                                           0.0,
                                           -0.001875347309108993,
                                           0.01350635221134535,
                                           -0.03077777973062021,
                                           0.05617283834958303,
                                           -0.024663154510213378,
                                           0.0,
                                           -0.05824554220939381,
                                           -0.030777779730620514,
                                           -0.00436568821521807,
                                           0.024128721956105385,
                                           0.01988935772066799,
                                           0.0,
                                           -0.061051220791907015,
                                           -0.004365688215218035,
                                           0.007010637163603362,
                                           0.0017187572184425572,
                                           -0.017050145433900037,
                                           0.0,
                                           0.01784022037137409,
                                           0.007010637163603887,
                                           0.017081709519737264,
                                           0.0008123092171996393,
                                           0.0078072520606488165,
                                           0.0028742634562730052,
                                           8.080184803762066e-05,
                                           -0.003614436537806812,
                                           -0.009380504829527592,
                                           0.0,
                                           0.018210763662281162,
                                           8.08018480384761e-05,
                                           -0.015180921505771864,
                                           0.0006852882806011477,
                                           0.030034021798225865,
                                           0.0,
                                           -0.027647255215121846,
                                           -0.015180921505771846,
                                           -0.01805667444947568,
                                           0.009864970855928013,
                                           0.017858337994564925,
                                           0.0,
                                           0.016729251688429807,
                                           -0.018056674449475646,
                                           0.00016469021034915275,
                                           -0.006392554442071806,
                                           0.004133077018271302,
                                           0.0,
                                           0.000851238043981335,
                                           0.00016469021034905957,
                                           -0.005755410931692727,
                                           0.0018965450125547178,
                                           -0.0017562196251981695,
                                           0.0,
                                           0.006533092903009581,
                                           -0.00575541093169323,
                                           -0.002689919641866087,
                                           0.009432226148914157,
                                           -0.0032017324550198565,
                                           0.0,
                                           0.002662562481431017,
                                           -0.002689919641866078,
                                           0.05003090903718167,
                                           0.028572861952536777,
                                           0.16683807397855088,
                                           0.0,
                                           -0.18274493445947487,
                                           0.05003090903718138,
                                           0.0,
                                           0.011538311515290726,
                                           -0.008894540249394855,
                                           -0.001012347812681366,
                                           0.0,
                                           0.0,
                                           0.01708170951993127],
                      'mode': 'torque',
                      'phase_fit_counts': [0, 626, 51600, 522, 0],
                      'spec': {'family': 'SCALE_ACTIVE',
                               'gate': 'r1_active',
                               'id': 'scale_active_state_log',
                               'ridge': 0.01,
                               'scale': 'state_logscale',
                               'target_dims': 2}},
            'threshold': 3.2086891525415253}}

_R1Detector = Detector
_R1NormalResidualModel = NormalResidualModel


class _R5TrackedNormalResidualModel(_R1NormalResidualModel):
    """R1 model with a passive record of the exact active argument."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.last_active = False

    def update(self, t, mass, com, wrench, active):
        self.last_active = bool(active)
        return super().update(t, mass, com, wrench, active)


class Detector(_R1Detector):
    """Unchanged R1 detector OR one private fixed R5 monitor."""

    def reset(self):
        super().reset()
        mode = getattr(getattr(self, "ctrl", None), "MODE", None)
        original_normal_enabled = self.normal_model is not None
        if original_normal_enabled and mode in MODEL_PARAMETERS:
            self.normal_model = _R5TrackedNormalResidualModel(**MODEL_PARAMETERS[mode])
        payload = R5_PAYLOADS.get(mode)
        self._r5_monitor = (FixedMonitor(payload)
                            if original_normal_enabled and payload is not None else None)
        self.r5_base_detect = False
        self.r5_extra = False

    def update(self, t, th, thd, tau_m, tau_s, grip):
        residual, baseline = super().update(t, th, thd, tau_m, tau_s, grip)
        self.r5_base_detect = bool(baseline)
        monitor = self._r5_monitor
        if monitor is None:
            self.r5_extra = False
            return residual, bool(baseline)
        packet = self._shared
        self.r5_extra = monitor.update(
            t, self.box_mass, self.box_com, th, packet.r_joint,
            packet.last_wrench, self.normal_model.last_active, grip)
        return residual, bool(baseline or self.r5_extra)


_R5BothModeDetector = Detector
_R5_ENABLED_MODES = ('servo',)


class Detector(_R5BothModeDetector):
    def reset(self):
        super().reset()
        mode = getattr(getattr(self, "ctrl", None), "MODE", None)
        if mode not in _R5_ENABLED_MODES:
            self._r5_monitor = None


_R8BaseDetector = Detector
import io as _r8_io
import base64 as _r8_b64
import mujoco
from sim.nominal import Nominal


def _clone_nominal(source, share_model=False):
    """Create a Nominal-compatible model/data owner without parsing XML."""
    nominal = Nominal.__new__(Nominal)
    nominal.m = source.m if share_model else copy(source.m)
    nominal.d = mujoco.MjData(nominal.m)
    nominal._M = np.zeros_like(source._M)
    nominal.tau_max = source.tau_max.copy()
    nominal.qd_max = source.qd_max.copy()
    nominal.B = source.B.copy()
    nominal.K = source.K.copy()
    nominal.elastic = source.elastic.copy()
    return nominal
from sim import config as C, trajectory as T
WINDOWS = ((1.4, 1.7), (3.6, 4.0))


def critical_step(position, velocity, target, pole, dt=0.001):
    error = position - target
    decay = np.exp(-pole * dt)
    return (target + decay * ((1 + pole * dt) * error + dt * velocity), decay * (-pole * pole * dt * error + (1 - pole * dt) * velocity))


class _R8Channel:
    """Unchanged positive/empty C1 thresholds, window entry veto and latch."""

    def __init__(self, mass, com, model):
        self.mass = float(mass)
        p = self.mass * np.asarray(com, float)
        self.regressor = np.r_[1.0, self.mass, p, 0.0 * self.mass, 0.0 * p]
        self.scale_regressor = np.array([1., mass / 8., mass * com[0] / .24, mass * com[1] / .16])
        self.coefficients = model['coefficients']
        self.thresholds = model['thresholds']
        self.threshold_coefficients = model['threshold_coefficients']
        self.entry_bounds = model['entry_bounds']
        if self.coefficients.shape != (9, 6000) or self.thresholds.shape != (240,) or self.threshold_coefficients.shape != (240, 4) or (self.entry_bounds.shape != (2, 2)) or np.any(self.thresholds <= 0) or np.any(self.entry_bounds <= 0):
            raise ValueError('invalid normal model')
        arrays = (self.coefficients, self.thresholds, self.threshold_coefficients, self.entry_bounds)
        if not all((np.isfinite(a).all() for a in arrays)):
            raise ValueError('nonfinite normal model')
        for a in arrays:
            a.setflags(write=False)
        self.mass = float(mass)
        self.width = float(model["width"])
        self.support_N = float(model["support_N"])
        if not 0.002 <= self.width <= 0.006 or not np.isfinite(self.support_N) or self.support_N <= 0:
            raise ValueError('invalid normal contact calibration')
        self.raw = []
        self.baselines = {}
        self.latched = False
        self.phase = None
        self.disabled = self.previous_open = False
        self.count = 0
        self.diagnostics = dict(active_samples=0, contact_blocked=0, support_blocked=0, entry_disabled=0, alarm_latched_samples=0)

    def threshold(self, indices):
        bins = np.asarray(indices) // 25
        value = self.thresholds[bins]
        value = value * np.exp(np.clip(self.threshold_coefficients[bins] @ self.scale_regressor, -10.0, 10.0))
        return np.maximum(0.3, value)

    def grid_index(self, t):
        k = int(round(t * 1000))
        if abs(k * 0.001 - t) > 1e-12 or k != len(self.raw) or not 0 <= k < 6000:
            raise ValueError('complete ordered 1ms stream required')
        return k

    def update(self, t, k, phase, signal, height_distance, support_force_N, grip, base_alarm, statistic):
        if not np.isfinite(statistic) or statistic < 0:
            raise ValueError('finite nonnegative membership statistic required')
        self.raw.append(float(signal))
        self.latched = bool(self.latched or base_alarm)
        if phase != self.phase:
            self.phase = phase
            self.disabled = self.previous_open = False
            self.count = 0
        if phase is None:
            return False
        a, b = WINDOWS[phase]
        if phase not in self.baselines:
            lo, hi = (round((a - 0.12) * 1000), round((a - 0.02) * 1000))
            if k < hi:
                return False
            self.baselines[phase] = float(np.mean(self.raw[lo:hi]))
        baseline = self.baselines[phase]
        in_phase = a - 1e-12 <= t < b - 1e-12
        geometric = bool(grip and height_distance <= self.width)
        support = bool(grip and self.mass > 0 and (support_force_N > self.support_N))
        self.diagnostics['contact_blocked'] += int(in_phase and geometric)
        self.diagnostics['support_blocked'] += int(in_phase and support)
        opened = bool(in_phase and (not (geometric or support)))
        if opened and (not self.previous_open):
            indices = np.arange(k - 100, k)
            error = np.asarray(self.raw[k - 100:k]) - baseline - self.regressor @ self.coefficients[:, indices]
            limit_abs, limit_var = self.entry_bounds[phase]
            prior_ratio = abs(error[-20:]) / self.threshold(indices[-20:])
            if len(error) != 100 or np.max(abs(error)) > limit_abs or np.var(error) > limit_var or (np.max(prior_ratio) > 0.8):
                self.disabled = True
                self.diagnostics['entry_disabled'] += 1
        self.previous_open = opened
        if self.latched:
            self.diagnostics['alarm_latched_samples'] += 1
        if not opened or self.disabled or self.latched:
            self.count = 0
            return False
        self.diagnostics['active_samples'] += 1
        error = float(statistic)
        self.count = self.count + 1 if abs(error) > self.threshold(k) else 0
        if self.count >= 4:
            self.latched = True
            return True
        return False

class Membership:
    """C1's three stiffness hypotheses, evaluated in one filter bank.

    The six nominal rigid-body models depend on known mass/COM, not on the
    stiffness hypothesis. Each inverse-dynamics call sets q, qd and qdd in
    full; only the observer/filter states must be distinct between rows.
    """

    def __init__(self, mass, com, model):
        self.mass = float(mass)
        self.empty = Nominal()
        self.full = _clone_nominal(self.empty)
        self.full.set_payload(mass, com)
        inv_k = np.zeros(6)
        elastic = self.empty.elastic
        inv_k[elastic] = 1 / self.empty.K[elastic]
        self.inv_k = inv_k / np.array([[0.8], [1.0], [1.2]])

        self.geometry = _clone_nominal(self.empty, share_model=True)
        self.body = self.geometry.m.body('link_6').id
        self.bottom_offset = np.array([0., 0., C.BOX_POS[2] + C.BOX_SIZE[2] / 2])
        self.heights = np.array([self.bottom(T.reference(t)[0]) for t in (1.4, 3.9)])
        self.epsilon = 0.001
        self.offset_coms = []
        self.scratch = None
        self.box = self.full.m.body('box').id
        self.j1 = self.full.m.joint('joint_1').id
        self.omega = np.empty(6)
        self.acc = np.empty(6)
        if mass > 0:
            self.bottom(T.reference(1.4)[0])
            rotation = self.geometry.d.xmat[self.body].reshape(3, 3)
            self.axes_local = rotation.T[:, :2].copy()
            self.scratch = _clone_nominal(self.full)
            for axis in self.axes_local.T:
                self.offset_coms.append(tuple(
                    np.asarray(com) + sign * self.epsilon * axis for sign in (-1., 1.)))

        self.channel = _R8Channel(mass, com, model)
        self.position = None
        self.velocity = np.zeros((3, 6))
        self.signal = np.zeros(3)
        self.signal_rate = np.zeros(3)
        self.sensitivity = np.zeros((3, 2))
        self.sensitivity_rate = np.zeros((3, 2))
        # Fixed per-episode history; store numeric rows, not Python objects.
        self.signals = np.empty((6000, 3))
        self.sensitivities = np.empty((6000, 3, 2))
        self.baselines = {}

    def bottom(self, q):
        d = self.geometry.d
        d.qpos[:] = q
        mujoco.mj_kinematics(self.geometry.m, d)
        return (d.xpos[self.body] + d.xmat[self.body].reshape(3, 3) @ self.bottom_offset)[2]

    def _com_sensitivity(self, args):
        """J1 derivative for a translated rigid payload, using nominal data only.

        With u = z_axis × r, the torque derivative along a world shift h is
        m h·(a × z_axis + u × alpha + omega (omega·u) - u (omega·omega)).
        The central difference of the moment of a *rigid* body is exactly
        this derivative in real arithmetic (quadratic terms cancel). Keep
        C1's direct difference for a nominal active constraint/nonvertical J1.
        """
        m, d = self.full.m, self.full.d
        axis = d.xaxis[self.j1]
        if d.nefc or not (axis[0] == axis[1] == 0. and axis[2] == 1.):
            result = []
            for negative_com, positive_com in self.offset_coms:
                self.scratch.set_payload(self.mass, negative_com)
                negative = self.scratch.inverse_dynamics(*args)[0]
                self.scratch.set_payload(self.mass, positive_com)
                positive = self.scratch.inverse_dynamics(*args)[0]
                result.append(-(positive - negative) / (2 * self.epsilon))
            return result
        mujoco.mj_rnePostConstraint(m, d)
        obj = mujoco.mjtObj.mjOBJ_BODY
        mujoco.mj_objectVelocity(m, d, obj, self.box, self.omega, 0)
        mujoco.mj_objectAcceleration(m, d, obj, self.box, self.acc, 0)
        wx, wy, wz = self.omega[:3]
        ax, ay, az, fx, fy, _ = self.acc
        rx, ry = (d.xipos[self.box] - d.xanchor[self.j1])[:2]
        dot = -ry * wx + rx * wy
        norm2 = wx * wx + wy * wy + wz * wz
        gradient = (fy + rx * az + dot * wx + ry * norm2,
                    -fx + ry * az + dot * wy - rx * norm2,
                    -ry * ay - rx * ax + dot * wz)
        rotation = d.xmat[self.box].reshape(3, 3)
        return -self.mass * (self.axes_local.T @ (rotation.T @ gradient))

    def update(self, t, th, tau_s, grip, support_N, base_alarm):
        k = self.channel.grid_index(t)
        # Once C1 has latched it can never emit another extra event. Preserve
        # the ordered-grid check independently, even after the final window.
        if self.channel.latched or base_alarm or t >= WINDOWS[-1][1]:
            self.channel.latched = True
            self.channel.raw.append(0.)
            return False
        q, qd, qdd = T.reference(t)
        measured = np.asarray(th) - self.inv_k * tau_s
        error = measured - q
        if self.position is None:
            self.position = error.copy()
        self.position, self.velocity = critical_step(self.position, self.velocity, error, 40.)
        acceleration = 40**2 * (error - self.position) - 80 * self.velocity
        states = zip(q + self.position, qd + self.velocity, qdd + acceleration)
        nom = self.full if grip else self.empty
        raw = np.empty(3)
        draw = np.zeros((3, 2))
        for i, args in enumerate(states):
            raw[i] = tau_s[0] - nom.inverse_dynamics(*args)[0]
            if grip and self.mass > 0:
                draw[i] = self._com_sensitivity(args)
        self.signal, self.signal_rate = critical_step(self.signal, self.signal_rate, raw, 25.)
        self.sensitivity, self.sensitivity_rate = critical_step(
            self.sensitivity, self.sensitivity_rate, draw, 25.)
        self.signals[k] = self.signal
        self.sensitivities[k] = self.sensitivity
        phase = next((i for i, (a, b) in enumerate(WINDOWS)
                      if a - 0.02 - 1e-12 <= t < b - 1e-12), None)
        statistic = distance = 0.
        if phase is not None:
            if phase not in self.baselines:
                a = WINDOWS[phase][0]
                window = slice(round((a - 0.12) * 1000), round((a - 0.02) * 1000))
                self.baselines[phase] = (self.signals[window].mean(axis=0),
                                         self.sensitivities[window].mean(axis=0))
            centre, slope = self.baselines[phase]
            errors = self.signal - centre - self.channel.regressor @ self.channel.coefficients[:, k]
            statistic = np.min(np.maximum(0., abs(errors) -
                                          0.03 * np.linalg.norm(self.sensitivity - slope, axis=1)))
            distance = np.min(np.abs(self.heights - self.bottom(measured[1])))
        return self.channel.update(t, k, phase, self.signal[1], distance, support_N,
                                   grip, base_alarm, statistic)


_R8_POSITIVE = 'P)h>@EdT%j2mk;8ApneerI7#s|NsC0|NjsG6aZsyWoBk+V`*h>baO6laCrd$5C8y}1^@s6001nE00000008XW<x>z}_$Y8hNx=XU6hROLK@@4whelLTK}iv$l@<|ckVZO1y1N_1hwkn!0sSflSRew|`!C%2-MKS&_VebMbN1}nv(K5`GrKQ#!xZl++*aR3@{GjctiFM@-jlN;%xA?6h0by@pEb02^7P3=b8U+!`Ud~IUiP7>wZXr7YvYHO2LH<Egt?dnE?&II!Nu&v{C|d;gg_t=2m}IwKp+qZ1OkCTAP@)y0)apv5C{YUfj}S-2m}J*f0%hs+FG+l1oD@JUQZql$NM$A<+YT;@ow)mvWCGhRJCmUBB2$A1yl~21D`_i*)83T5xY=K67(aR-xrEw8V8?d<%QsG)^F0+WJ2%?sW{!_crbb%b1{-I4#o$@mRUN#gYdx<YhiuPAdC#Vw)0FR5H(NG#py@{V7Zc0=Q%ZhT#0w7BQbf2bbabYPu%@b#PN=`a-1*jDziBGxyJ{WzF#~ycgzQOg)-3EDtqHWny3~7TQAHx#NYbB*AqWS2LHSi?tyP4ikSC>x}($l509wu7kK>or&o>%ZfKSHp~$Dm759CVD6dX-!7W((!`$kGRKIt4K3X{7g7fM0ues0h%2QYR5%OoqbY8C0THhKKH;*z1S6kwm{_bz0yG`-O$px{kzXoWP>we$)zBVT9`LWN=KnWj<cir2(!-rPRk4!FyN`P2Dr3GKCDt!0jS2<mv2b8L8;@puYFePZPT3BWULFUfn56WynRsC;;`LG?3(IqoHkadPaRF&>>a|4MB7q{Pka0l_L=3c?z1wKkkTbk}ZU{(M40{MU+czb^{7rz<+Z`ZY5BKv~iQ8UfScCHYpNY$qlvkiqSSH@VWzlMPa?QiPWkr6;cwi*0NCkpxmoanu!qaoR`Vi${446Hq!5#dvdgF4r^b0y9RkhOO)@X>T4kTci}@7a|Ms#dp7-DFRJ=3h*b?=(}vc+#d%B_$1*$R7nOZ=^%f$UH7R$OL1rw5Cg~SzsfiWgdPZ2Og<hd{-Nl3(Dd~o57U%Q1o}}%s2M}NJ>q(ci5>A$g-^8*9#WGLx(|8qn|~Pb$~kVcV02v4<1+6(J29L-|D3CgQYNBp=|S|wiKRnON5U-EQ1cm)o^{%au|@Pw}Yf|@cpGdZz@^=6lI?eeebIP(KZ{-D*Z~>XJwdia-$M_+M~W{1y(`6`-M|%C#u0A=YbY$QZ+0`3{7aCs)0=>S)Yy28qhrzI?udS10+02tEl@1&Ihq%DKx(U+t%VU3Cy+7d_72^>~Sq<>^FV3H>eh_tt)CNyVk;u^X$3ye6?^||MPpExHoX0TSLG<uLk<J9VrgpuZFALUuc*!{*B{K{5WNJIjHPA_2255QkY8!9+p)p0)-qw-7dvkV9e*RwvtbSUydBAB9bw{X}Pzp-^~+9av$HG_-c=&wsRM@PDP{l+Npz`?CEI#N#t=PSs@O#M+Gz;E<?=&awMNGSL5_4-`^+hB5I2Kr*>Yj0f)}B+*f_ugq7z+K97pFqT%x77DaYDrq@iW7m{`19M81A-t`{*OK!zUFWQG5X<T{aQvH}!`b+<a$N(-^zr6hA)F67DaTRO)K8T*>x|9z}htNKV_YcRzVRShfl_S3~j8erypM@MpQ1h0t&*p~_WW6kG`uFN6&OS~Qyy`NFRZRR+4cVjUWA0o#T>r1$VdJx2(<tiIR|XFhjbi8TuK&LJj-pj~?7?isQPe(LQdYKW6jgFlAJQa`ptPa4R|L-pCR7SPo5&tU22Re)J9NXSrLmvox$zLXQWV~w${obpUxR>uegH4=Q@=DPAHbBhxz?PM{W$Rbw9~$GeOP(7%3}OR54!EW%<jz7h5MT%O%DET$Cp7hn@>twQS<VMlAuO2GOrdzcC9wx{5G>$kbNDl+CSe#^Q{I~W<*$@jF)4~<(7KNk|K;U*^D*Zm4$ge;cm;P;?VTG-p+`TEAA8|InV0&Lc)9;ok?>deAyW<D><73Ogr)_JmtmkOV~$8rlSIqvnDvc_q_pw2HH=!PzRO656$E5HiBjG*I%mD%|O1x7cFq04OYC|T<O$0AnN;)x_xjLd{mYfnW*RiJMIW-dc!{O4+~yoHS7oF@!-84_XmLWzd!dwt_{Mj<9CnTIzI#zlq_l|$cEweYRv`Xp<yt}`|t1Cs}b;MnR;+UX%rTUG)-r>N1=gtZ{C}PG00P9x){$l4vkMax6^CKA@-Gzc8JIXxCK@pzLqlqAtc;iY7b1pu19ChS|3kBlBXfNMdc)L81HX=yfF#)$z9URxu?L4l<k_?y(y@eQg=CRIR$Y%*$gYrQ*bpgOnugU3NE%3JdFBR{<OqvP-{K~Uq~x1``(=b>&GvGhIppnc|`-8&8{ghoaNFEX`clCqwnM;yeDBR!9gcZWD=g9V&u(SoPe**2EKc|Ccr@T{z<b_6EN~gf8Dug9LkOxA7B(4hq+|Ui3J>kM#;8d2C*?XNM<oNRX7TD3K>6(PL0Cgsft=<pAoP~P~M*ZGz<folexy%hr##x_Edh#5V(pLGm&l#0=Hgc2g~(AFuKS;HWN7jN0;ZsD`)$`b?LzsbGCjkv(7e`HRywQ&b?H<*Lva8goSH;Q#ah4$QRub?t*k?=xfjHfFDX;WYu)-z*0UYecYiH62h)cf1Yaw8QSH%-x5tA5t|+HFrfil{2k?z*6X0yvnQ`K;VsDiC-XK$u?9GUC)X`@Dj-0IHBF$X6y&|M+m09%!Bw*NZ{&q@LHBfuu5Mu(*t?7h$Lqwyp%dy9T4o`@LGq@{P2dUmUXQ&wX6}k)kB)@Oe+$QbN`H1-x)L#Q^~Y4_!3>mPzt{U%HV<7wO&9|!iqQJb`W#(LDSoAIIr68w0)0GbI10O}(MS7u?(*AOq%t4*q@i4gQ86j8%#HOZac$DW@<Jodd_MehGrS4MAODxR@S_<sdz?h>-)cq4xb4N>j5h4eV&qc%cl~A3J5a0K=)_EqMn$KvE_@R%x~Hwb8>6T7z6?|J;_u>ddLPC<6i`0VSzOSE?QMZ6B$EB8Cz3T7)ZdT4xSuwAYYiYt%)4d3_X9|&UgQ4Nd=NwSue@~oci%cXb+lT=W(f7oO^SZ34WX|}okY0VFluHwyc=2?#&egGVpr8ikUntv3~R>-&YW<$=_N3VaY=>~JMp6!N?Ybtw|fkW_V+8v7>r^6*@AG3#xYcLe|xe0<T%olmxxwA9Y@db;}j}w<7k?ftp4oy1kTDb?5;DKKt4m?RE3fW^io?M^4Xj~QnO(t67fkaP_N*vb(=&j_lKk=^^=(Jc^{|N_9SjoG@cn1oWd*W6AjPJr?8*Z+)*}Z3Y~VnvYqUi!o$@p-1pX}FzM3HGAGkCs?8^saEec3z}|4ZquSGGOrLyp#(o;@PxAmx*fhS&`hq4I)0pzn)#-EPG+sI;`{-KRG%_VD=3gI~#y+)_*K_Zt(Oc%ebL-b>G!a(e%KJTy63j!3Qk&Bls&wbjz`yv&-c+_{V;Y~={kE+6F^yxlGa2`Pp2jDl91WX`)A()hifa1QG}70rz0?|*#sWRjgRITd==M3`Kv?NCx)=U!5l)&$vUrk3gO}5I<wO3jr6<#<sdZWSxau@EojvlNRCpSfXJiu;7^abLf-CI8pDENa*t&RYbPDArcHVmyO`+syq*}Po6mp#W361xs@Or^t{$QRdR1|h7lOUPG)B?V=Ki!j9{hy!H<>*N?``#%;uKO>)#@$*;9FxdS1-Ug}C-BMhk4o?V%}Zxe`gfye6L>@Bs)xVe1oAFwExP<3M~n22Iy~j$sI|DkD*tbOl{rUmGhZ4<R=(SjOe<sPS+uL~YT6ju-?R=E)f&SgCLaAU`Z1J#>Bnu;Hi~^&;~O;&qqwK>u@e1-QA{8;W>T6N!RgJ^r(rKgFl(>vC4GSr41Ua!wKzMBl+$Cf5kAAnZl?E5?BXy!dl+0&H#&rBg^j_5&O_KlRn$|!I)ro4?XG)U260>c%&`%}LF8fO)^R5v#3J#p?xU##$hmgt)NP3Y?32m4JUiKs?D{2sn$P-?%2X)A@jySia_SdAW*>t1Yu&LceaLzBq(yORFCJa`n$^+OgH{Jq$t%8hW69$0!1u?x@v7~A_nJk!aK={udgJ|0Gzw1;|KZeuUKHy}pX1vxdMzQvyrB&lIzmG3F1I41(;d?hvR2$&{}7YN(t?Bf<-5L!G^4@E%@>18P56+N+ur0+BmTdB{vi+u1OkCTAP@)y0)apv5C{YUfj}S-2m}IwKp+qZ1OkCTAP@)y0)apv5C{YUf$%@?qW|V>WLzi|Z4TKb*@Z&ct9mEL>!GkT-l{@D7YZ%sJG-LVLm;kFO7yi=2>3bF^kp9n0lHHjv3ZTb@B%t4CQXCk-XD|uUdMxB>b0j{U3?Hc=zac}j4=qpXa%SbL<WM@oll0a6abUE!mhCj1;Ck(d5-7q{@`u?@`6moOW3b*XZpjrm#}77Z5!X}2cyZVx)UmXz(}fi+<wUyZvQd)%IDw<ZZCFQDjxO)*5{<RIxBp@b?4OM8zw&BP->>%PUi#fcv=L;D!k#Spc~hBH*a|D{@JVSiZ`rADU)dQdqHn@m4e$dFEEb^7!P9d0&$hcuZz+=q4>iy2@@($h;c)X%f=osD6?>W<Ci=9()hb)In*7VetsDAhs+(+Z)RyTNW6fHv7h<$WZb~=X51Xf4;SD${#%8q%n<?yirChu?cjy&ltI!78(=LVxjW8i1qtgn3W9wf1E+KjONys1sA>BcUCvg5?%jKi9XZB=oV<}r+$w5VP@{A#X3hu$(_9{YF?xdb&4)NM`0X&LMBOiP)&<8>PTEr(bH{h5^4Tw3_rg={*BN&g_@Yu%0%x020NQ!!{ZUE|#_(%>`3K8FF|2;CrFwHXemf=pwPz+0$@h0Rs8B^C<#`^bM&20Q68miX#xxGkOJ$KIRVLujk<9zt6iKMcyVfVgpN#$kW!7gcQjpgE8BJA1DmrPXai7_hjw)X9qnfue(DhWf<H6)i9M9QUJ^4KglkFRCZeP#A)5ff(r5U-%%Ow*w@jDN@8Y#>bBn$A$kKWy%ZWiL_O6v+ik|HcO{5$1bQG{1zlXle1it*g^a?pX(C1`(U^z)6;666r@XEXIH#nPpSwl3i^JdrS5dS$r`+jUqX-i4H-S<tslt*aGCvP|->^nC@&&5$Hy`c)#dbN7{>t-?FyXDccjs{ZB8yrE3H8h4}w4RO61GbbuD9{SYaP@qNg>wRxf<&tg6g8v&7S!Q{VLs5$XidO^#Tx)UYV&UwOueB(MyVtYj-(rgg??qvcx9Gz^HP;yO77M*<ZajGO7KazYGVQl(F=O-%#kfo@-r9J@5GwWtfAng4X)IJ@u0%+qm0%SvC1v=G2vp#nQiJvTbEOz(R6*7xRE#8)@1%o5@=;@(R!mAH6H_#)KgJm)VF}%FB|7eKG&+_q%^_-y`H_o?`fOfs-`@LnjAjfBU69RPCP{~RZi)>qhkS4}D|^K4Q39m~5r+a>E8tZbN!yWKZy>w%v#Zb1I_S+$)YU!H0Mkh7Z+fN)uJ68^-E*)76x10kjF($srI*}sBDozF?pizaNOuB-$XELOwk|LpJd*N_y9We>f`rzsdg1nutiyZd`{2HRUsio<A3PV{yRydD4+l3|)(3<8K~bct?(}3o3{bSTf2JP*S{btlA<+T2C@YfUdTRi7R<6c1-Wh;8R<9$vk^^8<<l<<|JOFwou>&Vp`awf2J;5cTA8463UG47n1B2;<eA%@=s3jju_i^n5+5G>Wx|8%laD}`Q!*UNi%jtTZT-Xh%zs~RX)ae4D3jtK>KRaN7q`9@!svT-5S|cmRT7fRsvNG;`3#?a}Jso)51RnXNZ-S#5fHIaN=XPlwq*OD0aPX-GpMzW~b%9l|sut>`=vW46n3z|oRs_!^D57=cvtjiLzch7A5*%kN;+r)Lg^aM8TMEufP($xry2=rZp4vHlA#W4WfL583Q9la}OVgc_Hw&?4>zR<i)iNB`y-s_=p$Zpvl$xB=YmsKZfX}C?I^-KLd`~jnfM14>&UcM8VV~@?dp2z?c+E;vMKr$+^US)|Zu@i~Zr%yDxZ8yRCk5KK5A>iJSD~K!+g_|IOb?OD?L*yRGYv<}e!Tm6$UA{+0R8e+{ZG#hpz6<h8adxV+}H9YC5&qbRo`H2Xxk7ja5Y~X&>Y53R}V=`FAd}0GQ+dq^+u5Q4yp@|kKmCI)~lanMscxC?N?>yC{AjJ_`fC@!=D#-*j_4(;cIdq=dGYIv>da{xz#p?PaBNNe*79khFE)DamI0c{(61;0pB<l3+#FxAvKO6`oAh^u8(7Upw5dnxpCwO9m&(UGLAQc$R2ZXj$?ns+;R);IQ|k;JF578438dU{2fp+hLKgqf3#i3uuY+$<NcK}lzYNak^N&7Ew818>Zgq&y9uk}lHw>H+$Ho({M!i5?7j8C&vyhbZTc_&W*ET*-q)2QX~QU9^6s}K=P>Ss)P54o96}?9-C8s(LwLyO)~myjgZQxGyXi;rK@=%T<Fk1>fKIRW>%5xj$77XV3**xL7;bd!mw$R6HpGb<zueP@Ox1#Hcd!R}4edVATXZ9jN7$6#u`bjzK9#Ot+kxcMwowre+VNzux)}XND{`~(93Kj3L9bL>wcDqfaoi<zzBRWIW#u?ap9wTz+qZkiDX|WJ)yL1DdHELG3I$t#h1B4l3)5O*5tSHpx=!pxSQ+-f#aza~V)Xy+bnNxv0^FdJPdRcV3-PC%8vn@@yy>#e#;x)i!|vX^5x3=!k!deZUwUJJX`F1^{%daVOe=ID+&>)pPK(h$*+>A<R~FP4;?v;X@B<4co*cjrGl$ky3!&{}oz)t32{b!6KE07s26|2_E)Rq%p`Iz*A*HMuI5)>CYnW?+o@Iq}(FVaasYSPJtPV&(Eg|khJ=py98Eg930B4Z;bJ<KIJkr?rkgKT)HijwZ5@VWS_vrn4`-lIoOE$NKdiqwdzsERrtg;mnW8cz*-f4qU-og*HlWpL0O6|l~m39!<I&v<#xgFv}f9Tv|?|`GF`y+;IJ78a|<HAl=2c*+SIvIcK0K-uES1~6$f$7tW>^|{Mc+;aJ{7$124v|JL)S7p~HSgtzkq(_;QnHk*<Jt*UM?y|&ICaAOlbhocPddR)`7JGtUMFY=s1$s-)d_W}%AOa+I>F(_rHr7nonR1j{u}-Nf9(d%zKX4P0Hs{vCE?``P<Utb`OZiOgiB3N1+;ZQc8zNu+1n1-*%x`hs=5OtxC}gA)O7&;JyQ{z?hXiI=kG|K>wvc+CJW759l*;H<)i(t|F`A4KX2dfguYV=Mn!R*z>Rk<oc-7dk0)r`g(bUS`|Zj9KBRZS4}}w=G7R0odfmkQ%Ij`0F%J`Y%hm(IZ`F5yt?7Zzudf-$b$Wq>-P(?cx({~66a_9k?gOn4@|uzZePDE<u<Fm%eqjH>aYd%MA0FSMRQSX-0Cp#mcrFzTzyZ1YVcEiiAhv7dL}14tBt3uB*!OS<;&-~8FOv*IfNGhwOxiG9+B6<tRUCnzhZ&<qwnm`&C~rh(;V6U#RMm|ckAW$>b5$ejI4C<i(=pAD!<I?IW=`@1*o>E@GZ{_7w@=1-T3l1G@imhxWo-(m1McwnR!>9R&Er?=-Df~V;N1$9!Yo`?-upd)dJecKX5=f(=U|GpQDS{y4)(9ON399WgTky|@w@1Gi2BxhY5LDReE)bwYFd2(94H$Uv3vn$B%hJn9DE0*wPwGMJbnjeyX+6p*1ZGoXP2lX=@vn{p=Ga}&LSlJAZ1d@UW8Ot-geh-i@>s$d`psd3Fz&}Nh~dvz-)nK;a27nY%%>#Ntjv!p@JlusJ+XOvbvx#!M6-PKU*Ges4YX^FOS#Kw##t;<gN3nLCav=lB-3XybPP<sbAakm*MxJ*IQg=%kZYAQf1~}{FPxtT~N3Twms3g-&2=Czv7Z|e&{mrSx2#cv|EN?51N*Kjb#|Bu*#O>UxxkM`hUMtE(6z3$#tobC0Obuojwq^1Tyo#>#u1ofizda<hi3uU~?c%wxVMZ7(K<Vb=fY0zND0EBGV#tG)x8d;J-XPQK)m!d<TqjJY-K-7vQ+D<A#{q0-S&Oy;OYf0=&5=EbbjN53B}{_O<Ps2N8|O4Vk`kP$GjgKeuOLN1Y{iV|E7k4`jE_S4_i$wY2lGK2yM#=Au4xXA;7yNE<HFPXNcZ<t_H^F$mjr%GKI!6s8TfN|Y~-fQ;<z!fWG0u>7Bd_z9;$Fkb)E9K|>Qe|Jr7j2HL8s(^`NwRaD!+}@j2n9vCWYbh7FbKAh$d#LehK{M<>!FE_Srvchz-zreaAWX8fH21MpfwxS|tl?TQeCg&rwq=+FynA_{m%oXIyW0&6FAAigO);_kp4TfRX~>O}n$1MvKi5ouR2JjYGGBPUUWo@E{jw6~L_Ed%{O#xT2JEjd8^2K4j9I3Ww=xRaklsDjHz&Ch_d1+@;OyCh#cQSIGueIkhR$OvnrZ;;SN0~|eK3f-nqu8^GltN+pj@W>-#SauB*mF}bQCX2UcBKdHinM;VNaRvj$?@_jew5U1jdH-U;c4#5}!zjT$mM`!s(8-3v_hTsFN@)|8Qs;t$Cxab-$QFdlT-^+>^5yv9_^Pm^zD(7gv}Y_s(IrGu>xT(K#fMe_EdKa1J9kcWYjEn!~7U&oSztIovvs_+&kD4kd>oT*SlYFjMho<CfPPa{vAB)p?6K{B`5PfRe%-o_c(BS?~0}{psJ`UYef8r}lHZ@(pKkR?~jjWp)N%WpZB@x;BH%HSaAblBUrwLASg4;}mM{=}l*5pTf{=%h=;8lQ`7(fFsjs0!2x`4*JH9W31dYmU|^*xRg?t#M(59WgAcKC-#ltaZqDV9Un$R>f;d;b3^D%o4fn_`$0S<?IhsxWdKL6m)l<c*^iNvFaP}6?!zm8o;{wX?L%wlA6G+UdQg$;%vn?WF6@XGW99z0{u*(9cT1yd#}K1VIXS0RobJS4)$L|vqbk0Hp-mWn*i6v!Y9sDBbfLvzPXk)>(SZL#9iCi0?|Y*daaiVkd)!DZu6_N(HOT)4<xVnDD0ft2z6$l3?Z_(BytljlNJ1r&ZHjBYnyJ7dgZ&#4w<@qbK5%|?wHyzT?d3J-E=R$O^Wo$3<w#@IK<UU;fu)W$VxH+07;8_ixGGnP!B?_79??{xg0^F!%J(Yb`#uB$fj}S-2m}IwKp+qZ1OkCTAP@)y0)apv5C{YUfj}S-2m}IwKp+qZ1OkCTAP@)y0)apv5C{YUfj}S-2m}IwKp+qZ1OkCTAP@)y0)apv5C{YUfk61b;@-`pl)Jw3;Cr2qWf7`e*gu^7n4^IcL;k2z*xlj4>LzjP|K79X-AaD<wh%TvLz)%ZtacjplFy1!Bru}?mqUFly!+9(X4di^{C!J0A3A=$bo#Bn9qY5Np`WfQlu-ZoO?wx(w_Y^6oVXiSy_Z?Vqe<XE;!7R2mn3j(yN1p|pA?QVEghLRAcNiBQVIV5QNxc42A;ZTTB!RYI5V=fAIQ0x^9GgogHNv7hqlzcu+1j%i@BW&b`GEAo0Ovh%(z)?o=6LuZ6zXq#1Fv5Yb7<Er8L0lmo%>YksKV17Djh}p#-aYj7C9v2cc_IjZH%7D8!8Df6L=O0axCVH0}yz1_zam>7Z~nSPxfVsTAV?KY>206cQfT<kP-?!2J>|(m(hfy5}+!rUkou+wcPeJf=^w5QHbo&wuKQ3d7z+7E_W#q7ePJ#^SS;7_cAScZ&S<75KS$F`_E<3S2oKu>AYi6*wZjc(|SPDp37adUUz%3N$@nzV-gP7@X99nn`n71eBS=e3)W|ASya7&+Dfkuq{gs|2!uIxolyI3zkAKJM!4C-CGFqy3D?+y%2(>D^}K3uZ5wcNcwg(g(w`o#WKv_E&@v%lQi1zgu$W>+&*#$!^W^?O=g!6{9-hzdvfSse^Lh>zmW@q<%gFS8s!Bbc&~n1hdw_fMx-uuQ}KhldbOCr5I-D^$o($NCIGu{9mz-)<%i{+SHCtbE`wub<_Fk+5o~%t)N4hbgTx*FS0(pYp=-_2E7b55MEv*aSD^MW@Uz*=`tZyFD0}+*?GqMq@SYZ8W~|w|`qLnAD#>c>D(Sa{twXT(HZfYMDMy<W#rC(TJfx$=j<BKb$@)V$gz4*{e;Cll^di%BMi!J}x}mbDa0cmh`x(Z#IWRi!)f44uF0{<f^Vqb#geEsb_5J7t@B!VQzp`XvC~3JT4$8z)PxEWwN{A$W`edYdkW3mKD*j&W8IeZ)iqmhNbIPEZ%xag3jtoi(xk)8t%b;ZO_w>zC8FWlaJkESf7LS^|(NbcS!zTuR&wBC8qrZ0e`N7#6So*uX(sfG_4~+>qd05>+{`SWaY9E!cCh!Q!0=+8I9DL!Z$gGKTHEEAp{%WIbt-Z)pkuH)6sLhqXGQhBReuf?`#(4chRi~1oDFS8pj^$xX{3E}|C!ga99#A}cMOpnR;@i?MN7J66g0`$e#Th$%xIN5FGvR>bLxTIvTb)sy-TwyPwkv|^pQ<C(FHrKF(Rtb^51f>d7~_fZ!mzTruI4Hq3<}Jd+q>h3BtdkPRMr95$HI2piYgetF;0^1HV?r(XGdeN|3dKv({oYDXW_VKbm#fvfk=!%1DgBQQAjQ?OY+n@8ZWEg3MUbZK`Nbx6kiX-p%Ym&qrz%D9=0|RSvOC_lTtst_n9Q2XLj%C7n5X^u1U+Ov`9g+YdiLVHmSHr)8tu`QyPj=zKAgKNyp^_J*ReuX5e+7mhRV4nfO!D`RUi!S!j_{m>m(5jlK8ZvByN`pp}qBmUC1tPOz<flMl~BHUIy@w*vC<(V6Oo1or~G?(*)oqHrOeEeqh;omYqf#pe!QI$4Ci^cIxHzC}nSr^b7DwFv)Hy?@x@ZZUd{d6IrBDMpJhhFd24N|5r`d8%x^5}X{NJ*`?=f_sMC#)@`I@bB@Ch#=`w3_f7O*8HLrbt6rJcdAQ~J8U(X<Z~&Ge5kbe?`Rn^S$?}`DO!dZerKIz)ynYB$KcaK=4BX8waBXMScWB;pGQR9%P>N(nlA8Pe3Q$V80J)lM%$*&epY2ja=#$0@<AC|+)VnuAXSDmV#-OoPnTg#xR=8AdMOShy4bC>mg1nmD*LIRQnYq;Q^->+Me14ec>{`4R2L9?bh@eppN-4CI%ZIU{w;U7eQ8S2w%K%uGrt(6b^gw8$rq!#yN?~&ToIPE(zFI!6yd1M2>p|dLIk(*^n1>Q_(kQIQSw#+=6j6%(fgT?rWri*1B-chlF6;%dv`8A?0>~#RhWZKpNzTueY3H-bjjeHMi%CoH5q<l%|sSXva<h{(y{n%fE`(48eY|SrgP<PDz=$Nr3mgxLEr1@R6GSq$R~Bv<(_IH9&ziFY}<^(lgrXpQF1Yu>!EEF75)n8A_Hf4%|+nIna#dYwlK8ooHVY`3&uo|Pb_Q_FOhGr=)UPnPjq_7q!4u36|Z#6*FQ10!76S4=YjMF*b8qm$jtY{L6u}L#W@2o5!=`w=3)b~zYQk^W?W!?&!{HSc!F+i*=K{}ey|br>%Z1RL69z+G<IPp1pIk&mkS<;L;RVua(P{ma3HQS&y6h_Do8|Vc3H>3@HOFm1?6$Dz_OZha3ukD^h<2sP$WTr_FvMieaR50CmK|_F9mwc^(f@1QlY0m@togID$u3!XheQVgV>A@x%Y?DA=FFbdR$os3|`_QEf35D$A_m+az4z0o5$ZTC7sU(qbDRM=09b_QNDoYmq|G=YJHccMIje9be=T%ea(gWp7+m8z4Aaz&h4rneLnp1xfWCSDj!}XTblYGE`Tw~%D?w~3*i3yy|lZ37C`T7q$%Jkgm)CyDfcxBA#f@5^N$yWAl=*Vt(8#-Ii?p>Bijl=ap(F?rR72p>$E;EPf`Rg(^>;}A1MM?-<6XjXN#awSydtJauK+B)o6!_6~QRU>H$;9BCw?;O?8wmg1f(WZ}tBxpS=`XUV60%?9?Jn9fXTO^Zs3CN!}tb=)U!L>t8=X%F#i|3`MXz=Jh2;sv=18zror6y$~+HFz^_kD1`7R{?EK`3*iF$H-Wneg+N`+s#5P%2onu&G&KH=U*1pGx=NrBHhX#toDLSkmhn&3iopWd7H2*G)aBoKEAu4&$5sGAM{PGBHRprpjj!pW5A#96KG?qCcOJZlGtAB2c_8PQA0JGY2ZEj<w+=?-g6wn?9mA1a$njA>-yf0#+6=2!_S8Ax@nUxHkxMq5l|L^p{w)iJu2uNvXl23i>r69+9hpFJqTcZ*e<t*ctQHF<WWbU86}fEW84!_QH85<P4i92t@3*$4L0qxfCEIgpQ1esAfX_P>ZaFKxp;}A<8<H>;0Z4(9f+;hiQOO|b;c0#EV-jcvBwOhVCqb##yDodLM5sF~TZX9#@P)Sf9*J-~*!4U=>(?3!7bGq(hu?_-c@zHss79mV&3eh~IgM8^-?^W9WjYe}`yHu2WE27K&#9~&{u2f|tVP|qA)(;lr5e$4HUw<^`N$601_4G^{R(;Q4+pbC^^R8fg4$<Zy02Yc&=Sh}_S-vm$Q|c7rc3DtK4)aZ^-Ue2mhDhCQ|mLh`9S@y`*ABEIr;h+9~uH}DSH&Ty$Y0dD0BLV@4|(cAK$3mRq?^>b-~@@M(EdF`>apN3jG)D4h1hiLsJr}PYJq?*kf?^=dG`<IG^#(>PwD0w)09#+%@vTNA~k9ZYO+^=F6G<|L`TczX^M%s}hKR2NVxpvJOW6eW&K-t3vR_wq0P?-%zw{$j_P<3CDP0iN}hz5!lA{HS2h3B)+0G>?Qpgh1{!ej?b}0<Ep${`;O9UJffDF$nFq>x?Sp_X|b`W`@&U7y(SJz?@DS^PR66i!EcSNe-cnB-`;D(FA+178@afTCZTQ6VVq4$LXzc2)A1b1$S&oS*qN7%xg{%u(>y6SPd*Y}Sd@a6Lemlkms3$c{6Afz;#Ax#_&}BFLK<#c%$mQ-NW-<UDn3)DbR0gQdpkNb9UtsjHakI<feB16$b_F|;0C8}ZvDFqEZG&aOII-y-Oh~!%T;7zp1(;G>!~bkOwT`Z)H4g!Q`k=rEM?)(zwi7+CA0CQXN-JXY&Pzl>`edgGaEg7zdt&mkb_P~sOrs<a&VkF>|p0w4#qm9ZZSya;=W<ZME~Gil#zwm<mp^w2MOCb_B^DhUfK7>BoA4S+wK<4$;03#b!OZ5d3b{%WN?NxA6GAH)cn-W$E0wnQ~WXc_=WXKY~4^kvYGFCoJ3tf{GXFRAP@)y0)apv5C{YUfj}S-2m}IwKp+qZ1OkCTAP@)y0)apv5C{YUfj}S-2!#I+c|6xSu5nlhm=Be!wQCAOD_4$hE{7m|KSTB;W=;^UF6fHReGmfEPps;{3q(LomEU_#?g~^yGka9Ai-WW()5Vw-39t)#u^Ou+1v0hUbZ7QU!=doH{+j~QAXd#s_nKZBl>N0>pD{|qwf4~U%5f<uB|owXrc&TgpvdCEAq8$Dg2sveNx-7S*?kv+0igDPrjCdh_@yb09?uYl{$DTU{nP~^^!=HHzk>V_WhD9~vWOShX<PA}Cl3g^)C9%_aKH~$Qt?Wi(;%f^<Hpr-5E!iJEAKq4efy_s=uAfLK4dSRr>wigj1(8=j<<H4!yVGDbLz$y(O``J<)N+1X!Xg;;NqA7KB4e<q~9rmZ|#iEs@DUy@{4_EHkL$=G9kI=(bCv(ELxKDuq=xF)>@3Mmc`q8J=@A|vdCN4{ivlu26t|`Ig*me;2kYZ2Obw`tpD-3g*Hwa3)?T>WE+sinu@r-ONXT~GF@;k*-{coKT<ur#V>(k(N3lZ=f#nB(9ZDFV{!C3*ZY-HN*o_ZhB-!fi6cqj&;ZlFb`|kYS4!I@&<1Jls|`yaML3)7bddylzpKsTm5{*nfxT8nG2(~@k2*acisPuiMR|0zIPU7QmlHZCfnP63O7icO#E@m?((+14+!nL%u4j_Mlr?`6l$OHz=!=^69a8w>M83w1oHW|1M(C!fNn>Ml#=Rn4X$(&d`$FR+jZYc>o|lP`Mvj7KsgFvf@xIls!*5HZ@$SgG6#9SB%lnCTgn|r~73A;H`z3=<#O@w^#x9HgWOZJ%D>B$Tok@AZSq7En3!A9!$Y7Vqfa~FVGWh!LOzt&N8Jy*folFyx!FzZ`x5Q5dllSq(e19p6>7(_uCcCcTtNiB*9VYTvd5)WX=lTttRg#ml`>udzoW=j%3Av3`VL{rC*OYMQrrWZ^urgW<DxUjgdk;P3x6jJhsUk;Vta5R?I=1xp*nCgCj|+9wbNQLtxbIR&m67~IEc?Xs+N4t#U2POvNd)w97o)ucJ~F`VGX>LY#)h~LE)6`TFv5Yn)!WXpMtJe=ovUWNMkptoe%bG>A@X&EMU?j&pk#U10}~y6%x7rMf0Uq$ocz<|!KFILq+GupdO!<5+g6(UlBi*p?18BqS|y}D9&0x*Acup2FTZdf;KQz7afbkP7SN5SqhKwRf<K1@86{Tl0PDdsbPC!U@X+{#XQbgnc-`!`BpPJ^iR4xWN0%P~7uAwl*DZ6%AA@bL>sEk!MNBT-wFZe(;enLbZJ?JmjQ72`Eu<>tM&ygxK?YB84+=PdX_JLD<7Fo>zb$>$lg9;$->XjU3vz|vy}J3e-fnPgwTXJk>jm^?eoc68?+*2v9sU;D9x$4a=q)MW2`8UFPm84R0-L^+m$wJKKtFJCMK9PJuI#Fb9=hoRr^}XZZ|(R1sX<e!TZ%6%v`eonO8J3zN^XS3vLATL#nC<XcnRObx+~fb`2&1#f3F(r4?e3>0auR&!2L&VKD?d*u*WRDhVs7vAU~t~b3rN)Ug;h6WQYg^HJ)a<yEB0hmE$^ZdnyQaW6@5Wau9sL&K)+pAfO0S6W@sqf~?OMXXf4pf#}Tc1IEKaFn^nOb7>+7@`nS=R)&JWUG!5zPiqiJXVi(86$HWZ)6Da1!9l>Rar)SrK@d=;vpM_n1i_*4MxUZ@fk1b^c2zDd5RB96i>2iQp*HCHM~3A9@HQC{Q?dUy?lF6HX|e#wUp*#U@8b{UrIjqVHeW)HPmisL>PsMXA>G52;s-Ch&iGAy@P(GlKY?@Se1WBsUXMfD2i%|c|GeSv4V~eXxSit#mh^ijE?0Yk>QSeZ)?yDZWjaoo5#SEj6vZpqAG`pg7W%1+XWbxu_KWCbkqh{?z5YeH?FfotFXZ;$wTDH!Pb&23wt#8({&r72g*|?cjY@x50sJ{XS=ej_fz;-Tik?Owyy<^?iSHr&?3RhD)4T`G>ON`G!r~x%yZcYf3qIsw5qF4lR6_OZSCT%^=%AxTxmezcA%5ca*?iS+hH4_+YYB!=uyayZ;MorwoZuT`4Dho<QJ?rGJ{Birkdt|0sN{-od+AOT?R$Y0hKvf59qvfQlGV)?<cT88=l-s$dt*Y=Cxb#(U%c+5&9SoPhs9f^@$4o3`1N4t%#cnXw#Hd+dyog?JFDT_k(0rwq#yXEvM>a%+zn{&aSz2Ib4jfSx5Mz@<-pUV4B;4<db0V?yKs!rhz~uH8G$#(`|Nu4BJt(Cy-)D*D14OtBKvYj6ej4!eOrC<3Qud6Kf6U2jVL7;w3Qc)AAWDfFG{_}u$R?m!+T%j&0<I0o;xw99xAZv(iMXq<lnTDuf(F3XG3ygdMqv<3$!t%jl(NLyEjjm$6?J0{++?*IJ}dh@kIJ)Jg(<#e+hdSk9?kSk(H_O=#uxbyKg=o&6DMr#TgQ?>T=Km+l>VL`2I1+SK9=<Ds=29X>tNeFdF;(Zc4x-W_LX<&Lp6|*Vu8vl>~I14(N$kOTbI=sR9E3V#&diyq5p!<EJQ@j=f93blx;N_u&MLyl|PK{NMgu#v7Ii1qqlZ_B?o3R07i8-seE?kbo)$a_v-_38<GEb>t6U0*W7NQ`_E?fcw0kQ#}}n$KK<D#X<@3=zMJKyt7d}HV7DK!nt_t7G>NrosYxMPukx+c@>9(G8La@Zp2}7;jS~L8?l&otDpH*QY@Cg5i!z{h(&iL&u=#-WANo`3HR?dF&Lb>Dp_|h28H8~DA8xXMzgZolIK@m<6!DZ-Kp+qT=&eDiB^k7{qFEP5654j+U>s~Val&?k!H|fcUKgS=5s$Okcz_XLWavZ6_M!9ul&NEBN9c+wcgSuMxgkOninBQB9P<vs}gUYaQtC^$FA*X7;4OT3kPV1VYd}qr*usyk_$Em!l6*SPkwo8*f0bST>BKHQWA`Lb`=f`Ye6VJ@`+8>ArMaooEY4{<ByUw9Qw|-FHvcMt3hPR7t5)v@4db5gBG>yxkI^Ls8ew7O42@06#7Yikl(}|70lR6Xq(*dej(+>2W+mmb^6cunNv=v;_VSkM`@4M?-y%kR-d5<=`a8MUTdsmjC1f$vBclS)Q${mkMWn`t62I9ee^vR{wIo63ztuPSrKiyjo-+MXRqp9!jB_@Y<C}vLsoPk=kemZprvtA>#vm#P#w4(RqbO4e24RvGUiO-#iPpS!8ukCzi<7?;`^tN@?(F;7Q=J+xXZzMhTj2}Q&xm4Z#cteql$dCbXS0$s53%zFF@OC!$I|_JM5fxD$ASp0K>}K5Mc!`m}8J;m8$WE^Zg7o8fSf>?atmV-7r58)jr?ky!{er{jHn$v;*Lg$6++C4utTxmr|<t2Se`-;cIP8!SHT{Cx*u+1PbZNQx{H$LhXj+Ugv>OVCfot<meFwwEHi`2n&aU^hyR(;7T}nE0pc_kB$KC$0dGUQjzd!N!*!gEfV4`9U0<JiUL|8<`1*CUO^F+#_X=0S0L2yxlcAb8gQ&aDEH25a9?vw2>$gNR{o%=YitZm4eMj0bS(7mvR7POiG_jv{5j5%aj>6De}>^|JPhc+Qf`@z2eOab>nB|kz}$qT+~{B;bSyW$Q_)L=+e4-wXG;@7kLATvc8Vm>m)*@nubBk1hx;Uk@{>ShuNrv|X);K@>OAD9k_-)|VOO=XlA(<1YK_+4WT+i}lFV>B1=56hIxeK9fVHWY0sHS1h(7sf*FCvZpx_|CXcL(V*Zekq1udt-@NF5d7``-+f5sQ+;F<;-Y%H8{9ci$YKEG#%E*&04dYP}Pr^CUCcU<!c>7X+}9!<NL4uw`}CKPNLkhHhjYx_Y4tnJrm{T!76P2IA8YWg#P<xD^6Z>mh_zeBtCx_BnY{eH4RWs?a~Z{mZ3GcrM?=62=ISSDE1km`z2WdV&Aca4ic7Kjerq+flI1*K0Zs9QX<pyu+&I?Ty}htlE9ecf4bhWGk_&`K8g>^V#6Oq~tqeSF3z&Srx?dqDN1cs8UxrhR-!H5*9oMmt2BW&^Wwi1r!hY_NVoMz-zyFV^z>Ne<42F(I8?+JDi?Bl$&|cQ!mdJfNoZEE{y!N&A(xvH{QTu0D7*8v<J{ywyLJ4I80@_iq0??)b;AZ_TT-U|-PSy}e#pKyl=Gc=N3+X#Gunzw>AoER8Q6o}I~rH6zacrzx3`2fCkF9%aIQ0f+B`yqWNO&z@ZR?F@*DmRmpEodGrt#m&hH8F2OK(KI#d42Vn>{~RHo0h9a*E;Ik~d;F7IFvIQ)DBkY3yS@A`|5Mbr&-SK+xxAZ{O>H{RHp><rDNctk?6kt0IqAT&pY7P5>~#3MbW(iqUpY%&o@4jF?M>b{_h?n8!{oAKI5eh1bcGb7VRt%cKCI-`8BYhjMcXgG-lv1bZ#@;RjdXYrVA%7HCIb>>RH9N?GT?5tys4*f1{}FTpR#o~158Du#8WIY;Im!o_*K6QXx)8t#WFDi^bC~xO-nPNy76XALvsei##&nW4Q2qJVq*X8=?oYgrD1>bE(1oKDc5M1Gk{%tFy+663`l)=g|l=b1L~us!XvseATTyLVyHR;9<4YO^QLEjeD~$&)Bzc=M?P11-Xa64{JMFs-pYV`q1=c1&u2jIyeZkno`2&V+AI{BNC!%XjZBWbbjaaLnf`5`4vCJfn(7kiP}TBLtePwxA}d|giaOJv+|uZEQg9l)xcTSx`#Wi18eP!Ea3BqwWiH&U>rRCvX|KYHm#OfgfaLCxYpL*P-$jKRJ1GzaXNHcHra&RzryUpL6fkuy<$QiD1?Iwwy4aeN;b#BPnt@d^<ZCChQ=Ci&_b>E@;w?$=={MCnxp@*4U+1`1ekch}v({gzuS|r_!p!eOs)=xP<er(vQUY)bW_>%xm;muf-A{e*#zWNtl^0oP9LVh}mp1N?g~Y0r(!B>_;f}VtYVyq(@IC8iQ0Ms?{$IZzAP@)y0)apv5C{YUfj}S-2m}IwKp+qZ1OkCTAP@)y0)apv5C{YUfj}S-2m}IwKp+qZ1OkCTAP@)y0)apv5C{YUfj}S-2m}IwKp+qZ1OkCTAP@)y!vCL=UON2cU5`L-$#rk)od}emw@*#wjl}XYS8x%EL?!dNEuGm26g@X8WxyYS!c8j;ZEE3o?yOpZCr=pKjz5v%{1}41BaU>rNx@hoCqdo76olSKm28AY1CX|X^o*#wKdyHq1sqxM#ewsjj-2A&n9<O9CgrsUKBzGc3G#45<}M|j?SoEeeY)`HJ5F0X?JJiYwr+)N!6WM&T_&jh`rKYVIz#*ktY3u!_3@Wx(O}eo0aClxXOsUlLiYk!gVS8b_`pNY3e*g6|IS~|w-Q<?@oT-4t3wvOPMbU!T08@M7K)9+b21P+p6t9~a}!utQXELH$ilNbeuDfS$Ki^z<;S!q68Ms{v(sQh8Gmyuiazeu!szXf&y_fJ@cYA8g$`ftqe?w(+19!m`rhd{OkH#v_i32j<>otqZ4QoQmDcj`gPJq?aGVBoQ{=2`vFpR!Op_~_@*{YD&Sq$<+zdXPKecju%MvK;0zZ4dwg!u^{;36KTiE<upw1@b2)`+(>O%S5V4+LySx>nK$i#UL1S<K!0)uR3mFr8Oef2V&L@yBhrFcU{eS$&NdtwjWXbABBZMMsn41?|k@qqNHaQHS?&5-623AM6w)GAw1P#oE@t)&?aZ<US|pPzpX73`V~`cbj49yXbp{5T#m&mKGfjwumZ_g1!!P$WTphRaW*o+P0BZY+~%nhfiLamxvNQb4+6M)77q3W&rcHhNO0LUj(K{f0^^RPAF|z^GIhD;F=WUQPu%r9<yDY0`k}#NXueXVZY)Wp@daN*WydLd)|uHVx*lp33<+n+DX*AG&fl)8VWCN6r6&(xLh3AvfV4>Ckq@L+A`g2Gl)#r)kEW0lur<@x13VV2xVQw*OuRu&xhDdX{8>muk|FtzDVG;M6M+Es+WK7rFJ`xMYGZ+uORa=u8+oD@AhOFB3krD9|jcW&+=ppe6>IOxRTN&%IZb0S>))uChpFfVb#kgh+5YbY*>r;_6I;-)3ZE4tvs|bt2K#o+}kDKm75R%pnCBrqnhfkEcMl#T9MtqbYEOf%54kvlO`e<xNrWw-gw>)0H+6lL`eDBPV2R)8JRB+3WDAbT}a>TQGMr6ILrN(Q-Nqe*ai_s$7->yANe|yVT{uLicPR<wOC9RJmBOQ5Hehn`4T~BE{fq6J_oHv;^j+96DQyOF=Da@MP(yGElrK%6?X$0#a@~XzS0f1S-bohwEjlf#YaG+QG#dU^UI@a|o}6k|y5JAw7gM!uMX<rPM)LpyU^&xO$L$)%3J4r~w|vXp&xVZ3G9c^58pGP4I|3cFjh&8B%nZi&%_WAcMKcdfKrS)`D2sd;;6xYVU?UWlTHh*hufmDeQnJJQw4(hdP1WT7TAotQ#UiY{zrNd!YNi2K|wjy>OQ8k+Z8^A6y6wX1RH#AGU88Bo&bkfC1y;p4HX?I9%@_5Fa!MB2y$b3(7-~Tc$AaVBawK3!MuoelrY~DjnPfMk4@1Odk{Xje@AA%pc0EQTP~rQf6In3_`+b92aZG;OpM9XTJr;fvIzsfJx3cOjZ6I2Zjll=KbP*$YTOFt7C6+{Wk$#0%GkN*C#<V`rLo?X_N3c*QLJx$0YQ4$nyOsJO%y{!WR~wPJyR|#V+Q=DY&;o`!S?;3PkUUB)4==LB7J_KYqPaP#5>cR-kPP($}0wol2*G(lpR)B5Vo@O_eMQ^`@Yo_nRRZ(-d?jog+7EorGqkc`G^HNq9igdZz!w1lalAZu_o10rX5Atbc~b;f0f(uYlAzJj$onvWy*rJ!M~i{9PUe^Kuo=mUE+U(wnBn{NV`D<o`G(8!!xF=HKr4qz{3<dKrmT(ICiCCnX*z9sqgDvG-lM{ooj@H+m?m52P1XwA^m?!fnP8`{y^h;RM~<BW9US2-4_q?7G|zM^kqn%%^V!lBzdWHx~ZQf9mx0L-`HB+q~5nYEuXOFB@A59=`!6`ueV;8x?T(eQdBuNeKvtaOuR56+oAmNuhX3I-LI~ye}jo7QQL?b<`JogEZCuV(%`$s`{e6frEmgprWKAjiMj|3I?DPTQC3#2}MZ}`6&{TN`rJucXxM8y6cbvQYs(<k}48%{Ue?`o-yv&pEutz=UHc*J=WT5pL5>q{o#%gxd~piOJ1a4`+(+hTe4SU<1ldR&g|(O>8MI(|9P1)7hmv(2wgi=h`|ght$D*G$U*sXIPOgaN{+^UA$O`mF{gDORjC@ZHWjQnzgde9|2!#l4y{MlyR7Vq{EcY%<<3*n+$J>f<vzf8tOdK+^y{?UTCvtJBg6Dd8?K4-_zduLpa#{ghtcw#$g=NG#J96u*eYm0zLL>}zg~z>QnHe;P;&adrXvZjv*fJAe<tCjD9RZH(Qb_CxwjZ;-;G)FYNSVP-I#KL(doyo9t`U6Ob-$5L7l1OyN@(`kT<hh(#*RD%Q9?vpJETbSiEQT;#&`D@gMZ+-QA1B_A+12@%G}aQq#c8*Lra(SFZGzS}%r#kd79;?#0VzMdanId+|mPS6}+S1UR=fW8RDHyzi2qz3j!xg9-vos=dg1y}jzw)m~(I8QElcsuyQ>x@l}3>cySKzm=JmdaxjfR=lpQ2cI$7W?qZ$K?;qbh+mdH7(4&!%f5R(I5n&lCCb=?4msMaTEpGA>GZktLUcFgtKVBl)9uEkl2T9I)7|)Z^6a+iPZC!B8ZXl-BB9~3lX#jb36C!f(X5^!ArsX7VVdv4ez|+$bFp2hC{D5zQRu??-yGp=e>#!Ynzd8%VkeR$=3F|=JCMrYrcwd6W0P7=*I$Zu>>M=udH7lzMlk0!(Ob4+zghc<Cj~9|?5e1x^-427@)mE~eWDq)CfKfTKWM_1pudqN=8bq0Y)!_(8}MDI<lmj;^~l38ccx;n4i6V?sjsfrqNm<lw?)Q(=gD5ak|~KAq|DHdVt-YQ_oQvM7-Oq2`{SIyaWdkmv)t#C_f;a30$W#`NjcJ1Ju>AlEyX)Nr^@-JigEdBvd=I}5mNi*${)O6fQp$G+L{`9Se-3b7NnSiE%bECMnRcqDqGCPQJ#wJX==;o*Avm-z}-TcFAkqhxF?@Ch(vXM+n=@fKcHHpq5zYkC!ROq+0mf;4wbU&*4StO#aaJr{xE6@>UB%^s$1P5Q+6Zc>v#ZYKR=ai=Mx4Jv-yi45CaU~2J&vk#lsf|v114KCxcYX!PM){X)x2dUaz*1397S~GMrE4z?NFhw?Biq5MZchGw+=bG`o(vpTArPS`p&?zuJo6Dhs>bz`YV+C|~ln7%GLCm?wg<TIFz%`<dgD<_dU|7N?+p7{MjF|K|hoDme4_kc`QnDj0>e(S)*U&<gL)kbYeQ5#=9mAy+Mk{53DD?W=_-%IkR_-Rhvn=anC?SUqIW@@+pEs0Yj9O%86I2Dte4O_0fa18lWt+bF(kget*!dDlZt@W*L6WGt!)(w7IVYB`&Mt|VJq+@~3MFH)K{uQbE%2V!a7YAx_lkM@~%dkZM+njrJK(h6c`(b0W5t&lg78FKbm8)!EqZ*#}CLHIROU_Q|fclSEpKa|@JYzqgs=7c-oTgDmvrqm8_oZ8shd7=|aWbb}&jOzsFNnw4Y>H@KB`v=~lT`>7ayoz463$FfXRCs6B1(u8%hdhG1K$*|XQ?<GaD!qo(<d(bOvV5IkBnJsRD!ry7q)1@-j%VKIB?;UWifdgwNpR~mqvXX55|kXOThHhu!JR?M#mof~BwCC)UD(?VE;AQjoIcVG9B#iKl<;)Jfxd)ijr`qUcy+J(C4p`@?pSf~6kj*YYEs?V;{Mkcy)v)B)(yF#H#}SzyCK^7u68(8H{2?JKS8%f0tTIoMaS<XILf2)<K-v`a-<uV9s5Zz@q2S5pF{$4Ek)M7?f;JR+>*0S3kjP2WQBE`N$`QDjXA%b1X<?X??sRV^1161K9wYh>2MYsuOdP6o7@*iYX2So?5rl!zw!N%f!uYCBoODm`#GtU1do5tuAIpv!O};A$kbR8Fa<Kdx!_L%&$`>0+qNX&(F%GOrbmLEuXO_1?vOybfvPTupF}*LlRzL42m}IwKp+qZ1OkCTAP@)y0)apv5C{YUfj}S-2m}IwKp+qZ1OkCTAP@+I|AUeu&u%(j48WuBl>})21mH>gZ>@d|AMj?`yd8~05OQ3)bMaAVFh2kJ>b7D@2>#xMnf<k)c;|wjK7~^lMmNb6Q%Hs5E|XU?qp}hBa-NUk;DJaiJ8f`P+&B__B-VU(`bDC1;p%GWqezVKER6cR9)U4EBAvdj5g4_<Kw;<aaP*e7370<@j=Dp}6thdAIJP);dto&M`C`}TmFYu}rS6_;k906LnM}QS7!ib)pI1{E_6OmL(5I$Joey}i<P3+cP$0_j_V-bh`y=Q3cN)WteyCk8va>JL2YX|*xR%d)qq%de+2Ro|Y&^Db)Gpr>3r}(8wzPX9?+;Zy>t#=@NRsjCO!36f9r~RPGae|~#-x3*%^lt7I}ICI-O+`!`=m>i8%~M5q&=nOiXCTt%oBKA@Uz{Jp!+2!yn0TQlfT^oZNv@<4Yt@JpP78?hN=ykT%(|2*0e%t4{zo{KU4g;9?^FBmOk>e{`@7Rb_JC;8s2u6XhDtois^#-8z4E2XGSpC!PA9TJi{f<z#QQctkdoec5V8uHH+RL<+5l=ancWFo#LdLmjd9I!@?Qw;ULf^5B=S36bdDegpBV^hC`{OZ~XH0DCoF%G*r4L8iX7Cx}TTCfOzr35zvf<QGL1jV<+Q4=yB{l5sG+Vva}*AT8xKNo&!Ob+7n=IXy5M}Dv5CGebi&8u0$A6Fj1bnoCGX)^V!)mli*px)oB6MWN2QBC|qMmf%Lqs!riGU5VU>kMhSH)aHa3a^0}A_2L?77WA3Ix>B_yeT`yC?_7%gY+d-*-^sZ_r5>nyOopHSv|MKr^wVeT>sX#$hG+N`B3X26a7c5Lt;cbz@=H2)I##PVdUr$K|J(+yd6KSas8MODy>o2LmOp<CoB9jJt_bGEPC#1p4=M3D=$J4>pRP>MDXgWyhi;XFHWWYwHLrpP7Cd^22Qs~HKg68pJ8S$7*C_ddx!Mm0TK7qUYd#`6fKBFMzbDu2O=`(oeMqCy=USISUP0s@Nl(R;O>sb)*p5u`zo(=Z<DE98K%m#(=k^9}r+3+&FqhPWn8)|3|a5eR0L!9uw*OOh@K*z$@eYiUtnhr{Ae67!hPbzVf8PVC0O|5XP#wi<A^**ls`j>^SO~>!MWJ4B}igx*jY*?h~4_a;b*MF&>9(*VVwkSj5?knfOo9~Nba)CJ@{x(nM>vRsJG@Q^5c$y3DiDe0!>$yN-5>|XHI1d(lvd8SX^5F~r>X?joKAiTtVRD0%4<zN*bHRV}Axw$cdHz-bTt3>cWdET6vi1l4_|sJYJfhkugMSMk56;OS<tT)YLyg*-mkL4Y_jilW_X@!>_+{-q<wD?fRnc!yEQE)JA*t=`g%FmS$<W?b0Nb63ijhwWV9iinX1FjP{EglmV;so?4Z~B%GX`?O;kos>(Ucr`{we(%hjBL8cG&N?&ddaXa}&G=y3>Jg)%HK0nN(<a_=?iJAsLLvaOgx@A{>-1yS3;N51d&UZ_8C<!Fc)k_RBNT@Qh)8(S$A>iatIgYoHB==oRvZU#tV*&^Z4uX?9<DT~u%+z{4H9XH1T9M>#_Hx!YHk)hr?Ih6p`^HoU1=bXMk<!@l`5$*(C4F(h+bg_79{i}L>KxUy)EdRqDieAnKi6L0q4;|1QR$u|}7exE-UC|U1%w;6~m^Lt`4HiFSiCW-0Mr!b7zv2$^{DH1j8-ba&Pia}R3J9ioRIJ|WugZ1vS1e9ABzoq#h2~XX04LiD;f(w(OZKt%;@j-65Y;<ELmiCHWBR9{+!TZ@uLwY&5!((%|(5+m|cr4~U!JLN-EeC&7t>hu;ptaz~*?dg9{;`E&r2r4JUVLlxw-EUk?wwpdP>dH=nC9oUicxBW>|#Vw3Hl!yuoqP>#reA5V$utxDE7QIYV}<iQWQjgN@Fa?nkQ+qGHK;_%`|TG$;k>7y!%!9b6f@9ihp*slBN=QwR3M!Kd3|zLFo}E=SqC6F*7bvT8a14>m+ElD{-Qd&h>{l;w^#Uyk=R%oJMZEdIxb_<!R^M3y6!dKRu}TAxicPzx!KKiBt6At_$jw7<yBKVT7s@>89n5U5>B7C+2L*H_um~0nWL$ek(^e-p6TE)#X@mi9a^hxg1X>d%hH4EyrCve<TX4%aHMx?IB;EGV~2RRr1N847sA6&WhYFL&3;3e^I6~q-UBaeekIi_n!I`cOb45ZwsvJ|1m2?muB~IKlxI8wk@nubEy<-DSNj1E|%i0kR}#a&Qe_Gtc;o7F2Qnk0j1f`CHQvy_a_@H!I$r7$!>d;VBI^WzsK&DAa&-w4)Q}KSXi<D)V_gYY`L9o|0b;%P3}CIH}EY+>eJ`4d5w$FnekdT^P6J)Dj6d5+OHT{Zz}cn1{I?+Z^VaRAByqNn^CFI>|(6kG*%)XD#kvaWiFdv#W?;vb(V^{1ed(!Um4Stptr<{KK0+lDD#w8Tcx-dL#Hfg?*3Pd<vOC1Ke&o<cOJ#xhM^*??Vjo0A6|qMyt%iG^@{MVqO|G3Q$^UA&ObQPUx*_2H(K^M72<ASdChme5VO;Ed~}#8z)$;4j-{s;;Eh14F!HwrIHr7bZS`IO9^gDz$g#HoU5!5p(pThT%eadACyjixvx|82Zaoh}X{6rKC*)z2a!u`Zmpr_%AsNvBuU?qacu>=xhnD=c-)H`1o}0b9d3hdE8<Hio=I3G1>atMNznUR_!=a=u4+ZX=yp@)ohw)C&em-%?!_FS-;I}e)D7-LbZ+SQmO@;I>x{T-Iu}i;-7ZY>w#2YROY3p39Q=6j`(#geq-N`0(i(Hi4D|6{}XfD#qv};J%<|2t=ankQXE~@C?QNFvCgSS*(uy5DrAnVQ$>Y9KY+-!H=`O6^(#k};BguQYwc9$bxNnj4HYy}u!``6c`(`L6f{@4HFLhtfA2k%}<(oomW!DufRsS<-6{N~5w!|$Afq1x6yZ!>bxPfKKFvm*xwtrz@b{^eFd88>%M4vNZ8&{5dtpoYZ8aGP)r_GU;<IF)7Ni_y!0ANORV*4|^1S8XyeOu*=SWM4XV?`zGLs!qj@gEciG?~`$NnHB4k2Z{LfJladn#9`Hmiy~KrVz5E<+nj@KIIi3np}scq0q=yShEyl`;#%!M`i`5f`115f><ZaC{9#N^cC+#+W*yOf<Rha8r#KhPnU`#UwSmEG{JkqIakEo@7V(9nlrf5X+5+LVH-D&LOekFMZeAVpj)LyIh7}v*SjafSWY1`v0C#0d{T7mv!Rx>5TCRWJzlmXIbB=0d!hZ)H)FcbCA(i@eYSV*U__C5d%l9!4{^V=AQJ*V-o6io_-Y_Zz{nlvv5#J&p=ewKmBe57R<%G<qN0&g2M(lohk5X7qj}V}>ECa<H-sr^}<>1~@t6BZG97^JMmq`^>K)%O2p<LZc;F0eq8)iay@YOP~s2{=M3M0MaohsOITBTk%sS31oj@Z8>uZC#{#;t*u)u7#MwCeh|8oKDXvoGe<z_sg#tptr~AvxVMQ0ICbP(SkTa6DfRT_$J3Y3??Fp#pzgk8dOFBONZP-`@=R+pmP#3R}R`z_6&$wGB8qS_3b5bU=!Mz@v>PUC^|<;Na^35|I6we{KG$8;*}LmQ#xNf~!l6s!UQJa54|fiXR?;nl@7Efq;*YB3<#3{>Tv6I!ql4$r*;?^IEB?DkE@NEbi`Kx-rOn`CVbea2#Zw-;{a%XabCP2fVm>`!gJ~``dIx^a~VeUbSN2{t82SQRZ9wCSj}S2aoOQB*>hh;uRU1f{492#_g5U;LWXDTO0NbSp1!AhF{LW%G~NM_DkP^^G<@o#-3TwbG{RIwtE)#I#Eg;3Y>%P5VGN!2lLQ=gVRos>IY<Ysh(fY{{ipBc;9L$FTme47AwK!1<2m3+CG0}5#*NLiXS*G0#kjO&SPAJ2iJ^0*sm@^v7uaq4c8Ks)+v5EE42i|moM+_(_4ZIE8Dh|c1tkssAAmbzXYURGxIm1mjLQZ`WX_JfK}^DwNug(#H7)tX~r*sApiOMG2u&)*d5hp=(7YatC1S?|K9)OgLvw0!zB=tu<5LSyaap5#yGzH`<&L!q37mjm%wVtRW|X+5(wSOsMa7`g4|<iy7H5YFvck@G|;#RJ1*1*zfW2OS;gwL9H&Jnm3yW6L30r#bnM?ni!TD{=y=aD+C>O)q@ns=y8sr;4?D#)7J%p2uX74(Kj7k^-d%6^A5i+l>h=K34-i#LzVDPc57+m3DU1uuLtF<q4`<Io&Vwb(6oEN7sTa=cnLG=x4#*T49h-#<`;!iFBz%XB!H>W0bAN~AtdTDB@EK?n^qXKfI0LkoD|&65zJZE}rR?C^G`NZHU%RFEZ#}==Rto8x0+tT_wHe7NND<Y2WR^M!>Uh@4lztLgbT9emy!#5i9$bT-Q(s_5@!1lAJ6|C3e#jrkoX^maS*-Hz@Ml;!#PrzRZUQ`;*W6ORk3*KSZsiZ9ad>)j&yO~@G5AX3$dsWw3Y76w;%PTW;L<g+rdGaB5a+LRiHd$09%k#i%kCHg-MDk|Gb<lKW%7O|=iC6aot#oK`q2+!htfwB{`5ic-r?2}nm#bfyJbgqvKQtTvN_1_^nmOiC%kFY4RZz!WpuG5D0kz$esH7<Ug}$0sdINhcf-(x)no_M%8Pj=dAGwG-z*WTTWw%;>FeJC`c{b5k=S3l+zbyTTonDfo8a!HdUes<Mu=oM%2OZI0KZ<07M}^P2RgcElW*$kpyuL6zU1#(kQa|jUOZh3Dw#j;tG}rML*2?9lKs{2{B_n*#hcY|I{l@tYfcpi=<SHzqOSst0QzjY`AQ&nJaX2upaRx@2SpW{mP0yY+W8K_GKlFbEcab40Rdm8(qC=GK>ac1^^=Ao*st^>^I%;eqy`6`TS_Z{k!ug*R2}mny^K7^?m-^VM~HnfWzL1XQ8Q=8{%qp@J_G`RKp+qZ1OkCTAP@)y0)apv5C{YUfj}S-2m}IwKp+qZ1OkCTAP@)y0)apv5C{YUfj}S-2m}IwKp+qZ1OkCTAP@)y0)apv5C{YUfj}S-2m}IwK=@x_Y%6hmX1^2iFCW>T9_Wa>bC-5$vN+<wQ7MB7N=LMJ;0kGMcSM6YTY8mDCwyv>|K|0&6AEzoetlHxgh!nCYo70N!nFe>frp+rqN@BH_Iz}}lRQ0^a`cY)IpMJ&pR5z^cyH%pOL9ixcORAZ$i2t<QniKGy{`B{Hu5IRwj2JwSo&W_tp}E~IvzdW?}f{YE!?rS-e~jFE;`uO8|ybdKQdeOM7O11ax0ncSpM<Qp5J_K$YmfQAW!9rav$H{o5^)SLC^G3<0sB2(Dmu^=UGP_q&hp(ujhb!Y>$(#o7-WT?%Bs__BMElE&R$Q4J-6lpxq^EX^Mg)hP^g=|DiHhXj{Xf$7o^37ipd-2n~P6B0m^uz>vsIIpJCZc#!lVgR|HOym~lyXSSL_GoP5`?!VT+Mn`AxThtatl1$(GjM_o?<X`)DA`ajkp}?Zl?*JODV=odeIRYE&oiC<Lj<7iH*LLyW>n>Dcwu2dVAkG!2Gbm>ZhyF$s_xRetSiP!m7?}+uxxVjZZF~pn-Y2KBoor#BSj@KcH+$H)xtg-e&Iy1oh1JvlJtz%s?6;D4haInPhYo9bK>F_^3~a)laBxz`I8omVtPKTsveEf~pb5(tDRp1q95DD&{J{?*?yMQykn@NCHpG9`oBP8lqiU63;{KppVQ7<o*dM+<yBgBp=?5$HKOt<x7ktiGzZ7csfxWyT@_Jid(CE|4w?Xd#eN=yTG6%atlzc8#l!^-+DV4ubndJya?f2>&k#qnF*K5;nXdK{;B3+Z7rvv!9?d+N@cL2Hct?q^`dw9VYVPINs2T$#V-&MBSfX%`+X`|=XunGAQmz6ESoL1wNQh^2Ngsy}dO<6+4bV2D=`nPaKg>fc6_#HGU1&Zr&*}`KN+TdT=c5s}(w}|Jo1DN+SrSsJ}f@|=!&dw2M2uW>u#+LXVcGxtOhuFKpmo4u`u|;<%q8z;DnC=c|e^8Vih<5{htQTOXcY}-4RqS7PxkIeTcq?U<2dIkuP`Ulq6CPjMdoO9s12jK5t3Gyf16{^Qr8RD6_%G9?S@4)G@Wk?3s+yUA!0sbo4$o-9_mQ`eabtiRe4-_k6)#YUik5vg(h~P+F1!Coc0^<P$iVdLo@mx!`OqoCAOB2#+TkD`f&*IBxfC){D8jqHCSpDoJtnggOU)ATzM%7+36@mk9JHAkZOK6YBa<m^W7)_Sj(U~MdAO8draNj>fLGJ!_3ammFyURseMyf})U;Vl*5jzaqW!I8$z_O=M_i4S1gbG4x8Y2$Z4LfC!NoSxUW<A3<AV8A^~fo6!l~|T15!J)TT@Cm;_eUJHeUCdaCokEEbU@5zPwv&e2=~b$K4}dKOSzursofXMXg)WF}U2KouUnohEY&v8n&Uul+jn(!ZwV3RVgSt(T0{ZWjC*`v|$4OsL8&*HgsgL^)-IqhDx~+2BcGMSZ>D~&XC!P`^COn{-tb1%1zzDhZkG$=}w*?b+Kj)p8ue#M%jdy#G)vzf*R0vg3DZCXFd8RdXSP8YO(2Tr<ap+HR{CD43UW;R!KGUyq7G)dGWG8B^!lUSiIBek4i2c6=q^29ZbW3qFg5)?ikcLmR5JR(-q%tt5n;YIzmMI(Pc)6flYfOs|$r0aHwGmnQj+=(fbC*eS6B`<xJ?MI)`d-DX9pS*j*2A>RFPT!<(RAa^a%9avRvR+`fK(Zx?7i>fPcQBf+Snyhdbd513U>396a&LFbjD(LZktK#<V=C>xeRSPXvR&$%!JD#2$Hx6404hHsa`pwlSu7zfjq@QlMQ3Xb!0+vAW~zCCQwH32W5yZ#mp{R}UXKRlFp`URwSY_}Za`U)&8^hf#!zJeT0=(M}-BusKYIN`}P1)HJA*-lkW0ol)o)LN?3U@pfhG&esDTt$1;=-z$<7V5#^xji#r9O`-8&~FBuiX<0p4t@u<H-SxioWFz9e)T5S)$j1xcv#$0br#a3oxVINn*|po$46J_=b(+7qW7=*90aMxj2lGGfpt>k<=Wmk=s6siyF@V$lUa2fYeMsomFm9wS7{zn4d%7Jna;yNP>ElF=R6#r6Fig@GY?z6g#pXi^KhVp@$$a1d64)hMbU!u@HzW@rAhfb{O6H>nl67Hj(LlcXC=?W(yAnNeb78CCT1Ndw3!E{Xa2wVwdSGn_o!6g)p;0wq-bHmF%SJu$!)SX=io+j_2<6sIZ&WXlX(&|2cxu==F$dpP@tc8_>AZr?BVEm=Ce5qXE#;N^Q&j!%K?y6aF_*S1ya<A<SdlkmRgP5_zo;7e1n;J-(jTpa6+%%cc|R2D?WJSJN$7qU-NIAfqR#Sr2=2ifUc+aQ(dMRuw!eK)GGZ3PotwQO{ji@8tZe(vTM@-zd79*1E*o_^zfnJ)6;N#c}b<HZVC!`vRML@r{F=WsgTFaBn0yRxGZEf39Qe$PsHq-1h@Rei0bIC5byhoyixco40&9LqQox{Sgt3meg6vx`pS69j(>)D$Ga?UZ$87EV|<&<fzR-k)wEzDX#(DIP2L`b35aO;O*=X;4ocSk#^(&j;nhWlS3bMP;qtFKxsZ!vAT@Q-L*~gSTxl{4+wC+0<`<KC7*jt1_xyO6QqwT-^~4?5m>L4UT9G5-R6|fq{_w0J-yq!n?6QYP>LY}W*mtMD9Dw=(-uLZM{SdT%pwqIm4_1pAZYJ;OgXrM#VwMZNK;gXT&#v18D4sho@7@i2I&#OD6G;%(r9i1!)dlN`b37h(ogi*G70^xI0Z#@ddybuNgH~3Wf^LfzcwI`K{IaVFNY8Ys_wzIYOYnK#Cl>W!wj9=Zv8NXH<_5m%KT!i3k@Ab4E>&P?7DRIRQUO*weYOu)ltRRd@XZqzMR3QyxI&&cA95Z<*qzwU2LCG$>lw+?!OYbB+_+#OxKpbwO;<-l(S5cQvHybL%4SQ-DOPvzY4&w4O?(Mg#i&x1SKpw;l7N8CBQJdHvpHa+6^zaPz5*40qcJk8kM7|_0!nrN4aqi7#e9u`6^ZFgJTNOGbUrKx^G(OuN5b-OB1^bg%Bld*iR<JAoG-%5eyr-%W5rmk@QpLiz7(sBWBbpsm1Eejxrt6w1u8P<h@l6fd!xTrV^S6BuLK*IT2<rIkD@!S-l)MII|rK1owc}n@LRZiK`nAhYcjEF*C8KGfW>wCdbE0cI#RW+9v^s-@d(>AU|{5S<pO^r4zjMQDit?kSC(9<_?af8nO35d2yenG&&sJZ{x+c$PVR5gZ^l_u<AeMY&3L!CZFR3&3o<XA?j)aV!M9(WA@Xf29t}->!M?u@BQ8{jGX=F_2iw5F3Tr#MNlgF!FRUHg)F!Su9PGf99w%zcw;iZ%m8xtv*MZJrF;V~Nc4D_W?Jv{GPW*YK$Z-917y7i%wcdQzh1p@>E7)VYkS#^Kw0NQme?Cb*TXCF(YDX4V$uvpmmK@Ep&!2>6%58RSSCBAh`ff++R}#v#ShV?4cH^CAD+im|yU|GZHk-udZcMp(^xGrlZuC@8WWM;K8|7Kf9Okg=MvrrcDx(9sk>!)Zm59i04BK~;L>JMGvw5#yOa1HD_{jDiaq7l%W=mC$rrnsH&HCwwRyRJoci2tyPB-p2#CehaOgE<fvKHMz*Nv>&^ej&EBz(V@Vq3PEgkp1dstS`yI2>-}+2%>Yog~V?SKg3N{{-vse~(C*ZzSt#agBs&S@zQ_XGmC78)5UClZ4+)OVTs;laS`rpI;#xUFf`V_rCaS7b@P_P*namAFjUBLKUC7@NngV3jM$A+hZ6M(%XfnS(cf38@kYM_e%+bvMxLqHzVMh+lAt_D)+<Vx^U+slk6G)E{ts<Z%%dXLUR%GfDQXDlyKr;NVe?4kfX=WP#Sfi9JhzK$G<!mz@K{1q6;0^uLjK6c46ZrBSnsL7hdMncE0ew3kyzor?<W9!m^XnSysAT`2YL=4}m}+5C{YUfj}S-2m}IwKp+qZ1OkCTAP@)y0)apv5C{YUfj}S-2m}IwKp+qZg#Y)-uW)RwF}^|Nx8~O*NajeEvYB*!(i&;(y#oEUZIE7Gp8G?h4Kg_M%w#aw;M=QJhxDVYk-O)1RM8c4e4w^hCxzS?*{Z~{L^EFEQ%|msTu1aUIC}pm@AOloiu;=>^7#?QR@(lTn(zca(Nk#ZsA{2EU##z$Z@M_sL$ceKrHk{DREawEn)svEo5`B#5mqIcDttNr01K|hEVal<qv#>wLm@o!D9<`Xo*exYUyjq=CV!}pmSjvfj!zq*X-wbkrZW~eaoXN^t=0zfC}Npb{2b8Rc{RDE*#$quhU`Bz<BsiQf)xGzUTE7n@%YhB9~{3i#_2`lgBzRgZt2naU`A@cSf`>7S`1STR<ruz1D|SFw_AS5&H61UEXp6t?+&xpNClv9-9*_PKY!eJROoibdp~TMuvpDh@kN<Kcf;aTy)h-VsBCz|4JWO#fOEwOw>9SXx*6JG!|5l6_K(f+Y35h5p9;^hR=3@E?28C)G*Gc&m=;Vv)``9_VhkHkmRi{ptsrE!wZu!&4nprOmYGmH!g`k4DsP4(1V1@eE6d>o8*dGS)aIPPi{t6vsWoTV@)*g!nfV^>rf2CVqZ`}}d6mKX#T_iQTv&TG-NA9#^;Ac$J22e8zTH9V1qHu-Y95CBK;d2sANBx0@O8;o({2a=wSg=;x2+H0WJPyp-6|NmKF9iVx`Y5(l)snWcqmjIn&D633x`upQP=OWMS^PT1C8U%(U3P<rZ?Fd4V4n^J9-|*z{rgkuFIP-5P8g>G?ozy9hZ|U7r5i#!KvKc*L>pOd8%0d=3*Q~G7nz2hIlwn8!tijDjrJvbV@Ji#6yODf1Vg$JnYqIlG5^zgW%(Hjmo33z&QWcZscSv+}|!g!f6l#)q$|i(HIR1Yv)woutmdWkojqN5Ct#iJNgPQMM2)98{>Z{qhPD-;m&BrNC+)`t0>tN4jL-td?%X2AWVXT@oIZ0u>2IeeVIK3NWD!g4hMqZ{P*rp7Hom=U^Q=QJkB3_*>AMGc<cw33hg;HXMBNHMRYTh)(3<nfx{X-fo5$eWkkjuuKdZ}n{?R~sN5M9Ls;K~@s)YKz&9?iH~rk-3{e+|esRxKz0L*n4OKLADP2K*ety)0%N>+=`M(Gb^@QOdhPYK8AJ~eWwTye=3(}kdN7p?4pln&^!bOe%aM;HqXifD2s;@j?Ua$*-Cd+8UjWfYullL?9$k$+a6lbCoN)-w>`W>739K%4Fxga-9FannL+A3(oM}qj{PTAG;D2V0|(GEBk4NVuB>s)oBL5f=Jv|3*@T>ZIggj*p7TJl+TJCDVH-I-8w8Od0n{WB_ZN+1?0k1oarH^ji%lMdvYiZM_Z`X7z;Ml{&mGY+h8j)v>%FU!Wmqv6^e@}0Y+qk(#<HBI1TG$gN|Z;GamhEUCz(!Ba8m@TKeb}=IgjLOqmy1qq0CYyRv*u`jIwK551_lXAUJ1Wh!U!%cdx%yK2R5YyHnryg4L<56xZSW7SXo!jqdm6A62{Y_&-M6G7;CtK8#8SmDaNFA~x6%;;GB)amtM!AS$!2LUPss;3PPXT=7)=1E+X=A`i1@-uuu<!l@_>lri8<S!9AN^#(p=ZD1Peut5%~~p$Z%XsGPY30!LwHz+rrGScHtJM*nKBd=xy>0WAMh#FRyhj@C4v+XL`fbn_%>9tQlU74M)0C8eLiQXcRwqs3^5378_QUjwzi_z@b0>70pgbxbiDG<!5{fzFOCPqEwfLDfsE-ed<i)*hlvzhcg?)f@XOtPUm3h`}Dr7Yq@xM$l_(8b{_uuc;)YB!+gwoX)%%cqyY2dX^wlIDa89**Q8l~72*-YgXbSq6(L)?lFuEVVywN|lp$qPf<vkH_x^iRiqv0gmHyl;!`(((M^Dq1<9l<NF5QuGEb=|-O72mCtgOGc6d5Y9I-*YeT6QJA+PaibrimyzcX#&YA4JuSgE}Blh5YRgDn!hxaOk_h!{Wk!^>b!ip{Pch1s?ir`DzS)!BM-@xf<Kn{$7ZQtH#cmPwmXj)kyY2x&7o)HEu0eeDG$eL9vtWKTTx+rGT%ilVc5vnq=i|rq<xl4-I|W@*4c)K(AE!FB5Yf<w!Nv;M0(URlbuoX!S}&BA=`lxwDHh6qsspi)@QVfu|OQm+Pm_bJwDZ@By`~gS99<UuqVyUV}DLhYfPZYS2{Q|7C7b4ce+s9If%FK>?>OnOf5tESS1>IzYDu@7sv(J9NDU`S&u1HgVTrt5Lsk`o4eT(i?}g)~fL#S8?R|*=oG}I+*?IU^TjGoqpBdR*komZ@N`9R%45<j#^7;HEM6~so09B##j6rnuqnPk%<-$Pl{IKxs6xTcXwCg;fGN8s=EqvIG+xbhF9Se`J~sv|5f3s-@d$==c~|l+W1w_CgQTT*^8}6MBNPG_+6I}dnsxBF11$TQ@L%yh&z=y<jFOn>tBIn9;?1t8Rht?P9n{vu?#IY{DOk|O7WPa@n}zP3G(D!i|PAZj5=0IqqIAV(YsD|_vyn$XjUN3mcd$x`=9hw+K3b&kEzkXr@Q&syh^7h|2z-hp5ZyWGb9(eHpsn?7v|tvPF2~i+-$UwcZIB1Sr|x567rnNKw%fD&GT;Qh(48@J1EnT!?)$ht2ZfVzZIIbi!~XSoF=6jdlE75*zfr2>;xP<a(9ZVFCHJ7iT>wwHXiAm%3XcG#$i{rXODM#9BxM>cyqbOq1f@5_2&(-7_y4`w+CWy^Xi{^aiJL8b0En=yd@d~#+Ofc*+k>O`V#urMB)7luhtCqMIu|vjc<q7!;w9_g6xz;IKJPge?jRUhE)edk4Hp>qNC6f!@%Vbtol0UTHG3hvKkWnnIRwW%|w-Ov0ETIts3N!X#}8K;K14^8-G+4yeJ%Tz#q@H+@X9G;)hj#y66mE_+r5YQ|%Xby|MQpo>y~r!<Y}H()Ll#=uIh8x~tC-r5l+WqbD3tFl&=3Qo#W^v&nNTqU`a{tI6%023y=^eO{?n$r5K|{C6|;8e(6Br)hPRGVVD0h}xf836>A3-S-oE1p!%2;h{IophD@?w~JS7;LfkCEWf=D@QU`Mv2}|R6sXa=l)ZfqjN?p~4#m2I2={A$e@Y+7<aMc}x$XyrIcKTp!ver+@HcBtR}iRD>zY?`gu!+~Tm`LWBzzY!**Lo$4e#4>?9cs*h1{Z%9S5iqpvoy{`pJPLn9=0>KE##+=7CbY_HWW4L|K7Lcq9Wps-BJB-p&Fk5%~j*j5(0c-T2F5cP@<HHo4JIlnc$DFU1UK=Rt-1-g(#K`5;R%e=}x0AEM~@*iQKrfMv>KS2pKD7*bnQyI5TW0orZke$&O!AZl|x>1zp4)sOd<50%1k)qvnDb!Bk5Xy;~;OF1YTSo@Sdt$@zr7s2$#mB4+YQ1rShf~7#i5zm`dP~cFrYZj|uVfWNpkw7)bS{a%?46cUF)=*bRsv0;tu@<lBQUlQ*m-GD()<U^KLm+2lE&P4a{MY<U9hla#J=CtMgU8EuIm_4TVbAlQLoN07z<ln>&eq2bFiJC<uR-1jR@(wiuVWj5i7eR3Nva9nym%aVX{iaQMH>5D!<wOo;`zkkn=KH2A&lg_rxnz*Vw(r1TS51sr7Xu#8w@6t&v1Qihu**IE$4rBK-TT-ff33sFn-J0bIzp;5-VotT**mr;zhUPNjnmFlVVepc6LLKX&pD$`)-IbOXR1Z=>gS%=v)h@9+)n;tAoFKAbxAQ;l5Tc%y{xqdA0OH>Aa#4p6LT0ffddxL4EMn{9F#-N+0NKsU01^*AEs&=h{|d`+;r2$oA-LKOA2y_4#yS037;7nY7dgAZ&k*sJP1jNbN|K97!GkhNPF@PS*}Vy~?}X{LTTOO<_!FX#3aKZS4#$9sp)q{#MV>0hms3|D0hp04+fU-})a9fTDl7UiY~H=o;Eh$GY7Q<of$`DysUy=f)Vhu1!DW>Fgx$6Yd9o?d8w(gMCoQUsuj;-Um-IobE5t_JLs7caDkJUhuMIlL{2>g?)9`sk<6_pmCpqG~?4ASbCOd$g|ZAD}_g=jAOeY$v19sRkj;KuCUyEPTmc7#u9Q4=aS%LV4~`r4hgoUt9TZUlAtZU`EGh|7aSezJYDPD1$TsAGgOFnfoL{=SXzB2JPGh=&^y!#m3&K-Y3dzN8{5R_n%@po@2U8OHrqhnz&bgIs|}diV}u5TTj4ni^(l#SE#M>87gs3S3{`#|f*le~pqx{oAtK%gDGs&os82LNsp|0^dk)ltn!C-*7b~?;y_N4JJ6Z#!!|C@*K32o1OSoJ1SQXGw$_jn1L+IvHJhxO(0WMC|aoT}p@FVg=->_K;7+$hDe9of?bYJ|rbGxAc=6_}ITqny1x-<NcdNmhpKRWat2+amjuaBON2Qr~H=f^iDwlsL@Ggfh2B^jCm?XSj1C%|gpGjiFPSZE%YP0i+s2D^-4iIl2PP;`EOn%?l=kL(7nBbU8>;C{qttAzJ^a3*^%Op|I07ISP*2E~lv<IT(l#|3E+>wKM?;GvD8s>kiuX5Zik+TQ624Lckk_b_G@aK^z~J~`n-u9zKbTdDfn6YW*0A2gixMINu(t?DH|Y={$Q)0GcE#Vh<rj(-ls#v3AyVhusKUgPmiVNVF2(dYg$=@NoZ1JvhsJP5_&DI>0ZreWyCq?a^i5RRQXU;8V<Bk&7`3E4mtj?o-R;Jh7;VxAlY-`k@x)a#IXr(+CK4Nj?;^2TDsuMzfshgfvT-$xzwJr)PXBqo2#$KeT^z`Ak&ICOF{mHyEihh)Vva}So{Q0tr3%ZZLSWa?)+Jc@DXbM7()_e2~P(!a^(rHjYY!_W4(--yTNXDT~cbmP(0zDjB{D;{GCsrq8BC*Wn->~FQl5>Z9sz<Wo@B#d*rt52t%jJ(3yy^W7j@W0vyx8-xG#Ql8;1OkCTAP@)y0)apv5C{YUfj}S-2m}IwKp+qZ1OkCTAP@)y0)apv5C{YUfj}S-2m}IwKp+qZ1OkCTAP@)y0)apv5C{YUfj}S-2m}IwKp+qZ1OkEZzk+_P>-gfmr|?{w{?>zaP59YGlU8Y^10xL*J8fE?L9b3KL&Ue|KweDWwH>4n4?gOewtRmMH->6s9<l4eH<|B2>eV_B!nN@=ms1PorrB@N-+2ldulGOZ(s=~mJiHIw8Bzv1@uQmk_Z2~t?1T5dr3aAfs1&v6A_rcf1>tRl_n}g4(#9a-2H0-j_ek&-0X|V(w(C3ifokL4?blCFL*7w|a}mp@f%}H9V((RQDCFiHZ6=ezth(8+iY#}ricZGFkVzi-KGj1Xmpmqy*6pI^kj0%R_@bpP?xP=#(=m}jc|6Lp>n;b`Bdn76kwG4%ffbFhAy+qb@x&{Ita@5QjNP+d6O&|up9hZDEOMHm1%s|+U8gzPu#Nv_K5c>2`^V`TL(R}zYM+6F`5V+Lm)0lOH%0fc?7l{4W6W|{|Ea|P8U>sFqujk^fFGqS2PFe^kt?r=bC07szA$w@x{Pw@FX_JCMRglL#|BpM_TR!4X+~1K)pcAX^L#$yCW-1we$ghrk~nwa{66KX%V_9-R^%&)Vp=S9jnSv8==L3kCHo&>sbnL6;8#WbZ?@(b!<rJBnax^o>Zzdm<LuixHfreQVUxP+lLr1gEo`)!_8fCgee98@G(^?tV#!In*C;%9jXl`S2(d-)Rm2Npbf@?0XR<d&+eW`|!!{#S{&U4VPsj*QzPtXq`2K5Td-%7|mEu42y&=_hV_g@sCKuf<#yrIXR4LDgv*oaxa}U+!F%Dc1{(J16%5A7;9<o_oR0Qffa@JJeRY8oqq4`^-8eD#)Gjcg!9aQAlwy(X_0(-sH@WkThVAPrGHg?4jj;cHFH<UJkdIQg)3$HEUEv@Hu_c9wWzEwbOeZ>LXrAB{oI5@+uQNi5e9#^Qn5G?b;*b{bzr)&*}`hdy)YfLG&eh{`lP5Zu40FWgV$E`Ac0NI&Pg%f%~5PR%_7F$9voY=ohFJUPJu2%m@n|mJy9~cYEn`k1SU9?o*Ts;zoPd6M6EsX*(UM9L^n`k)3LCOA{IR>&il*tpyV_-etva7aaEJ#F_3Qgz6g5cz+ACGt(@J<QF=jO$M{zsCP#lv_|FFBM`sGa~%eSXiRJx_$P?c}|!my_V^>yti)(@Bv2Y^K>rHyKze&d)zvNCvN4zsf*01wODU!LMg2u=~~%-Jdck;PB+2S({V}OfI;(8{A2O9?|)TFSk;F&0x5g?@|hsPcOXw$&~_(_jgH*d`X7!eZ%fY-X+6E;M$S|eKJh@_1Wx?N`lKm7g<bCCPCX@m-p8?6X68)ZQgxm|I*~}x$HBEkR9XJButSAA8B-q-cBXJ3zo$1`}-3h^h?eRcSr(I7VpsKy_Wz*vJ*Mq_!3}GZL9yIe;Hb^UMa$!0KCQ#&zZO4VaFREc2~Lt&|1u267o#|Yr3DqDf)>JyFPfL#~}%1GsG`8_9R1Sz_(S`yQ#oT^J45?OB$SgyW?Tfe;M$>f$7Npb6H@%(yKgVo(($JC>8n@av(-N=AxizE?ltRC-C-Y9%O0kI?q6s54ukV$k+PwVY=xq`^o46(CK6q#QTMiYUay&WT_B#Dc(Ny)TIb2=B%~6X^WwBN?Gl$e=+PHRLVMayaeoJ&A)7uN+5!#w;|N16rP)34PhrMgVL_$5TUnaK>aRZ({8m44)bJ5AG0b4-Zy05)^}FG0L5fum_-GQ-E-Mr{jCB%%K6w|cvK0}Z2Sw&{gu%3^oK7qAapy3ddc}C6wqk?e%6KXp^R~vWgg*l=c-K3FN6xdRfade5mvu<XHIP*FzlBQW?x5ejR`xbvxJ~!hIe=T8^341I@F;LLDE5TaV`|$n(+<EleZCGfBvxl%zPyjPP-k=b*+Rnxqyv4wn`|{{*>)gT>;un%$4+}6_6DdEOzjC1(-@*{z>0Z4%<ZmuXvx7L#Nq0qY0{VV2_s$+o&jmt<<{12W-orc}JMc_d8|4emr<Ek*y3k{){Jv&Xj_m@B5&g;iYi?n#3&;u~Lw$3yXg^Rsx3a)sfP#1Rl?bF5ia|cv0;0i8Nab_WWDZV)4Zg`6}YJnrbl=_XS-r+bjZQN!zhI5k=5(*CyBcauL|<FxjN|w_f@i*W{tD5G-;mL+&LP0xykAa;8rqJY{RRK=r&3Ag3U@o4F9AXAbB6$}52Wo|t~_#|7}>#|?=g+5#vDZT!4anGYQT3IU5&`S9PLRO6bP`CuB|q|3;b4`m^fM_)4LgF)f4?%0WZm{(l#JtdS692+O?@3H5@_W9>=x(D(>R+-XS{@?kf7;8|=@;MJ;PA#3T%FhEa8i_r}ZSo-Fk+hw>N*=^%d}R45lm|-E))o9qxgbCtY${)o3&M|gn`V0Eg3kL1mgon$z!|M+UBi|OdFoX~Uh_H7yd`%=qdEsBUhEc^^UMJW8bKGWdpV%suy0OkR}Mr-P5G^6WdrNF%Gs+1+3+~o&+D96Ht4Fcnou0dhT?opK`+{DcrKn6GA5J_-QELu;Z-)s{ym%#QJf70w?2R6qRj!B*=HZoJ_o2R|1>du%>mXueg3M~a={M<7;e|*!dH^~eAKx-P;k7(wjPiN2VQDiKK?5YE-hMcaVX@&&LDk<49tg6S6cWkQ2=t)ODiub3qVWLWa;3)eI_}|UYj;m2(?}g!lRBw;7U^5W5Hhxyk==TJ!*@A|H_7Zgm4MeeL9}YY*qrMJ7j|c^GYDaKB@ETVhIRrrQ2QLD1`_apO5NSO2KVi@|d4yDHKg*XL7if!bG7f^V#xJ&`~}=nD?g?{+mwPbzQ6syn5V4O!Uj3r-_4d-R<8#m1t33&MAYi(|uXFGiAVeVWvHStsJyQx$e6ul!GKK*+;*ya-hG<d?5RCIeZJ#Hkde30o@+qkp=(u2UFo<+PmTkIA3qtIQpdmPP9`NTsl+<Eytg3i^*2Pt*xb1dH+hNGSXRIX{dzgzG=Ir<w{V>$i2dT1YuCB;%ys00=0ODTag$-tl&xAnKK9?F9LR4r$%5FIk%(YTP2u;ets+5SqbN!Ir}JQR>D_9WlwpVN{CrHjoddYp^`HF`&GtDm~5->7h0?UFVpLRZ0!{g{r2x)@2m>gJSlyq)UyJ9y()DnP_KYEmXorSrz(JFuR?zMzw44M(^sCV#d1(Ptnk=jp&Zt%svh6|S3k;q;2ymCZyu_BTNJ0ufloes+^@DACVl7ZuZEOE*8$<Y(SP&AQl;8+`*#@>T8LeD^(g~#mz%Y-TxF21#UwCYQ3}Uz$Jjpmw?1m&0xA4tr65h7XTLk71P(53Da)NL0m|2J<LpSqAfP<SsqwBDcG>QSK(XR~*J+_M8e2usHvUe$uCNHK9>zEyF#J~=EZFy*D1xDEtCXRZLJ+8o11F_IICDTIb8Dag_Liy~{;E^}?)>DZOZxIbUq4v0S~VXy<x8{TCh|aB*6P%Xavs>49Y6E1CKogfT-TzG<$_x2nIl$qIiR!0xWIEb8~n7R2W=&@;i<udLWOM>GznZ5|DBczB17!4dGi@?J;uUw2Y&{{piJorjda+eWxV7PmIex)0|mH{3bE-)8+?|jV8swGp30aC6W@)l&Q_;@#Km*=e>79T^W;kK=U>T?MH;*J#y1(TIN6wCI|<$dI+~o4O9ECMWzAPniJ)C~ZshG`0<6%*(?xP7z}@lPk2MtHq5SXTh6(35Sbpj>!!{KQpZF;$xOHRU$I(q`x7HZg{UppL`BV(-Vz<o+4~&MN)=ys2Z$`m~4=)8den-MF<{0zN(~;0~{rLR_z6faK|H8Zy5C$Hu5<fFCLco%At1QtY2&^2&tiIg{gmN#vr4}-OsHvHJwI%BV&^)TH7VZWD8e{$C4;|s?+srnZ7AweXFE;c){R(WvN>fH#rC`WK*Lh4?6WI!u<C*rG;PuBFmPzjKFuhaGYLeXv9i<O()V+2`kMHx3{v-3nbS_4z2QmRjrgf)}sxpXpJ|}@dAP@)y0)apv5C{YUfj}S-2m}IwKp+qZ1OkCTAP@)y0)apv5C{YUfj}S-2>%C#*oL)PpY?~MI|F+i4+a2*?Jq5hpa6(*a8jJz3;?RhT_=R@2f{v&t7iTRfza_d((S_U05~iqs5&GQ07Bam0oLyR5ZdNdaV5nM<eNHHDGGg|e%QnQHl;6E%Z>TbD*M0}nVxw?@P^GvSB9)!PiSO*@+<F@2V9~`-94D@1|PKkryV>0szH1I04_pq$e5B0p#dd|qV$A>tHhP5bW0+)47pNNl1_ut+^N$bohHqr({N6)8>9>=mB`RQ5t$N_x_tUCzU#Zz`}NCS`}t+>wfA1n^Q=AC`%&JXt|lkuDP{XpOQQV~Z*PBc_$Bw{>q0*|Ip(?7;;%k*q3_Wj_TM<ve$KhS%+-TjTVvdekDeusH}Tp1Jtyg{T+S?ww-&VJ_R)`berqvp?qKlhM>cHP@#Qi;drq^9!mw7|wQlTC&D$*&5}s^BwQO!=zZY{19p_|S?918`)RZH<{8^{&(hO6DKvrriCSy1unCVs=v2&;oW{O4P2bzpS*w|1hcCs~uo%uKK<{#;y>}7RBnBT)NmXz(XZt55=)1K3GQf^ZO3%=gBKEolBZF4yqux=-hO;|WhIyr~O>c?xXQc2>m@Wgo<@pe4MKPkOoc19$t8LAf_j*VcPYYwK36C#*k;ht5G%DL>Te7WFF0GBxrE>T+-#bwK8|F~UqYb3tpoM>@2m!%{sKW!ciXBz{1jhE+!v3Bp^Y`;H3S)GsZ?WrUC4>-_0YqwJXvq<b*5O~;+?Qf?ZuUp=1`gqL=Mw2~RlB@rNLzZr=y=u_$bgLs<qPAj>Z^lt}cS*^AnX9+5q`i$-kBl;<np=+|_Iuk>r|OcKd%rr6?vus~b51(a!EZkjo7<hJwEMyo&bhO6^_RD~Q~EhlERN5rw00pezWbsx2hY)Ij>(I`x^u*Bd%kH7=R9RLKA3S>^CEqFtFu7;lo#y`ecLm?-<PJPepfwb>`(I+Ua=}IxlD`O{&Xrh7C-~?yQ`i(3!t8mx_dWL11U?cecgVuAPTzM<$O*fn8wc=A7Z#Igu-_2@4BN9O7A}HQqx=%MxN*Qo8^jy)8?TC>eu_i$#>L~;j=wly4^FbxK9v4ZDC#wH$O&_tfkAm-4~+B*tWJcT{VW(`-g@G@5PdxV>gxLUm<(Xw9OMgULk$e;=csiaio(QV5eslM~W*YUd2hrQT;NdL0k7L)Va5C!-d>f3XaR5f3sr9Sx5hweMvN_XD-}w`c)L=XE!u9&W<AKOf$W1Gaead&&@a66G?h^wY>({MbM4a0S9{Bxipvi#o+YQaJn0IBKpnqFzS6_YA7}~jGmo3RK?#ON~bew$IOclq1m|wGU>BI=ysg_hvxS|WNp3BUeYL#{;dg=t!TSUJ=((yRKzdS%sfkNtxx_mTpg08x^*PJ)#YLQ8-My>P}{U(u|NGvvTRH8yhJ7ru9ITc`cRO1>%BDz95V3iQhT<@gI->`_3owBS&A6AG*@(E<o?YMP2qIg(z=M%bKTD$BFU}ZGsMCyN%O0Mv$ymv@*N*}Wx{+_`fJS*tMy!MHmj~XMzMc0yLG2+n$wH*tbR%Um<fH$C~=*B<6y=vQl9>(C4AE%GEGR~nAO=+kWo|b>#9>UE^L+2kLga7e)xX<HE&nSwNc*Em*GJYUTeqNjpk5v!Qy?1&%7v0)Mc+W&X@j)r@trF`O#5}(hZNL0;soEerfOSAX19Zli3j-LK8mr<w^5HDe!=7KiP&6XU(GO;??2w?0Uz?0ZA^IHMnJQs<|}!o$|{G^%0~z$6{l&3Xd`ceEz)PD7x2kUPD1Kh6=iM`rp2frKV#Jg=V+nXpZrdZfUQ0s+RHcwaMbs4CPH#I~)_}=kl-djcSRs+iw21X{(axlc?){%GqSP?~?YVqAZ0hAFIlH7hEG1$K6{C|4gL=Duqi{)~AwZAZsa8Pa}!0rQ)hP)5t8RolD2kDB$<8<3|N)q<e9mQI%FYJ(Wq4@o7${f`?B}2#;jYyD?Wb4t8eHe>GQaiw|Z}hLfbE<>V|fn^O5<{L3sl;;Q(}S(Hsn7BSN`p*i$qaAx1+&|GR<p#9;*(L9=~_1X55N<Jw_=3bcdE}uFI=J@;wDIn!lYCD!1TqgxJkJ2Nf1oY0*u=c!2Kr=m7M6N$qNGC3CmsDR?M9p#6lQbU}k*?uTx}RGyX$jVxm@HF5(?kEL?a3=4hjjxD0zDyl_)Y(|vRX)ArGv`Smx-t#ZrA!ldqm_OuXkA6Ohgl3tRD5&c4R)=Zrz`*BI4GFEfu(nNUrnW`f1K0s?7F#TCr0^!SPWi8y1Ub&|ZA>zS$x=tAAhShMb55BDLFNWJT0)So4cul!#8Zw|wKg5Yj9$eTB7kLUQYrDlva5q^eyXI1%rKRQPAF!l^bPZQW?qJF!YgK|%N2PUH&7^hDO#4?H2&@jA<t9E7CfdT6e?x{&<eypVYHvV^h%UoRMNE}{9KEgr6aQ%rx~zO&26w3t-x3~$t|Dk7^ZB%Zdah^h|;*1ygyBq!NhR*R+=Qi`ZWEb4-Qyi7iyjvOx__YU=*2HthrToCrOPVYLsniE#uI#56c@@2ZAMFn(B+F+l}nF3mrW4`~BMgdvqt!=E4D4>TIgj+VZ<dcN?lLftD`E-QSF-F@cpL+Xif>TE2(|h^;M8n)XD)AfiF0{y_Z8gy=y=C*rb)c)cp)8l|YOFqS>~cwMd*{pmxm@a#+uS9doI};Gr<hM4&ZhQ9Q&({}Ws_z{?Z@b(EZWE^dc!ZzB)deh`P|YBVs7(07Ivmnzz^%k4O7x7MABla+wWAeFr9pHZ0j|O`EZ}(l9NJfU)k==v`8l7RSjM>%aZ7wl$H3=iHTG<Ys^&b-+T&t|5Mrdc|1kxn>4RJ7)Qpt6r~JQV`-U`OIAi~G@ab4<nVKM6fLRL+b-4<N$MJ3C;l1}LA8koI$|D&QL0M1TGXo$(krq#G=l=^aKD`I@#UB3xkGC1==~hJ;^pte?{Xtw!^>j>AD^aoTRJUU9@&t2%&(R^+N-G`;h!swj(gdY2|jW~Zb#UmrqDp~@h6$-5AVMOd`DImXrLBdagJ^9s1(=NcW1@YyyC?@9xPWa@1$RlCp#WbzB{D6S;2<$E!*mSS;*v-le!K3S!L|d9^UN$*20T8n;#a;T3i~(iMXL`lH~8qbWS*H6{?1x_`zlS+s-_^y&#gE5Vx+@m*BCj(MNTxGkI)T#4uT}iDJFFYB^<9QEX#U-t^m+(JUcq%bAyb(JZd*a#o^w4D;%qyfDi@h81QC>mHwqVcNdsG6JO-mbO{PIP-2aQ}59^Y_KVs{f<B1bjc@*-53ZjyZVO5x{CS^rt)|!Hep8H8CM=l)Ea$kD&sK^_qTgaiAS+Thihb<Wuusq`C7|s^P|`<#g(bcM*5dF&iWpQnn-3E-uzxiA(91qnm<&jjbJx9|NLI@If7lxp8x8>j!0HHDTAZ3DU#*dE9E_&8p-}yx4gqqJ(3mdGfQ<&jAVv|pZ?pkg2$dX@Y}|B@|evSiIPXpqnP%k&TRo%(d?{=!i-Y07}i{_a_-lxSa#~8D82e!ER&E>tsk!v$38fH2}wD0l|{zS5Nwu-XP+ldakWc|XQOV6<x0x)*=z;5c7GE-({dVR+Q8?t{k{zY(lZm-lRsHsoqGb?>G#uaX?p_W$+~BVCMB}GjLBZ53lrJTT1QRMnnb4TJ4r)AKasuBv()o5PGluZqsWsIneFVY%%d-XojUIEeY$@FOWAgK_l0!{>;$jL-2W4wnY^g1e0G7)eoPGvjPs0VG1FEkeYh0Ilzwih46Th}B`!Ozb*S*zHMM|miX-o!_Qe-L&(~nKV*AzCs;@4w`ISOP^ZzcgD;lfo-nTk26%U*Hs>4>S@32}zyugB<+sU8pNEzw7EKl<9X?fDji0YNvMt;=Y`#Np6T?joF_j%f^&83`fv4WYyJlZ>4Y$qNROG`Jd@Dv{@lFdu)za`fsP`bTTeNlB1rQQ=tz3fS$7q`qzj_agRk-m(bFE@h<XWL4be#s)Mu<{)otz5cjGFY-OGLLp0XtKF_CZ9H)^SPL7R6ui=Yu;^Jd!0@OMw-j777&HGTJM`%NXj0iGyQuC$=ECQaVfuuG*kn}UfWwt3$zaCuNo?**<Es~u|Xx2KYyC)r=>zlc5Lqb(JiDGD{5bwTp7vpiY?O|wMxlUarUU6H%rORRw_$Jw~YF?865~IE2FFGD#Lx-Z;;wWr8kC6H|VTJo!q{)H>qdHW#D4mO$u`q<#-R>r1tOaWgUAMwVD>V@Uj?H-$~s!FwAI3IFQ+{R8C7@@WYmwmeX_Au;0L`oId^c%X(T+Ihk0P#Q3C@)3Huh<C`Vrr0(`ps;{h^id(PF-%wRfr;ewRu&JEPzWus?_Cq-}u2WgKZE^*DiL2K9qA@a7z2I2wub`G6+ai8@S5VHRF-E+c3fgV6%S*Aof*v0j=o@}rLErBjwL9~vf~1zaezpHxK_}!>wDsOr(7?d=3$9NpD0FMF(KuE?ca4_kpNOrX@O|$~#7<SvrQWrrhQ<}7D_yx(YIy~1SN4^<EK@;Uf8`|~96z$&Q_+3P{_>GLYBP|~UQS13pUH2$T~3@n*(Y|%<#hX-*k^%rIi+RS3p;ensdo%tQ}Y`mW!nm!vVDxazPMbHPq|5pIWKi%$KRx>M<34F<~?%Wd2eP5ewR@|!ltgu<7MR8WqZf1sgy3S$#_3qqm;gDi02IliAcfyv%2+LA${m*_;6oGNKrb=)%|@-$RcVN-{L_rbyUoh-a4<CCZ73M$>>xO?O0K1U0YO0Z#(`P3Yk|(k_DWxT)u$HmCKGf%@mM|`Gf?8i0dRA{r6N)<&ir1?Lc31VFA6;wO=}GO97pEP*r`uHJ=PUWjFL#<kQ6|K1ckT@<?w}h(efV9u4&mc|KT?OA^kj)2)?q=#;S6`{&+lS{eRRt35P}mO1KJ32tPP?B5#v?8h0DVI#4XTc1u%lbc;1XQok>c8!Hna4K={NCqtTxkh{aiv3-3Q)t#7+o#UgN}(OQ#n=9vl0u4t{IL_?C)1L4lL-ofWQsTR?LK`une@NcxT@JD(@~r2yncgZ+Lm|5&S+jT)n8b@$z3v;0<!}~>&;B2|Lgw&00000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000008*^>3>j50|YGq000O8001EX+da!p|NsC0|NsC04geGYbZByAb7*gDWOFWVaCrd$5C8xO000000058y00000005ip7wQ`j$;eQ~P_3SlTAW;@Zl$1ZlV+i=qoAIaUsO_*m=~X4l#&V(cT3DEP6dh=XCxM+0{I$7CI&j1ItsN4WCJe2SGtWerx)6e(u0$paP?3E5BCB4Cl}SG-9EX+Uc!3Iva4r~+b4GQu$4ESvEMxBb)3WGOZJBk?kg8;e{6rvVDY|4pZE3#jt|cB?)qkb-nK;e)O1D%#*^M29+%l1eE-BU=S*gIu(eCrvVIr4LocVr4TbA04y6trU%r0&Zr}TGbIQGs%nn|4rEh%OIUEMs|D*KKrXQ`GU%P$&ar=q>RWW}*9<V=Mn0Bmr(sg^^{eKo__&l;t(Ty_g+4;;~C7&}*-1C+FUYF}~|8Kmrch|U*^2qduy&%(6pLxlQ4*NI`L|J;WIH*Oo7un2UbvV6e%fE^#tPU(o^%G|`GCNr9Um}#Y;<x>p%RPRp{{OJ&sWW>aJNcLW%T|{;oNYhsXO|SD);9jM?>#dqWOv$6d-lvHH#)a{wjb#58Kp-y4FFI}0|YGq000O8001EXw}eyd|NsC0|NsC08UPdkbZByAb7*gDWM5-%WoBk+V`*h>baO6laCrd$5CDK400000004#q00000008aM>rc~X6u|Kg6%ibuQ8SCW0xeZ8Q(Km)bUjRIVhn1B7!;+YND)T{U5ilLf?z8XRMJ{5)zHZlkqK2nP%+AJkVrel8w^E<D{j^)R1J@0496V%7xtzJ{hc?Tob&FR9OH(~>xElCpykli>{O{@hn&5N!RDo{V23l<X<71Yxg=AZB~O+9caN9vmQI`%=@OZAVqMIMTo%rVToTU6WBlLozPO&&&0Rx~a&lR%Dz$Iq6I7ac!4=g6`=Xh9{_Z0<sA&1jYlI;4!f1`t_buveBRjW0BUtI1{*gVN;FQXTM@nuH%wG~Uw0HJfmv>OQr_+y*y)temIKSD+dR|8`!tZ6#%m)OeD|2Mbrvw)yco`Qt2u@qr&Y@cgn%<2zTGa$&{20!j6oPj@kx4Rl5sak8^;nV!e#opn)gUIg!b3O0oY;Fyykj;BT;ECQ-WKaL$)R=gC?}bn?`v~@ugn1DWx1(G-vL2f&V|dj?!n$)bJ~)u-Jt31TBOV9hOk+&!bg2JDD{Z`$@9D(lIW^$2g*BP^fj;Gd2|oBx`Wa^JyOCKb-k|yzS{a%yPypguN`yz`DrH<P5nVQBC>&qrcv&1vBJlpe)awLu0zPato-sjXTUL>f9dqqTA2HJ0{6=`7O({LW%B}0LPpf1F9Z%V4Bx)=TW+xl?n{-0`U(;C-;^%3#(5)G(xK#0PA=zbNT+a@8l!^N{WCS~I6Pk39C&zk8O+!uiRq6!0L21xqL-rxTC=BFez|FYj+#s}Gxji)2jnGwxBMgoidW6tQ?(CV-9hP|CSLiO7AY=7%U9KW`}J>7J15p--KZQ>eZE&W7AxRggYAr&l?R(t+0VusazR#aQw`%l+s*;|Yv1-V+?<(m_E`8qY)ovrIb_hn?eP~Ob>2px4=uf;2{VA_mH`KScpLTKlrHtqvZtjPmSoCFrsbT`^XuMx=ocFD;+qQbL_!rkc<e0tiyAmSQO(FJxEf$?X~M_jthMvJPa*h-mUpT2D2o+5&~uwnG58~^=6ox<x`Wa^eYi!|I_sbox77!A9r7<kV@S-){<Qs=k{((>pKHMT+_)!4wFY$h=(t%=G#Iwv3act28OMVprqllA;H>*Y#fdlpR)zL1EusRh`CbdqRheM0akahoO&J7x@tH!I7+l>+>E8YeP)h>@EdT%j2mk;8Apo{g-0}ba|NsC0|NjsG6aZyzbaHuLVsCYBWOFWVaCrd$5CEV600000003wJ00000005ip7wQ`j$;eQ~P_3SlTAW;@Zl$1ZlV+i=qoAIaUsO_*m=~X4l#&V(cT3DEP6dh=XCxM+0{I$7ItoUbItsN4WCO0G%RNfX9AE5{k`g`sy4<jrJ2H`N>z}vwp;K69$r~N92LMn@0|YGq000O8001EXnud0j|NsC0|NsC02>=uTcWGpFXfAGWc>w?r0Ehqp0000007?J=000000GsR=>KhQr$WX>mt)7xvoLr=CrJ!z;W}&X5pq`drR8o|f7oT60k_r-cOUx-w1&SAEBo?Fs`5Kx!3bhIp0WQW%fkGy;Q|tjyO9KQg000080000X0H%b*1poj4|NsC0{|x{X0CRP4aBp&SUrsJ=aCrd$5CDh(00000002t>00000005ip7wQ`j$;eQ~P_3SlTAW;@Zl$1ZlV+i=qoAIaUsO_*m=~X4l#&V(cT3DEP6dh=XCxM+0{I%6ItsN46alWkXUpH6cq{Dy08mQ-0xbhA000080000X0E~L2kSvS<005W<01yBG00000000000Du7i0001EZ)Ij?X=7<+Zgg`lZg6=}O928c11$gm00;m803iU|J<Cpz0RR912mk;M00000000000001h0d9-{0CZ?_Wpik6Y-DpTZg6=}O928c11$gm00;m803iUkgj4K>0{{Sk9smFu00000000000001h0X~iZ0CZ?_Wpik6Y-C?!Z)Ij?X=7<+Zgg`lZg6=}O928c11$gm00;m803iUjQrz)q0001>0000G00000000000001h0T7Y^0A+4;a(Q23Z*^{Db1rUhc~DCM0xbhA000080000X0Gft&l}Z2r0Ehqp00{s900000000000Du9yk^lgAX=HS0E^csnP)h*<Edwn8000O8001EXri8=<O8@`>hyVZp4FCWD0000000000fB`>~0047!aBy#ObYD&`Zg6=}O9ci10000600sbZ0RRBWlK=n!00'

_R8_EMPTY = 'P)h>@EdT%j2mk;8AplW!_A&qe|NsC0|NjsG6aZsyWoBk+V`*h>baO6laCrd$5C8y}1^@s6002=R00000008aW<yTN!)aYT1W1$|quv-BeTc=$Z7#u}xz-|vmY!p!xu@JFaQ4s^d0C55$DM*J%D1z99U9b0FxZ@r7e)&CL=9p`*z1LcM%`w*4U$*;5%Mlh<#SH#4Sk=PTZkf%}7Tp`R=sB}%3zLQ|W-eO#@6zcDr!HD*YxjTJ2Th;9%<fP7vf0xY+x^K~b~9<%xm~+<txOuOY50GqoB;p;000000000000000!2h!9-A*)HmbFj)PydWFYrRj8)lY3S*2PneTY9Y={%NoJZ(Lfm-e#{_wtSnkJj_E4<3h*n`n*SD-ZtN3{@z`UvqtT_{B*Y-yZ^RT*w0-$Z{W6x%j)e?!<jW5d-dF@ma7LH^L)Bp`>gDiofqY%)qlM6>w0aQ&Knd^vhPAy^{Di_vW?|dop-8WqV?=8`l63z=6Yurb<CMEqtS)U>RqwawzXNC)N*k~^J!0<^+ixfkLF)CYM-*}-$hl~s0ROpbSw6JgF3peZMNsYdQGVobFjFvqgtL1I5^RNjXI7#owIezDowF?|EBBe6>7Y2QOT{I%QdA>*sc3M_G(c2_^&lvmZ@Wv5^GzOTB6>?Ol*cMp05Va?f3QEJ5vn?m2Ww*$uu=Mb8lC?+i3M}?Nr%&LRs~;y}881ezX|8vkE_EJwv?vIz^OiF<*>*on7)DEfs?m!-Ez%E*I}ACl@bzyh^;C%Z1l2=_roTxn@y@8zg1@kxGjmIY~<DmMei9oW;X^#c#ico5jO$^0S*sE;4WD(_4-^w@UTp2W;PW+9n2L49ZrHa+4PUsXHdE-XYW0Z>@Vcb*K0b*_7{9YqvNW{+j7I!ChRBzMj0_caNBxSDKeMV6XIvsBUm1$5Z@s99AE4-Y@g|#`UN;@}M|IxGp@`^00V|QAR?$qhgqNW6=1k$0UB=#XWDbj?28p<%{2Ga#9XH^S7<P;*{8}I8*;n+G+WinI7}0kCzw~8VpRna#pM(B7$eOJ}>PoK6~aJ@RmRq&lcNC`H1~^tHgGTF3Qd`YyEG;Uy=n4ipSgSyDSbLuVq#raz#3()fm+B#}x_mKbDby#aALHRyuXU=Bmv8x2egh>V9JOE~1CUT|Y6N+V5qP#s2cM$J1@~>s*ug=GFeK5_(NctH#WEGyS?a|JpHP{I~0JIH<{^?438{@?6uTkIiq&NAuuo|6IQ*uZ+fg9os8FOx<^XJbybtY#NvEv~)nAES}NTFE=_+a-D`*P9Ank96I+N<Z$<vjCHWS)Vo)Zw2L!2()n_bScJ^_>eleKEM5_L;Ga#m#i_xC<Rg){W%l76?@Gqs7Ng9eowD}amg*}a{pVJ`Eglo|KBx5vl7X2G{xU0fOYTm7|6iwN0kW)F)W);RZ^)3mRKp6juSwSmF4i5V{8{ggX@k#=xhy?gFLY|~>YSV&9a8S3`w5wr`|!)~`5xl<qhfp8|CUMc{pxR?J#|sDZ*ups!x1(3zAw6|+gVMT7jm>pnJap*armqE>Hb=4eTs#}#Q-&qD~OF=s_JDnZB+NaL$zLo>yb~!hv~hcCdGcZhU@Ub$E)7(iqu)3{wbF36|HT9x<>bOxu<JJ#`<m@5~oMwYJ^XEdS9~^HTl)pH(tF=M{WNyIYF;|Thj7CQG#}Kd$GgHK2fheY1^}IPNH5)8T(*J`y`E;5O!eGf+Q{1ZDq+x?n&CJkAwe{V@djT!zVXek0jl@-7Bg6(j<LdZSBC5U6a&o*|w99pCzhe#X04gZBEqdQ`&{6S4`AR{wJ32*pQ%I2KjB9_9R~Wb*>e$wsX8%*)JXu_AX8zy@?JuGd5O#Z0c;#-aSUQ?CEIw+9yiq@9o`puUCZbS(@DWjq_ctYtc8i#E3hZY<Z}p%kN-4T(R_~%RPhi@MmN5lVfh^+44P3?HzDca}QK+>)Pdlp5Eo?(mC|FI@YgI|I)r~>Uis1qsOh*i2a$ga`T%V7LOL+2Um4HCm%bRt|=?N;;wg(#`s<r<6ah%dejb*9iN&<-R>G9FZ-MI`ks47GK@`P3pa$zEyE(4kw#JCsw<u2&c;aPhUfZQwu+OKe!D*O7#T04>uedY(J?`W^jkmN+dEMjnY}h{5St`fvl}-u%1f4@zI|Oszey3F_wB2=yi1iHPvY*_7N*IN56xCMC8bO3l<AS<{4&J#SEsTww>}W-?tc#~HZD^PXWoAHs?tLlX!7sLGa(NprfoOZHZ!xtEppIbZGUEo-{7-$^*3cp`|U#pPcEJ#FUEFgvu8<;+-&=?WV6H^xzjUjU}oi9aVlrq>_^93Sy<!Qmv5tT<<|5P<DKntW#HoP@nQewO37u(+k)5R%8R7A4v~(zQZH;s4OyEjCE7mUBmd>fh|aEuHqXwLp%2ToGP2B-ppS#Dx%AAHwB~&dw{DOt=cZP3Tl_6YzSf%`Zh0|B42xwhuRbwH{wjOGaO{t4nS69;Se!?;bS%4dQm^KJVx#Mad-`Na%Ub>FbT!G6A#o4qMxS~plP5o{99{FFln8x3vhK!AsZ{uOpwaUO61-)?hz7$RNQpId*1b8EA<G6@{@DF3U3T@Y@uA$Xbg}zjwxH6UG<h@SYGIoPsbV-F;rX6=sdDn=hmMn{rbtz*9*vqDO_oJlo$vO1m?S+41|95FGfB3VyVYpx$VBOXV&7Qf4GHpS=D^?1SL4Me{9cM%<bBEMQtSP?)^TE;-8tQN+dZlKy_?IR=h4zA`sS+3{iDR7{#4JVM<c|fa9l;7?|0=<E9=rv#)Zk>c02FyIv*<68($rj@mGk<o-{GQBJsA^Y<cyj=b=C`FP1sKqwNi8mtkXV+R<Nb%<4U)SQTFx7_dhw{qm7Y9`AyizxI-b1u1j)KRzz@dp}he=;tZkWs6LfkJ&8VOaHn(rKq#c8*S7~rnqR#vcMfbA9!kDndM21{f}$AiZ5Jd&-K#$ny)hsR`k)>4iiSaGryvD>%0t_YT&0|zSbPS*z1})oLKU_+Q6IIGT^2`<?KMsJ-*=F&BeEMUBbN%6V?Q4U7Nl=?gfOXv&FWdyWWKA@>!GLY_A`t$#LQ1&knq+)#}@pDrFO{y+Th{s^<`)>xwS*eC`sd)lHlF4)%=F@rN4CA9FlfJ^MDEc<N+~7A1WMPC9%~lZ)*&DB2yXy)4^A?s1CKR-Lw)?_79aV?2u<uWc2t-*0~^Jmnd$V>eCh^W7jps~mkg^TEsn4Lf0%av?fF^ADL+duN=eJKA-#9lJGAGgHDMmlh=IlK;9s-qklr*B^CD7_&b~o#gCi&lgGRyS7@*ZatFKHt)@~ep`}t=-H*Ue<UYsv6{XUgX*QI!<{xE1~XDLrA8Yo=Tj-_+O|}B$)_p$wRx;(hnA`8zx3Z0t7oO^qpOdnw>*`qHybwz8k(J|)BYRT=}nC^y)wbdW7)_wJ#00|>g$Fy{qFFvV3vQH#(S+RH?bg1>t!x)X<98^XIIH>|4!1?(09?<H?z}qXq3m-z-{U3^w+NE<1eJ^=l&mj$3&)U*PT9>mp@9^Q$Jq~{{1;!2X6d5DAO=Q`&izyGpd)Nu^a5h7PZb$kEPpxCUwcsNmUJ_y$57y)d9X^+FN93wS}q0t*kQiY-a4M5>qnt>N%$gC;vq6VI{gpOwQ2AvsrHpCuOM5>beWIPWW^Ebj&8BF&Vn;e5pm@L;vJ!`&hp0lcAF%jdnKdoS_p^R}cTyGDG{i__uFgCqq*bZ|=QX`cLk9x_0^Z=^D9w!LpzS>3X@0?aNZ3>3XN{(eG|v>DqGB;7g;ormK5ELEq8y(>2&>bzy}O>AE`kT%(n(({+gP%hGL1rR$4Rp_X4BrD@8c`|q5B(lmdhxkok6G_@Ol>`t+TY5H-U+wXe)|Gei?BhN3cnWnJ=D^I)dJXI?MeYtTpFjfD4-|nm7wp6ug*0hq_q*V3TKX=9+<5caDHZ92E+n@dXJ)rx?$P}G5HX)$UBSrsecdq-_Kl^>=L;nFm?NYSYp}f%6zml~<#h&%b#w2UToCx<_p2^y3!>>Z;8Oho=t@3&YlVq*fET>VoPf1#$K}y<~+evz2$;$>Gos-nBp!L4vBa(E${ok((swU}do5&Bf(-Sq~oBxmN`x5o}$DW7w{*$OTM=Th2xk{oA9eyAxBj(RJoV2k=vp?sgS$yKS<jx6tWX-fsK85jmIk<9jkG=7_Z1(Pd&-9DeRg**79eZ$Jm#2NXUds5sK5;&0*xeyct4=g4ybv6#5huT{ODP_!tAZLPpB#8khmKoxzU`J6ZPnG^DmErsZ{3;F(79r?UaB1U@zm%jZT8Rm(rzA+TK8=6$a$F&>agBEs(G^r4Ju_EIDb*Nwtr+8JL}qAy=QoPv+3tB{rBkVO})*+)Tl$Pn+@#mXpz1-*6UiR{{Q~|0RR91000000000000000000000RQ_ON6k0>FK(~&S=A@*?~$J3QL(qZV}z&p_it|$&~Kj@L>T+n9NQ;`RSMReE7&K~Ze~uZV!2<ulWO<tcYnW_Uo3vaxBUS*)<X(Qcpi}InUlJh|2QB9ohSMSj5#RgUh9vWtU4%`-Mzc6ntM=ObE7XU=x|W_q&tPB#UBu}|3+EO9CkoFCVh_BeR03IHcrcY`DLFNKWfowVXb}QIO}MsGxa^?qxqerE=~7}>+i|sO11J3*Yg)FHk$5{kJk?Q-f8YG{$FkP*Q&Q$ye&3n1lQXo4Qt-^h_1F%=6#r*Qo7@Iv8-14<h$M5#QVd%L04+IisjhRM-Cm|BF3E#dhKrIBHnSepZe-1>GRj)<3~n1i{-1S%Wvm7iKERox1v=W<i(TxDUI(sie+I!QsVtJVo>`+1G{D`#j#nF%=LHd#c}1QxUlJq#d~;<z9p>f#9;n_Ly51)h(UVM1Dipe)$(-QunLc-s^x0W_e;($R7ca1!HM7g(-fa2Z|ox0>b%Vb*?E1PH06pxt6AMQt7T}5@ne^}YOtMU^{Kf#^hJo<*LSA7wU5>1s=Fud(S}D`I6auLSB=Z0j;c3)pL&N#vDSSLsDJ+Z=liN1QvX?@cFR%^>#<IQZmryPR2w#Ze}B^TW12g?rTK(&$N!8ceP(Vyp~pJUJLI(Lq}m6MdR$}fDK$8I?`KJ?(`px#J7L&}Ga8@e(dT@BFI{|R)0oHsXSG>1HMbdeP6G?~w=O&TyxJXiu=xF-w;H-NzP@+eg+KEys<3RMkJ{gDRR8zRi@JEy_vKrBFR8~%^Mrtfm(`+baGUbQujqtBXNOnccSSoo$e9Lhd^JA%TCXX0ebr$2x++ztUsdz)9?oljUe(iAeNu0E`srA+kpAo2`0Jf_Uc29h`K$T9@<+c;xTZ~B8Fd&?a7`y0b$M}d;dSl0<j(S^Z?EgXS0!5fUU)-o-jDG*{pL?Bf4H#z;+tyU#{YMVXE$|~MbP#O69Y8Rv~}T=@Bkf=V^K4>b)bIOlK#bDQ=ks?&)@SRF;IsL8o1BC;w`P^Fr@GPZnxB8Ol;x7p?~s!v-9ek+|sHJ)wW-`AE+@&PNC0D19eS!^fI5`0cz2zdF7+OZm9W>qK2_^uIu6-ZkaEg{MGtra*tmluIgRWlSACXFYCM;AIdrxc&q>8b8AbVKCOKmLkgOCA68@MHHXeL-LBrx4|nw2&`#zZFnDyo%1*JoTVY~O^J8N8baTP01!ra1l78>n+g=ufq<&lG9rBYB2g-f3$-N=9mfBo=*dRzs=E#amr-Q|_dCrMP7eeLDq=WGWXToH0>E4}ecZW-)%{S}9Ig!$2ll$~e&7;LEGpWJ7@EF+_7NzseW2Ntf&lAJ_;^f7o2g|1ZzAt;1x+MLzCtmux2Bmt{NRWQ(_I9h`k{}}sQ+l~QOOP4QE;_C3mMGSTwyv(~kSI^vPq7<(I#Fgvm0lB|i85z&G22p+iDG6y@~wSPqMY)uxEg#cQ6700|7QJfqBK8c@viKj`&YPEtCaCNK~fh~eXz+RLA;uWZ#{3CAoi#J)aA{1*>SV&*J@qjrIlq&lgMHB<$KoK8h6XYNr_LbelIz9Px6m6nrhZ2MuxsEmDc`1lz2Dx$vIvWA#Pqjtg813mjUzU<b<yZ6Qehq^SYi1l?>l<Jw1bi#mAuK67yq0@^f=`uODY`%Iq3>C&Ksp%Z-o_n|#MBGOxxh`8)i)eD7;qShVPb95ejbY{DmZF<7^xdwkPb@}g(-j}!LWv|&Vt;1;uw>R-b~KPWfqtp2>SWk<-L`ae7{zs35Ce!6C?U(2C+H`K0SUiPVyL7H6m%99gYf_1*r({W?AgsS)Ez$aZdglSe_x2cv(!nNVys^`v*j#RsH!Fygcj@AXAO5ghZBu4Yh!#u4#V|9%~qtcPx?(35#7L~W##p{Ur!M#2<Pta1v9hR8nCTN!HkIcQ>6LrylA2P>xO43I0cWX>fO41f{TK{e{H(8^vyLayVIax=QDVx84Wr{AJS~`DPQHqXgGQ!<*Myh_T=eq2`om92aPPeYqO4BA4#;)sOlcsBItp?9InWjTalqhh_NYhvA!?qiiPS>3aOMS0toUXmHUhg*^l&%)le=km-kgg?Xc1eFTBVF&9I1X{LN!O;8rpy>K@lU^EN$(mDO;^)FNt3R%OV=-^uNN$>l&(f4ukGIRG))&puD$1eElr!PE54$~>NFiY`b^(zUDMQ}-tAEvUZ?8fi8)1Y52fm!E3PKL2c+uDf=x3%Jx$TspWLVK-k73s4K4S$)=5#98{a2Jo=w*H{XgDTX_u^~8Hc)zy^^FUcRrQ~Xq}|30`@H~dn8d$1(n_0y?mnH+Of+m#XdpHjIFR|TV}lW|J%Cp+Me;c#$!_Ax3>4S!P*8PO|oNk-kWDnJ8r(GZ=3J?tGG#wt}9>mYDQd?X8k?&QO1HuHLhn>a$1=P9qsHe@axgL+T~iJ+p*4JdgFB4@SNyStsc9v&BaL}>hsDk?|YW&jk2eWUj4YGcgGEXTBUY?76uh1WOTW%fxF7ApFP%3W9QEDn>qW6uIlZxJoC*3ec`it=kDTOYP4u}hxg@=tB1YArJzxJH8|J$WtDZ#>izm$erU@<;ytHI()r#l;@^A6%^4n^;+VZ;?xKyy#eVpDyUWI2;t|_^bI02k#BRropJuhMNZ_k8L(GG&is|``a{rCECKEOr)>{*MLq5z|(eX~7KxwjgyQ#;iAUSrrkL&G^Ds}tEec4z&L|T+{vrB0ZDlx8Rj&<tXk=$Z~Ym_MwCJX9*SyC}GOlB|oXGNPccf~HI``#Hd!sWu+YL>^#M@VF0i$;}BMo7Lx%Z{^}M~Y|mvNaVCMoPU;z8z;3i;{z7bi@ejC^7u6h2A?CB`Ffn?B3HTSzmwCPV4H?a(BYKI-k2l%i_7ij7N`-mNMU+J`J;pmToaFH}dC3%lPx{lUB@(mPQMo9WMS)w8R8@UNq_*Eju@T`(3tiv<#2we}B-gD9Qa=eYj0>l*GTR@}RU=lx*p}YG~k!C|QtKeNL_6QL@HmlV=m7D5>U?;_&T7q*Plx{%T}!q<n}um~F5>QoaYYs@BLpQfjVue_m#Eq}YX)N!i~vQcT}veo3kxDUJ)DTz*|LQi^%6ZM)z{gxL8U>*w@6LTume?e(HWq|C3k&h2>3KmGe!UZ34HQhpbftMbJ*QeL!sop<ri^E)ka?+}z0DU*-v^IO^^N_MVy{kQbmC~0O;ZBx6#D4FHg{p8?j(Ne5mFT2eJ(K31HiMdDC#YmIlZ|XNRy(eXMpQz>W?w+g|oi|}%RICI|-{>Fa9Vf2lqmMTAxGzr~+L~<cA1`Kc=T1BiiI?<4$DVf`lpr6L@0!syF+tW2Gs_=7Gf{5*)3BXQu_Sr$#W~#Zc#@o(eaq!@|73Z(>*#f}XUTH&{w$M~?kSSlbY<a${;BfuV9SJeUs9#0#KO!;erfW1P)Jh9vUIW7-X{K)S%#PxH*7Tidxk`wvCCZ&_CSW%x9rq^N2VC0c3d;h`k`DJ@L}BVmRT~?G4}Jx&skz!$!3CiWVVc*ot?gGe~!2$R-c<<mMixrH0cz7Bv-1xo#DTz^dniZqH0m$!bh_I`mW+@k{?NlhwZ06>hf616sz58=)T8dd#l8jp`RX0?b}-hSdGq;RZn&g3qPAD=1<1iefXXy4WG4;@xz`-pEBi7E#LD*n!Oy+Z%M`zc~SR3>kl>a<$|NZn6no7;*#-WQg7#cS@L+m{>lFN(rikT=S4aBQuo8vI^F&%kh&#9PNz35kZJQKY^&J2K&<ktx{VuMAY}$$Y1MH`frK2jXxrYlKnyyWl%HW!AQwWnEF3kdK-%qH*C2XSf!xZgWVNzqfvhj*V0EZbfdtu2Y}weLKz>)+Hr_TZUmQ&0zHjx)msRmKtK43a|EGQe_r;p!ONE-fwru_IL_!h=-(7Y7iQKpE@XmbV6FKCP?e1RUiCk)4Y}$6OJgKy4MA49edGhaj`-2XdkEO}4hjXgVdn_BZv-9VoM{;QHgdYza9*OnXqM7YJ<%;X<iZ$vi%oX`qqUMIM961<zXh7vN+2Xv(_TAidSyIL$I3!`zLoxka>eYoBnPOG^Q+V%`3@P#EiAj66bg58no@qd@H0d&LW0!%2DPkX0>A?uMWN|C-oAcHrN!q#Q^_~@(ARk{w_53|GUW^l3*xbpFm2OwQ8if>#ky}?X>ffsxDPO~em0MN!t}H01yFaaNsFbOE&9+DT+u~BO#>L9bZb+|}m4gRV_Lac7(Y2Fbc}ev<p`A>vJjJ`ir8QNY#;D`>XS<$l*slZsIr*k_&RI2$>sIxggRfpKKhz=D@rF+DwX|)v;kMeodwh9#w@}Tm=yzvMi@WMA8;q8hi`1$$JYB!%MC+H6|I~;&6RQ;hH#baf6R$b-?6zdCO;8WxT?5X9CF<be8?GA|B<rQoLtL-6Nl~M{!_V!tNY$~uxA?cWO;eY`AI6Te()HQYVd)>NGPK$Kjy*4Rf1pkm4IlWF&D5(uJ58+@k*Oof^-E~9^r0TFsv|#@%hKvG?|rB2%u?6RAuB_RWvhRcO9_vrW^1)+4{9Wz&(@@Vh2!2n&eqMRx*W?dlcTE^+!!&SZjP?sIKtqbQI3vtyqeLgbdJ8VkDhxZCtC~mTV-XP&ej`;U!C}SVzwHUi(Nk9XO;#f9l2BS&p7cO2S@a-lBJmiCabPHJ=Cg(Uq+UElBuf-mfWh-IaAHT{I+MWexR?OYMB4^40S8&Q8p((U27d^T7F5{bnO(}&fTVInp(Gy+iKP=RlPE2UD`i1MKiV?F&zF+vX*SHv5nu<B;9#y{pxcw6SdFMkt3(fPtd8khkxcRjn^g9yX~I-IZjI)Ydr5*g;;e7TX&#y>liH>P=Ds`aZzerqp<I=oe^qcSb2Y|hj;ao@B1b`UBk3inFaQB&W7rT&MyZnHwn?)8AkV;Y*TGw)u>DS>s$J9Umv$VTLN^#`Ka>WC*9D@ZxuIgopnu5o;hKC{k)%69UWQHs?k+Vng3wfo`@^@w%vj?DHkqlW|hRP*5@y&!N$y&!xvvv;|g_mpE~EG)@H{GuUY%(_mp3ix4Zjj;FZOnPMTlT__lYx_j0(TxfLu-{G2Xpy?Vua6)d}=hpktf8s+Z$rw*na%1OP-|9t=e00000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000030Py7#1O9KQg000080000X00Qxuz5oCJ|NsC0{|*2Y0CZ?_Wpik6Y-DpTZg6=401yBG2mk;800014000000001+>=)`A5Xs0;#!#)El3JWxq;934Zj)xAuA`uymS0p-l$aNvUzCyx5_e0?DNY577iT0EqyqUGMkWS2nmP)#3S<K=V=%aEKT40%qx2{}N{`Z`H2r7*P)h>@EdT%j2mk;8ApmD*hsyu||NsC0|Nj~Q6aaK+a%FR9Z){{=V{c_<W@%$-Wo~qHE^csn0RRvHfF1w<00000YybcN00000?Z`0-gFqAj(Y5syzxxrk5V4SuI$gxZ!b%ohNfBb$l{Dfha$(ICQigfm%%^T|O_!LrS#|Vr2znO1OqE*EX%0(p&wdW0|Ng6cM1TBvcQ1eRsVH+RWoBiQAEgEW00000000000Pq)fP)h>@EdT%j2mk;8AprVbKn(x?|NsC0|NjsG6aZyzbaHuLVsCYBWOFWVaCrd$5CEV600000003D400000005ip7wQ`j$;eQ~P_3SlTAW;@Zl$1ZlV+i=qoAIaUsO_*m=~X4l#&V(cT3DEP6dh=XCxM+0{I$7ItoUbItsN4WCJd)>x&C)xvty)s>|9_GP%PZoeuy|O9KQg000080000X0Gft&mH+?$|NsC0{|NvT0C#C*bZ9PaaCrd$5CDh(00000002q=00000005ip7wQ`j$;eQ~P_3SlTAW;@Zl$1ZlV+i=qoAIaUsO_*m=~X4l#&V(cT3DEP6dh=XCxM+0{I%6ItsN46ag;AOMyZrvs3H=P)h>@EdT%j2mk;8Apm5i_}Tye|NsC0|NjjD6aaH|aBy#ObYD&`Zg6=401yC(000000000-000000001+>=)`A5Xs0;#!#)El3JWxq;934Zj)xAuA`uymS0p-l$aNvUzCyx5_e0?DNY577iT0EqyqUGnmP)#3KRh@1~3qD002-+0Rk-pEdT%j2mk;8AplW!_AyZ(0001(1^^HM0000000000004ji00000V{c_<W@%$-Wo~qHE^csnP)h*<Edwn8000O8001EX0`Zx>SpWb400;m84gdfE0000000000fB};q004Aoa%FR9Z){|9E^csnP)h*<Edwn8000O8001EXXJ?1XYybcNfF1w<8UO$Q0000000000fB`HZ004Aoa%FR9Z){{=V{c_<W@%$-Wo~qHE^csnP)h*<Edwn8000O8001EX`d&Z`SpWb4pa1{>5C8xG0000000000fC1zn003ofbaHuLVsCYBWOFWVaCuNm0Rk-pEdT%j2mk;8Apn|&c9lv1004*p000R90000000000004jie<1(>cWGpFXfAGWc~DCM0xbhA000080000X0A!~4*+&2X0Ehqp01W^D00000000000Du7mA^-q$b#QQRa&%u#E^csnP)h{{000001^@;CasdDUiXs32000'


class _ReferenceGrid:
    """Only the frozen networks' *reference-dependent* terms are cached.

    Wrench, grip, geometry gates, persistence and both EMA states stay live.
    Finite episode-local tables are rebuilt on reset. Non-grid calls use the
    original scalar calculation, including at floating-point phase boundaries.
    """

    def __init__(self, mass, com, normal, monitor):
        self.mass, self.com = float(mass), tuple(com)
        n = 6000
        self.r1 = np.empty((n, 2))
        self.r5 = np.empty((n, 2))
        self.scale = np.ones((n, 2))
        q = np.empty((n, 6))
        qd = np.empty((n, 6))
        qdd = np.empty((n, 6))
        previous_qdd = np.empty((n, 6))
        for k in range(n):
            t = k * C.DT
            q[k], qd[k], qdd[k] = T.reference(t)
            previous_qdd[k] = T.reference(max(t - 0.001, 0.0))[2]
        sine, cosine = np.sin(q), np.cos(q)
        normalized_mass = float(mass) / 8.0
        normalized_com = np.asarray(com, float) / 0.05
        features = np.concatenate((
            np.ones((n, 1)), sine, cosine, qd / 0.8, qdd / 5.0,
            qd * np.abs(qd), np.full((n, 1), normalized_mass),
            np.broadcast_to(normalized_com, (n, 3)),
            normalized_mass * qd / 0.8, normalized_mass * qdd / 5.0,
            normalized_mass * sine,
            (normalized_com[None, :, None] * sine[:, None, :]).reshape(n, -1),
            (qdd - previous_qdd) / 0.001 / 50.0,
        ), axis=1)
        times = np.arange(n, dtype=float) * C.DT
        phases = phase_index(times)
        self.supported = monitor.phase_fit_counts[phases] > 0
        normal_rows = np.clip((features - normal.mean) / normal.scale, -8.0, 8.0)
        monitor_rows = np.clip(
            (features - monitor.r1_mean) / monitor.r1_scale, -8.0, 8.0)
        for k in range(n):
            self.r1[k] = normal_rows[k] @ normal.weights
            self.r5[k] = monitor_rows[k] @ monitor.r1_weights + monitor.location[phases[k]]
        phase_onehot = np.zeros((n, 5))
        phase_onehot[np.arange(n), phases] = 1.0
        for grip in (0, 1):
            grip_column = np.full((n, 1), float(grip))
            augmented = np.concatenate(
                (features, phase_onehot, grip_column, float(mass) * grip_column), axis=1)
            normalized = np.clip(
                (augmented - monitor.feature_mean) / monitor.feature_scale, -8.0, 8.0)
            for k in np.flatnonzero(self.supported):
                value = np.r_[1.0, normalized[k]] @ monitor.logscale_weights
                self.scale[k, grip] = float(np.clip(np.exp(value), 0.25, 1000.0))
        for values in (self.r1, self.r5, self.scale, self.supported):
            values.setflags(write=False)

    def index(self, t, mass, com):
        k = int(round(t / C.DT))
        if (0 <= k < len(self.r1) and t == k * C.DT
                and mass == self.mass and tuple(com) == self.com):
            return k
        return None


def _r8_model(empty):
    encoded = _R8_EMPTY if empty else _R8_POSITIVE
    with np.load(_r8_io.BytesIO(_r8_b64.b85decode(encoded)), allow_pickle=False) as z:
        return {key: z[key] for key in z.files}


class Detector(_R8BaseDetector):
    """R7 unchanged OR fixed L2d for the exact servo mode only."""

    def reset(self):
        super().reset()
        mode = getattr(getattr(self, 'ctrl', None), 'MODE', None)
        self._r8_member = (Membership(self.box_mass, self.box_com,
                                    _r8_model(self.box_mass == 0.))
                           if mode == 'servo' else None)
        monitor = self._r5_monitor
        if (mode == 'servo' and monitor is not None and monitor.spec["family"] == "SCALE_ACTIVE"
                and monitor.spec["scale"] == "state_logscale"):
            grid = _ReferenceGrid(self.box_mass, self.box_com, self.normal_model, monitor)
            self.normal_model._reference_grid = monitor._reference_grid = grid
        self.r8_extra = False
        self.r8_base_detect = False

    def update(self, t, th, thd, tau_m, tau_s, grip):
        residual, base = super().update(t, th, thd, tau_m, tau_s, grip)
        self.r8_base_detect = bool(base)
        extra = False
        if self._r8_member is not None:
            extra = self._r8_member.update(
                t, th, tau_s, bool(grip),
                max(0., self._shared.last_wrench[2]), bool(base))
        self.r8_extra = bool(extra)
        return residual, bool(base or extra)


# --- Fixed, authored R31 torque inference (loaded only in torque mode). ---
from functools import lru_cache as _submission_cache
from types import SimpleNamespace as _SubmissionNamespace
_submission_r10 = _SubmissionNamespace(Detector=Detector)

@_submission_cache(maxsize=1)
def _submission_torque_class():
    _normal_table_r21_yaw_impulse_normal_bins = '{"schema":"r21-yaw-normal-bins.v1","threshold_Nm":[[1.767264480407721,1.767264480407721,2.0122173094615703,2.0122173094615703,2.0122173094615703,1.8337280125357311,1.8337280125357311,1.9650984381823116,1.9650984381823116,2.180249307541482,2.180249307541482,2.376108709214208,2.376108709214208,2.376108709214208,2.2286915663709554,3.23779270273884,3.23779270273884,3.23779270273884,3.23779270273884,3.23779270273884,3.23779270273884,3.23779270273884,3.23779270273884,3.23779270273884,3.23779270273884,3.23779270273884,3.23779270273884,3.23779270273884,3.23779270273884,3.23779270273884,3.23779270273884,3.23779270273884,3.23779270273884,3.23779270273884,3.23779270273884,3.23779270273884,3.23779270273884,3.23779270273884,3.0582029924995466,3.418925992202723,3.445490195885792,3.445490195885792,3.445490195885792,2.9679568956599796,2.9679568956599796,2.913147298587709,2.598390331527533,2.598390331527533,2.598390331527533,2.3389354199738146,2.2805979471894635,2.273702179376777,2.273702179376777,2.2031752492973826,2.2031752492973826,2.2212297489313615,2.2212297489313615,2.2212297489313615,2.1487072994973055,2.102242239736751],[5.111942835466052,5.111942835466052,5.111942835466052,5.111942835466052,5.111942835466052,5.111942835466052,5.111942835466052,5.111942835466052,5.111942835466052,5.111942835466052,5.111942835466052,5.111942835466052,5.111942835466052,3.5891728888065115,3.5891728888065115,3.5891728888065115,3.5101791804684277,3.5101791804684277,3.5101791804684277,3.433441479975124,3.661354430583063,3.661354430583063,3.661354430583063,3.482650448260615,3.482650448260615,3.6398983258349302,3.6984427431846796,3.9143706165250864,4.47447195434656,4.47447195434656,4.47447195434656,4.3754105543787905,4.109095707528455,4.130954633209454,4.130954633209454,4.551026390244765,4.551026390244765,5.9464286945673726,5.9464286945673726,5.9464286945673726,5.111942835466052,5.111942835466052,5.111942835466052,5.111942835466052,5.111942835466052,5.111942835466052,5.111942835466052,5.111942835466052,5.111942835466052,5.111942835466052,5.111942835466052,5.111942835466052,5.111942835466052,5.111942835466052,5.111942835466052,5.111942835466052,5.111942835466052,5.111942835466052,5.111942835466052,5.111942835466052]]}'
    _normal_table_r22_release_model = '{"schema":"r21-release-momentum-bounds.v1","threshold_Nm":[[1.7672644804077617,1.7672644804077617,2.0122173094616107,2.0122173094616107,2.0122173094616107,1.8337280125357716,1.8337280125357716,1.9650984381823522,1.9650984381823522,2.180249307541523,2.180249307541523,2.3761087092142485,2.3761087092142485,2.3761087092142485,2.2286915663709963,3.488937509676121,3.488937509676121,3.488937509676121,3.488937509676121,3.488937509676121,3.488937509676121,3.488937509676121,3.488937509676121,3.488937509676121,3.488937509676121,3.488937509676121,3.488937509676121,3.488937509676121,3.488937509676121,3.488937509676121,3.488937509676121,3.488937509676121,3.488937509676121,3.488937509676121,3.488937509676121,3.488937509676121,3.488937509676121,3.488937509676121,3.6966350028231134,3.6966350028231134,3.6966350028231134,3.445490195885768,3.445490195885768,2.967956895660102,2.967956895660102,2.913147298587797,2.5983903315276216,2.5983903315276216,2.5983903315276216,2.338935419973884,2.280597947189482,2.2737021793767886,2.2737021793767886,2.2031752492973724,2.2031752492973724,2.221229748931446,2.221229748931446,2.221229748931446,2.148707299497259,2.102242239736864],[5.111942835466052,5.111942835466052,5.111942835466052,5.111942835466052,5.111942835466052,5.111942835466052,5.111942835466052,5.111942835466052,5.111942835466052,5.111942835466052,5.111942835466052,5.111942835466052,5.111942835466052,3.5891728888065115,3.5891728888065115,3.5891728888065115,3.5101791804684277,3.5101791804684277,3.5101791804684277,3.433441479975124,3.661354430583063,3.661354430583063,3.661354430583063,3.482650448260615,3.482650448260615,3.6398983258349302,3.6984427431846796,3.9143706165250864,4.47447195434656,4.47447195434656,4.47447195434656,4.3754105543787905,4.109095707528455,4.130954633209454,4.130954633209454,4.551026390244765,4.551026390244765,5.9464286945673726,5.9464286945673726,5.9464286945673726,5.111942835466052,5.111942835466052,5.111942835466052,5.111942835466052,5.111942835466052,5.111942835466052,5.111942835466052,5.111942835466052,5.111942835466052,5.111942835466052,5.111942835466052,5.111942835466052,5.111942835466052,5.111942835466052,5.111942835466052,5.111942835466052,5.111942835466052,5.111942835466052,5.111942835466052,5.111942835466052]]}'
    _normal_table_r22_tangent_model = '{"bounds_N":[4.60253623135071,5.391369656814549,5.588662513397839,11.84113032836651]}'
    _normal_table_r24_loaded_normal_response = '{"coefficients_Nm_per_N":[-0.20407491043242484,0.008739510645516974,-0.0095460084529884,0.03727150600946697,0.04008027147389957],"schema":"week09-r24-load-response-model.v1"}'
    _normal_table_r24_loaded_dc_drive = '{"eligible":[true,true,true],"high":[[1.027808230101643,1.0081654880539301,1.1145764529661761],[0.9016413577804369,1.0452341157868708,0.9781167745970802]],"low":[[0.9821843282240097,0.9946683978787147,1.057961795625697],[0.8937332515831427,1.0402701537538652,0.9735006635028695]],"schema":"week09-r24-dc-slope-calibration.v1"}'
    _normal_table_r27_load_change_settled_model = '{"joint_bounds_N":[1.0393334089992114,1.6769614425828194,1.5374594665485537,0.9464752353530933,0.927950930731815,1.1402139272446856,1.2279640953165287,0.8946755674788502],"separate_bounds_N":[[2.660038078073421,1.9695901860233729,3.5953131493698187,5.1680649024099985,3.9549676061220667,1.5958079922177253,2.2830199552086134,3.478176820774621],[1.493695691483243,1.8226215239714845,1.8177429754202703,1.6440937654474457,1.4011463883934796,2.250620353685652,2.006889012390892,2.2693821454279686]]}'
    _normal_table_r27_load_change_manifold_model = '{"location":[0.22728082533008737,0.40277182273167633,1.4990887170736955,-0.00017976337358034856,1.2317326436600957,0.22696324337009544,0.7945541112251403,0.1480051775811536,-0.23728765312606148,0.0,0.08688563056383214,0.7963517449609436],"nodes":[[0.2896366414867498,0.258197089471793,-0.27650277687030395,0.04733768837615846,0.2928911564625863,0.28984159605184134,0.19393042190969717,0.08794890587373348,-0.13314447592068632,0.0,0.09756097560976958,0.1932742054693285],[0.44179528220578973,0.7035797700323391,-0.4996596972666885,0.02037318233910617,0.18180272108843631,0.441935437184766,-0.4881569207994066,2.4290650193697307,-0.28753541076486394,0.011984224905356571,-2.0168855534708725,-0.4911308203991124],[-0.5591607338147212,-0.5344277757815307,0.5584912338015897,0.017976337358034857,-0.5549999999999998,-0.5590559443089637,-0.489267209474462,-1.4385928175060365,0.8031161473087873,0.011984224905356571,0.18761726078799704,-0.4911308203991124],[0.4415055803813924,0.5006961911606181,-0.45071055210715427,-0.029361351018123597,0.3995578231292529,0.4415808554826291,-0.48889711324944585,0.9737200293163149,0.16005665722379805,0.02396844981071317,-1.0656660412757841,-0.49076127124907604],[-0.4778197040695863,-0.4745104204096299,0.47973156920396365,0.020972393584374,-0.46561224489795855,-0.4770896802251956,0.007031828275350401,-0.048162496073709114,0.056657223796046784,0.0,-0.026266416510310545,0.007390983000739014],[0.4415273080182222,0.44163223140495866,-0.44139986932375025,-0.027563717282320106,0.4415306122448988,0.441645982734042,-0.48630643967431536,-0.5151293058318592,0.5594900849858438,-0.011984224905356585,-0.2720450281425766,-0.4889135254988891],[-0.17419046446444997,-0.16338034495149129,0.16349504519220295,-0.0671116594699968,-0.17581632653061285,-0.17352432502840273,0.4877868245743896,0.35179562349492904,-0.4079320113314371,0.011984224905356531,0.2626641651032097,0.4855875831485587],[-0.5590520956305721,-0.5276006108515989,0.5551698791244689,-0.007789746188481772,-0.549625850340137,-0.5585855808265372,-0.4888971132494441,-0.5507276724950364,0.5481586402266293,-0.023968449810713114,-0.2908067542213624,-0.4900221729490016],[0.4412303636482151,0.5243666906216311,-0.45414080365893467,0.06950850445106811,0.34261904761904843,0.4423768552221201,-0.4874167283493708,1.7631661606114684,-0.12889518413597756,0.035952674716069805,-1.4427767354596377,-0.49076127124907604],[0.441237606193825,0.43273895077254754,-0.4300745943591415,0.0461392658856228,0.42921768707483066,0.4422466007192943,-0.49148778682457284,0.30153910585279636,0.41501416430595606,-0.047936899621426284,-0.5590994371482026,-0.49076127124907604],[0.39723914161349433,0.35615792310456346,-0.3837117499727754,0.10486196792187,0.3895578231292516,0.39854621502123877,-0.14248704663212325,-0.21149617841063925,0.20396600566571854,-0.023968449810713062,-0.0881801125703506,-0.14005912786400493],[0.1127084947817459,0.11788088393819626,-0.11786725470978962,0.05093295584776543,0.09976190476190465,0.11191394519179974,0.4481865284974091,0.32666736467386026,-0.3781869688385283,0.023968449810713114,0.24202626641651234,0.45048041389504784],[-0.3562101207332353,-0.36850970176068976,0.35969998911031204,0.08448878558276383,-0.3527551020408157,-0.3554681563922397,0.2953367875647674,0.1361114019474426,-0.2308781869688434,0.02396844981071317,0.1369606003752528,0.29083518107908374],[-0.5590810658130119,-0.5314184333453107,0.5619214853533702,0.043742420904551486,-0.5899659863945586,-0.5592802715082748,-0.4874167283493708,-0.9129934038320738,0.6133144475920673,0.0719053494321394,0.09005628517824484,-0.49002217294899986],[-0.5336235179941046,-0.5211776859504132,0.5387808994881844,0.062917180753122,-0.5387414965986406,-0.5328168983508332,-0.23686158401184224,-0.3350434509475499,0.29036827195467413,0.02396844981071317,-0.11069418386490747,-0.2339246119733923],[0.44157800583749174,0.5252200862378729,-0.45634596537079397,-0.020972393584374003,0.3724829931972789,0.44171110998545493,-0.4888971132494441,1.3338917390849336,0.016997167138821906,0.02396844981071317,-1.339587242026255,-0.49076127124907604],[0.4416504312935911,0.6006333093783688,-0.4761107481215291,-0.019174759848570515,0.2979251700680278,0.44171834634672297,-0.4881569207994083,2.0437650507800478,-0.20396600566571854,0.011984224905356585,-1.7936210131331807,-0.4911308203991124],[0.4414621251077329,0.4481899029823932,-0.4439044974409237,-0.02157160482964183,0.43418367346938797,0.44192820082349793,-0.4933382679496675,-0.05863260391582232,0.584985835694057,-0.08388957433749604,-0.35459662288929666,-0.4907612712490743],[0.4414983378357825,0.4796757096658281,-0.44760699117935293,-0.013781858641160057,0.4101700680272112,0.44203674624251943,-0.49000740192449954,0.6093602764108587,0.27337110481586535,0.023968449810713142,-0.8105065666040991,-0.4911308203991124],[0.3533203450348729,0.3520706072583544,-0.3551807688119349,-0.02876213977285577,0.3594897959183684,0.3527834663617746,0.026276831976313815,-0.05863260391581747,0.031161473087820482,0.011984224905356585,0.0018761726078768417,0.02697708795269817],[0.4261875964163884,0.42721433704635303,-0.42520145921811997,-0.0167779148674992,0.4241156462585049,0.42641344226469163,-0.30717986676535997,-0.3538896450633527,0.37393767705383263,-0.011984224905356531,-0.17448405253282437,-0.30672579453067167],[-0.025486518001347106,-0.053022817103844794,0.0349695088750952,0.10725881290294131,-0.0331632653061213,-0.02592426424296806,0.514433752775722,0.3748298607475694,-0.4461756373937634,0.023968449810713277,0.2682926829268403,0.5125646711012563],[-0.42408001564389847,-0.3979743083003951,0.4088669280191658,-0.05932191328151503,-0.428333333333333,-0.42346100686730687,0.15877128053293832,0.006282064705265984,-0.10339943342776438,-0.02396844981071317,0.09005628517824484,0.1577974870657797],[0.2099251845038494,0.2128323751347468,-0.21236251769574202,-0.04014715343294452,0.21710884353741505,0.20961205867241714,0.33826794966691326,0.22196628625275247,-0.2719546742209668,0.02396844981071317,0.17636022514071859,0.335550628233555]],"radius_squared":[0.14425634785730612,0.22731709111386877,0.31011237162532096,0.2242038295173418,0.28349908540442437,0.2204848119173981,0.22159052151886446,0.313561630662432,0.22543979512736795,0.22320300922643566,0.14379225773567625,0.12621053282674696,0.2277248062721696,0.5153810740750502,0.2911770035639702,0.23245796080660922,0.23198074134074537,0.20431381283499156,0.23812430741388743,0.14778746856240196,0.14946320714336697,0.22605869279277127,0.2201535307639063,0.14536232279143663],"scale":[1.654697885357298,0.26681678329285874,0.4402045492235578,0.02,0.17616810610874145,1.65611202389613,1.619068784713673,0.28615333017764827,0.423043139159085,0.05,0.3193795937277563,1.6214656296947454],"upper_bounds_N":[[[0.3023835917482868,0.21530304059554797,0.44910782386047104,0.7901607227448738,0.9284450271630675,0.6373606453474279,0.43756684654338496,0.38399623136639466],[1.0791628934532644,1.0077248365756577,0.7137074966261066,0.1947514763107575,0.0,0.0,0.17459598730645784,0.7492078263889702]],[[0.33042828422975046,0.5186750305276013,0.5422386448122253,0.6115654936582384,0.7858533753651217,0.7893979978220044,0.5794296431298667,0.3894230931653049],[1.1276194824671715,1.3700106805541903,0.9152809442821436,0.19383837706696555,0.0,0.0,0.0,0.4207877340218682]],[[0.3999566154104025,0.37638537418590357,0.3988506140980855,0.4787936609532912,0.3456795879905485,0.31579088014226403,0.2329510610225599,0.3503526280813933],[0.2553474252586893,0.10103534933097123,0.11574031119965134,0.06264616848085831,0.21497787550379507,0.4140629093543088,0.5565622067939844,0.5463792290094495]],[[0.5676259358332658,0.9086633257523501,0.72979186011597,0.40707666177040547,0.8544716759381334,1.0428409135448067,0.7533998049763673,0.22101667222675592],[1.1865882474678062,1.7473618656933474,1.4446806268522292,0.45763315997613074,0.1789661757288879,0.2226212553582299,0.35531262962454285,0.4236201528196425]],[[0.4276865989146586,0.2971603340371365,0.473060448049255,0.5122085645667243,0.3079928296963152,0.30969530299821946,0.5650125767970688,0.58227934250332],[0.5418001408176372,0.19204860375369445,0.008213220631020526,0.0,0.06619229941119326,0.5052067662470825,0.7693147546133703,0.7706615671576211]],[[0.7374814612474315,0.5929935878545065,0.8500835064775081,1.0452476683361913,1.1805049272542294,1.4163694501976931,0.9050065931810846,0.8287048997758242],[1.2428497607793303,1.7650855025504784,1.4225706165547718,0.2672605569462429,0.01477465144312518,0.049902735239210856,0.060845382889614844,0.5894455441834977]],[[0.35718573256018005,0.305273962066124,0.5740148541418234,0.7800880498377284,0.6568445063620272,0.389581237338186,0.30257036071732935,0.3869256568309842],[0.8458745226824975,0.7198758059286853,0.4398963994688305,0.0,0.0,0.3062517764387121,0.7098800378380757,0.9169649251319523]],[[2.4821004583419466,0.5628888342574373,1.8822928257674953,2.6091648620583587,1.887835113164883,0.4245995366625849,2.208400663494031,3.232916217724726],[0.675308324023247,0.32883558062478585,0.7771840280753832,0.8722195647261042,0.7894225359264072,0.5478990496850681,1.0695560734115366,1.1051016382047207]],[[0.42127164907667236,0.3676962705916068,0.19673837953106396,0.245990432091958,0.7385163663475537,1.040536182998369,0.8076218929160587,0.4991536551570267],[0.9978490201756752,1.443572134325879,1.1479592427593226,0.36943534115828663,0.0,0.0,0.0,0.5246863280686959]],[[0.6182023682491318,1.0010115523108103,0.8858015408703003,0.4444670834596087,1.1511036036352627,1.283714271253865,1.0224171458691254,0.30046648259430697],[1.4163603165807581,1.8883207911866646,1.4576477574690325,0.38764858305785227,0.0,0.0,0.0,0.5777045650997512]],[[0.46089817330765215,0.29740044138604227,0.467813705097219,0.7424379925211361,0.8633065080443434,0.6081306884043118,0.6113581321531621,0.4716837344470062],[1.1434111155059314,1.1326822430308827,0.738408868537417,0.39196333624566465,0.0,0.0,0.042740089046808274,0.6660643319945868]],[[0.40896645669219167,0.30576130404013124,0.29184000132738325,0.4024942099628783,0.5520073411054789,0.5950785091373814,0.4637584431203202,0.5703529540001138],[1.0751726687028786,0.7392955826519305,0.40055903854975294,0.0,0.0,0.0,0.4089194628578485,0.8797266323038379]],[[0.22709149575719684,0.3175168999583715,0.6002751422355405,0.7800880498377284,0.6536341974830254,0.26643553133621123,0.2117607349542124,0.1801777276120907],[0.658099020597272,0.3382873993828009,0.13496913778661174,0.0,0.00880647104465035,0.4068099631293432,0.7441208893010711,0.8799353760448823]],[[2.4821004583419466,0.5628888342574373,0.5599511902403738,0.7205497219093041,0.4566723530248931,0.31579088014226403,2.208400663494031,3.232916217724726],[0.2702841485296873,0.29640196946084907,0.7771840280753832,0.8722195647261042,0.7894225359264072,0.4140629093543088,0.5565622067939844,0.5463792290094495]],[[2.3044885270196693,0.44023961557359337,1.8822928257674953,2.6091648620583587,1.887835113164883,0.4245995366625849,1.8184830198147057,2.884907865303249],[0.675308324023247,0.32883558062478585,0.32663279990455757,0.36439185874447017,0.24965023446509627,0.5478990496850681,1.0695560734115366,1.1051016382047207]],[[0.5093198759972722,0.5069806691344998,0.3703137582374881,0.2572418787981468,0.857438416589452,1.040536182998369,0.8076218929160587,0.4991536551570267],[1.1865882474678062,1.7473618656933474,1.4446806268522292,0.45763315997613074,0.0,0.0,0.0,0.5246863280686959]],[[0.3672932208802414,0.5186750305276013,0.5422386448122253,0.6115654936582384,0.7858533753651217,1.0542842771025636,0.857687754218891,0.3894230931653049],[1.1276194824671715,1.443572134325879,1.1479592427593226,0.36943534115828663,0.0,0.0,0.0,0.4207877340218682]],[[0.5140762690947307,0.5838844768855967,0.7709256330523347,0.7524598295860128,1.1805049272542294,1.4163694501976931,1.0224171458691254,0.30046648259430697],[1.358570942229482,1.861792022992951,1.4061149386415077,0.2672605569462429,0.01477465144312518,0.049902735239210856,0.060845382889614844,0.5777045650997512]],[[0.6182023682491318,1.0010115523108103,0.8858015408703003,0.4444670834596087,0.8544716759381334,1.0428409135448067,0.7533998049763673,0.18898253191122688],[1.4163603165807581,1.8883207911866646,1.4576477574690325,0.38764858305785227,0.1789661757288879,0.2226212553582299,0.35531262962454285,0.4236201528196425]],[[0.1379614960187712,0.2028492725482051,0.467813705097219,0.7901607227448738,0.9284450271630675,0.617688846516498,0.21470438494020797,0.06707291708064224],[1.1434111155059314,1.0279065259406124,0.738408868537417,0.35543137273569575,0.0,0.0,0.12403028370678934,0.6670607033840211]],[[0.7374814612474315,0.5929935878545065,0.8500835064775081,1.0452476683361913,1.0532649720482583,0.6273034281923214,0.8569548177917488,0.8287048997758242],[1.1042848435201988,1.237188990105083,0.8645672388737179,0.38792098088271476,0.0,0.0,0.0,0.5894455441834977]],[[0.40896645669219167,0.305273962066124,0.4781481935762772,0.7227412038194341,0.6568445063620272,0.5180283536380355,0.4637584431203202,0.5703529540001138],[0.8986614542016718,0.6993647633852688,0.3324750085703004,0.0,0.0,0.32991560713606527,0.6823907033912763,0.9169649251319523]],[[0.2836319694178987,0.3175168999583715,0.6002751422355405,0.6818388044626782,0.5160077689023229,0.30969530299821946,0.4507394899305267,0.48353005679157296],[0.5826266380052232,0.2887697202910082,0.1001886497571917,0.0,0.0,0.43946811998835483,0.7693147546133703,0.7706615671576211]],[[0.35164685999359985,0.21530304059554797,0.4010900560970763,0.6361705053846745,0.6931665413452281,0.5950785091373814,0.41812221728302706,0.5010674988019453],[1.0751726687028786,0.9185068266698087,0.6209859366522572,0.1690577711934223,0.0,0.0,0.28430983692666906,0.8797266323038379]]]}'

    # Source: experiments/r15_delayed_wrench/observer.py
    def _make_r15_delayed_wrench_observer():
        """Causal, delayed virtual vertical force for the course's nominal 6-axis arm.

        Uses only motor encoder, joint-torque sensor, grip and the two public nominal
        models. It is an estimate of *all* vertical external wrench, not a contact
        classifier. The delay is part of the output contract. No force clipping.
        """
        from collections import deque
        from dataclasses import dataclass

        import numpy as np


        N = 31
        # Encoder is a nodal state. Rigid-axis JTS at that state was formed from
        # the previous solved interval. f_table[j] is logged after solving step j.
        STATE_LAG = 2 + N // 2
        LAG = STATE_LAG + 1
        X = np.linspace(-1., 1., N)
        KERNEL = (1. - X * X) ** 2
        KERNEL /= KERNEL.sum()


        @dataclass(frozen=True)
        class Sample:
            time: float
            effective_time: float
            force_n: np.ndarray


        def vertical_displacement(model, q):
            """Virtual displacement that admits vertical force and rejects other wrenches."""
            import mujoco
            model.d.qpos[:] = q
            mujoco.mj_kinematics(model.m, model.d)
            mujoco.mj_comPos(model.m, model.d)
            jp, jw = np.zeros((3, 6)), np.zeros((3, 6))
            mujoco.mj_jacSite(model.m, model.d, jp, jw, model.m.site('tcp').id)
            jac = np.vstack((jp, jw))
            if np.linalg.cond(jac) > 1e4:
                return None
            return np.linalg.solve(jac, [0., 0., 1., 0., 0., 0.])


        class NormalForce:
            """A fresh instance per episode, owning its nominal workspaces."""
            def __init__(self, empty, loaded, *, dt, viscous_spring, jts_hz):
                if not np.isclose(dt, .001, rtol=0, atol=1e-12):
                    raise ValueError('this design requires 1 ms samples')
                self.empty, self.loaded, self.dt = empty, loaded, dt
                self.elastic = np.asarray(empty.elastic, bool)
                self.k = np.asarray(empty.K)[self.elastic]
                d = np.asarray(viscous_spring)[self.elastic]
                if np.any(self.k <= 0) or np.any(d <= 0) or jts_hz <= 0:
                    raise ValueError('strictly positive elastic and sensor parameters required')
                self.spring_memory = np.exp(-dt * self.k / d)
                self.unfilter = 1. / np.expm1(2. * np.pi * jts_hz * dt)
                self.prev_th = self.prev_sensor = self.deflection = None
                self.clock = self.prev_grip = None
                self.pos = deque(maxlen=3)
                self.residual = deque(maxlen=N)
                self.midpos = deque(maxlen=N)
                self.offsets = []
                self.bias = None

            def update(self, t, encoder, jts, grip, *, project=True):
                """Return Sample or None (no full window/calibration, singular, or no load).

                Call on every regular sample. project=False skips only the output
                projection; it neither skips samples nor changes the integration.
                """
                th, raw = np.asarray(encoder, float), np.asarray(jts, float)
                if (not np.isfinite(t) or th.shape != (6,) or raw.shape != (6,)
                        or not np.isfinite(th).all() or not np.isfinite(raw).all()):
                    raise ValueError('finite time and two finite six-axis signals required')
                if self.clock is not None and abs(t - self.clock - self.dt) > 1e-8:
                    raise ValueError('a dropped or reordered sample requires a new observer')
                self.clock = t
                # A change of payload invalidates every differentiation/averaging stencil
                # that straddles it. Bias belongs to the unloaded arm, not the payload.
                if self.prev_grip != bool(grip):
                    self.pos.clear()
                    self.residual.clear()
                    self.midpos.clear()
                    self.prev_th = self.prev_sensor = self.deflection = None
                self.prev_grip = bool(grip)
                if self.prev_th is None:
                    self.prev_th, self.prev_sensor = th.copy(), raw.copy()
                    return None
                aligned = raw + self.unfilter * (raw - self.prev_sensor)
                previous = self.prev_th.copy()
                self.prev_th, self.prev_sensor = th.copy(), raw.copy()
                equilibrium = aligned[self.elastic] / self.k
                self.deflection = (equilibrium if self.deflection is None else
                                   equilibrium + self.spring_memory * (self.deflection - equilibrium))
                previous[self.elastic] -= self.deflection
                self.pos.append((previous, aligned))
                if len(self.pos) < 3:
                    return None
                (qa, _), (q, tau), (qb, _) = self.pos
                v = (qb - qa) / (2. * self.dt)
                acc = ((qb - q) - (q - qa)) / self.dt**2
                model = self.loaded if grip else self.empty
                self.residual.append(model.inverse_dynamics(q, v, acc) - tau)
                self.midpos.append(q)
                if len(self.residual) != N:
                    return None
                state_time = t - STATE_LAG * self.dt
                # MuJoCo's nominal inverse dynamics is affine in acceleration. Integrate
                # the residual before projection; endpoints and slope of the kernel
                # vanish, so isolated quantization differences are not used as force.
                averaged = KERNEL @ np.asarray(self.residual)
                if self.bias is None:
                    if not grip and .075 <= state_time < .425:
                        self.offsets.append(averaged)
                    elif state_time >= .425 and len(self.offsets) >= 300:
                        self.bias = np.median(self.offsets, axis=0)
                        self.offsets.clear()
                if not project or self.bias is None:
                    return None
                import mujoco
                model.d.qpos[:] = self.midpos[N // 2]
                mujoco.mj_kinematics(model.m, model.d);mujoco.mj_comPos(model.m,model.d)
                jp,jw=np.zeros((3,6)),np.zeros((3,6));mujoco.mj_jacSite(model.m,model.d,jp,jw,model.m.site("tcp").id);jac=np.vstack((jp,jw))
                if np.linalg.cond(jac)>1e4:return None
                wrench=np.linalg.solve(jac.T, averaged-self.bias)
                return Sample(float(t),float(t-LAG*self.dt),wrench[:3].copy())
        return _SubmissionNamespace(KERNEL=KERNEL, LAG=LAG, N=N, NormalForce=NormalForce)
    _r15_delayed_wrench_observer = _make_r15_delayed_wrench_observer()

    # Source: experiments/r27_load_change/event.py
    def _make_r27_load_change_event():
        """Adjacent-window vector innovation, separate source ownership and actual Boolean output."""
        from collections import deque
        import math

        import numpy as np


        WIDTH = 40  # Four native persistence intervals of one normal-model 10-tick EMA time scale.
        DIRECTIONS = np.stack((np.cos(np.arange(8) * np.pi / 4), np.sin(np.arange(8) * np.pi / 4)), axis=1)


        class Changes:
            def __init__(self):
                self.clear()

            def clear(self):
                self.past = deque()
                self.present = deque()
                self.a = np.zeros(2)
                self.b = np.zeros(2)

            def feed(self, value):
                if value is None:
                    if self.present:
                        self.clear()
                    return None
                v = np.asarray(value, float)
                if v.shape != (2,) or not np.isfinite(v).all():
                    self.clear()
                    return None
                if len(self.present) == WIDTH:
                    old = self.present.popleft()
                    self.b -= old
                    if len(self.past) == WIDTH:
                        self.a -= self.past.popleft()
                    self.past.append(old)
                    self.a += old
                self.present.append(v.copy())
                self.b += v
                if len(self.past) != WIDTH:
                    return None
                return self.a / WIDTH, self.b / WIDTH


        def sensor(detector, grip):
            n = detector.normal_model
            packet = detector._shared
            if (not grip or n is None or not n.last_active or packet.last_table_allowed
                    or not math.isfinite(packet.load_fraction) or packet.load_fraction < 0.8):
                return None
            return n.last_error


        class Event:
            def __init__(self, bounds):
                self.bounds = np.asarray(bounds, float)
                if self.bounds.shape != (8,) or not np.isfinite(self.bounds).all() or np.min(self.bounds) <= 0:
                    raise ValueError('eight finite positive fitted normal-only bounds needed')
                self.windows = Changes()
                self.clear()

            def clear(self):
                self.owned = False
                self.baseline = self.direction = None
                self.peak = 0.
                self.reported_native = False
                self.native_was_high = False

            def feed(self, value, native):
                means = self.windows.feed(value)
                if means is None:
                    self.clear()
                    self.native_was_high = bool(native)
                    return bool(native)
                before, now = means
                change = now - before
                if not self.owned:
                    scores = (DIRECTIONS @ change) / self.bounds
                    axis = int(np.argmax(scores))
                    # Moving back toward the ordinary model prediction is not proof of a new impact.
                    if scores[axis] > 1 and float(np.dot(now, change)) > 0 and np.dot(now, now) > np.dot(before, before):
                        self.owned = True
                        self.baseline = before.copy()
                        self.direction = change / np.linalg.norm(change)
                        self.peak = float(np.dot(now - self.baseline, self.direction))
                        self.reported_native = bool(native or self.native_was_high)
                else:
                    level = float(np.dot(now - self.baseline, self.direction))
                    self.peak = max(self.peak, level)
                    # Nearer the preceding level than the excursion peak; no fixed alarm or inhibition time.
                    if 2 * level <= self.peak:
                        self.clear()
                    elif native:
                        self.reported_native = True
                self.native_was_high = bool(native)
                return bool(native or (self.owned and not self.reported_native))
        return _SubmissionNamespace(Changes=Changes, DIRECTIONS=DIRECTIONS, WIDTH=WIDTH, sensor=sensor)
    _r27_load_change_event = _make_r27_load_change_event()

    # Source: experiments/r27_load_change/manifold_event.py
    def _make_r27_load_change_manifold_event():
        """Set-valued normal force certificate in observable pose/velocity, not elapsed task time."""
        from collections import deque

        import numpy as np

        DIRECTIONS = _r27_load_change_event.DIRECTIONS; WIDTH = _r27_load_change_event.WIDTH


        class Coordinates:
            def __init__(self):
                self.x=deque(maxlen=WIDTH+1)

            def feed(self,encoder,eligible):
                if not eligible or not np.isfinite(encoder).all():
                    self.x.clear()
                    return None
                self.x.append(np.asarray(encoder,float).copy())
                if len(self.x)!=WIDTH+1:return None
                return np.r_[(self.x[0]+self.x[-1])*.5,(self.x[-1]-self.x[0])/(.001*WIDTH)]


        class FeasibleNormals:
            def __init__(self,model):
                self.center=np.asarray(model['location'],float)
                self.scale=np.asarray(model['scale'],float)
                self.nodes=np.asarray(model['nodes'],float)
                self.limit=np.asarray(model['upper_bounds_N'],float)
                self.radius=np.asarray(model['radius_squared'],float)
                if (self.nodes.ndim!=2 or self.nodes.shape[1]!=12 or self.limit.shape!=(len(self.nodes),2,8)
                        or self.radius.shape!=(len(self.nodes),) or self.scale.shape!=(12,) or self.center.shape!=(12,)
                        or np.any(self.scale<=0) or np.any(self.radius<0)
                        or not all(np.isfinite(a).all() for a in (self.center,self.scale,self.nodes,self.limit,self.radius))):
                    raise ValueError('invalid force/kinematics envelope')

            def unexpected(self,coords,levels,axis):
                if coords is None or levels is None or not np.isfinite(coords).all() or not np.isfinite(levels).all():
                    return False
                distances=np.sum((self.nodes-(coords-self.center)/self.scale)**2,axis=1)
                neighbors=np.argsort(distances,kind='stable')[:2]
                if np.any(distances[neighbors]>self.radius[neighbors]):
                    return False  # Unknown geometry/motion is not positive evidence.
                bound=self.limit[neighbors,:,axis].max(axis=0)
                return bool(np.all(np.asarray(levels) @ DIRECTIONS[axis]>bound))
        return _SubmissionNamespace(Coordinates=Coordinates, DIRECTIONS=DIRECTIONS, FeasibleNormals=FeasibleNormals)
    _r27_load_change_manifold_event = _make_r27_load_change_manifold_event()

    # Source: experiments/r27_load_change/settled_event.py
    def _make_r27_load_change_settled_event():
        """Wait for a corroborated new force plateau; a native event retains full authority."""
        from collections import deque

        import numpy as np

        Changes = _r27_load_change_event.Changes; DIRECTIONS = _r27_load_change_event.DIRECTIONS; WIDTH = _r27_load_change_event.WIDTH; sensor = _r27_load_change_event.sensor


        def channels(detector, grip):
            native_error = sensor(detector, grip)
            if native_error is None:
                return None
            projected=detector.r27_projection.from_integrated(detector)
            return None if projected is None else np.array([native_error,projected],dtype=float)


        class PairedWindows:
            def __init__(self):
                self.windows = [Changes(), Changes()]

            def feed(self, values):
                means = [w.feed(None if values is None else values[i]) for i,w in enumerate(self.windows)]
                if means[0] is None or means[1] is None:
                    return None
                before, now = np.stack(means, axis=1)
                changes = (now-before) @ DIRECTIONS.T
                return before, now, changes


        class Settled:
            def __init__(self, model):
                self.bounds = np.asarray(model['joint_bounds_N'],float)
                self.each = np.asarray(model['separate_bounds_N'],float)
                if (self.bounds.shape!=(8,) or self.each.shape!=(2,8) or min(self.bounds.min(),self.each.min())<=0
                        or not np.isfinite(self.bounds).all() or not np.isfinite(self.each).all()):
                    raise ValueError('positive signed normal-only envelopes for two independent estimates required')
                self.windows = PairedWindows()
                self.native_window = deque(maxlen=WIDTH)
                self.clear()

            def clear(self):
                self.state = 'quiet'
                self.axis = None
                self.reference = self.peak = self.change_peak = None
                self.native_seen = False
                self.signal_left = 0
                self.notified = False

            def feed(self, values, native):
                self.notified = False
                if values is None:
                    self.native_window.clear()
                self.native_window.append(bool(native))
                obj = self.windows.feed(values)
                if obj is None:
                    self.clear()
                    return bool(native)
                before, now, change = obj
                if self.state == 'quiet':
                    ratios = np.min(change,axis=0) / self.bounds
                    axis = int(np.argmax(ratios))
                    if ratios[axis]>1. and np.all(now @ DIRECTIONS[axis] > 0):
                        self.axis = axis
                        self.state='rising'
                        self.reference=before @ DIRECTIONS[axis]
                        self.peak=now @ DIRECTIONS[axis]-self.reference
                        self.change_peak=change[:,axis].copy()
                        self.native_seen=any(self.native_window)
                else:
                    self.native_seen |= bool(native)
                    value = now @ DIRECTIONS[self.axis] - self.reference
                    self.peak = np.maximum(self.peak,value)
                    self.change_peak=np.maximum(self.change_peak,change[:,self.axis])
                    if np.any(2*value <= self.peak):
                        self.clear()
                    elif (self.state=='rising' and np.all(change[:,self.axis] <= self.each[:,self.axis])
                          and np.all(2*change[:,self.axis] <= self.change_peak)):
                        self.state='settled'
                        if not self.native_seen:
                            self.signal_left=4  # the incumbent's native four-sample decision width, not a duration estimate
                            self.notified=True
                if native:
                    self.signal_left=0
                active = self.signal_left>0
                if active:self.signal_left-=1
                return bool(native or active)
        return _SubmissionNamespace(Settled=Settled, channels=channels)
    _r27_load_change_settled_event = _make_r27_load_change_settled_event()

    # Source: experiments/r20_horizontal/candidate_detector.py
    def _make_r20_horizontal_candidate_detector():
        """Normal-bounded horizontal residual detector, OR'ed with unchanged R10.

        The auxiliary channel uses the same world-frame horizontal residual already
        computed by R10.  Its threshold is 1.02 times the largest absolute x/y
        residual in the early-transfer portion of a normal-only 4-mass x 5-COM
        calibration grid.  The phase window is a public trajectory transfer phase, not
        a collision time.  Both horizontal directions share the threshold.  A release
        branch was rejected because a prior collision produced a larger later normal
        transfer residual than the official 10 N cases.
        """
        import numpy as np

        my_detector = _submission_r10


        # The early-transfer normal maximum was 18.0150567392 N.
        PHASES = (
            (1.35, 1.75, 18.3753578739),
        )
        # The threshold is a phase-wide maximum bound, so no additional consecutive
        # exceedance assumption is needed to preserve the calibration negatives.
        PERSISTENCE = 1
        CLEAR_SAMPLES = 8


        class Detector(my_detector.Detector):
            """Preserve every baseline alarm and add a causal horizontal branch."""

            def reset(self):
                super().reset()
                self.horizontal_count = 0
                self.horizontal_clear = 0
                self.horizontal_detected = False
                self.horizontal_score = 0.0

            @staticmethod
            def threshold(t):
                for start, stop, value in PHASES:
                    if start <= t < stop:
                        return value
                return None

            def update(self, t, th, thd, tau_m, tau_s, grip):
                residual, baseline = super().update(t, th, thd, tau_m, tau_s, grip)
                threshold = self.threshold(float(t))
                if threshold is None:
                    over = False
                    self.horizontal_score = 0.0
                else:
                    horizontal = np.asarray(self._shared.last_residual_wrench[:2], float)
                    self.horizontal_score = float(np.max(np.abs(horizontal)) / threshold)
                    over = bool(self.horizontal_score > 1.0)
                self.horizontal_count = self.horizontal_count + 1 if over else 0
                self.horizontal_clear = 0 if over else self.horizontal_clear + 1
                if self.horizontal_count >= PERSISTENCE:
                    self.horizontal_detected = True
                elif self.horizontal_clear >= CLEAR_SAMPLES:
                    self.horizontal_detected = False
                return residual, bool(baseline or self.horizontal_detected)
        return _SubmissionNamespace(Detector=Detector)
    _r20_horizontal_candidate_detector = _make_r20_horizontal_candidate_detector()

    # Source: experiments/r21_yaw_impulse/state.py
    def _make_r21_yaw_impulse_state():
        """Estimate external base-axis impulse without differentiating joint velocity."""
        from collections import deque

        import numpy as np
        from sim import config as C

        AVERAGE = 12
        SEPARATION = 32


        class YawBalance:
            def __init__(self):
                self.dt = float(C.DT)
                self.elastic = np.asarray(C.ELASTIC, bool)
                k, d = C.K[self.elastic], C.D[self.elastic]
                self.decay = d / (d + k * self.dt)
                self.in_gain = self.dt / (d + k * self.dt)
                self.phi = None
                self.prev = None
                self.mass_row = None
                self.grip = None
                self.integral = 0.
                self.terms = deque(maxlen=AVERAGE)
                self.history = deque(maxlen=SEPARATION + 1)

            def feed(self, th, tau, mass_row, grip):
                # D*phi' + K*phi = -tau_link. Only public encoder & calibrated JTS.
                measured = np.asarray(tau)[self.elastic]
                q = np.asarray(th, dtype=float).copy()
                if self.phi is None:
                    self.phi = -measured / C.K[self.elastic]
                else:
                    self.phi = self.decay * self.phi - self.in_gain * measured
                q[self.elastic] += self.phi
                row = np.asarray(mass_row, dtype=float)
                if self.prev is None or self.grip != bool(grip):
                    self.terms.clear()
                    self.history.clear()
                    self.integral = 0.
                    value = None
                else:
                    # M[0] dq/dt is vertical angular momentum of links plus known carried
                    # payload. For the cyclic base yaw, gravity has no generalized torque.
                    p = float((self.mass_row + row) @ (q - self.prev)) / (2 * self.dt)
                    self.integral += float(tau[0]) * self.dt
                    self.terms.append(p - self.integral)
                    value = None
                    if len(self.terms) == AVERAGE:
                        self.history.append(sum(self.terms) / AVERAGE)
                        if len(self.history) > SEPARATION:
                            value = (self.history[-1] - self.history[0]) / (SEPARATION * self.dt)
                self.mass_row = row.copy()
                self.grip = bool(grip)
                self.prev = q
                return value
        return _SubmissionNamespace(AVERAGE=AVERAGE, YawBalance=YawBalance)
    _r21_yaw_impulse_state = _make_r21_yaw_impulse_state()

    # Source: experiments/r21_yaw_impulse/detector.py
    def _make_r21_yaw_impulse_detector():
        """A base-axis torque witness, not a general six-axis safety detector."""
        import math

        Horizontal = _r20_horizontal_candidate_detector.Detector
        YawBalance = _r21_yaw_impulse_state.YawBalance


        class Detector(Horizontal):
            bounds = (math.inf, math.inf)  # diagnostic default until calibrated
            model_scope = 'R10 torque controller only'

            def reset(self):
                super().reset()
                self.yaw = YawBalance()
                self.yaw_value = None
                self.yaw_count = self.yaw_quiet = 0
                self.yaw_active = False

            def update(self, t, th, thd, tau_m, tau_s, grip):
                residual, detected = super().update(t, th, thd, tau_m, tau_s, grip)
                p = self._shared
                self.yaw_value = self.yaw.feed(th, tau_s - p.bias, p.mass_prev[0], grip)
                # Initial zeroing is already performed by the unmodified R10 observer.
                if self.yaw_value is None or t < .08:
                    self.yaw_count = self.yaw_quiet = 0
                    self.yaw_active = False
                else:
                    above = abs(self.yaw_value) > self.bounds[int(bool(grip))]
                    self.yaw_count = self.yaw_count + 1 if above else 0
                    self.yaw_quiet = 0 if above else self.yaw_quiet + 1
                    if self.yaw_count >= 3:
                        self.yaw_active = True
                    elif self.yaw_quiet >= 8:
                        self.yaw_active = False
                return residual, bool(detected or self.yaw_active)
        return _SubmissionNamespace(Detector=Detector)
    _r21_yaw_impulse_detector = _make_r21_yaw_impulse_detector()

    # Source: experiments/r21_yaw_impulse/frozen.py
    def _make_r21_yaw_impulse_frozen():
        """Normal-only torque bounds for the global yaw-balance ablation."""
        GlobalBalance = _r21_yaw_impulse_detector.Detector


        class Detector(GlobalBalance):
            bounds = (3.23779270273884, 5.111942835466052)
        return _SubmissionNamespace(Detector=Detector)
    _r21_yaw_impulse_frozen = _make_r21_yaw_impulse_frozen()

    # Source: experiments/r21_yaw_impulse/coarse.py
    def _make_r21_yaw_impulse_coarse():
        """Coarse normal amplitude correction, no per-tick signed signal prediction."""
        import json


        import numpy as np

        Global = _r21_yaw_impulse_frozen.Detector


        class Detector(Global):
            def reset(self):
                super().reset()
                src = json.loads(_normal_table_r21_yaw_impulse_normal_bins)
                table = np.array(src['threshold_Nm'], dtype=float)
                if src['schema'] != 'r21-yaw-normal-bins.v1' or table.shape != (2, 60) or not (
                        np.all(np.isfinite(table)) and np.all(table > 0)):
                    raise ValueError('expected normal-only yaw bounds')
                self._by_tick = table.T

            def update(self, t, th, thd, tau_m, tau_s, grip):
                idx = int(round(1000 * float(t))) // 100
                self.bounds = self._by_tick[idx] if 0 <= idx < 60 else Global.bounds
                return super().update(t, th, thd, tau_m, tau_s, grip)
        return _SubmissionNamespace(Detector=Detector)
    _r21_yaw_impulse_coarse = _make_r21_yaw_impulse_coarse()

    # Source: experiments/r22_tangent/state.py
    def _make_r22_tangent_state():
        """Finite impulse statistic for world-planar force, with explicit availability and hysteresis."""
        from collections import deque
        import math


        def load_class(mass, grip):
            if not math.isfinite(mass) or not 0 <= mass <= 8:
                return None
            if not grip or mass == 0:
                return 0
            return 1 if mass <= 1 else 2 if mass <= 4 else 3


        class Evidence:
            def __init__(self):
                self.history = deque(maxlen=12)
                self.sx = self.sy = 0.
                self.above = self.quiet = 0
                self.active = False
                self.score = None
                self.group = None

            def feed(self, vector, group, bound):
                if vector is None or group is None or group != self.group:
                    self.history.clear()
                    self.sx = self.sy = 0.
                    self.above = self.quiet = 0
                    self.active = False
                    self.score = None
                self.group = group
                if vector is None or group is None:
                    return False
                x, y = float(vector[0]), float(vector[1])
                if not math.isfinite(x) or not math.isfinite(y) or math.isnan(bound) or bound <= 0:
                    raise ValueError('expected finite planar signal and positive bound')
                if len(self.history) == self.history.maxlen:
                    a, b = self.history[0]
                    self.sx -= a
                    self.sy -= b
                self.history.append((x, y))
                self.sx += x
                self.sy += y
                if len(self.history) == self.history.maxlen:
                    self.score = math.hypot(self.sx, self.sy) / len(self.history)
                    self.above = self.above + 1 if self.score > bound else 0
                    self.quiet = self.quiet + 1 if self.score < .8 * bound else 0
                    if self.above >= 3:
                        self.active = True
                    elif self.quiet >= 8:
                        self.active = False
                return self.active
        return _SubmissionNamespace(Evidence=Evidence, load_class=load_class)
    _r22_tangent_state = _make_r22_tangent_state()

    # Source: experiments/r22_tangent/detector.py
    def _make_r22_tangent_detector():
        """A separate research witness, not a default controller or an R21 change."""
        import math

        from sim import config as C
        NormalForce = _r15_delayed_wrench_observer.NormalForce
        Yaw = _r21_yaw_impulse_coarse.Detector
        Evidence = _r22_tangent_state.Evidence; load_class = _r22_tangent_state.load_class


        class Detector(Yaw):
            bounds_tangent = (math.inf,) * 4

            def reset(self):
                super().reset()
                self.tangent_force = NormalForce(self.nom, self.nom_box, dt=C.DT, viscous_spring=C.D, jts_hz=C.JTS_CUTOFF)
                self.tangent = Evidence()

            def update(self, t, th, thd, tau_m, tau_s, grip):
                residual, baseline = super().update(t, th, thd, tau_m, tau_s, grip)
                s = self.tangent_force.update(t, th, tau_s, grip)
                g = load_class(self.box_mass, grip)
                flag = self.tangent.feed(None if s is None else s.force_n, g, math.inf if g is None else self.bounds_tangent[g])
                return residual, bool(baseline or flag)
        return _SubmissionNamespace(Detector=Detector)
    _r22_tangent_detector = _make_r22_tangent_detector()

    # Source: experiments/r22_tangent/frozen.py
    def _make_r22_tangent_frozen():
        import json

        Unbounded = _r22_tangent_detector.Detector


        class Detector(Unbounded):
            bounds_tangent = tuple(json.loads(_normal_table_r22_tangent_model)['bounds_N'])
        return _SubmissionNamespace(Detector=Detector)
    _r22_tangent_frozen = _make_r22_tangent_frozen()

    # Source: experiments/r22_normal/candidate.py
    def _make_r22_normal_candidate():
        """R21 with one-sample early-wrench spikes rejected before thresholding.

        The stock course collision is a 20 ms ramp, so a horizontal collision signal is
        temporally coherent.  The inherited R20 branch instead thresholded a single
        unfiltered sample.  This branch keeps its frozen threshold and state machine,
        but applies a causal three-sample component-wise median before the absolute
        horizontal maximum.  R10 and the R21 yaw-balance branch are unchanged.
        """
        from collections import deque

        import numpy as np

        my_detector = _submission_r10
        R21 = _r21_yaw_impulse_coarse.Detector
        Global = _r21_yaw_impulse_frozen.Detector


        class Detector(R21):
            """Composable replacement for the R21 detector; only R20 horizontal input changes."""

            def reset(self):
                super().reset()
                self.horizontal_samples = deque(maxlen=3)
                self.horizontal_raw = np.zeros(2)
                self.horizontal_filtered = np.zeros(2)

            @staticmethod
            def robust_horizontal(samples):
                values = np.asarray(samples, dtype=float)
                if values.shape != (3, 2) or not np.all(np.isfinite(values)):
                    raise ValueError("three finite horizontal wrench samples required")
                return np.median(values, axis=0)

            def update(self, t, th, thd, tau_m, tau_s, grip):
                # Bypass only the inherited R20/R21 composition; the unchanged R10
                # detector still owns the shared estimator and its baseline result.
                residual, baseline = my_detector.Detector.update(
                    self, t, th, thd, tau_m, tau_s, grip)

                self.horizontal_raw = np.asarray(self._shared.last_residual_wrench[:2], float)
                self.horizontal_samples.append(self.horizontal_raw.copy())
                threshold = self.threshold(float(t))
                if threshold is None or len(self.horizontal_samples) < 3:
                    over = False
                    self.horizontal_score = 0.0
                else:
                    self.horizontal_filtered = self.robust_horizontal(self.horizontal_samples)
                    self.horizontal_score = float(np.max(np.abs(self.horizontal_filtered)) / threshold)
                    over = bool(self.horizontal_score > 1.0)
                self.horizontal_count = self.horizontal_count + 1 if over else 0
                self.horizontal_clear = 0 if over else self.horizontal_clear + 1
                if self.horizontal_count >= 1:
                    self.horizontal_detected = True
                elif self.horizontal_clear >= 8:
                    self.horizontal_detected = False

                idx = int(round(1000 * float(t))) // 100
                self.bounds = self._by_tick[idx] if 0 <= idx < 60 else Global.bounds
                packet = self._shared
                self.yaw_value = self.yaw.feed(th, tau_s - packet.bias, packet.mass_prev[0], grip)
                if self.yaw_value is None or t < .08:
                    self.yaw_count = self.yaw_quiet = 0
                    self.yaw_active = False
                else:
                    above = abs(self.yaw_value) > self.bounds[int(bool(grip))]
                    self.yaw_count = self.yaw_count + 1 if above else 0
                    self.yaw_quiet = 0 if above else self.yaw_quiet + 1
                    if self.yaw_count >= 3:
                        self.yaw_active = True
                    elif self.yaw_quiet >= 8:
                        self.yaw_active = False
                return residual, bool(baseline or self.horizontal_detected or self.yaw_active)
        return _SubmissionNamespace(Detector=Detector)
    _r22_normal_candidate = _make_r22_normal_candidate()

    # Source: experiments/r22_release/state.py
    def _make_r22_release_state():
        """Preserve the yaw impulse history across a modeled payload release."""
        from collections import deque

        import numpy as np

        AVERAGE = _r21_yaw_impulse_state.AVERAGE; YawBalance = _r21_yaw_impulse_state.YawBalance


        class ReleaseBalance(YawBalance):
            """Remove outgoing modeled momentum instead of clearing sensor history."""

            def __init__(self, mass):
                super().__init__()
                self.loaded = float(mass) > 0
                self._positions = deque(maxlen=AVERAGE + 1)

            def feed(self, th, tau, mass_row, grip):
                switching = self.prev is not None and self.grip != bool(grip)
                if switching and not self.loaded:
                    # An empty gripper command changes no modeled payload.
                    self.grip = bool(grip)
                elif switching and not grip and len(self._positions) == self._positions.maxlen:
                    # Use only positions preceding the release.  The inertia-row change
                    # is already part of the public R10 model packet.
                    dq = (self._positions[-1] - self._positions[0]) / (AVERAGE * self.dt)
                    self.integral += float((np.asarray(mass_row) - self.mass_row) @ dq)
                    self.mass_row = np.asarray(mass_row).copy()
                    self.grip = False
                value = super().feed(th, tau, mass_row, grip)
                self._positions.append(self.prev)
                return value
        return _SubmissionNamespace(ReleaseBalance=ReleaseBalance)
    _r22_release_state = _make_r22_release_state()

    # Source: experiments/r22_combined/core.py
    def _make_r22_combined_core():
        """Compose the isolated-spike correction and release momentum continuity without two R10 updates."""
        import json


        import numpy as np

        Median = _r22_normal_candidate.Detector
        ReleaseBalance = _r22_release_state.ReleaseBalance

        BOUNDS = _normal_table_r22_release_model


        class Detector(Median):
            def reset(self):
                super().reset()
                model = json.loads(BOUNDS)
                table = np.asarray(model['threshold_Nm'], dtype=float)
                if (model['schema'] != 'r21-release-momentum-bounds.v1' or table.shape != (2, 60)
                        or not np.isfinite(table).all() or not (table > 0).all()):
                    raise ValueError('invalid frozen release-continuity bounds')
                self._by_tick = table.T
                self.yaw = ReleaseBalance(self.box_mass)
        return _SubmissionNamespace(Detector=Detector)
    _r22_combined_core = _make_r22_combined_core()

    # Source: experiments/r22_guarded/candidate.py
    def _make_r22_guarded_candidate():
        """Sensor-guarded release continuity composed onto the frozen R22 core."""


        import json


        import numpy as np

        YawBalance = _r21_yaw_impulse_state.YawBalance
        Core = _r22_combined_core.Detector
        ReleaseBalance = _r22_release_state.ReleaseBalance


        NORMAL_BOUNDS = _normal_table_r21_yaw_impulse_normal_bins


        class GuardedReleaseBalance(ReleaseBalance):
            """Continue a loaded release only when the preceding sensor report was active."""

            def __init__(self, mass):
                super().__init__(mass)
                self.pre_release_alarm = False
                self.release_continuity_selected = None
                self.use_reset_bank = False

            def feed(self, th, tau, mass_row, grip):
                switching = self.prev is not None and self.grip != bool(grip)
                releasing_payload = switching and self.loaded and not bool(grip)
                picking_payload = switching and self.loaded and bool(grip)
                if releasing_payload and not self.pre_release_alarm:
                    self.release_continuity_selected = False
                    self.use_reset_bank = True
                    value = YawBalance.feed(self, th, tau, mass_row, grip)
                    self._positions.append(self.prev)
                    return value
                if releasing_payload:
                    self.release_continuity_selected = True
                    self.use_reset_bank = False
                elif picking_payload:
                    self.release_continuity_selected = None
                    self.use_reset_bank = False
                return super().feed(th, tau, mass_row, grip)


        class Detector(Core):
            """R22 core with sensor-evidenced continuity at positive-mass release."""

            def reset(self):
                super().reset()
                model = json.loads(NORMAL_BOUNDS)
                table = np.asarray(model["threshold_Nm"], dtype=float)
                if (model["schema"] != "r21-yaw-normal-bins.v1" or table.shape != (2, 60)
                        or not np.isfinite(table).all() or not (table > 0).all()):
                    raise ValueError("invalid frozen R21 normal-only yaw bounds")
                self._normal_by_tick = table.T
                self._continuity_by_tick = self._by_tick.copy()
                self.yaw = GuardedReleaseBalance(self.box_mass)
                self._previous_combined_report = False

            def update(self, t, th, thd, tau_m, tau_s, grip):
                self._by_tick = (self._normal_by_tick if self.yaw.use_reset_bank
                                 else self._continuity_by_tick)
                self.yaw.pre_release_alarm = bool(self._previous_combined_report)
                residual, detected = super().update(t, th, thd, tau_m, tau_s, grip)
                self._previous_combined_report = bool(detected)
                return residual, detected
        return _SubmissionNamespace(Detector=Detector, GuardedReleaseBalance=GuardedReleaseBalance)
    _r22_guarded_candidate = _make_r22_guarded_candidate()

    # Source: experiments/r22_light/witness.py
    def _make_r22_light_witness():
        """Extra force evidence only while lifting a light object; base detection is unconditional."""
        import math

        from sim import config as C, trajectory as T
        NormalForce = _r15_delayed_wrench_observer.NormalForce
        Calibration = _r22_tangent_frozen.Detector
        Evidence = _r22_tangent_state.Evidence; load_class = _r22_tangent_state.load_class

        TARGET_N = 10.
        GRAVITY = 9.81


        def eligible(mass):
            return math.isfinite(mass) and 0 <= mass * GRAVITY <= TARGET_N


        def lift_stop(waypoints, grasp):
            return min(t for t, _ in waypoints if t > grasp)


        class LightWitness:
            def __init__(self, mass, nominal, loaded):
                self.mass = mass
                self.enabled = eligible(mass)
                self.nominal, self.loaded = nominal, loaded
                self.stop = lift_stop(T.WAYPOINTS, T.T_GRASP)
                self.state = Evidence()
                self.calibrated = None
                self.observer = self._new() if self.enabled else None
                self.in_lift = False

            def _new(self):
                return NormalForce(self.nominal, self.loaded, dt=C.DT, viscous_spring=C.D, jts_hz=C.JTS_CUTOFF)

            def feed(self, t, th, tau, grip):
                if not self.enabled:
                    return False
                if self.calibrated is None:
                    # The inherited estimator accepts its static startup calibration before 0.5 s.
                    # Failed calibration disables only this optional port, never the base detector.
                    if self.observer is not None and t < .5 and not grip:
                        self.observer.update(t, th, tau, False, project=False)
                        if self.observer.bias is not None:
                            self.calibrated = self.observer.bias.copy()
                            self.observer = None
                    return False
                open_window = bool(grip) and T.T_GRASP <= t < self.stop
                continue_event = self.in_lift and bool(grip) and self.state.active
                if not open_window and not continue_event:
                    if self.in_lift:
                        self.observer = None
                        self.state = Evidence()
                        self.in_lift = False
                    return False
                if not self.in_lift:
                    self.observer = self._new()
                    self.observer.bias = self.calibrated.copy()
                    self.in_lift = True
                sample = self.observer.update(t, th, tau, grip)
                group = load_class(self.mass, grip)
                return self.state.feed(None if sample is None else sample.force_n, group, Calibration.bounds_tangent[group])
        return _SubmissionNamespace(LightWitness=LightWitness)
    _r22_light_witness = _make_r22_light_witness()

    # Source: experiments/r22_light/candidate.py
    def _make_r22_light_candidate():
        """Evidence-scoped weak uplift; the guarded full-cycle base is never muted."""
        Guarded = _r22_guarded_candidate.Detector
        LightWitness = _r22_light_witness.LightWitness


        class Detector(Guarded):
            def reset(self):
                super().reset()
                self.light_witness=LightWitness(self.box_mass,self.nom,self.nom_box)

            def update(self,t,th,thd,tau_m,tau_s,grip):
                residual,base=super().update(t,th,thd,tau_m,tau_s,grip)
                lifted=self.light_witness.feed(t,th,tau_s,grip)
                return residual,bool(base or lifted)
        return _SubmissionNamespace(Detector=Detector)
    _r22_light_candidate = _make_r22_light_candidate()

    # Source: experiments/r23_detached/state.py
    def _make_r23_detached_state():
        """Admit a newly rising planar disturbance, not a residual already high at startup."""
        import math


        class OnsetGate:
            def __init__(self):
                self.quiet = 0
                self.armed = self.accepted = False

            def feed(self, xy, above, incumbent, bound):
                if xy is None:
                    self.__init__()
                    return False
                if self.accepted:
                    if above:
                        return True
                    self.accepted = self.armed = False
                    self.quiet = 0
                if incumbent:
                    self.quiet = 0
                    self.armed = False
                elif math.hypot(*xy) < 0.8 * bound and not above:
                    self.quiet += 1
                    if self.quiet >= 3:
                        self.armed = True
                else:
                    self.quiet = 0
                if above and self.armed:
                    self.accepted = True
                    self.armed = False
                return self.accepted
        return _SubmissionNamespace(OnsetGate=OnsetGate)
    _r23_detached_state = _make_r23_detached_state()

    # Source: experiments/r23_detached/candidate.py
    def _make_r23_detached_candidate():
        """Study-only addition to R22 LIGHT for an unloaded arm after actual detachment."""
        import math

        from sim import config as C
        NormalForce = _r15_delayed_wrench_observer.NormalForce
        Light = _r22_light_candidate.Detector
        Evidence = _r22_tangent_state.Evidence
        Tangent = _r22_tangent_frozen.Detector
        OnsetGate = _r23_detached_state.OnsetGate


        class ReleasedForce:
            def __init__(self, mass, empty, loaded, require_onset):
                self.enabled = math.isfinite(mass) and 0 <= mass <= 8
                self.empty, self.loaded = empty, loaded
                self.require_onset = require_onset
                self.calibrator = self.new_observer() if self.enabled else None
                self.bias = self.prev_grip = self.observer = None
                self.evidence = Evidence()
                self.gate = OnsetGate()
                self.vector = None

            def new_observer(self):
                return NormalForce(self.empty, self.loaded, dt=C.DT, viscous_spring=C.D, jts_hz=C.JTS_CUTOFF)

            def feed(self, t, encoder, jts, grip, incumbent):
                grip = bool(grip)
                detach = self.prev_grip is True and not grip
                self.prev_grip = grip
                self.vector = None
                if not self.enabled:
                    return False
                if self.calibrator is not None:
                    if t < .5 and not grip:
                        self.calibrator.update(t, encoder, jts, False, project=False)
                        if self.calibrator.bias is not None:
                            self.bias = self.calibrator.bias.copy()
                            self.calibrator = None
                    else:
                        self.calibrator = None
                    return False
                if grip:
                    self.observer = None
                    return False
                if detach and self.bias is not None:
                    self.observer = self.new_observer()
                    self.observer.bias = self.bias.copy()
                    self.evidence = Evidence()
                    self.gate = OnsetGate()
                if self.observer is None:
                    return False
                sample = self.observer.update(t, encoder, jts, False)
                self.vector = None if sample is None else sample.force_n[:2]
                bound = Tangent.bounds_tangent[0]
                above = self.evidence.feed(self.vector, 0, bound)
                return self.gate.feed(self.vector, above, incumbent, bound) if self.require_onset else above


        class Free(Light):
            require_onset = False

            def reset(self):
                super().reset()
                self.detached = ReleasedForce(self.box_mass, self.nom, self.nom_box, self.require_onset)

            def update(self, t, th, thd, tau_m, tau_s, grip):
                residual, incumbent = super().update(t, th, thd, tau_m, tau_s, grip)
                extra = self.detached.feed(t, th, tau_s, grip, incumbent)
                return residual, bool(incumbent or extra)


        class Detector(Free):
            require_onset = True
        return _SubmissionNamespace(Free=Free)
    _r23_detached_candidate = _make_r23_detached_candidate()

    # Source: experiments/r23_detached/free.py
    def _make_r23_detached_free():
        """Default-name adapter for the magnitude-only research comparator."""
        Detector = _r23_detached_candidate.Free

        __all__ = ['Detector']
        return _SubmissionNamespace(Detector=Detector)
    _r23_detached_free = _make_r23_detached_free()

    # Source: experiments/r24_loaded/drive.py
    def _make_r24_loaded_drive():
        """Identify motor transmission from past encoder/command/JTS data, before any grasp."""
        import math

        import numpy as np

        from sim import config as C


        class DriveEstimate:
            width, windows = 48, 20

            def __init__(self):
                self.last_velocity = self.last_jts = None
                self.begin_velocity = self.begin_jts = None
                self.command = np.zeros(3);self.resistance=np.zeros(3)
                self.x=[];self.y=[];self.i=self.blocks=0
                self.closed=False;self.gain=None;self.excitation=None

            def feed(self, velocity, torque, jts, grip):
                if self.closed:return
                if grip:
                    self.closed=True
                    return
                v=np.asarray(velocity)[:3];j=np.asarray(jts)[:3]
                if self.last_velocity is None:
                    self.begin_velocity=v.copy();self.begin_jts=j.copy()
                self.last_velocity=v;self.last_jts=j
                self.command+=np.asarray(torque)[:3]
                self.resistance+=j+C.VISCOUS[:3]*v+C.COULOMB[:3]*np.tanh(v/C.FRICTION_EPS)
                self.i+=1
                if self.i < self.width:return
                duration=self.width*float(C.DT);lag=float(C.DT)+1./(2.*math.pi*float(C.JTS_CUTOFF))
                y=self.resistance/self.width+(C.B[:3]*(v-self.begin_velocity)+lag*(j-self.begin_jts))/duration
                if self.blocks:
                    self.x.append(self.command/self.width);self.y.append(y)
                self.blocks+=1;self.begin_velocity=v.copy();self.begin_jts=j.copy()
                self.command.fill(0.);self.resistance.fill(0.);self.i=0
                if len(self.x)==self.windows:
                    self.closed=True
                    x=np.asarray(self.x);y=np.asarray(self.y);x=x-x.mean(0);y=y-y.mean(0)
                    xx=np.sum(x*x,axis=0);self.excitation=np.sqrt(xx/self.windows)
                    if np.all(self.excitation>1.) and np.isfinite(xx).all():
                        gains=np.sum(x*y,axis=0)/xx
                        if np.isfinite(gains).all():self.gain=gains


        def departure(gain, low, high):
            if gain is None:return False
            values=np.asarray(gain,dtype=float);lo=np.asarray(low);hi=np.asarray(high)
            return bool(values.shape==(3,) and np.isfinite(values).all() and np.count_nonzero((values<lo)|(values>hi))>=2)
        return _SubmissionNamespace(DriveEstimate=DriveEstimate)
    _r24_loaded_drive = _make_r24_loaded_drive()

    # Source: experiments/r24_loaded/control.py
    def _make_r24_loaded_control():
        """Signed support-yaw attribution in the original detector; all quantities are causal sensor estimates."""
        import math



        GuardedReleaseBalance = _r22_guarded_candidate.GuardedReleaseBalance
        Released = _r23_detached_free.Detector


        def subtract_aligned(value, prediction):
            """Never add a yaw alarm or reverse its sign by subtracting a modeled reaction."""
            if not math.isfinite(prediction) or value * prediction <= 0.:
                return value
            return math.copysign(max(0., abs(value) - abs(prediction)), value)


        class LoadedYaw(GuardedReleaseBalance):
            def __init__(self, mass, packet, sliding):
                super().__init__(mass)
                self.packet, self.sliding = packet, sliding
                self.was_away = False
                self.before_projection = None
                self.reaction = 0.

            def feed(self, th, tau, mass_row, grip):
                value = super().feed(th, tau, mass_row, grip)
                s = self.packet
                self.before_projection = value
                self.reaction = 0.
                if not grip:
                    self.was_away = False
                    return value
                if not s.last_table_allowed:
                    self.was_away = True
                    return value
                if not (self.was_away and self.loaded and value is not None and s.last_table_wrench[2] > 0):
                    return value
                jac, _, _ = s._point_jacobian(s.q_prev)
                support = s.last_table_wrench
                if self.sliding:
                    # From the measured link motion, friction at the box bottom dissipates
                    # tangential power; torsional resistance opposes spin around world Z.
                    speed = jac @ s.v
                    planar_speed = math.hypot(*speed[:2])
                    if planar_speed <= 1e-12:
                        return value
                    fz = float(support[2])
                    fx, fy = -s.friction_mu * fz * speed[:2] / planar_speed
                    tz = -math.copysign(s.torsion_radius * fz, speed[5]) if speed[5] else 0.
                    sliding_yaw = jac[0,0]*fx + jac[1,0]*fy + jac[5,0]*tz
                    measured_yaw = float(jac[:,0] @ s.last_wrench)
                    if sliding_yaw * measured_yaw > 0 and support[:2] @ speed[:2] < 0:
                        self.reaction = math.copysign(min(abs(measured_yaw), abs(sliding_yaw)), sliding_yaw)
                else:
                    self.reaction = float(jac[:,0] @ support)
                return subtract_aligned(value, self.reaction)


        class Projected(Released):
            sliding = False

            def reset(self):
                super().reset()
                if getattr(self.ctrl, 'MODE', None)=='torque' and self.box_mass > 0:
                    self.yaw = LoadedYaw(self.box_mass, self._shared, self.sliding)


        class Sliding(Projected):
            sliding = True
        return _SubmissionNamespace(subtract_aligned=subtract_aligned)
    _r24_loaded_control = _make_r24_loaded_control()

    # Source: experiments/r24_loaded/response_state.py
    def _make_r24_loaded_response_state():
        """Finite-history predictors have no knowledge of trial times, force schedules or phase labels."""
        from collections import deque
        import math


        class SupportHistory:
            width, taps = 8, 5

            def __init__(self):
                self.pressure = deque([0.]*(self.width*self.taps), maxlen=self.width*self.taps)
                self.sums = [0.]*self.taps

            def push(self, normal):
                # Accumulate exact non-overlapping, sliding causal bands; oldest crosses band boundaries.
                x = float(normal)
                for i in range(self.taps):
                    previous = self.pressure[(i+1)*self.width-1]
                    self.sums[i] += x - previous
                    x = previous
                self.pressure.appendleft(float(normal))
                return [s/self.width for s in self.sums]


        class ConeHistory:
            def __init__(self, width):
                self.window = deque(maxlen=width)
                self.width = width

            def signed_credit(self, projected, measured, cone):
                self.window.append((projected, measured, cone))
                signed, observed, capacity = (sum(row[k] for row in self.window)/self.width for k in range(3))
                if signed*observed <= 0 or not math.isfinite(capacity):
                    return 0.
                return math.copysign(min(abs(observed),max(0.,capacity)), signed)
        return _SubmissionNamespace(ConeHistory=ConeHistory, SupportHistory=SupportHistory)
    _r24_loaded_response_state = _make_r24_loaded_response_state()

    # Source: experiments/r24_loaded/memory.py
    def _make_r24_loaded_memory():
        """Physically scoped support memory for the cyclic yaw evidence. Full R23 FREE otherwise unchanged."""
        import json
        import math


        AVERAGE = _r21_yaw_impulse_state.AVERAGE
        GuardedReleaseBalance = _r22_guarded_candidate.GuardedReleaseBalance
        Free = _r23_detached_free.Detector
        subtract_aligned = _r24_loaded_control.subtract_aligned
        ConeHistory = _r24_loaded_response_state.ConeHistory; SupportHistory = _r24_loaded_response_state.SupportHistory

        MODEL=_normal_table_r24_loaded_normal_response


        class MemoryYaw(GuardedReleaseBalance):
            def __init__(self,mass,packet,weights):
                super().__init__(mass)
                self.packet=packet;self.weights=weights;self.away=False
                self.contact_history=SupportHistory() if weights is not None else ConeHistory(AVERAGE)
                self.raw_value=None;self.reaction=0.

            def feed(self,th,tau,mass_row,grip):
                value=super().feed(th,tau,mass_row,grip);s=self.packet
                self.raw_value=value;self.reaction=0.
                if not grip:
                    if self.away:self.contact_history=SupportHistory() if self.weights is not None else ConeHistory(AVERAGE)
                    self.away=False
                    return value
                if not s.last_table_allowed:self.away=True
                if not self.away:return value
                n=max(0.,float(s.last_table_wrench[2])) if s.last_table_allowed else 0.
                if self.weights is not None:
                    features=self.contact_history.push(n)
                    self.reaction=sum(w*x for w,x in zip(self.weights,features,strict=True))
                elif n>0. and value is not None:
                    jac,pos,_=s._point_jacobian(s.q_prev)
                    radius=math.hypot(*pos[:2])+math.hypot(s.half_x,s.half_y)
                    cap=n*(s.friction_mu*radius+s.torsion_radius)
                    self.reaction=self.contact_history.signed_credit(float(jac[:,0]@s.last_table_wrench),float(jac[:,0]@s.last_wrench),cap)
                else:
                    self.reaction=self.contact_history.signed_credit(0.,0.,0.)
                if value is None or not s.last_table_allowed:return value
                return subtract_aligned(value,self.reaction)


        class Cone(Free):
            calibrated=False
            def reset(self):
                super().reset()
                if getattr(self.ctrl,'MODE',None)!='torque' or self.box_mass<=0:return
                weights=None
                if self.calibrated:
                    model=json.loads(MODEL);weights=model['coefficients_Nm_per_N']
                    if model['schema']!='week09-r24-load-response-model.v1' or len(weights)!=5 or not all(map(math.isfinite,weights)):
                        raise ValueError('invalid normal-only response')
                self.yaw=MemoryYaw(self.box_mass,self._shared,weights)


        class Response(Cone):
            calibrated=True
        return _SubmissionNamespace(MODEL=MODEL, MemoryYaw=MemoryYaw)
    _r24_loaded_memory = _make_r24_loaded_memory()

    # Source: experiments/r24_loaded/dc.py
    def _make_r24_loaded_dc():
        """Explain table yaw only in a transmission regime supported by both DC impulse and AC-slope evidence."""
        import json


        import numpy as np

        from sim import config as C
        GuardedReleaseBalance = _r22_guarded_candidate.GuardedReleaseBalance
        Free = _r23_detached_free.Detector
        DriveEstimate = _r24_loaded_drive.DriveEstimate
        MemoryYaw = _r24_loaded_memory.MemoryYaw; SUPPORT = _r24_loaded_memory.MODEL

        MODEL=_normal_table_r24_loaded_dc_drive


        def measurements(est):
            if est.gain is None:return None,None
            mean_x=np.mean(est.x,axis=0);mean_y=np.mean(est.y,axis=0)
            valid=np.isfinite(mean_x)&np.isfinite(mean_y)&(np.abs(mean_x)>C.COULOMB[:3])
            dc=np.divide(mean_y, mean_x, out=np.zeros_like(mean_y), where=valid)
            return np.stack((est.gain,dc)),valid


        def changed(value,valid,lo,hi):
            if value is None:return False
            same_side=np.all(value<lo,axis=0)|np.all(value>hi,axis=0)
            return bool(np.any(valid & np.all(np.isfinite(value),axis=0) & same_side))


        class SelectiveYaw(MemoryYaw):
            def __init__(self,mass,packet,weights):
                super().__init__(mass,packet,weights);self.admitted=False

            def feed(self,th,tau,mass_row,grip):
                if self.admitted:
                    return super().feed(th,tau,mass_row,grip)
                # Use the same balance and past history when inactive, not a second
                # observer and not a filtered/suppressed R23 output.
                value=GuardedReleaseBalance.feed(self,th,tau,mass_row,grip)
                self.raw_value=value;self.reaction=0.
                return value


        class Detector(Free):
            def reset(self):
                super().reset()
                self.drive=None;self.changed_drive=False
                if getattr(self.ctrl,'MODE',None)!='torque' or not self.box_mass>0:return
                m=json.loads(MODEL);s=json.loads(SUPPORT);w=s['coefficients_Nm_per_N']
                self.lo=np.asarray(m['low']);self.hi=np.asarray(m['high']);self.good=np.asarray(m['eligible'],dtype=bool)
                if (m['schema']!='week09-r24-dc-slope-calibration.v1' or s['schema']!='week09-r24-load-response-model.v1'
                        or self.lo.shape!=(2,3) or self.hi.shape!=(2,3) or self.good.shape!=(3,) or not (self.hi>self.lo).all()
                        or not np.isfinite([self.lo,self.hi]).all() or len(w)!=5 or not np.isfinite(w).all()):
                    raise ValueError('invalid frozen torque-port models')
                self.drive=DriveEstimate()
                self.yaw=SelectiveYaw(self.box_mass,self._shared,w)

            def update(self,t,th,thd,tau_m,tau_s,grip):
                if self.drive is not None and not self.drive.closed:
                    self.drive.feed(thd,tau_m,tau_s,grip)
                    if self.drive.closed:
                        data,valid=measurements(self.drive)
                        self.changed_drive=changed(data, None if valid is None else valid & self.good,self.lo,self.hi)
                        self.yaw.admitted=self.changed_drive
                return super().update(t,th,thd,tau_m,tau_s,grip)
        return _SubmissionNamespace(Detector=Detector)
    _r24_loaded_dc = _make_r24_loaded_dc()

    # Source: experiments/r24_repeat/trial.py
    def _make_r24_repeat_trial():
        """Causal pickup model-switch continuity trial on the selected R23 detector."""


        import numpy as np

        AVERAGE = _r21_yaw_impulse_state.AVERAGE
        GuardedReleaseBalance = _r22_guarded_candidate.GuardedReleaseBalance
        Free = _r23_detached_free.Detector


        class PickupContinuousBalance(GuardedReleaseBalance):
            """Keep one active yaw event continuous as the known payload is attached.

            The model's generalized momentum changes algebraically when the grip state
            selects the loaded inertia.  If yaw evidence was already active on the
            preceding sensor report, compensate that modeled momentum step instead of
            erasing the measured event history.  Quiet pickups keep the inherited reset.
            """

            def __init__(self, mass):
                super().__init__(mass)
                self.pre_pick_yaw_active = False
                self.pick_continuity_selected = None

            def feed(self, th, tau, mass_row, grip):
                switching = self.prev is not None and self.grip != bool(grip)
                picking_payload = switching and self.loaded and bool(grip)
                can_continue = picking_payload and self.pre_pick_yaw_active and (
                    len(self._positions) == AVERAGE + 1)
                if can_continue:
                    dq = (self._positions[-1] - self._positions[0]) / (AVERAGE * self.dt)
                    self.integral += float(
                        (np.asarray(mass_row, dtype=float) - self.mass_row) @ dq)
                    self.mass_row = np.asarray(mass_row, dtype=float).copy()
                    self.grip = True
                    self.pick_continuity_selected = True
                elif picking_payload:
                    self.pick_continuity_selected = False
                return super().feed(th, tau, mass_row, grip)


        class Detector(Free):
            """Selected R23 free detector plus sensor-evidenced pickup continuity."""

            def reset(self):
                super().reset()
                self.yaw = PickupContinuousBalance(self.box_mass)
                self.pick_event_active = False
                self.pick_event_quiet = 0
                self.pick_continuity_consumed = False

            def update(self, t, th, thd, tau_m, tau_s, grip):
                self.yaw.pre_pick_yaw_active = bool(self.yaw_active)
                residual, report = super().update(t, th, thd, tau_m, tau_s, grip)
                if (self.yaw.pick_continuity_selected and
                        not self.pick_continuity_consumed):
                    self.pick_event_active = True
                    self.pick_continuity_consumed = True
                packet = self._shared
                continuation = max(float(packet.last_score),
                                   float(packet.last_fast_score)) > 0.55
                if self.pick_event_active and (report or continuation):
                    self.pick_event_quiet = 0
                elif self.pick_event_active:
                    self.pick_event_quiet += 1
                    if self.pick_event_quiet >= packet.release_persistence:
                        self.pick_event_active = False
                return residual, bool(report or self.pick_event_active)
        return _SubmissionNamespace(Detector=Detector, PickupContinuousBalance=PickupContinuousBalance)
    _r24_repeat_trial = _make_r24_repeat_trial()

    # Source: experiments/r24_composed/detector.py
    def _make_r24_composed_detector():
        """Compose two disjoint physical-state corrections; evaluate the joint behavior rather than adding scores."""
        DriveDetector = _r24_loaded_dc.Detector
        MemoryYaw = _r24_loaded_memory.MemoryYaw
        PickupDetector = _r24_repeat_trial.Detector; PickupContinuousBalance = _r24_repeat_trial.PickupContinuousBalance


        class JointBalance(MemoryYaw, PickupContinuousBalance):
            """Both components share one causal momentum history, including at pickup."""
            admitted = False

            def feed(self, th, tau, mass_row, grip):
                if self.admitted:
                    # MemoryYaw delegates to PickupContinuousBalance before attributing
                    # any post-transport component to observed support.
                    return super().feed(th, tau, mass_row, grip)
                value = PickupContinuousBalance.feed(self, th, tau, mass_row, grip)
                self.raw_value, self.reaction = value, 0.
                return value


        class Detector(DriveDetector, PickupDetector):
            def reset(self):
                super().reset()
                if self.drive is not None:
                    self.yaw = JointBalance(self.box_mass, self._shared, self.yaw.weights)
        return _SubmissionNamespace(Detector=Detector)
    _r24_composed_detector = _make_r24_composed_detector()

    # Source: experiments/r25_fragment/event.py
    def _make_r25_fragment_event():
        """A two-hypothesis continuation of an existing learned horizontal-force alarm; never initiates one."""
        import numpy as np


        class SignedEvent:
            """Compare each independent signal with its measured event and zero-load models.
    
            Means are estimated only while the original alarm is present. Under equal
            covariance and prior odds, ||x-m||² < ||x||² iff 2 x·m > m·m.
            A sharp disagreement or change of grip ends the carry; it never mutes an alarm.
            """
            def __init__(self, predict=False, require_stationary=False):
                self.predict = predict
                self.require_stationary = require_stationary
                self.reset()

            def reset(self):
                self.n = 0
                self.mean = None
                self.bridge = self.observed = False
                self.last_grip = None
                self.history = []

            def feed(self, observed, ref_signal, force, grip, width, lag, bound):
                if self.last_grip != bool(grip):
                    self.reset()
                self.last_grip = bool(grip)
                if force is None or ref_signal is None or not (np.isfinite(force).all() and np.isfinite(ref_signal).all()):
                    self.n = 0
                    self.mean = None
                    self.bridge = self.observed = False
                    self.history.clear()
                    return False
                value = np.stack((ref_signal[:2],force[:2]))
                self.history.append(value)
                if len(self.history)>2*width:
                    del self.history[0]
                if observed:
                    if not self.observed and not self.bridge:
                        self.n = 0
                        self.mean = None
                    self.n += 1
                    if self.mean is None:
                        self.mean = value.copy()
                    else:
                        self.mean += (value-self.mean)/self.n
                    self.observed = True
                    self.bridge = False
                    return False
                if not (self.bridge or self.observed) or self.mean is None or len(self.history)<2*width:
                    self.bridge = self.observed = False
                    return False
                self.observed=False
                # Refuse an event defined only by noise in one estimator or opposite directions.
                m=self.mean;energy=np.sum(m*m,axis=1)
                valid=bool(min(energy)>0 and np.dot(m[0],m[1])>0)
                if valid:
                    avg=np.mean(self.history[-width:],axis=0)
                    if self.predict or self.require_stationary:
                        before=np.mean(self.history[:width],axis=0)
                        # A least-assumption first-order prediction for a delayed smoother.
                        # Do not amplify a growing force; only extrapolate a fall toward zero.
                        slope=np.sum((avg-before)*m,axis=1)/width
                    proj=np.sum(avg*m,axis=1)
                    if self.predict:proj=proj+lag*np.minimum(0.,slope)
                    valid=bool(np.all(2.*proj>energy))
                    if self.require_stationary:
                        # When the primary model already explains the fall as a ramp ending,
                        # hold only if both redundant estimates still exceed its frozen decision level.
                        valid=valid and bool(np.all(slope>=0) or np.all(proj/np.sqrt(energy)>bound))
                self.bridge=valid
                return valid
        return _SubmissionNamespace(SignedEvent=SignedEvent)
    _r25_fragment_event = _make_r25_fragment_event()

    # Source: experiments/r25_fragment/stream.py
    def _make_r25_fragment_stream():
        """Apply the signed continuation only when the original model is observing the carried payload."""
        from sim import config as C
        LAG = _r15_delayed_wrench_observer.LAG; NormalForce = _r15_delayed_wrench_observer.NormalForce
        Joint = _r24_composed_detector.Detector
        SignedEvent = _r25_fragment_event.SignedEvent


        class ContinuationMixin:
            def reset(self):
                super().reset()
                self.continuation = SignedEvent(predict=True)
                self.witness = None
                self.static_bias = None
                if getattr(self.ctrl, 'MODE', None) == 'torque' and self.normal_model is not None:
                    self.witness = self.new_witness()

            def new_witness(self):
                return NormalForce(self.nom, self.nom_box, dt=C.DT, viscous_spring=C.D, jts_hz=C.JTS_CUTOFF)

            def update(self, t, th, thd, tau_m, tau_s, grip):
                residual, report = super().update(t, th, thd, tau_m, tau_s, grip)
                if self.static_bias is None:
                    if self.witness is not None:
                        self.witness.update(t, th, tau_s, grip, project=False)
                        if self.witness.bias is not None:
                            self.static_bias = self.witness.bias.copy()
                            self.witness = None
                    return residual, report
                n = self.normal_model
                if not n.last_active:
                    # last_error is retained even when the old predictor is inactive.
                    # A stale value is not an independent observation of present force.
                    if self.witness is not None:
                        self.witness = None
                        self.continuation.reset()
                    return residual, report
                if self.witness is None:
                    self.witness = self.new_witness()
                    self.witness.bias = self.static_bias.copy()
                project = n.detected or self.continuation.observed or self.continuation.bridge
                sample = self.witness.update(t, th, tau_s, grip, project=project)
                if not project:
                    if self.continuation.n:
                        self.continuation.reset()
                    return residual, report
                extra = self.continuation.feed(n.detected, n.last_error, None if sample is None else sample.force_n,
                                               grip, self._shared.release_persistence, LAG, n.threshold)
                return residual, bool(report or extra)


        class Detector(ContinuationMixin, Joint):
            """The R24 joint detector plus phase-valid signed horizontal force continuation."""
        return _SubmissionNamespace(ContinuationMixin=ContinuationMixin)
    _r25_fragment_stream = _make_r25_fragment_stream()

    # Source: experiments/r25_pickup/candidate.py
    def _make_r25_pickup_candidate():
        """Complete a reported pre-pick yaw event at a physically consistent model switch."""


        import math

        R24Joint = _r24_composed_detector.Detector


        def signed_completion(before, after):
            """Return the consumed sign when compensation preserves sign and reduces magnitude."""
            if not (math.isfinite(before) and math.isfinite(after)):
                return 0
            if before == 0.0 or before * after <= 0.0 or abs(after) > abs(before):
                return 0
            return 1 if before > 0.0 else -1


        class PickupCompletionMixin:
            """Turn a compensated, already-reported pickup impulse into a completed event.

            The inherited detector is updated exactly once.  Its yaw observer and model
            compensation remain untouched.  When the actual grip/model transition shows
            that the signed yaw estimate is already progressing toward zero, only that
            reported yaw route is consumed; all other detector routes remain live.  The
            route is immediately released for an above-bound opposite-signed yaw event.
            """

            def reset(self):
                super().reset()
                self.pick_completion_active = False
                self.pick_completion_decided = False
                self.pick_completion_sign = 0
                self.pick_completion_selected = False
                self.pick_completion_reason = None

            def _non_yaw_report(self):
                packet = self._shared
                normal = getattr(self, "normal_model", None)
                normal_active = bool(normal is not None and normal.detected)
                light = getattr(getattr(self, "light_witness", None), "state", None)
                detached = getattr(getattr(self, "detached", None), "evidence", None)
                return bool(
                    packet.detected
                    or normal_active
                    or self.horizontal_detected
                    or (light is not None and light.active)
                    or (detached is not None and detached.active)
                )

            def update(self, t, th, thd, tau_m, tau_s, grip):
                before = self.yaw_value
                pre_active = bool(self.yaw_active)
                selection_before = self.yaw.pick_continuity_selected
                residual, report = super().update(t, th, thd, tau_m, tau_s, grip)

                selected_now = (selection_before is not True and
                                self.yaw.pick_continuity_selected is True)
                if selected_now and not self.pick_completion_decided:
                    self.pick_completion_decided = True
                    sign = signed_completion(float(before), float(self.yaw_value))
                    if pre_active and sign:
                        self.pick_completion_active = True
                        self.pick_completion_sign = sign
                        self.pick_completion_selected = True
                        self.pick_completion_reason = "same-sign compensated momentum receded"
                        # R24's generic latch is replaced only after causal completion.
                        self.pick_event_active = False
                    else:
                        self.pick_completion_reason = "signed completion confidence absent"

                if not self.pick_completion_active:
                    return residual, report

                value = self.yaw_value
                bound = float(self.bounds[int(bool(grip))])
                reversal = bool(
                    value is not None
                    and math.isfinite(float(value))
                    and self.pick_completion_sign * float(value) < -bound
                )
                if reversal:
                    self.pick_completion_active = False
                    self.pick_completion_reason = "above-bound opposite-signed yaw released route"
                    return residual, report

                if not self.yaw_active:
                    self.pick_completion_active = False
                    self.pick_completion_reason = "consumed yaw route physically cleared"
                return residual, self._non_yaw_report()


        class Detector(PickupCompletionMixin, R24Joint):
            """R24-compatible detector with one causal pickup-completion update."""
        return _SubmissionNamespace(Detector=Detector)
    _r25_pickup_candidate = _make_r25_pickup_candidate()

    # Source: experiments/r25_composed/detector.py
    def _make_r25_composed_detector():
        """One physical simulation update: preserve live sensor routes while fixing two different event boundaries."""
        ContinuationMixin = _r25_fragment_stream.ContinuationMixin
        Pickup = _r25_pickup_candidate.Detector


        class Detector(ContinuationMixin, Pickup):
            """Apply measured load continuation after the causal pickup ownership decision."""
        return _SubmissionNamespace(Detector=Detector)
    _r25_composed_detector = _make_r25_composed_detector()

    # Source: experiments/r26_long/candidate.py
    def _make_r26_long_candidate():
        """Causal identity for one signed yaw event across estimator route dropouts."""


        import math

        import numpy as np

        R25 = _r25_composed_detector.Detector


        class SignedYawOwnership:
            """Classify an inactive sample against the owned event and zero hypotheses.

            Ownership can only start from the frozen detector's yaw alarm.  Its signed
            yaw mean and the independent learned horizontal-residual mean are learned
            while that alarm is present.  A route handoff is admitted only when both
            current measurements are closer to their event means than to zero.  Once
            admitted, signed yaw owns the event until zero becomes nearer.  A sign
            reversal, grip/model switch, non-finite sample, or zero-closer sample
            releases ownership immediately.
            """

            def __init__(self):
                self.reset()

            def reset(self):
                self.active = False
                self.bridge = False
                self.mean = 0.0
                self.support_mean = None
                self.n = 0
                self.grip = None

            def _begin(self, value, support, grip):
                self.active = True
                self.bridge = False
                self.mean = value
                self.support_mean = None if support is None else support.copy()
                self.n = 1
                self.grip = grip

            @staticmethod
            def _support(value):
                if value is None:
                    return None
                value = np.asarray(value, dtype=float)
                if value.shape != (2,) or not np.isfinite(value).all():
                    return None
                return value

            @staticmethod
            def _nearer_event(value, mean):
                return bool(2.0 * float(np.dot(value, mean)) > float(np.dot(mean, mean)))

            def feed(self, observed, value, support, grip, blocked=False):
                grip = bool(grip)
                support = self._support(support)
                if blocked or value is None or not math.isfinite(float(value)):
                    self.reset()
                    self.grip = grip
                    return False
                value = float(value)
                if self.grip is not None and self.grip != grip:
                    self.reset()
                self.grip = grip
                if observed:
                    if not self.active or self.mean * value <= 0.0:
                        self._begin(value, support, grip)
                    else:
                        self.n += 1
                        self.mean += (value - self.mean) / self.n
                        if support is not None:
                            if self.support_mean is None:
                                self.support_mean = support.copy()
                            else:
                                self.support_mean += (support - self.support_mean) / self.n
                        self.bridge = False
                    return False
                if not self.active:
                    return False
                # Equal covariance and prior odds: event is nearer than zero iff
                # (value-mean)^2 < value^2, equivalently 2*value*mean > mean^2.
                yaw_agrees = bool(2.0 * value * self.mean > self.mean * self.mean)
                # A route handoff becomes owned only when the independent learned
                # horizontal residual agrees with the same event-vs-zero hypothesis.
                # Once established, ownership follows signed yaw until zero is nearer;
                # instantaneous support noise cannot create extra output edges.
                support_agrees = bool(
                    self.bridge or
                    (support is not None and self.support_mean is not None and
                     self._nearer_event(support, self.support_mean)))
                self.bridge = bool(yaw_agrees and support_agrees)
                if not self.bridge:
                    self.reset()
                    self.grip = grip
                return self.bridge


        class YawEventOwnershipMixin:
            """Keep a redundantly witnessed signed yaw event continuous; never initiate one."""

            def reset(self):
                super().reset()
                self.yaw_identity = SignedYawOwnership()
                self.yaw_identity_bridge = False

            def update(self, t, th, thd, tau_m, tau_s, grip):
                residual, report = super().update(t, th, thd, tau_m, tau_s, grip)
                # R25 pickup completion intentionally consumes its yaw route at a real
                # payload-model switch.  Event identity must never undo that correction.
                blocked = bool(getattr(self, "pick_completion_active", False))
                normal = getattr(self, "normal_model", None)
                support = (None if normal is None or not normal.last_active
                           else normal.last_error)
                self.yaw_identity_bridge = self.yaw_identity.feed(
                    bool(self.yaw_active), self.yaw_value, support, grip, blocked=blocked)
                return residual, bool(report or self.yaw_identity_bridge)


        class Detector(YawEventOwnershipMixin, R25):
            """Frozen R25 plus causal signed-yaw event ownership."""
        return _SubmissionNamespace(SignedYawOwnership=SignedYawOwnership)
    _r26_long_candidate = _make_r26_long_candidate()

    # Source: experiments/r26_yaw_acquisition/candidate.py
    def _make_r26_yaw_acquisition_candidate():
        """Consume only a confidently same-source yaw reacquisition, never its gap."""


        import math

        R25 = _r25_composed_detector.Detector
        SignedYawOwnership = _r26_long_candidate.SignedYawOwnership


        class YawReacquisitionMixin:
            """Separate continuous event identity from the detector's Boolean alarm.

            R25 is updated exactly once.  A native yaw fall is always exposed.  If the
            frozen signed identity remains certified during that gap, a later yaw-only
            reacquisition of the same source is consumed.  Any independently live R25
            route is preserved.  Unknown route provenance fails open.
            """

            def reset(self):
                super().reset()
                self.yaw_acquisition_identity = SignedYawOwnership()
                self.yaw_acquisition_episode_active = False
                self.yaw_reacquisition_consumed = False
                self.yaw_reacquisition_count = 0
                self.yaw_reacquisition_provenance_complete = True
                self._yaw_acquisition_previous_active = False
                self._yaw_acquisition_gap_certified = False

            @staticmethod
            def _same_signed_value(identity, value, grip):
                """Use the frozen identity's signed event-vs-zero decision."""
                return bool(
                    identity.active
                    and identity.grip == bool(grip)
                    and value is not None
                    and math.isfinite(float(value))
                    and identity.mean * float(value) > 0.0
                    and 2.0 * float(value) * identity.mean > identity.mean * identity.mean
                )

            @classmethod
            def _same_signed_source(cls, identity, value, grip):
                return bool(identity.bridge and cls._same_signed_value(identity, value, grip))

            def _r25_route_provenance(self, report):
                """Return effective R25 route states and whether their OR is complete.

                The reconstruction includes the outer continuation route and the
                pickup-completion ownership rule, not merely PickupCompletionMixin's
                `_non_yaw_report`.  A mismatch with R25's actual output is uncertainty,
                so suppression is forbidden for that sample.
                """
                try:
                    packet = self._shared
                    normal = self.normal_model
                    light = self.light_witness.state
                    detached = self.detached.evidence
                    continuation = self.continuation
                    pickup_completion = bool(self.pick_completion_active)
                    routes = {
                        "r8_base": bool(self.r8_base_detect),
                        "r8_extra": bool(self.r8_extra),
                        "directional": bool(packet.detected),
                        "normal": bool(normal is not None and normal.detected),
                        "horizontal": bool(self.horizontal_detected),
                        "yaw": bool(self.yaw_active and not pickup_completion),
                        "light": bool(light.active),
                        "detached": bool(detached.active),
                        "pickup_hold": bool(self.pick_event_active and not pickup_completion),
                        "continuation": bool(continuation.bridge),
                    }
                except (AttributeError, TypeError, ValueError):
                    return {}, False
                return routes, bool(any(routes.values()) == bool(report))

            def update(self, t, th, thd, tau_m, tau_s, grip):
                residual, report = super().update(t, th, thd, tau_m, tau_s, grip)
                yaw_now = bool(self.yaw_active)
                blocked = bool(getattr(self, "pick_completion_active", False))
                normal = getattr(self, "normal_model", None)
                support = (None if normal is None or not normal.last_active
                           else normal.last_error)
                identity = self.yaw_acquisition_identity

                reacquired = bool(
                    yaw_now
                    and not self._yaw_acquisition_previous_active
                    and self._yaw_acquisition_gap_certified
                    and not blocked
                    and self._same_signed_source(identity, self.yaw_value, grip)
                )
                if self.yaw_reacquisition_consumed and yaw_now:
                    if blocked or not self._same_signed_value(identity, self.yaw_value, grip):
                        self.yaw_reacquisition_consumed = False
                elif not yaw_now:
                    self.yaw_reacquisition_consumed = False
                if reacquired:
                    self.yaw_reacquisition_consumed = True
                    self.yaw_reacquisition_count += 1

                bridge = identity.feed(yaw_now, self.yaw_value, support, grip, blocked=blocked)
                self._yaw_acquisition_gap_certified = bool(not yaw_now and bridge)
                self.yaw_acquisition_episode_active = bool(identity.active)
                self._yaw_acquisition_previous_active = yaw_now

                routes, complete = self._r25_route_provenance(report)
                self.yaw_reacquisition_provenance_complete = complete
                independent = bool(any(value for name, value in routes.items() if name != "yaw"))
                consume = bool(
                    self.yaw_reacquisition_consumed
                    and complete
                    and routes.get("yaw", False)
                    and not independent
                )
                return residual, bool(report and not consume)


        class Detector(YawReacquisitionMixin, R25):
            """Frozen R25 with signed same-event yaw reacquisition ownership."""
        return _SubmissionNamespace(Detector=Detector)
    _r26_yaw_acquisition_candidate = _make_r26_yaw_acquisition_candidate()

    # Source: experiments/r29_detached_release/candidate.py
    def _make_r29_detached_release_candidate():
        """Expose return to normal only for an already-delivered, detached-only force event."""
        import math
        Tangent = _r22_tangent_frozen.Detector
        C3 = _r26_yaw_acquisition_candidate.Detector


        class Endpoint:
            """The acquired long-window route and its short input must both certify low force."""

            def __init__(self):
                self.low = 0
                self.published = False

            def update(self, report, detached, vector, bound, mean_quiet, sole):
                if not detached:
                    self.__init__()
                    return bool(report)
                try:
                    x, y = vector
                    finite = all(math.isfinite(v) for v in (x, y, bound)) and bound > 0
                    inside = finite and math.hypot(x, y) < .8 * bound
                except (TypeError, ValueError):
                    inside = False
                self.low = self.low + 1 if inside else 0
                retire = self.published and sole and mean_quiet and self.low >= 3
                answer = bool(report and not retire)
                self.published |= answer
                return answer


        class Detector(C3):
            def reset(self):
                super().reset()
                self.detached_endpoint = Endpoint()

            def update(self, t, th, thd, tau_m, tau_s, grip):
                residual, native = super().update(t, th, thd, tau_m, tau_s, grip)
                routes, complete = self._r25_route_provenance(native)
                witness = self.detached
                detached = bool(not grip and witness.evidence.active and
                                (not witness.require_onset or witness.gate.accepted))
                sole = complete and routes.get('detached', False) and not any(
                    live for route, live in routes.items() if route != 'detached')
                report = self.detached_endpoint.update(native, detached, witness.vector,
                            Tangent.bounds_tangent[0], witness.evidence.quiet > 0, sole)
                return residual, report
        return _SubmissionNamespace(Detector=Detector, Endpoint=Endpoint)
    _r29_detached_release_candidate = _make_r29_detached_release_candidate()

    # Source: experiments/r27_load_change/projection.py
    def _make_r27_load_change_projection():
        """Read the incumbent's sensor integration window; project it using private nominal scratch."""
        import mujoco
        import numpy as np

        KERNEL = _r15_delayed_wrench_observer.KERNEL; N = _r15_delayed_wrench_observer.N
        from sim.nominal import Nominal


        class Projection:
            def __init__(self,mass,com):
                self.nom=Nominal()
                self.nom.set_payload(float(mass),tuple(com))
                self.jp=np.empty((3,6));self.jw=np.empty((3,6))
                self.site=self.nom.m.site('tcp').id

            def from_integrated(self, detector):
                # C3 intentionally skips its own Jacobian projection when no native signal is active.
                witness=detector.witness
                if (witness is None or witness.bias is None or len(witness.residual)!=N
                        or len(witness.midpos)!=N or not witness.prev_grip):
                    return None
                self.nom.d.qpos[:]=witness.midpos[N//2]
                mujoco.mj_kinematics(self.nom.m,self.nom.d);mujoco.mj_comPos(self.nom.m,self.nom.d)
                mujoco.mj_jacSite(self.nom.m,self.nom.d,self.jp,self.jw,self.site)
                jac=np.vstack((self.jp,self.jw))
                if np.linalg.cond(jac)>1e4:return None
                return np.linalg.solve(jac.T,KERNEL@np.asarray(witness.residual)-witness.bias)[:2]
        return _SubmissionNamespace(Projection=Projection)
    _r27_load_change_projection = _make_r27_load_change_projection()

    # Source: experiments/r27_load_change/manifold.py
    def _make_r27_load_change_manifold():
        """Only add a settled notification if two measured forces also exclude the local normal set."""
        import json


        Base = _r26_yaw_acquisition_candidate.Detector
        Coordinates = _r27_load_change_manifold_event.Coordinates; FeasibleNormals = _r27_load_change_manifold_event.FeasibleNormals
        Projection = _r27_load_change_projection.Projection
        Settled = _r27_load_change_settled_event.Settled; channels = _r27_load_change_settled_event.channels


        class Detector(Base):
            def reset(self):
                super().reset()
                # The two calibrated fields below were embedded at packaging.
                self.r27_projection=Projection(self.box_mass,self.box_com)
                self.r27_gate=Settled(json.loads(_normal_table_r27_load_change_settled_model))
                self.r27_coords=Coordinates()
                self.r27_normals=FeasibleNormals(json.loads(_normal_table_r27_load_change_manifold_model))
                self.r27_certificate=False

            def update(self,t,th,thd,tau_m,tau_s,grip):
                residual,native=super().update(t,th,thd,tau_m,tau_s,grip)
                pair=channels(self,grip)
                coords=self.r27_coords.feed(th,pair is not None)
                extra_or_native=self.r27_gate.feed(pair,native)
                if self.r27_gate.notified:
                    # The exact two 40-tick mean world forces on which the notified plateau is based.
                    levels=[window.b/len(window.present) for window in self.r27_gate.windows.windows]
                    self.r27_certificate=self.r27_normals.unexpected(coords,levels,self.r27_gate.axis)
                elif not extra_or_native or native:
                    self.r27_certificate=False
                return residual,bool(native or (self.r27_certificate and extra_or_native))
        return _SubmissionNamespace(Base=Base, Detector=Detector)
    _r27_load_change_manifold = _make_r27_load_change_manifold()

    # Source: experiments/r29_native_event_link/candidate.py
    def _make_r29_native_event_link_candidate():
        """Link a pose-certified optional notice to the exact same continuous C3 route."""


        import numpy as np

        manifold = _r27_load_change_manifold
        DIRECTIONS = _r27_load_change_manifold_event.DIRECTIONS
        channels = _r27_load_change_settled_event.channels


        LINKED_NATIVE_ROUTES = frozenset(("r8_base", "normal"))


        class ContinuousForceLink:
            """Measured force identity with fail-open native-route provenance."""

            def __init__(self):
                self.clear()

            def clear(self):
                self.active = False
                self.native_linked = False
                self.grip = False
                self.axis = None
                self.reference = None
                self.peak = None

            def start(self, axis, reference, peak, grip):
                reference = np.asarray(reference, float)
                peak = np.asarray(peak, float)
                if (axis not in range(len(DIRECTIONS)) or reference.shape != (2,) or peak.shape != (2,)
                        or not np.isfinite(reference).all() or not np.isfinite(peak).all()
                        or np.any(peak <= 0.0) or not grip):
                    self.clear()
                    return False
                self.active = True
                self.native_linked = False
                self.grip = bool(grip)
                self.axis = int(axis)
                self.reference = reference.copy()
                self.peak = peak.copy()
                return True

            def observe(self, levels, grip):
                if not self.active or bool(grip) != self.grip:
                    self.clear()
                    return False
                levels = np.asarray(levels, float)
                if levels.shape != (2, 2) or not np.isfinite(levels).all():
                    self.clear()
                    return False
                value = levels @ DIRECTIONS[self.axis] - self.reference
                if np.any(2.0 * value <= self.peak):
                    self.clear()
                    return False
                return True

            def consume(self, native, routes, complete):
                if not native:
                    return False
                active_routes = frozenset(name for name, value in routes.items() if value)
                if not self.active or not complete:
                    self.clear()
                    return False
                if not self.native_linked:
                    if active_routes != LINKED_NATIVE_ROUTES:
                        self.clear()
                        return False
                    self.native_linked = True
                elif not active_routes or not active_routes <= LINKED_NATIVE_ROUTES:
                    self.clear()
                    return False
                return True


        class Detector(manifold.Detector):
            """R27 pose certificate plus conservative continuous-force/native identity."""

            def reset(self):
                super().reset()
                self.r29_link = ContinuousForceLink()
                self.r29_optional_starts = 0
                self.r29_native_links = 0
                self.r29_consumed_samples = 0
                self.r29_native_consumed = False
                self.r29_routes_complete = True
                self.r29_active_routes = ()

            @staticmethod
            def _levels(gate):
                values = []
                for window in gate.windows.windows:
                    if not window.present:
                        return None
                    values.append(window.b / len(window.present))
                return np.asarray(values, float)

            def update(self, t, th, thd, tau_m, tau_s, grip):
                residual, native = manifold.Base.update(self, t, th, thd, tau_m, tau_s, grip)
                pair = channels(self, grip)
                coords = self.r27_coords.feed(th, pair is not None)
                extra_or_native = self.r27_gate.feed(pair, native)
                levels = self._levels(self.r27_gate)
                if self.r27_gate.notified:
                    self.r27_certificate = self.r27_normals.unexpected(
                        coords, levels, self.r27_gate.axis)
                    if self.r27_certificate and self.r29_link.start(
                            self.r27_gate.axis, self.r27_gate.reference,
                            self.r27_gate.peak, grip):
                        self.r29_optional_starts += 1
                elif not extra_or_native or native:
                    self.r27_certificate = False

                continuous = self.r29_link.observe(levels, grip)
                routes, complete = self._r25_route_provenance(native)
                self.r29_routes_complete = complete
                self.r29_active_routes = tuple(sorted(name for name, value in routes.items() if value))
                was_linked = self.r29_link.native_linked
                consume = bool(continuous and self.r29_link.consume(native, routes, complete))
                if consume and not was_linked:
                    self.r29_native_links += 1
                self.r29_native_consumed = consume
                self.r29_consumed_samples += int(consume)
                optional = bool(self.r27_certificate and extra_or_native)
                return residual, bool(optional or (native and not consume))
        return _SubmissionNamespace(Detector=Detector)
    _r29_native_event_link_candidate = _make_r29_native_event_link_candidate()

    # Source: experiments/r29_disjoint_events/candidate.py
    def _make_r29_disjoint_events_candidate():
        """Compose two mechanisms with disjoint measured-gripper ownership; no new gate or threshold."""
        Detached = _r29_detached_release_candidate.Detector
        Held = _r29_native_event_link_candidate.Detector


        class Detector(Detached, Held):
            """Held calls the physical C3 update once. Detached can retire only an already
            delivered native event with the gripper open. With the gripper open the held
            observer returns no pair, resets, and passes the native result unchanged.
            """
        return _SubmissionNamespace(Detector=Detector)
    _r29_disjoint_events_candidate = _make_r29_disjoint_events_candidate()

    # Source: experiments/r29_loaded_rising_certificate/candidate.py
    def _make_r29_loaded_rising_certificate_candidate():
        """Publish one pose-certified notice while the existing two-force excursion rises."""


        import numpy as np

        DIRECTIONS = _r27_load_change_event.DIRECTIONS; WIDTH = _r27_load_change_event.WIDTH
        Coordinates = _r27_load_change_manifold_event.Coordinates
        channels = _r27_load_change_settled_event.channels
        Composed = _r29_disjoint_events_candidate.Detector


        NATIVE_NOTICE_SAMPLES = 4  # The inherited native decision width, not a new duration threshold.


        def rising_means(gate):
            """Return the exact adjacent 40-sample means already owned by ``Settled``."""
            pairs = []
            for window in gate.windows.windows:
                if len(window.past) != WIDTH or len(window.present) != WIDTH:
                    return None
                before = window.a / WIDTH
                now = window.b / WIDTH
                if (np.asarray(before).shape != (2,) or np.asarray(now).shape != (2,)
                        or not np.isfinite(before).all() or not np.isfinite(now).all()):
                    return None
                pairs.append((before, now))
            return np.asarray([pair[0] for pair in pairs]), np.asarray([pair[1] for pair in pairs])


        def rising_certificate(gate, normals, coords, levels, grip, native_active, routes_complete):
            """No threshold: both measured levels must exceed the frozen local normal set
            while both existing signed 40-sample changes are still strictly increasing.
            """
            if (not grip or native_active or not routes_complete or gate.state != "rising"
                    or gate.axis is None or gate.native_seen):
                return False
            means = rising_means(gate)
            if means is None or levels is None:
                return False
            before, now = means
            axis = int(gate.axis)
            if axis not in range(len(DIRECTIONS)):
                return False
            signed_change = (now - before) @ DIRECTIONS[axis]
            return bool(np.all(signed_change > 0.0)
                        and normals.unexpected(coords, levels, axis))


        class Detector(Composed):
            """Keep the linked and detached mechanisms; move only an optional notice.

            ``super().update`` is called exactly once.  The added observer reads the two
            existing means after that call.  It never changes the underlying C3 update.
            """

            def reset(self):
                super().reset()
                self.r29_rising_coords = Coordinates()
                self.r29_rising_announced = False
                self.r29_rising_signal_left = 0
                self.r29_rising_starts = 0
                self.r29_rising_start_times = []
                self.r29_rising_axes = []
                self.r29_plateau_samples_suppressed = 0

            def update(self, t, th, thd, tau_m, tau_s, grip):
                residual, report = super().update(t, th, thd, tau_m, tau_s, grip)

                pair = channels(self, grip)
                coords = self.r29_rising_coords.feed(th, pair is not None)
                levels = self._levels(self.r27_gate)
                native_active = bool(self.r29_active_routes or self.r29_native_consumed)

                if self.r27_gate.state == "quiet":
                    self.r29_rising_announced = False
                    self.r29_rising_signal_left = 0

                if (self.r29_rising_announced and self.r27_gate.state == "settled"
                        and self.r27_certificate and report and not native_active):
                    # The same excursion was already announced while rising.  Suppress
                    # only its later optional plateau pulse; a native route is invariant.
                    report = False
                    self.r29_plateau_samples_suppressed += 1

                if (not self.r29_rising_announced
                        and rising_certificate(self.r27_gate, self.r27_normals, coords,
                                               levels, grip, native_active,
                                               self.r29_routes_complete)
                        and self.r29_link.start(self.r27_gate.axis,
                                                self.r27_gate.reference,
                                                self.r27_gate.peak, grip)):
                    self.r29_rising_announced = True
                    self.r29_rising_signal_left = NATIVE_NOTICE_SAMPLES
                    self.r29_rising_starts += 1
                    self.r29_rising_start_times.append(float(t))
                    self.r29_rising_axes.append(int(self.r27_gate.axis))

                active = self.r29_rising_signal_left > 0
                if active:
                    self.r29_rising_signal_left -= 1
                return residual, bool(report or active)
        return _SubmissionNamespace(NATIVE_NOTICE_SAMPLES=NATIVE_NOTICE_SAMPLES, rising_certificate=rising_certificate, rising_means=rising_means)
    _r29_loaded_rising_certificate_candidate = _make_r29_loaded_rising_certificate_candidate()

    # Source: experiments/r30_detached_lifetime/candidate.py
    def _make_r30_detached_lifetime_candidate():
        """A completed detached notice cannot reopen in the same uninterrupted native acquisition."""
        import math
        Released = _r29_detached_release_candidate.Detector; Endpoint = _r29_detached_release_candidate.Endpoint


        class AcquisitionEndpoint(Endpoint):
            def __init__(self):
                super().__init__()
                self.retired = False

            def update(self, report, detached, vector, bound, mean_quiet, sole):
                answer = super().update(report, detached, vector, bound, mean_quiet, sole)
                # The parent resets through self.__init__ when the witness drops. Its next
                # acquisition is a different event even one tick later. No dead-time is imposed.
                if not detached or not sole or not report:
                    return answer
                try:
                    x, y = vector
                    if not (math.isfinite(x) and math.isfinite(y) and math.isfinite(bound) and bound > 0):
                        return answer
                except (TypeError,ValueError):
                    return answer
                if not answer and self.published:
                    self.retired = True
                return bool(answer and not self.retired)


        class Detector(Released):
            def reset(self):
                super().reset()
                self.detached_endpoint = AcquisitionEndpoint()
        return _SubmissionNamespace(AcquisitionEndpoint=AcquisitionEndpoint)
    _r30_detached_lifetime_candidate = _make_r30_detached_lifetime_candidate()

    # Source: experiments/r30_source_association/candidate.py
    def _make_r30_source_association_candidate():
        """Assign the first optional notice by measured yaw observability; never mask a new yaw rise."""


        import math
        import numpy as np
        Coordinates = _r27_load_change_manifold_event.Coordinates
        Composed = _r29_disjoint_events_candidate.Detector
        NATIVE_NOTICE_SAMPLES = _r29_loaded_rising_certificate_candidate.NATIVE_NOTICE_SAMPLES; rising_certificate = _r29_loaded_rising_certificate_candidate.rising_certificate; rising_means = _r29_loaded_rising_certificate_candidate.rising_means


        class MeasuredCoordinates(Coordinates):
            """Expose the exact coordinates already evaluated, without doing another inverse solve."""
            def __init__(self):
                super().__init__()
                self.last = None

            def feed(self, encoder, eligible):
                self.last = super().feed(encoder, eligible)
                return self.last


        def prefer_planar_notice(changes, yaw_column):
            """For both planar innovations require more energy orthogonal to yaw than along it.

            J_xy[:,0] is the translational Jacobian column at the sensor witness pose.
            Its dot product with force is the torque about joint 1. The scale of this
            column cancels in the comparison. Ties, opposing normals and unknowns use
            the existing detector/plateau, not an early notice.
            """
            if changes is None or yaw_column is None:return False
            delta = np.asarray(changes, dtype=float)
            col = np.asarray(yaw_column, dtype=float)
            if delta.shape != (2,2) or col.shape != (2,) or not np.isfinite(delta).all() or not np.isfinite(col).all():
                return False
            cx, cy = map(float, col)
            if not math.hypot(cx, cy) > 0:return False
            along = delta[:,0]*cx + delta[:,1]*cy
            across = -delta[:,0]*cy + delta[:,1]*cx
            return bool(np.isfinite(along).all() and np.isfinite(across).all()
                        and np.all(np.abs(across)>np.abs(along)) and across[0]*across[1]>0)


        class Detector(Composed):
            def reset(self):
                super().reset()
                self.r27_coords = MeasuredCoordinates()
                self.r30_owner = None
                self.r30_notice_left = 0
                self.r30_optional_starts = 0
                self.r30_deferred = 0

            def update(self, t, th, thd, tau_m, tau_s, grip):
                residual, reported = super().update(t, th, thd, tau_m, tau_s, grip)
                gate = self.r27_gate
                if gate.state == 'quiet':
                    self.r30_owner = None
                    self.r30_notice_left = 0
                native = bool(self.r29_active_routes or self.r29_native_consumed)
                if (self.r30_owner == 'planar' and gate.state == 'settled'
                        and self.r27_certificate and reported and not native):
                    # Only our redundant optional plateau is consumed. Native yaw is never suppressed here.
                    reported = False

                if self.r30_owner is None and gate.state == 'rising' and not native:
                    levels = self._levels(gate)
                    if rising_certificate(gate, self.r27_normals, self.r27_coords.last, levels, grip,
                                          native, self.r29_routes_complete):
                        before, now = rising_means(gate)
                        if (prefer_planar_notice(now-before, self.r27_projection.jp[:2,0])
                                and self.r29_link.start(gate.axis, gate.reference, gate.peak, grip)):
                            self.r30_owner = 'planar'
                            self.r30_notice_left = NATIVE_NOTICE_SAMPLES
                            self.r30_optional_starts += 1
                        else:
                            self.r30_owner = 'incumbent'
                            self.r30_deferred += 1
                optional = self.r30_notice_left > 0
                if optional:self.r30_notice_left -= 1
                return residual, bool(reported or optional)
        return _SubmissionNamespace(MeasuredCoordinates=MeasuredCoordinates, prefer_planar_notice=prefer_planar_notice)
    _r30_source_association_candidate = _make_r30_source_association_candidate()

    # Source: experiments/r30_yaw_headroom/candidate.py
    def _make_r30_yaw_headroom_candidate():
        """Only pre-announce a force when measured yaw has room for the pending signed innovation."""
        import math
        import numpy as np
        Composed = _r29_disjoint_events_candidate.Detector
        NATIVE_NOTICE_SAMPLES = _r29_loaded_rising_certificate_candidate.NATIVE_NOTICE_SAMPLES; rising_certificate = _r29_loaded_rising_certificate_candidate.rising_certificate; rising_means = _r29_loaded_rising_certificate_candidate.rising_means
        MeasuredCoordinates = _r30_source_association_candidate.MeasuredCoordinates; prefer_planar_notice = _r30_source_association_candidate.prefer_planar_notice


        def has_yaw_reserve(yaw, bound, force_changes, jacobian_column):
            """Guard a lead notice, never the native yaw signal.

            Check the present measured moment and both continuations obtained by
            adding the measured planar force change projected on joint 1. Fractions
            of each change lie between those endpoints. This is a first-order screen,
            not a bound on an arbitrary future collision or an exact delay model.
            """
            if yaw is None or bound is None or not all(math.isfinite(x) for x in (yaw,bound)) or bound<=0:
                return False
            if not prefer_planar_notice(force_changes,jacobian_column):
                return False
            projected=np.asarray(force_changes,float)@np.asarray(jacobian_column,float)
            predicted=float(yaw)+projected
            return bool(abs(float(yaw))<bound and np.isfinite(predicted).all() and np.all(np.abs(predicted)<bound))


        class Detector(Composed):
            def reset(self):
                super().reset()
                self.r27_coords=MeasuredCoordinates()
                self.pending_owner=None
                self.pending_width=0
                self.planar_notices=self.deferred_notices=0

            def update(self,t,th,thd,tau_m,tau_s,grip):
                residual,reported=super().update(t,th,thd,tau_m,tau_s,grip)
                gate=self.r27_gate
                if gate.state=='quiet':
                    self.pending_owner=None
                    self.pending_width=0
                native=bool(self.r29_active_routes or self.r29_native_consumed)
                if self.pending_owner=='planar' and gate.state=='settled' and self.r27_certificate and not native:
                    # This optional plateau belongs to the earlier notice; native yaw is unchanged.
                    reported=False
                if self.pending_owner is None and gate.state=='rising' and not native:
                    levels=self._levels(gate)
                    if rising_certificate(gate,self.r27_normals,self.r27_coords.last,levels,grip,native,self.r29_routes_complete):
                        before,now=rising_means(gate)
                        if (has_yaw_reserve(self.yaw_value,self.bounds[int(bool(grip))],now-before,self.r27_projection.jp[:2,0])
                                and self.r29_link.start(gate.axis,gate.reference,gate.peak,grip)):
                            self.pending_owner='planar'
                            self.pending_width=NATIVE_NOTICE_SAMPLES
                            self.planar_notices+=1
                        else:
                            self.pending_owner='incumbent'
                            self.deferred_notices+=1
                optional=self.pending_width>0
                if optional:self.pending_width-=1
                return residual,bool(reported or optional)
        return _SubmissionNamespace(Detector=Detector)
    _r30_yaw_headroom_candidate = _make_r30_yaw_headroom_candidate()

    # Source: experiments/r31_held_identity/candidate.py
    def _make_r31_held_identity_candidate():
        """Assign individual native acquisitions to a previously announced measured force, not to a clock."""
        import math

        import mujoco
        import numpy as np

        KERNEL = _r15_delayed_wrench_observer.KERNEL; N = _r15_delayed_wrench_observer.N
        Changes = _r27_load_change_event.Changes
        rising_means = _r29_loaded_rising_certificate_candidate.rising_means
        AcquisitionEndpoint = _r30_detached_lifetime_candidate.AcquisitionEndpoint
        Headroom = _r30_yaw_headroom_candidate.Detector


        class SpatialProjection:
            """Return the identical xy solve used by the parent while retaining its already-computed z.

            The nominal model and scratch arrays are shared with the original projection.
            One Jacobian/SVD/solve is evaluated, never a second plant integration.
            """
            def __init__(self, previous):
                self.nom,self.jp,self.jw,self.site=previous.nom,previous.jp,previous.jw,previous.site
                self.last_force=None

            def from_integrated(self, detector):
                self.last_force=None
                witness=detector.witness
                if (witness is None or witness.bias is None or len(witness.residual)!=N
                        or len(witness.midpos)!=N or not witness.prev_grip):
                    return None
                self.nom.d.qpos[:]=witness.midpos[N//2]
                mujoco.mj_kinematics(self.nom.m,self.nom.d);mujoco.mj_comPos(self.nom.m,self.nom.d)
                mujoco.mj_jacSite(self.nom.m,self.nom.d,self.jp,self.jw,self.site)
                jac=np.vstack((self.jp,self.jw))
                if np.linalg.cond(jac)>1e4:return None
                self.last_force=np.linalg.solve(jac.T,KERNEL@np.asarray(witness.residual)-witness.bias)[:3]
                return self.last_force[:2]


        class Source:
            """A horizontal-force ray measured independently in two channels in a single gate generation."""
            def __init__(self, generation, past, current, yaw):
                self.generation=generation; self.before=np.array(past,float)
                delta=np.asarray(current)-self.before
                self.size=np.linalg.norm(delta[:,:2],axis=1)
                self.rays=delta[:,:2]/self.size[:,None]
                self.yaw=yaw

            @classmethod
            def start(cls,generation,past,now,yaw):
                if generation is None or yaw is None or not math.isfinite(yaw) or past is None or now is None:
                    return None
                a,b=np.asarray(past),np.asarray(now)
                if (a.shape!=(2,3) or b.shape!=(2,3) or not np.isfinite(a).all() or not np.isfinite(b).all()
                        or not np.all(np.linalg.norm(b[:,:2]-a[:,:2],axis=1)>0)):
                    return None
                return cls(generation,a,b,yaw)

            def coordinates(self,levels):
                if levels is None:return None
                levels=np.asarray(levels)
                if levels.shape!=(2,3) or not np.isfinite(levels).all():return None
                delta=levels-self.before
                along=np.sum(self.rays*delta[:,:2],axis=1)
                side=self.rays[:,0]*delta[:,1]-self.rays[:,1]*delta[:,0]
                perpendicular=np.hypot(side,delta[:,2])
                return delta,along,perpendicular,side

            def agrees(self,generation,levels,routes,yaw,column):
                if (self.generation is not generation or not routes or not set(routes) <= {'yaw','normal','directional','r8_base'}
                        or not ({'yaw','normal'}&set(routes))):
                    return False
                data=self.coordinates(levels)
                if data is None:return False
                delta,along,off,_=data
                # Both sensors must remain closer to the originally measured force than to zero,
                # and the common ray must explain more than all transverse force together.
                if not np.all((2*along>self.size)&(along>off)):
                    return False
                if 'yaw' in routes:
                    col=np.asarray(column)
                    if col.shape!=(2,) or not np.isfinite(col).all() or yaw is None or not math.isfinite(yaw):
                        return False
                    # Generalized joint-1 momentum must move with both measured planar projections.
                    if not np.all((float(yaw)-self.yaw)*(delta[:,:2]@col)>0):
                        return False
                return True

            def distinct_load(self,levels):
                data=self.coordinates(levels)
                if data is None:return True   # missing compatibility information fails open
                delta,along,off,side=data
                # Merely returning to zero along the original ray is not a new positive
                # excitation. Opposing or perpendicular displacement has to exceed the
                # originally certified onset and agree between the two instruments.
                return bool(np.all(-along>self.size) or (np.all(off>self.size) and np.all(off>np.abs(along))
                            and float(side[0]*side[1]+delta[0,2]*delta[1,2])>0))


        class NativeAcquisitions:
            """Never consume an already exposed route or an unrelated newly active route."""
            def __init__(self):
                self.previous=set()
                self.accounted=set()

            def feed(self,routes,complete,grip,generation,levels,token,yaw,column):
                live=set(routes); new=live-self.previous
                self.previous=live
                self.accounted.intersection_update(live)
                if not complete or not grip or (token is not None and levels is None):
                    self.accounted.clear()
                    return False
                if token is None:
                    return False
                if self.accounted and (generation is not None and generation is not token.generation
                                       or token.distinct_load(levels)):
                    self.accounted.clear()
                if new and token.agrees(generation,levels,live,yaw,column):
                    self.accounted.update(new)
                return bool(live and live<=self.accounted)


        class Detector(Headroom):
            def reset(self):
                super().reset()
                self.detached_endpoint=AcquisitionEndpoint()
                self.r27_projection=SpatialProjection(self.r27_projection)
                self.z_window=Changes()
                self.source=None
                self.native_acquisitions=NativeAcquisitions()
                self.associated_native_samples=0

            def update(self,t,th,thd,tm,ts,grip):
                count=self.planar_notices;width=self.pending_width
                residual,reported=super().update(t,th,thd,tm,ts,grip)
                optional=bool(width or self.planar_notices>count)
                gate=self.r27_gate;projection=self.r27_projection
                usable=bool(projection.last_force is not None and all(w.present for w in gate.windows.windows))
                z=self.z_window.feed([self._shared.last_residual_wrench[2],projection.last_force[2]] if usable else None)
                means=rising_means(gate) if z is not None and (self.source is not None or count<self.planar_notices) else None
                past=levels=None
                if means is not None:
                    past=np.column_stack((means[0],z[0]));levels=np.column_stack((means[1],z[1]))
                if self.planar_notices>count:
                    self.source=Source.start(gate.reference,past,levels,self.yaw_value)
                associated=self.native_acquisitions.feed(self.r29_active_routes,self.r29_routes_complete,grip,gate.reference,
                                                         levels,self.source,self.yaw_value,projection.jp[:2,0])
                if associated:
                    self.associated_native_samples+=1
                    reported=optional
                elif self.source is not None and self.r29_active_routes and self.r29_routes_complete and grip:
                    # The older broad link may have elided this raw native rise. With an
                    # optional ownership claim we must expose uncertain/new raw sources.
                    reported=True
                if (not grip or (self.source is not None and gate.reference is not self.source.generation
                                 and not self.native_acquisitions.accounted)):
                    self.source=None
                return residual,bool(reported)
        return _SubmissionNamespace(Detector=Detector)
    _r31_held_identity_candidate = _make_r31_held_identity_candidate()

    return _r31_held_identity_candidate.Detector



# --- Fixed, authored R32 servo inference (loaded only with the supplied servo controller). ---
@_submission_cache(maxsize=1)
def _submission_servo_class():
    # Source: experiments/r15_delayed_wrench/observer.py
    def _make_r15_delayed_wrench_observer():
        """Causal, delayed virtual vertical force for the course's nominal 6-axis arm.

        Uses only motor encoder, joint-torque sensor, grip and the two public nominal
        models. It is an estimate of *all* vertical external wrench, not a contact
        classifier. The delay is part of the output contract. No force clipping.
        """
        from collections import deque
        from dataclasses import dataclass

        import numpy as np


        N = 31
        # Encoder is a nodal state. Rigid-axis JTS at that state was formed from
        # the previous solved interval. f_table[j] is logged after solving step j.
        STATE_LAG = 2 + N // 2
        LAG = STATE_LAG + 1
        X = np.linspace(-1., 1., N)
        KERNEL = (1. - X * X) ** 2
        KERNEL /= KERNEL.sum()


        @dataclass(frozen=True)
        class Sample:
            time: float
            effective_time: float
            force_n: np.ndarray


        def vertical_displacement(model, q):
            """Virtual displacement that admits vertical force and rejects other wrenches."""
            import mujoco
            model.d.qpos[:] = q
            mujoco.mj_kinematics(model.m, model.d)
            mujoco.mj_comPos(model.m, model.d)
            jp, jw = np.zeros((3, 6)), np.zeros((3, 6))
            mujoco.mj_jacSite(model.m, model.d, jp, jw, model.m.site('tcp').id)
            jac = np.vstack((jp, jw))
            if np.linalg.cond(jac) > 1e4:
                return None
            return np.linalg.solve(jac, [0., 0., 1., 0., 0., 0.])


        class NormalForce:
            """A fresh instance per episode, owning its nominal workspaces."""
            def __init__(self, empty, loaded, *, dt, viscous_spring, jts_hz):
                if not np.isclose(dt, .001, rtol=0, atol=1e-12):
                    raise ValueError('this design requires 1 ms samples')
                self.empty, self.loaded, self.dt = empty, loaded, dt
                self.elastic = np.asarray(empty.elastic, bool)
                self.k = np.asarray(empty.K)[self.elastic]
                d = np.asarray(viscous_spring)[self.elastic]
                if np.any(self.k <= 0) or np.any(d <= 0) or jts_hz <= 0:
                    raise ValueError('strictly positive elastic and sensor parameters required')
                self.spring_memory = np.exp(-dt * self.k / d)
                self.unfilter = 1. / np.expm1(2. * np.pi * jts_hz * dt)
                self.prev_th = self.prev_sensor = self.deflection = None
                self.clock = self.prev_grip = None
                self.pos = deque(maxlen=3)
                self.residual = deque(maxlen=N)
                self.midpos = deque(maxlen=N)
                self.offsets = []
                self.bias = None

            def update(self, t, encoder, jts, grip, *, project=True):
                """Return Sample or None (no full window/calibration, singular, or no load).

                Call on every regular sample. project=False skips only the output
                projection; it neither skips samples nor changes the integration.
                """
                th, raw = np.asarray(encoder, float), np.asarray(jts, float)
                if (not np.isfinite(t) or th.shape != (6,) or raw.shape != (6,)
                        or not np.isfinite(th).all() or not np.isfinite(raw).all()):
                    raise ValueError('finite time and two finite six-axis signals required')
                if self.clock is not None and abs(t - self.clock - self.dt) > 1e-8:
                    raise ValueError('a dropped or reordered sample requires a new observer')
                self.clock = t
                # A change of payload invalidates every differentiation/averaging stencil
                # that straddles it. Bias belongs to the unloaded arm, not the payload.
                if self.prev_grip != bool(grip):
                    self.pos.clear()
                    self.residual.clear()
                    self.midpos.clear()
                    self.prev_th = self.prev_sensor = self.deflection = None
                self.prev_grip = bool(grip)
                if self.prev_th is None:
                    self.prev_th, self.prev_sensor = th.copy(), raw.copy()
                    return None
                aligned = raw + self.unfilter * (raw - self.prev_sensor)
                previous = self.prev_th.copy()
                self.prev_th, self.prev_sensor = th.copy(), raw.copy()
                equilibrium = aligned[self.elastic] / self.k
                self.deflection = (equilibrium if self.deflection is None else
                                   equilibrium + self.spring_memory * (self.deflection - equilibrium))
                previous[self.elastic] -= self.deflection
                self.pos.append((previous, aligned))
                if len(self.pos) < 3:
                    return None
                (qa, _), (q, tau), (qb, _) = self.pos
                v = (qb - qa) / (2. * self.dt)
                acc = ((qb - q) - (q - qa)) / self.dt**2
                model = self.loaded if grip else self.empty
                self.residual.append(model.inverse_dynamics(q, v, acc) - tau)
                self.midpos.append(q)
                if len(self.residual) != N:
                    return None
                state_time = t - STATE_LAG * self.dt
                # MuJoCo's nominal inverse dynamics is affine in acceleration. Integrate
                # the residual before projection; endpoints and slope of the kernel
                # vanish, so isolated quantization differences are not used as force.
                averaged = KERNEL @ np.asarray(self.residual)
                if self.bias is None:
                    if not grip and .075 <= state_time < .425:
                        self.offsets.append(averaged)
                    elif state_time >= .425 and len(self.offsets) >= 300:
                        self.bias = np.median(self.offsets, axis=0)
                        self.offsets.clear()
                if not project or self.bias is None:
                    return None
                import mujoco
                model.d.qpos[:] = self.midpos[N // 2]
                mujoco.mj_kinematics(model.m, model.d);mujoco.mj_comPos(model.m,model.d)
                jp,jw=np.zeros((3,6)),np.zeros((3,6));mujoco.mj_jacSite(model.m,model.d,jp,jw,model.m.site("tcp").id);jac=np.vstack((jp,jw))
                if np.linalg.cond(jac)>1e4:return None
                wrench=np.linalg.solve(jac.T, averaged-self.bias)
                return Sample(float(t),float(t-LAG*self.dt),wrench[:3].copy())
        return _SubmissionNamespace(N=N, NormalForce=NormalForce)
    _r15_delayed_wrench_observer = _make_r15_delayed_wrench_observer()

    # Source: experiments/r31_servo_release/candidate.py
    def _make_r31_servo_release_candidate():
        """Deferred dual-observer release notice composed with unchanged R10 servo detection."""


        import numpy as np

        my_detector = _submission_r10
        NormalForce = _r15_delayed_wrench_observer.NormalForce
        from sim import config as C


        HORIZONTAL_N = 5.0
        DIRECTION_COSINE = 0.8
        PERSISTENCE = 3
        REPORT_SAMPLES = 8


        def coherent_horizontal(force, wrench, table_allowed):
            if table_allowed or force is None:
                return False, None
            force = np.asarray(force, float); wrench = np.asarray(wrench, float)
            if (force.shape != (3,) or wrench.shape[0] < 2
                    or not np.isfinite(force).all() or not np.isfinite(wrench[:2]).all()):
                return False, None
            a, b = force[:2], wrench[:2]
            an, bn = float(np.linalg.norm(a)), float(np.linalg.norm(b))
            cosine = float(a @ b / (an * bn)) if an > 0 and bn > 0 else -1.0
            return bool(an > HORIZONTAL_N and bn > HORIZONTAL_N and cosine >= DIRECTION_COSINE), {
                "delayed_horizontal_N": an, "shared_horizontal_N": bn, "cosine": cosine}


        class DeferredEpisode:
            def __init__(self): self.reset()

            def reset(self):
                self.onset = 0; self.pending = False; self.owned = False
                self.release = 0; self.pulse = 0

            def feed(self, evidence, valid, incumbent):
                if self.pulse:
                    self.pulse -= 1
                    return True
                if self.pending:
                    self.owned |= bool(incumbent)
                    if not valid:
                        return False
                    self.release = 0 if evidence else self.release + 1
                    if self.release < PERSISTENCE:
                        return False
                    report = not self.owned
                    self.onset = 0; self.pending = False; self.owned = False; self.release = 0
                    if report:
                        self.pulse = REPORT_SAMPLES - 1
                        return True
                    return False
                if not valid:
                    self.onset = 0
                    return False
                self.onset = self.onset + 1 if evidence else 0
                if self.onset >= PERSISTENCE:
                    self.pending = True
                    self.owned = bool(incumbent)
                    self.onset = 0
                return False


        class Detector(my_detector.Detector):
            def reset(self):
                super().reset()
                self.release_force = NormalForce(self.nom, self.nom_box, dt=C.DT,
                                                 viscous_spring=C.D, jts_hz=C.JTS_CUTOFF)
                self.release_episode = DeferredEpisode()
                self.release_evidence = False; self.release_features = None; self.release_extra = False

            def update(self, t, th, thd, tau_m, tau_s, grip):
                residual, incumbent = super().update(t, th, thd, tau_m, tau_s, grip)
                sample = self.release_force.update(t, th, tau_s, bool(grip))
                force = None if sample is None else sample.force_n
                self.release_evidence, self.release_features = coherent_horizontal(
                    force, self._shared.last_wrench, self._shared.last_table_allowed)
                self.release_extra = self.release_episode.feed(
                    self.release_evidence, sample is not None, bool(incumbent))
                return residual, bool(incumbent or self.release_extra)
        return _SubmissionNamespace(HORIZONTAL_N=HORIZONTAL_N, PERSISTENCE=PERSISTENCE, REPORT_SAMPLES=REPORT_SAMPLES, coherent_horizontal=coherent_horizontal)
    _r31_servo_release_candidate = _make_r31_servo_release_candidate()

    # Source: experiments/r32_servo_ownership/candidate.py
    def _make_r32_servo_ownership_candidate():
        """Defer weak servo notice and require two settled observer windows before rearm."""
        import numpy as np
        my_detector = _submission_r10
        N = _r15_delayed_wrench_observer.N; NormalForce = _r15_delayed_wrench_observer.NormalForce
        coherent_horizontal = _r31_servo_release_candidate.coherent_horizontal; HORIZONTAL_N = _r31_servo_release_candidate.HORIZONTAL_N; PERSISTENCE = _r31_servo_release_candidate.PERSISTENCE; REPORT_SAMPLES = _r31_servo_release_candidate.REPORT_SAMPLES
        from sim import config as C

        SETTLE_SAMPLES=2*N

        class OwnedSource:
            def __init__(self):self.reset()
            def reset(self):
                self.onset=0;self.pending=False;self.owned=False;self.release=0;self.pulse=0;self.awaiting_settle=False;self.settle=0
            def feed(self,evidence,valid,settled,incumbent):
                if self.awaiting_settle:
                    self.settle=self.settle+1 if valid and settled else 0
                    if self.settle>=SETTLE_SAMPLES:
                        self.awaiting_settle=False;self.settle=0
                if self.pulse:
                    self.pulse-=1;return True
                if self.awaiting_settle:return False
                if self.pending:
                    self.owned|=bool(incumbent)
                    if not valid:return False
                    self.release=0 if evidence else self.release+1
                    if self.release<PERSISTENCE:return False
                    report=not self.owned;self.onset=0;self.pending=False;self.owned=False;self.release=0
                    self.awaiting_settle=True;self.settle=1 if settled else 0
                    if report:self.pulse=REPORT_SAMPLES-1;return True
                    return False
                if not valid:self.onset=0;return False
                self.onset=self.onset+1 if evidence else 0
                if self.onset>=PERSISTENCE:self.pending=True;self.owned=bool(incumbent);self.onset=0
                return False

        class Detector(my_detector.Detector):
            def reset(self):
                super().reset();self.release_force=NormalForce(self.nom,self.nom_box,dt=C.DT,viscous_spring=C.D,jts_hz=C.JTS_CUTOFF)
                self.source_owner=OwnedSource();self.release_extra=False;self.release_evidence=False;self.release_settled=False
            def update(self,t,th,thd,tau_m,tau_s,grip):
                residual,incumbent=super().update(t,th,thd,tau_m,tau_s,grip)
                sample=self.release_force.update(t,th,tau_s,bool(grip));force=None if sample is None else sample.force_n
                self.release_evidence,features=coherent_horizontal(force,self._shared.last_wrench,self._shared.last_table_allowed)
                valid=sample is not None and force is not None and np.isfinite(force).all() and np.isfinite(self._shared.last_wrench[:2]).all()
                self.release_settled=bool(valid and np.linalg.norm(force[:2])<=HORIZONTAL_N and np.linalg.norm(self._shared.last_wrench[:2])<=HORIZONTAL_N)
                self.release_extra=self.source_owner.feed(self.release_evidence,valid,self.release_settled,bool(incumbent))
                return residual,bool(incumbent or self.release_extra)
        return _SubmissionNamespace(Detector=Detector)
    _r32_servo_ownership_candidate = _make_r32_servo_ownership_candidate()

    return _r32_servo_ownership_candidate.Detector


class Detector:
    """Course adapter: R31 with the supplied torque controller, R32 with the supplied
    servo controller, unchanged R10 for any other controller.

    The simulator supplies the controller and nominal models before reset.
    Reset selects one method; each physical sample updates that method once.
    """
    SOURCE = _submission_r10.Detector.SOURCE
    API = _submission_r10.Detector.API

    def __init__(self, nom=None, nom_box=None, box_mass=0.0, box_com=(0.0, 0.0, 0.0)):
        self._impl = None
        self.box_mass = float(box_mass)
        self.box_com = tuple(box_com)
        if nom is not None and nom_box is not None:
            self.nom, self.nom_box = nom, nom_box
            self.reset()

    def reset(self):
        ctrl = vars(self).get('ctrl')
        shared = getattr(ctrl, 'shared_estimator', None)
        contract = (getattr(shared, 'API', None) == self.API
                    and getattr(shared, 'SOURCE', None) == self.SOURCE
                    and callable(getattr(shared, 'update', None))
                    and callable(getattr(shared, 'reset', None)))
        mode = getattr(ctrl, 'MODE', None)
        if contract and mode == 'torque':
            cls, method = _submission_torque_class(), 'r31-torque'
        elif contract and mode == 'servo':
            cls, method = _submission_servo_class(), 'r32-servo'
        else:
            cls, method = _submission_r10.Detector, 'r10'
        engine = cls()
        engine.nom, engine.nom_box, engine.ctrl = self.nom, self.nom_box, ctrl
        engine.box_mass, engine.box_com = self.box_mass, self.box_com
        engine.reset()
        self._impl = engine
        self.selected_method = method
        # No routing, wrapper allocation or second update inside the 1 ms loop.
        self.update = engine.update

    def update(self, t, th, thd, tau_m, tau_s, grip):
        raise RuntimeError('attach course models and call reset() before the first sample')

    def __getattr__(self, name):
        impl = vars(self).get('_impl')
        if impl is None:
            raise AttributeError(name)
        return getattr(impl, name)
