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

    def update(self, t, mass, com, wrench, active):
        if not active:
            self.ema, self.count, self.clear, self.detected = 0.0, 0, 0, False
            return False
        x = normal_features(t, mass, com)
        z = np.clip((x - self.mean) / self.scale, -8.0, 8.0)
        predicted = z @ self.weights
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
                normalized = np.clip(
                    (augmented - self.feature_mean) / self.feature_scale, -8.0, 8.0)
                scale = float(np.clip(
                    np.exp(np.r_[1.0, normalized] @ self.logscale_weights), 0.25, 1000.0))
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
_R5_ENABLED_MODES = 'servo'


class Detector(_R5BothModeDetector):
    def reset(self):
        super().reset()
        mode = getattr(getattr(self, "ctrl", None), "MODE", None)
        if mode not in _R5_ENABLED_MODES:
            self._r5_monitor = None
