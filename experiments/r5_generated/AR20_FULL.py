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
R5_CONFIG_ID = 'AR20_FULL'
R5_PAYLOADS = {'servo': {'mode': 'servo',
           'model': {'feature_mean': [1.0,
                                      0.06686947678056421,
                                      0.2919363502189908,
                                      0.9879689741346815,
                                      0.0,
                                      0.9539502431536606,
                                      0.06686947678056421,
                                      0.8430031988000874,
                                      0.9339559272806551,
                                      0.047854964413654444,
                                      1.0,
                                      0.24804573929036058,
                                      0.8430031988000874,
                                      0.00010570445848004342,
                                      -6.606528559524877e-05,
                                      -1.3213058223916598e-05,
                                      0.0,
                                      7.927834381922402e-05,
                                      0.00010570445848004342,
                                      0.0008340629392273633,
                                      -0.0005212893370170957,
                                      -0.00010425786740341364,
                                      0.0,
                                      0.000625547204420509,
                                      0.0008340629392273633,
                                      0.10596743739303588,
                                      -0.04297359841854573,
                                      -0.0019288473836106513,
                                      0.0,
                                      -0.022307955291677416,
                                      0.10596743739303588,
                                      0.5467312499992958,
                                      -0.024824999999972838,
                                      -0.05830833333327688,
                                      0.6148333333330693,
                                      5.779193071553156e-05,
                                      -3.611995617500256e-05,
                                      -7.223991839164317e-06,
                                      0.0,
                                      4.3343948014137435e-05,
                                      5.779193071553156e-05,
                                      0.00045600827334244204,
                                      -0.00028500517083902583,
                                      -5.700103416780965e-05,
                                      0.0,
                                      0.000342006205006836,
                                      0.00045600827334244204,
                                      0.03655963262709202,
                                      0.15961072567566767,
                                      0.5401535121897476,
                                      0.0,
                                      0.5215544088771158,
                                      0.03655963262709202,
                                      -0.0016600347610779885,
                                      -0.00724731989418661,
                                      -0.024526329782891636,
                                      0.0,
                                      -0.02368181478628815,
                                      -0.0016600347610779885,
                                      -0.003899047741947818,
                                      -0.01702232202068576,
                                      -0.05760682426682705,
                                      0.0,
                                      -0.05562324876121312,
                                      -0.003899047741947818,
                                      0.04111358330725802,
                                      0.1794921993263127,
                                      0.6074362575970869,
                                      0.0,
                                      0.5865204078322667,
                                      0.04111358330725802,
                                      0.0026598723584350618,
                                      -0.0016624202240177212,
                                      -0.00033248404480522265,
                                      0.0,
                                      0.0019949042688228616,
                                      0.0026598723584350618,
                                      0.21212121212121213,
                                      0.04713804713804714,
                                      0.36195286195286197,
                                      0.04208754208754209,
                                      0.3367003367003367,
                                      0.4208754208754209,
                                      1.8408459595969966,
                                      2.6755934374250714,
                                      -0.967482091471639,
                                      -1.027327883037332,
                                      2.2774700136718744,
                                      0.2901425709008504,
                                      -0.3186078345378093,
                                      0.1950499798483622,
                                      -0.1228776406135961,
                                      3.655127388140141e-06,
                                      -0.003074302181089109,
                                      -0.003509789318511606,
                                      2.89854148945495e-07,
                                      -1.1375902147713967e-05,
                                      1.037456527489021e-05,
                                      2.673849052108843,
                                      -0.9665150895652832],
                     'feature_scale': [1.0,
                                       0.5121237266488327,
                                       0.19668930741587273,
                                       0.015078092905123741,
                                       1.0,
                                       0.040184914998097544,
                                       0.5121237266488327,
                                       0.15034416680162863,
                                       0.06174795273016946,
                                       0.1462869086509522,
                                       1.0,
                                       0.1638212971919073,
                                       0.15034416680162863,
                                       0.9601496475884975,
                                       0.4952790391370911,
                                       0.2813597699646304,
                                       1.0,
                                       0.36248461590950415,
                                       0.9601496475884975,
                                       0.3247292691585976,
                                       0.4911495564821817,
                                       0.22794148486234933,
                                       1.0,
                                       0.2867780740484106,
                                       0.3247292691585976,
                                       0.9603974653062937,
                                       0.284523654760717,
                                       0.08383244785190351,
                                       1.0,
                                       0.15069262175581694,
                                       0.9603974653062937,
                                       0.21373123988684387,
                                       0.28316987429529344,
                                       0.29379205219109444,
                                       0.18993861874002077,
                                       0.5636300088930513,
                                       0.2907402299625416,
                                       0.16516468025732495,
                                       1.0,
                                       0.2127868382657685,
                                       0.5636300088930513,
                                       0.19062366081031432,
                                       0.28831614949869,
                                       0.13380691555850321,
                                       1.0,
                                       0.1683454029048026,
                                       0.19062366081031432,
                                       0.3009679866608098,
                                       0.1312422607458011,
                                       0.2113452600730152,
                                       1.0,
                                       0.20524905506290608,
                                       0.3009679866608098,
                                       0.14680056587109694,
                                       0.09979914414972074,
                                       0.27979587985907967,
                                       1.0,
                                       0.2703713766969079,
                                       0.14680056587109694,
                                       0.15464544075215206,
                                       0.10405273714174738,
                                       0.29029256512108803,
                                       1.0,
                                       0.28052133784848343,
                                       0.15464544075215206,
                                       0.32979805288307645,
                                       0.13818367980234095,
                                       0.18790414135918465,
                                       1.0,
                                       0.18302795551649817,
                                       0.32979805288307645,
                                       0.13175773839460983,
                                       1.0011271514792353,
                                       0.45602352509212185,
                                       1.0,
                                       0.5541671232904446,
                                       0.13175773839460983,
                                       0.4088102291891934,
                                       0.21193407382986676,
                                       0.48056527930881265,
                                       0.20078889632792832,
                                       0.47258144268039237,
                                       0.4936996060137451,
                                       2.427619421477933,
                                       2.297896549443801,
                                       2.665678062901783,
                                       1.1438278563359403,
                                       2.7715945060125087,
                                       1.3962129707998134,
                                       0.18492945734851515,
                                       0.26800423860619643,
                                       0.04326879357275608,
                                       0.0009868754328052922,
                                       0.0016231481376064191,
                                       0.0009543611887876668,
                                       5.112736315359128e-05,
                                       0.0007135580482148961,
                                       0.0006232923235031738,
                                       1.7379109886060549,
                                       2.3273380570183533],
                     'format': 'r5.normal-model.v1',
                     'mode': 'servo',
                     'phase_fit_counts': [30240, 6720, 51600, 6000, 48000],
                     'phase_scale': [0.8372513980113238,
                                     1.2254649401805096,
                                     1.1386633070686905,
                                     1.355194436239768,
                                     0.8986410959032833],
                     'spec': {'family': 'AR_FULL',
                              'gate': 't>=0.06',
                              'id': 'ar_full_lag20ms',
                              'lag_ms': 20,
                              'ridge': 0.01,
                              'scale': 'phase_mad',
                              'target_dims': 2},
                     'weights': [[2.6639980588119405, -0.966010020791667],
                                 [0.0, 0.0],
                                 [0.1043338955914039, 0.25132591413491756],
                                 [0.13526911231122424, 0.08703725113148827],
                                 [0.09226922498845855, -0.17172155778099243],
                                 [0.0, 0.0],
                                 [0.09849523704689402, -0.00605197485024863],
                                 [0.10433389559140338, 0.25132591413491634],
                                 [0.14866397642363738, -0.04093238166312984],
                                 [0.09073662141299585, -0.02431171868308328],
                                 [0.13564711138839605, 0.14444835116310467],
                                 [0.0, 0.0],
                                 [0.008304650094362718, -0.03048060433820606],
                                 [0.14866397642363738, -0.040932381663129014],
                                 [-0.23745214455330765, 0.16365549261978113],
                                 [-0.06425291956842058, 0.0004907912283013425],
                                 [-0.004886970496592132, -0.2459846034561701],
                                 [0.0, 0.0],
                                 [0.09158491065095181, 0.19026212371017687],
                                 [-0.23745214455329167, 0.16365549261980436],
                                 [-0.08297706308355256, -0.040902003592156344],
                                 [-0.11819520946099221, -0.004505369207920257],
                                 [0.24606354588598975, -0.40426167604609864],
                                 [0.0, 0.0],
                                 [0.006846529993099938, 0.3290377659179903],
                                 [-0.08297706308355474, -0.04090200359215105],
                                 [-0.06989770793330875, 0.09123412206414608],
                                 [0.13910712407603196, 0.07169973639651389],
                                 [-0.25823587304325174, 0.5153279616704306],
                                 [0.0, 0.0],
                                 [0.09290692077244955, 0.1038109510291847],
                                 [-0.06989770793330613, 0.09123412206414712],
                                 [-0.005413617576397315, 0.030069612774773578],
                                 [0.0089915394117845, -0.017011546387062156],
                                 [0.019550930198788008, -0.011815164660435367],
                                 [-0.010047778877864176, -0.011243550981403127],
                                 [-0.017809654627974545, -0.07478777258489502],
                                 [0.047987580068910136, 0.051591473802468674],
                                 [-0.02621556631768226, -0.20189384882660114],
                                 [0.0, 0.0],
                                 [-0.045219124051173176, 0.08621781393732794],
                                 [-0.017809654627975055, -0.07478777258489602],
                                 [0.14408738040932242, -0.09614208673542222],
                                 [-0.15132562427446034, 0.09805027095324273],
                                 [0.3798904555879223, -0.4805653277448688],
                                 [0.0, 0.0],
                                 [-0.042783162930300114, 0.21404497556578883],
                                 [0.14408738040932292, -0.09614208673542245],
                                 [0.06685115269032361, 0.006099673541939084],
                                 [-0.03528526211482594, 0.032862143132885754],
                                 [-0.005645436775536678, 0.0075793061104428945],
                                 [0.0, 0.0],
                                 [0.010454503609350642, 0.008292014459067],
                                 [0.06685115269032556, 0.0060996735419367575],
                                 [0.008160842332653958, 0.03731839762345355],
                                 [0.026680790653354273, -0.034390950361917964],
                                 [0.010961389519072091, -0.007523433508978447],
                                 [0.0, 0.0],
                                 [0.007813570537884245, 0.0010492331931063394],
                                 [0.008160842332656092, 0.03731839762344542],
                                 [0.008589911389234437, 0.03222499967912691],
                                 [0.004170499656660867, 0.00983817053292033],
                                 [0.026613128916551966, -0.01906490091802608],
                                 [0.0, 0.0],
                                 [0.01308035166571092, -0.004282039505563646],
                                 [0.008589911389234922, 0.03222499967912401],
                                 [0.016702356251687714, 0.09250000727333225],
                                 [-0.01297773473560245, 0.031546279949871414],
                                 [0.010702795578207443, -0.013637589698823787],
                                 [0.0, 0.0],
                                 [0.01005174768179052, -0.02666935489169877],
                                 [0.016702356251687665, 0.09250000727333269],
                                 [-0.02302288528565701, -0.033761824242029635],
                                 [-0.20570287864865067, -0.10669391074126158],
                                 [0.15811841558961642, -2.114449407072942],
                                 [0.0, 0.0],
                                 [0.24820340324411122, 1.998766901399845],
                                 [-0.02302288528565921, -0.033761824242034846],
                                 [-0.18013067931584903, 0.10520347372551676],
                                 [0.0033219676637848247, 0.10877335244308667],
                                 [0.25527390909690895, -0.010618058865479053],
                                 [-0.0776943742341139, -0.005616563395739593],
                                 [-0.07224232805983058, -0.1266038327351326],
                                 [0.020455348494863784, -0.4319037673011174],
                                 [0.16556361343447895, 0.2719704964810125],
                                 [0.1279147319032477, -0.2485310173920967],
                                 [-0.05941837160677259, 0.3339471762661344],
                                 [0.13398082820099264, -0.23133617381864685],
                                 [0.14668110829398356, 0.03552236676172712],
                                 [-0.40436779907473525, 0.27947930117423814],
                                 [-0.08101852076487208, 0.042766908217029914],
                                 [0.20407507185880014, -0.2644259139697853],
                                 [0.05199209016258295, -0.06766825144207819],
                                 [0.28569401661275645, -0.07084808254346293],
                                 [0.5507593228940386, 0.0354613334178757],
                                 [-0.10071663536291743, 0.4881373006454125],
                                 [0.057971103569833746, 0.059701750942072196],
                                 [0.5322933816812466, -0.5042682377448903],
                                 [-0.1796992723553012, 0.09267711971214854],
                                 [0.5690300371475917, 0.04184495709092058],
                                 [0.27855053715589423, 0.8200260197221274]]},
           'threshold': 7.225805072254827},
 'torque': {'mode': 'torque',
            'model': {'feature_mean': [1.0,
                                       0.06686947678056421,
                                       0.2919363502189908,
                                       0.9879689741346815,
                                       0.0,
                                       0.9539502431536606,
                                       0.06686947678056421,
                                       0.8430031988000874,
                                       0.9339559272806551,
                                       0.047854964413654444,
                                       1.0,
                                       0.24804573929036058,
                                       0.8430031988000874,
                                       0.00010570445848004342,
                                       -6.606528559524877e-05,
                                       -1.3213058223916598e-05,
                                       0.0,
                                       7.927834381922402e-05,
                                       0.00010570445848004342,
                                       0.0008340629392273633,
                                       -0.0005212893370170957,
                                       -0.00010425786740341364,
                                       0.0,
                                       0.000625547204420509,
                                       0.0008340629392273633,
                                       0.10596743739303588,
                                       -0.04297359841854573,
                                       -0.0019288473836106513,
                                       0.0,
                                       -0.022307955291677416,
                                       0.10596743739303588,
                                       0.5467312499992958,
                                       -0.024824999999972838,
                                       -0.05830833333327688,
                                       0.6148333333330693,
                                       5.779193071553156e-05,
                                       -3.611995617500256e-05,
                                       -7.223991839164317e-06,
                                       0.0,
                                       4.3343948014137435e-05,
                                       5.779193071553156e-05,
                                       0.00045600827334244204,
                                       -0.00028500517083902583,
                                       -5.700103416780965e-05,
                                       0.0,
                                       0.000342006205006836,
                                       0.00045600827334244204,
                                       0.03655963262709202,
                                       0.15961072567566767,
                                       0.5401535121897476,
                                       0.0,
                                       0.5215544088771158,
                                       0.03655963262709202,
                                       -0.0016600347610779885,
                                       -0.00724731989418661,
                                       -0.024526329782891636,
                                       0.0,
                                       -0.02368181478628815,
                                       -0.0016600347610779885,
                                       -0.003899047741947818,
                                       -0.01702232202068576,
                                       -0.05760682426682705,
                                       0.0,
                                       -0.05562324876121312,
                                       -0.003899047741947818,
                                       0.04111358330725802,
                                       0.1794921993263127,
                                       0.6074362575970869,
                                       0.0,
                                       0.5865204078322667,
                                       0.04111358330725802,
                                       0.0026598723584350618,
                                       -0.0016624202240177212,
                                       -0.00033248404480522265,
                                       0.0,
                                       0.0019949042688228616,
                                       0.0026598723584350618,
                                       0.21212121212121213,
                                       0.04713804713804714,
                                       0.36195286195286197,
                                       0.04208754208754209,
                                       0.3367003367003367,
                                       0.4208754208754209,
                                       1.8408459595969966,
                                       2.8406051816513287,
                                       -1.20395130761138,
                                       -1.244260296114435,
                                       2.4503022699966364,
                                       0.33365418410115505,
                                       -0.3210933027941879,
                                       0.20157254093777588,
                                       -0.12331572096589238,
                                       -0.00045969942709687636,
                                       -0.0015293134925687204,
                                       -0.004152492977074432,
                                       -0.00040954827647631697,
                                       -0.00026294559726360803,
                                       -0.00047508338561349264,
                                       2.8387859518835046,
                                       -1.2028296888325016],
                      'feature_scale': [1.0,
                                        0.5121237266488327,
                                        0.19668930741587273,
                                        0.015078092905123741,
                                        1.0,
                                        0.040184914998097544,
                                        0.5121237266488327,
                                        0.15034416680162863,
                                        0.06174795273016946,
                                        0.1462869086509522,
                                        1.0,
                                        0.1638212971919073,
                                        0.15034416680162863,
                                        0.9601496475884975,
                                        0.4952790391370911,
                                        0.2813597699646304,
                                        1.0,
                                        0.36248461590950415,
                                        0.9601496475884975,
                                        0.3247292691585976,
                                        0.4911495564821817,
                                        0.22794148486234933,
                                        1.0,
                                        0.2867780740484106,
                                        0.3247292691585976,
                                        0.9603974653062937,
                                        0.284523654760717,
                                        0.08383244785190351,
                                        1.0,
                                        0.15069262175581694,
                                        0.9603974653062937,
                                        0.21373123988684387,
                                        0.28316987429529344,
                                        0.29379205219109444,
                                        0.18993861874002077,
                                        0.5636300088930513,
                                        0.2907402299625416,
                                        0.16516468025732495,
                                        1.0,
                                        0.2127868382657685,
                                        0.5636300088930513,
                                        0.19062366081031432,
                                        0.28831614949869,
                                        0.13380691555850321,
                                        1.0,
                                        0.1683454029048026,
                                        0.19062366081031432,
                                        0.3009679866608098,
                                        0.1312422607458011,
                                        0.2113452600730152,
                                        1.0,
                                        0.20524905506290608,
                                        0.3009679866608098,
                                        0.14680056587109694,
                                        0.09979914414972074,
                                        0.27979587985907967,
                                        1.0,
                                        0.2703713766969079,
                                        0.14680056587109694,
                                        0.15464544075215206,
                                        0.10405273714174738,
                                        0.29029256512108803,
                                        1.0,
                                        0.28052133784848343,
                                        0.15464544075215206,
                                        0.32979805288307645,
                                        0.13818367980234095,
                                        0.18790414135918465,
                                        1.0,
                                        0.18302795551649817,
                                        0.32979805288307645,
                                        0.13175773839460983,
                                        1.0011271514792353,
                                        0.45602352509212185,
                                        1.0,
                                        0.5541671232904446,
                                        0.13175773839460983,
                                        0.4088102291891934,
                                        0.21193407382986676,
                                        0.48056527930881265,
                                        0.20078889632792832,
                                        0.47258144268039237,
                                        0.4936996060137451,
                                        2.427619421477933,
                                        2.312276894026216,
                                        2.7381616447898924,
                                        1.1412084883299474,
                                        2.7780388998626244,
                                        1.4047482259878377,
                                        0.18557962059864055,
                                        0.25570791636546353,
                                        0.043798917469356526,
                                        0.0030038017787372877,
                                        0.0036462288892256853,
                                        0.0023878905192791316,
                                        0.00031829586770494025,
                                        0.0029447565629378953,
                                        0.004525427392735831,
                                        1.7557494847011694,
                                        2.409020373037487],
                      'format': 'r5.normal-model.v1',
                      'mode': 'torque',
                      'phase_fit_counts': [30240, 6720, 51600, 6000, 48000],
                      'phase_scale': [0.8364764149931712,
                                      1.235345188140354,
                                      1.128942082359294,
                                      1.3201965756772895,
                                      0.8832687918626607],
                      'spec': {'family': 'AR_FULL',
                               'gate': 't>=0.06',
                               'id': 'ar_full_lag20ms',
                               'lag_ms': 20,
                               'ridge': 0.01,
                               'scale': 'phase_mad',
                               'target_dims': 2},
                      'weights': [[2.8457766498769543, -1.2088184202383647],
                                  [0.0, 0.0],
                                  [0.04992439597856419, 0.29075543591634767],
                                  [-0.01784285736908774, 0.13615449330361518],
                                  [0.07591996585076949, -0.21142556485422342],
                                  [0.0, 0.0],
                                  [0.08442082126819431, 0.037119667674131135],
                                  [0.049924395978563824, 0.29075543591634556],
                                  [0.08764488812510218, 0.09216100088236429],
                                  [0.12043758179145657, -0.1190915836496215],
                                  [0.055950129798941164, 0.24361236317248877],
                                  [0.0, 0.0],
                                  [-0.08777621131371546, -0.05386791843575403],
                                  [0.08764488812510252, 0.09216100088236477],
                                  [-0.21548218225161242, 0.15023215621310917],
                                  [-0.060468435944400184, 0.07254062367314072],
                                  [0.018946947850901993, -0.32649529105362546],
                                  [0.0, 0.0],
                                  [0.06791416485912749, 0.15430941656301236],
                                  [-0.21548218225163251, 0.15023215621311908],
                                  [-0.009455859966440307, 0.01954268327766071],
                                  [-0.08255554628472592, 0.034242817476882945],
                                  [0.11732889854664491, -0.1802443088274865],
                                  [0.0, 0.0],
                                  [0.048131282828088116, 0.08461878006594072],
                                  [-0.009455859966444744, 0.019542683277671193],
                                  [-0.0002294262515382588, 0.016645476907186093],
                                  [0.16070483162486554, 0.1092897445310045],
                                  [-0.22171028671565435, 0.5749318588671803],
                                  [0.0, 0.0],
                                  [0.20552582072271608, 0.04004574266383836],
                                  [-0.00022942625153938898, 0.016645476907184376],
                                  [-0.014503450791884141, -0.00017209773160012325],
                                  [-0.004170325590249359, 0.011428326223845944],
                                  [0.003244595797093635, 0.010898918763422031],
                                  [-0.0035695887889763255, 0.00863588401709047],
                                  [-0.14649600942649046, 0.0844886751606747],
                                  [0.07027165698094572, -0.03075994295346983],
                                  [-0.2724540297819265, 0.09200314174924462],
                                  [0.0, 0.0],
                                  [0.11546289800384539, -0.02938394429256225],
                                  [-0.14649600942648983, 0.08448867516067399],
                                  [0.022434082296716826, -0.010069973084081512],
                                  [-0.027437321647664874, 0.04110841648112577],
                                  [0.1493399999808381, -0.08571273440878907],
                                  [0.0, 0.0],
                                  [-0.07171031479701458, -0.0022766510393393124],
                                  [0.022434082296716784, -0.010069973084081247],
                                  [0.02363516092758882, -0.014425691750347856],
                                  [0.00931383550397083, 0.08155498078544574],
                                  [-0.003438736667975764, -0.006587170287376807],
                                  [0.0, 0.0],
                                  [-0.012951331458036501, -0.01661345582211969],
                                  [0.023635160927592457, -0.014425691750345436],
                                  [-0.0030025957898945983, -0.017187147554271042],
                                  [0.001592512673853472, 0.005170828027053589],
                                  [0.004650969855841963, 0.005262408444758031],
                                  [0.0, 0.0],
                                  [-0.004084709194323138, 0.003281239571693895],
                                  [-0.0030025957898863367, -0.01718714755426763],
                                  [-0.010880234879350744, 0.0012036570952335963],
                                  [0.00950799827404827, -0.026843712802486965],
                                  [0.0027652071479908284, 0.012399412688787827],
                                  [0.0, 0.0],
                                  [0.0017312490072343236, 0.009865600421829318],
                                  [-0.01088023487935364, 0.0012036570952302856],
                                  [0.004307669873499116, 0.043578897168504355],
                                  [0.01766653585680882, 0.05691423095113473],
                                  [-0.003071922955127901, 0.00478378008218023],
                                  [0.0, 0.0],
                                  [-0.0012973571643941503, -0.00465601716227761],
                                  [0.004307669873499226, 0.04357889716850324],
                                  [-0.009488090905571785, -0.0574048514569432],
                                  [-0.24948651186215823, -0.08177227844514771],
                                  [0.18001361015069173, -2.117348925214475],
                                  [0.0, 0.0],
                                  [0.2906409032785537, 1.9642959124627302],
                                  [-0.009488090905577486, -0.05740485145694747],
                                  [-0.0517826075827054, 0.17308651715571602],
                                  [-0.002819390433268034, 0.19043678649747753],
                                  [0.1585929255230021, -0.03963441905161387],
                                  [-0.09963117860706625, -0.01811073817964029],
                                  [-0.07288190240530625, -0.1871343517495708],
                                  [-0.011193743603755567, -0.23654829573956335],
                                  [0.05564073157626392, -0.06107492204512702],
                                  [0.11754669895778933, -0.17004076532790124],
                                  [-0.009540824505446783, 0.18310845654158528],
                                  [0.11274154830913242, -0.18947651969442664],
                                  [0.22193324770589076, -0.13625475040645108],
                                  [-0.5145096450028362, 0.384290950609834],
                                  [-0.040071174951460724, 0.05423653947565897],
                                  [0.26872181176162546, -0.29708036748670597],
                                  [0.09940293788915418, -0.02554361937877023],
                                  [-0.027577260340758376, 0.10148960012706043],
                                  [-0.01126193153877018, 0.013496196357620501],
                                  [0.0012042614790892668, 0.09177297258450039],
                                  [-0.10021329644957769, -0.10488368361007473],
                                  [-0.054868982172406724, -0.012081670415890193],
                                  [-0.023013991250177886, 0.003503528330195804],
                                  [0.7380412097014281, 0.11736382747649336],
                                  [0.22072674687338045, 1.146847659262834]]},
            'threshold': 7.622101519725345}}

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
