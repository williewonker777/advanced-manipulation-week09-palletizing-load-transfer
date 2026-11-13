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
from collections import deque
import mujoco
from sim.nominal import Nominal
from sim import config as C, trajectory as T
WINDOWS = ((1.4, 1.7), (3.6, 4.0))


def mass_weight(mass, lo, hi):
    if not 0 <= lo < hi <= 8:
        raise ValueError('invalid mass activation band')
    return C.quintic(mass, lo, hi - lo)[0]


def critical_step(position, velocity, target, pole, dt=0.001):
    error = position - target
    decay = np.exp(-pole * dt)
    return (target + decay * ((1 + pole * dt) * error + dt * velocity), decay * (-pole * pole * dt * error + (1 - pole * dt) * velocity))


class SignalBank:

    def __init__(self, mass, com, *, include_L3=True):
        self.mass, self.com = (float(mass), tuple(com))
        self.empty, self.full, self.geometry = (Nominal(), Nominal(), Nominal())
        self.full.set_payload(mass, com)
        self.inv_k = np.zeros(6)
        self.inv_k[self.empty.elastic] = 1 / self.empty.K[self.empty.elastic]
        self.l3 = None
        if include_L3:
            pass
        self.mean = deque(maxlen=5)
        self.position_error = self.velocity_error = None
        self.disturbance, self.disturbance_rate = (0.0, 0.0)
        self.bottom_offset = np.array([0.0, 0.0, C.BOX_POS[2] + C.BOX_SIZE[2] / 2])
        self.table_heights = np.array([self.bottom(T.reference(t)[0]) for t in (1.4, 3.9)])

    def bottom(self, q):
        nom = self.geometry
        nom.d.qpos[:] = q
        mujoco.mj_kinematics(nom.m, nom.d)
        body = nom.m.body('link_6').id
        rotation = nom.d.xmat[body].reshape(3, 3)
        return float((nom.d.xpos[body] + rotation @ self.bottom_offset)[2])

    def update(self, t, th, thd, tau_s, grip, *, cached_L3=None):
        q, qd, qdd = T.reference(t)
        measured_q = np.asarray(th) - self.inv_k * tau_s
        error = measured_q - q
        if self.position_error is None:
            self.position_error, self.velocity_error = (error.copy(), np.zeros(6))
        self.position_error, self.velocity_error = critical_step(self.position_error, self.velocity_error, error, 40.0)
        acceleration_error = 40 ** 2 * (error - self.position_error) - 80 * self.velocity_error
        predicted = (self.full if grip else self.empty).inverse_dynamics(q + self.position_error, qd + self.velocity_error, qdd + acceleration_error)
        raw = float(tau_s[0] - predicted[0])
        self.disturbance, self.disturbance_rate = critical_step(self.disturbance, self.disturbance_rate, raw, 25.0)
        self.mean.append(float(tau_s[0]))
        if cached_L3 is not None:
            l3 = float(cached_L3)
        elif self.l3 is not None:
            l3 = 0.0
        else:
            l3 = 0.0
        distance = float(np.min(np.abs(self.table_heights - self.bottom(measured_q))))
        return (np.array([np.mean(self.mean), self.disturbance, l3]), distance)


def regressors(mass, com, band=None):
    m = float(mass)
    p = m * np.asarray(com, float)
    w = 0.0 if band is None else mass_weight(m, *band)
    return np.r_[1.0, m, p, w * m, w * p]


def scale_regressors(mass, com):
    return np.array([1.0, mass / 8.0, mass * com[0] / 0.24, mass * com[1] / 0.16])


class _R8CalibratedChannel:

    def __init__(self, variant, mass, com, coefficients, thresholds, threshold_coefficients, entry_bounds, width_m, support_N, *, band=None):
        if variant not in ('a', 'b'):
            raise ValueError('only the two registered L2b candidates are allowed')
        self.variant = variant
        self.regressor = regressors(mass, com, band)
        self.scale_regressor = scale_regressors(mass, com)
        self.coefficients = np.asarray(coefficients, float)
        self.thresholds = np.asarray(thresholds, float)
        self.threshold_coefficients = np.asarray(threshold_coefficients, float)
        self.entry_bounds = np.asarray(entry_bounds, float)
        if self.coefficients.shape != (9, 6000) or self.thresholds.shape != (240,) or self.threshold_coefficients.shape != (240, 4) or (self.entry_bounds.shape != (2, 2)) or np.any(self.thresholds <= 0) or np.any(self.entry_bounds <= 0):
            raise ValueError('invalid normal model')
        arrays = (self.coefficients, self.thresholds, self.threshold_coefficients, self.entry_bounds)
        if not all((np.isfinite(a).all() for a in arrays)):
            raise ValueError('nonfinite normal model')
        for a in arrays:
            a.setflags(write=False)
        self.mass = float(mass)
        self.width = float(width_m)
        self.support_N = float(support_N)
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
        if self.variant == 'b':
            value = value * np.exp(np.clip(self.threshold_coefficients[bins] @ self.scale_regressor, -10.0, 10.0))
        return np.maximum(0.3, value)

    def update(self, t, signal, height_distance, support_force_N, grip, base_alarm):
        k = int(round(t * 1000))
        if abs(k * 0.001 - t) > 1e-12 or k != len(self.raw) or (not 0 <= k < 6000):
            raise ValueError('complete ordered 1ms stream required')
        self.raw.append(float(signal))
        self.latched = bool(self.latched or base_alarm)
        if self.mass == 0:
            return False
        phase = next((j for j, (a, b) in enumerate(WINDOWS) if a - 0.02 - 1e-12 <= t < b - 1e-12), None)
        if phase != self.phase:
            self.phase = phase
            self.disabled = self.previous_open = False
            self.count = 0
        if phase is None:
            return False
        a, b = WINDOWS[phase]
        if self.variant == 'a' and phase == 0:
            b = 1.6
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
        error = float(signal) - baseline - float(self.regressor @ self.coefficients[:, k])
        self.count = self.count + 1 if abs(error) > self.threshold(k) else 0
        if self.count >= 4:
            self.latched = True
            return True
        return False


class _R8L2c(_R8CalibratedChannel):

    def __init__(self, mass, com, coefficients, thresholds, threshold_coefficients, entry_bounds, width_m, support_N, *, band=None):
        super().__init__('b', mass, com, coefficients, thresholds, threshold_coefficients, entry_bounds, width_m, support_N, band=band)

    def update(self, t, signal, height_distance, support_force_N, grip, base_alarm):
        k = int(round(t * 1000))
        if abs(k * 0.001 - t) > 1e-12 or k != len(self.raw) or (not 0 <= k < 6000):
            raise ValueError('complete ordered 1ms stream required')
        self.raw.append(float(signal))
        self.latched = bool(self.latched or base_alarm)
        phase = next((j for j, (a, b) in enumerate(WINDOWS) if a - 0.02 - 1e-12 <= t < b - 1e-12), None)
        if phase != self.phase:
            self.phase = phase
            self.disabled = self.previous_open = False
            self.count = 0
        if phase is None:
            return False
        a, b = WINDOWS[phase]
        if self.variant == 'a' and phase == 0:
            b = 1.6
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
        error = float(signal) - baseline - float(self.regressor @ self.coefficients[:, k])
        self.count = self.count + 1 if abs(error) > self.threshold(k) else 0
        if self.count >= 4:
            self.latched = True
            return True
        return False


class HypothesisSignal(SignalBank):

    def __init__(self, mass, com, stiffness_scale=1.0, *, com_delta=None, epsilon_m=0.001):
        super().__init__(mass, com, include_L3=False)
        if stiffness_scale <= 0 or epsilon_m <= 0:
            raise ValueError('positive stiffness and difference step required')
        self.inv_k /= stiffness_scale
        centre = np.asarray(com, float) + (0.0 if com_delta is None else com_delta)
        self.full.set_payload(mass, centre)
        n = Nominal()
        n.d.qpos[:] = T.reference(1.4)[0]
        mujoco.mj_kinematics(n.m, n.d)
        rotation = n.d.xmat[n.m.body('link_6').id].reshape(3, 3)
        self.axes_local = rotation.T[:, :2].copy()
        self.epsilon = epsilon_m
        self.offset_models = []
        for axis in self.axes_local.T:
            pair = []
            for sign in [-1.0, 1.0]:
                model = Nominal()
                model.set_payload(mass, centre + sign * epsilon_m * axis)
                pair.append(model)
            self.offset_models.append(pair)
        self.sensitivity = np.zeros(2)
        self.sensitivity_rate = np.zeros(2)

    def update(self, t, th, thd, tau_s, grip, *, cached_L3=None):
        values, distance = super().update(t, th, thd, tau_s, grip, cached_L3=cached_L3)
        q, qd, qdd = T.reference(t)
        error = np.asarray(th) - self.inv_k * tau_s - q
        acceleration = 40 ** 2 * (error - self.position_error) - 80 * self.velocity_error
        args = (q + self.position_error, qd + self.velocity_error, qdd + acceleration)
        raw = np.zeros(2)
        if grip and self.mass > 0:
            for j, (minus, plus) in enumerate(self.offset_models):
                raw[j] = -(plus.inverse_dynamics(*args)[0] - minus.inverse_dynamics(*args)[0]) / (2 * self.epsilon)
        self.sensitivity, self.sensitivity_rate = critical_step(self.sensitivity, self.sensitivity_rate, raw, 25.0)
        return (values, distance)


def membership_statistic(errors, sensitivities):
    errors, sensitivities = (np.asarray(errors), np.asarray(sensitivities))
    if errors.shape != (3,) or sensitivities.shape != (3, 2):
        raise ValueError('three stiffness hypotheses and two offset axes required')
    return float(np.min(np.maximum(0.0, abs(errors) - 0.03 * np.linalg.norm(sensitivities, axis=1))))


class Membership:

    def __init__(self, mass, com, model):
        self.banks = [HypothesisSignal(mass, com, scale) for scale in [0.8, 1.0, 1.2]]
        self.channel = _R8Channel(mass, com, model['coefficients'], model['thresholds'], model['threshold_coefficients'], model['entry_bounds'], float(model['width']), float(model['support_N']))
        self.signals, self.sensitivities = ([], [])
        self.baselines = {}

    def update(self, t, th, thd, tau_s, grip, support_N, base_alarm):
        observations = [b.update(t, th, thd, tau_s, grip) for b in self.banks]
        signal = np.array([v[0][1] for v in observations])
        sensitivity = np.array([b.sensitivity.copy() for b in self.banks])
        self.signals.append(signal)
        self.sensitivities.append(sensitivity)
        k = int(round(t * 1000))
        phase = next((j for j, (a, b) in enumerate(WINDOWS) if a - 0.02 - 1e-12 <= t < b - 1e-12), None)
        errors, centred_sensitivity, statistic = (np.zeros(3), np.zeros((3, 2)), 0.0)
        if phase is not None:
            if phase not in self.baselines:
                start = WINDOWS[phase][0]
                lo, hi = (round((start - 0.12) * 1000), round((start - 0.02) * 1000))
                self.baselines[phase] = (np.mean(self.signals[lo:hi], axis=0), np.mean(self.sensitivities[lo:hi], axis=0))
            baseline, sb = self.baselines[phase]
            errors = signal - baseline - self.channel.regressor @ self.channel.coefficients[:, k]
            centred_sensitivity = sensitivity - sb
            statistic = membership_statistic(errors, centred_sensitivity)
        self.last_nominal_distance = observations[1][1]
        hit = self.channel.update(t, signal[1], self.last_nominal_distance, support_N, grip, base_alarm, statistic)
        return (hit, errors, centred_sensitivity, statistic)


class _R8Channel(_R8L2c):

    def update(self, t, signal, height_distance, support_force_N, grip, base_alarm, statistic):
        if not np.isfinite(statistic) or statistic < 0:
            raise ValueError('finite nonnegative membership statistic required')
        k = int(round(t * 1000))
        if abs(k * 0.001 - t) > 1e-12 or k != len(self.raw) or (not 0 <= k < 6000):
            raise ValueError('complete ordered 1ms stream required')
        self.raw.append(float(signal))
        self.latched = bool(self.latched or base_alarm)
        phase = next((j for j, (a, b) in enumerate(WINDOWS) if a - 0.02 - 1e-12 <= t < b - 1e-12), None)
        if phase != self.phase:
            self.phase = phase
            self.disabled = self.previous_open = False
            self.count = 0
        if phase is None:
            return False
        a, b = WINDOWS[phase]
        if self.variant == 'a' and phase == 0:
            b = 1.6
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

_R8_POSITIVE = 'P)h>@EdT%j2mk;8ApneerI7#s|NsC0|NjsG6aZsyWoBk+V`*h>baO6laCrd$5C8y}1^@s6001nE00000008XW<x>z}_$Y8hNx=XU6hROLK@@4whelLTK}iv$l@<|ckVZO1y1N_1hwkn!0sSflSRew|`!C%2-MKS&_VebMbN1}nv(K5`GrKQ#!xZl++*aR3@{GjctiFM@-jlN;%xA?6h0by@pEb02^7P3=b8U+!`Ud~IUiP7>wZXr7YvYHO2LH<Egt?dnE?&II!Nu&v{C|d;gg_t=2m}IwKp+qZ1OkCTAP@)y0)apv5C{YUfj}S-2m}J*f0%hs+FG+l1oD@JUQZql$NM$A<+YT;@ow)mvWCGhRJCmUBB2$A1yl~21D`_i*)83T5xY=K67(aR-xrEw8V8?d<%QsG)^F0+WJ2%?sW{!_crbb%b1{-I4#o$@mRUN#gYdx<YhiuPAdC#Vw)0FR5H(NG#py@{V7Zc0=Q%ZhT#0w7BQbf2bbabYPu%@b#PN=`a-1*jDziBGxyJ{WzF#~ycgzQOg)-3EDtqHWny3~7TQAHx#NYbB*AqWS2LHSi?tyP4ikSC>x}($l509wu7kK>or&o>%ZfKSHp~$Dm759CVD6dX-!7W((!`$kGRKIt4K3X{7g7fM0ues0h%2QYR5%OoqbY8C0THhKKH;*z1S6kwm{_bz0yG`-O$px{kzXoWP>we$)zBVT9`LWN=KnWj<cir2(!-rPRk4!FyN`P2Dr3GKCDt!0jS2<mv2b8L8;@puYFePZPT3BWULFUfn56WynRsC;;`LG?3(IqoHkadPaRF&>>a|4MB7q{Pka0l_L=3c?z1wKkkTbk}ZU{(M40{MU+czb^{7rz<+Z`ZY5BKv~iQ8UfScCHYpNY$qlvkiqSSH@VWzlMPa?QiPWkr6;cwi*0NCkpxmoanu!qaoR`Vi${446Hq!5#dvdgF4r^b0y9RkhOO)@X>T4kTci}@7a|Ms#dp7-DFRJ=3h*b?=(}vc+#d%B_$1*$R7nOZ=^%f$UH7R$OL1rw5Cg~SzsfiWgdPZ2Og<hd{-Nl3(Dd~o57U%Q1o}}%s2M}NJ>q(ci5>A$g-^8*9#WGLx(|8qn|~Pb$~kVcV02v4<1+6(J29L-|D3CgQYNBp=|S|wiKRnON5U-EQ1cm)o^{%au|@Pw}Yf|@cpGdZz@^=6lI?eeebIP(KZ{-D*Z~>XJwdia-$M_+M~W{1y(`6`-M|%C#u0A=YbY$QZ+0`3{7aCs)0=>S)Yy28qhrzI?udS10+02tEl@1&Ihq%DKx(U+t%VU3Cy+7d_72^>~Sq<>^FV3H>eh_tt)CNyVk;u^X$3ye6?^||MPpExHoX0TSLG<uLk<J9VrgpuZFALUuc*!{*B{K{5WNJIjHPA_2255QkY8!9+p)p0)-qw-7dvkV9e*RwvtbSUydBAB9bw{X}Pzp-^~+9av$HG_-c=&wsRM@PDP{l+Npz`?CEI#N#t=PSs@O#M+Gz;E<?=&awMNGSL5_4-`^+hB5I2Kr*>Yj0f)}B+*f_ugq7z+K97pFqT%x77DaYDrq@iW7m{`19M81A-t`{*OK!zUFWQG5X<T{aQvH}!`b+<a$N(-^zr6hA)F67DaTRO)K8T*>x|9z}htNKV_YcRzVRShfl_S3~j8erypM@MpQ1h0t&*p~_WW6kG`uFN6&OS~Qyy`NFRZRR+4cVjUWA0o#T>r1$VdJx2(<tiIR|XFhjbi8TuK&LJj-pj~?7?isQPe(LQdYKW6jgFlAJQa`ptPa4R|L-pCR7SPo5&tU22Re)J9NXSrLmvox$zLXQWV~w${obpUxR>uegH4=Q@=DPAHbBhxz?PM{W$Rbw9~$GeOP(7%3}OR54!EW%<jz7h5MT%O%DET$Cp7hn@>twQS<VMlAuO2GOrdzcC9wx{5G>$kbNDl+CSe#^Q{I~W<*$@jF)4~<(7KNk|K;U*^D*Zm4$ge;cm;P;?VTG-p+`TEAA8|InV0&Lc)9;ok?>deAyW<D><73Ogr)_JmtmkOV~$8rlSIqvnDvc_q_pw2HH=!PzRO656$E5HiBjG*I%mD%|O1x7cFq04OYC|T<O$0AnN;)x_xjLd{mYfnW*RiJMIW-dc!{O4+~yoHS7oF@!-84_XmLWzd!dwt_{Mj<9CnTIzI#zlq_l|$cEweYRv`Xp<yt}`|t1Cs}b;MnR;+UX%rTUG)-r>N1=gtZ{C}PG00P9x){$l4vkMax6^CKA@-Gzc8JIXxCK@pzLqlqAtc;iY7b1pu19ChS|3kBlBXfNMdc)L81HX=yfF#)$z9URxu?L4l<k_?y(y@eQg=CRIR$Y%*$gYrQ*bpgOnugU3NE%3JdFBR{<OqvP-{K~Uq~x1``(=b>&GvGhIppnc|`-8&8{ghoaNFEX`clCqwnM;yeDBR!9gcZWD=g9V&u(SoPe**2EKc|Ccr@T{z<b_6EN~gf8Dug9LkOxA7B(4hq+|Ui3J>kM#;8d2C*?XNM<oNRX7TD3K>6(PL0Cgsft=<pAoP~P~M*ZGz<folexy%hr##x_Edh#5V(pLGm&l#0=Hgc2g~(AFuKS;HWN7jN0;ZsD`)$`b?LzsbGCjkv(7e`HRywQ&b?H<*Lva8goSH;Q#ah4$QRub?t*k?=xfjHfFDX;WYu)-z*0UYecYiH62h)cf1Yaw8QSH%-x5tA5t|+HFrfil{2k?z*6X0yvnQ`K;VsDiC-XK$u?9GUC)X`@Dj-0IHBF$X6y&|M+m09%!Bw*NZ{&q@LHBfuu5Mu(*t?7h$Lqwyp%dy9T4o`@LGq@{P2dUmUXQ&wX6}k)kB)@Oe+$QbN`H1-x)L#Q^~Y4_!3>mPzt{U%HV<7wO&9|!iqQJb`W#(LDSoAIIr68w0)0GbI10O}(MS7u?(*AOq%t4*q@i4gQ86j8%#HOZac$DW@<Jodd_MehGrS4MAODxR@S_<sdz?h>-)cq4xb4N>j5h4eV&qc%cl~A3J5a0K=)_EqMn$KvE_@R%x~Hwb8>6T7z6?|J;_u>ddLPC<6i`0VSzOSE?QMZ6B$EB8Cz3T7)ZdT4xSuwAYYiYt%)4d3_X9|&UgQ4Nd=NwSue@~oci%cXb+lT=W(f7oO^SZ34WX|}okY0VFluHwyc=2?#&egGVpr8ikUntv3~R>-&YW<$=_N3VaY=>~JMp6!N?Ybtw|fkW_V+8v7>r^6*@AG3#xYcLe|xe0<T%olmxxwA9Y@db;}j}w<7k?ftp4oy1kTDb?5;DKKt4m?RE3fW^io?M^4Xj~QnO(t67fkaP_N*vb(=&j_lKk=^^=(Jc^{|N_9SjoG@cn1oWd*W6AjPJr?8*Z+)*}Z3Y~VnvYqUi!o$@p-1pX}FzM3HGAGkCs?8^saEec3z}|4ZquSGGOrLyp#(o;@PxAmx*fhS&`hq4I)0pzn)#-EPG+sI;`{-KRG%_VD=3gI~#y+)_*K_Zt(Oc%ebL-b>G!a(e%KJTy63j!3Qk&Bls&wbjz`yv&-c+_{V;Y~={kE+6F^yxlGa2`Pp2jDl91WX`)A()hifa1QG}70rz0?|*#sWRjgRITd==M3`Kv?NCx)=U!5l)&$vUrk3gO}5I<wO3jr6<#<sdZWSxau@EojvlNRCpSfXJiu;7^abLf-CI8pDENa*t&RYbPDArcHVmyO`+syq*}Po6mp#W361xs@Or^t{$QRdR1|h7lOUPG)B?V=Ki!j9{hy!H<>*N?``#%;uKO>)#@$*;9FxdS1-Ug}C-BMhk4o?V%}Zxe`gfye6L>@Bs)xVe1oAFwExP<3M~n22Iy~j$sI|DkD*tbOl{rUmGhZ4<R=(SjOe<sPS+uL~YT6ju-?R=E)f&SgCLaAU`Z1J#>Bnu;Hi~^&;~O;&qqwK>u@e1-QA{8;W>T6N!RgJ^r(rKgFl(>vC4GSr41Ua!wKzMBl+$Cf5kAAnZl?E5?BXy!dl+0&H#&rBg^j_5&O_KlRn$|!I)ro4?XG)U260>c%&`%}LF8fO)^R5v#3J#p?xU##$hmgt)NP3Y?32m4JUiKs?D{2sn$P-?%2X)A@jySia_SdAW*>t1Yu&LceaLzBq(yORFCJa`n$^+OgH{Jq$t%8hW69$0!1u?x@v7~A_nJk!aK={udgJ|0Gzw1;|KZeuUKHy}pX1vxdMzQvyrB&lIzmG3F1I41(;d?hvR2$&{}7YN(t?Bf<-5L!G^4@E%@>18P56+N+ur0+BmTdB{vi+u1OkCTAP@)y0)apv5C{YUfj}S-2m}IwKp+qZ1OkCTAP@)y0)apv5C{YUf$%@?qW|V>WLzi|Z4TKb*@Z&ct9mEL>!GkT-l{@D7YZ%sJG-LVLm;kFO7yi=2>3bF^kp9n0lHHjv3ZTb@B%t4CQXCk-XD|uUdMxB>b0j{U3?Hc=zac}j4=qpXa%SbL<WM@oll0a6abUE!mhCj1;Ck(d5-7q{@`u?@`6moOW3b*XZpjrm#}77Z5!X}2cyZVx)UmXz(}fi+<wUyZvQd)%IDw<ZZCFQDjxO)*5{<RIxBp@b?4OM8zw&BP->>%PUi#fcv=L;D!k#Spc~hBH*a|D{@JVSiZ`rADU)dQdqHn@m4e$dFEEb^7!P9d0&$hcuZz+=q4>iy2@@($h;c)X%f=osD6?>W<Ci=9()hb)In*7VetsDAhs+(+Z)RyTNW6fHv7h<$WZb~=X51Xf4;SD${#%8q%n<?yirChu?cjy&ltI!78(=LVxjW8i1qtgn3W9wf1E+KjONys1sA>BcUCvg5?%jKi9XZB=oV<}r+$w5VP@{A#X3hu$(_9{YF?xdb&4)NM`0X&LMBOiP)&<8>PTEr(bH{h5^4Tw3_rg={*BN&g_@Yu%0%x020NQ!!{ZUE|#_(%>`3K8FF|2;CrFwHXemf=pwPz+0$@h0Rs8B^C<#`^bM&20Q68miX#xxGkOJ$KIRVLujk<9zt6iKMcyVfVgpN#$kW!7gcQjpgE8BJA1DmrPXai7_hjw)X9qnfue(DhWf<H6)i9M9QUJ^4KglkFRCZeP#A)5ff(r5U-%%Ow*w@jDN@8Y#>bBn$A$kKWy%ZWiL_O6v+ik|HcO{5$1bQG{1zlXle1it*g^a?pX(C1`(U^z)6;666r@XEXIH#nPpSwl3i^JdrS5dS$r`+jUqX-i4H-S<tslt*aGCvP|->^nC@&&5$Hy`c)#dbN7{>t-?FyXDccjs{ZB8yrE3H8h4}w4RO61GbbuD9{SYaP@qNg>wRxf<&tg6g8v&7S!Q{VLs5$XidO^#Tx)UYV&UwOueB(MyVtYj-(rgg??qvcx9Gz^HP;yO77M*<ZajGO7KazYGVQl(F=O-%#kfo@-r9J@5GwWtfAng4X)IJ@u0%+qm0%SvC1v=G2vp#nQiJvTbEOz(R6*7xRE#8)@1%o5@=;@(R!mAH6H_#)KgJm)VF}%FB|7eKG&+_q%^_-y`H_o?`fOfs-`@LnjAjfBU69RPCP{~RZi)>qhkS4}D|^K4Q39m~5r+a>E8tZbN!yWKZy>w%v#Zb1I_S+$)YU!H0Mkh7Z+fN)uJ68^-E*)76x10kjF($srI*}sBDozF?pizaNOuB-$XELOwk|LpJd*N_y9We>f`rzsdg1nutiyZd`{2HRUsio<A3PV{yRydD4+l3|)(3<8K~bct?(}3o3{bSTf2JP*S{btlA<+T2C@YfUdTRi7R<6c1-Wh;8R<9$vk^^8<<l<<|JOFwou>&Vp`awf2J;5cTA8463UG47n1B2;<eA%@=s3jju_i^n5+5G>Wx|8%laD}`Q!*UNi%jtTZT-Xh%zs~RX)ae4D3jtK>KRaN7q`9@!svT-5S|cmRT7fRsvNG;`3#?a}Jso)51RnXNZ-S#5fHIaN=XPlwq*OD0aPX-GpMzW~b%9l|sut>`=vW46n3z|oRs_!^D57=cvtjiLzch7A5*%kN;+r)Lg^aM8TMEufP($xry2=rZp4vHlA#W4WfL583Q9la}OVgc_Hw&?4>zR<i)iNB`y-s_=p$Zpvl$xB=YmsKZfX}C?I^-KLd`~jnfM14>&UcM8VV~@?dp2z?c+E;vMKr$+^US)|Zu@i~Zr%yDxZ8yRCk5KK5A>iJSD~K!+g_|IOb?OD?L*yRGYv<}e!Tm6$UA{+0R8e+{ZG#hpz6<h8adxV+}H9YC5&qbRo`H2Xxk7ja5Y~X&>Y53R}V=`FAd}0GQ+dq^+u5Q4yp@|kKmCI)~lanMscxC?N?>yC{AjJ_`fC@!=D#-*j_4(;cIdq=dGYIv>da{xz#p?PaBNNe*79khFE)DamI0c{(61;0pB<l3+#FxAvKO6`oAh^u8(7Upw5dnxpCwO9m&(UGLAQc$R2ZXj$?ns+;R);IQ|k;JF578438dU{2fp+hLKgqf3#i3uuY+$<NcK}lzYNak^N&7Ew818>Zgq&y9uk}lHw>H+$Ho({M!i5?7j8C&vyhbZTc_&W*ET*-q)2QX~QU9^6s}K=P>Ss)P54o96}?9-C8s(LwLyO)~myjgZQxGyXi;rK@=%T<Fk1>fKIRW>%5xj$77XV3**xL7;bd!mw$R6HpGb<zueP@Ox1#Hcd!R}4edVATXZ9jN7$6#u`bjzK9#Ot+kxcMwowre+VNzux)}XND{`~(93Kj3L9bL>wcDqfaoi<zzBRWIW#u?ap9wTz+qZkiDX|WJ)yL1DdHELG3I$t#h1B4l3)5O*5tSHpx=!pxSQ+-f#aza~V)Xy+bnNxv0^FdJPdRcV3-PC%8vn@@yy>#e#;x)i!|vX^5x3=!k!deZUwUJJX`F1^{%daVOe=ID+&>)pPK(h$*+>A<R~FP4;?v;X@B<4co*cjrGl$ky3!&{}oz)t32{b!6KE07s26|2_E)Rq%p`Iz*A*HMuI5)>CYnW?+o@Iq}(FVaasYSPJtPV&(Eg|khJ=py98Eg930B4Z;bJ<KIJkr?rkgKT)HijwZ5@VWS_vrn4`-lIoOE$NKdiqwdzsERrtg;mnW8cz*-f4qU-og*HlWpL0O6|l~m39!<I&v<#xgFv}f9Tv|?|`GF`y+;IJ78a|<HAl=2c*+SIvIcK0K-uES1~6$f$7tW>^|{Mc+;aJ{7$124v|JL)S7p~HSgtzkq(_;QnHk*<Jt*UM?y|&ICaAOlbhocPddR)`7JGtUMFY=s1$s-)d_W}%AOa+I>F(_rHr7nonR1j{u}-Nf9(d%zKX4P0Hs{vCE?``P<Utb`OZiOgiB3N1+;ZQc8zNu+1n1-*%x`hs=5OtxC}gA)O7&;JyQ{z?hXiI=kG|K>wvc+CJW759l*;H<)i(t|F`A4KX2dfguYV=Mn!R*z>Rk<oc-7dk0)r`g(bUS`|Zj9KBRZS4}}w=G7R0odfmkQ%Ij`0F%J`Y%hm(IZ`F5yt?7Zzudf-$b$Wq>-P(?cx({~66a_9k?gOn4@|uzZePDE<u<Fm%eqjH>aYd%MA0FSMRQSX-0Cp#mcrFzTzyZ1YVcEiiAhv7dL}14tBt3uB*!OS<;&-~8FOv*IfNGhwOxiG9+B6<tRUCnzhZ&<qwnm`&C~rh(;V6U#RMm|ckAW$>b5$ejI4C<i(=pAD!<I?IW=`@1*o>E@GZ{_7w@=1-T3l1G@imhxWo-(m1McwnR!>9R&Er?=-Df~V;N1$9!Yo`?-upd)dJecKX5=f(=U|GpQDS{y4)(9ON399WgTky|@w@1Gi2BxhY5LDReE)bwYFd2(94H$Uv3vn$B%hJn9DE0*wPwGMJbnjeyX+6p*1ZGoXP2lX=@vn{p=Ga}&LSlJAZ1d@UW8Ot-geh-i@>s$d`psd3Fz&}Nh~dvz-)nK;a27nY%%>#Ntjv!p@JlusJ+XOvbvx#!M6-PKU*Ges4YX^FOS#Kw##t;<gN3nLCav=lB-3XybPP<sbAakm*MxJ*IQg=%kZYAQf1~}{FPxtT~N3Twms3g-&2=Czv7Z|e&{mrSx2#cv|EN?51N*Kjb#|Bu*#O>UxxkM`hUMtE(6z3$#tobC0Obuojwq^1Tyo#>#u1ofizda<hi3uU~?c%wxVMZ7(K<Vb=fY0zND0EBGV#tG)x8d;J-XPQK)m!d<TqjJY-K-7vQ+D<A#{q0-S&Oy;OYf0=&5=EbbjN53B}{_O<Ps2N8|O4Vk`kP$GjgKeuOLN1Y{iV|E7k4`jE_S4_i$wY2lGK2yM#=Au4xXA;7yNE<HFPXNcZ<t_H^F$mjr%GKI!6s8TfN|Y~-fQ;<z!fWG0u>7Bd_z9;$Fkb)E9K|>Qe|Jr7j2HL8s(^`NwRaD!+}@j2n9vCWYbh7FbKAh$d#LehK{M<>!FE_Srvchz-zreaAWX8fH21MpfwxS|tl?TQeCg&rwq=+FynA_{m%oXIyW0&6FAAigO);_kp4TfRX~>O}n$1MvKi5ouR2JjYGGBPUUWo@E{jw6~L_Ed%{O#xT2JEjd8^2K4j9I3Ww=xRaklsDjHz&Ch_d1+@;OyCh#cQSIGueIkhR$OvnrZ;;SN0~|eK3f-nqu8^GltN+pj@W>-#SauB*mF}bQCX2UcBKdHinM;VNaRvj$?@_jew5U1jdH-U;c4#5}!zjT$mM`!s(8-3v_hTsFN@)|8Qs;t$Cxab-$QFdlT-^+>^5yv9_^Pm^zD(7gv}Y_s(IrGu>xT(K#fMe_EdKa1J9kcWYjEn!~7U&oSztIovvs_+&kD4kd>oT*SlYFjMho<CfPPa{vAB)p?6K{B`5PfRe%-o_c(BS?~0}{psJ`UYef8r}lHZ@(pKkR?~jjWp)N%WpZB@x;BH%HSaAblBUrwLASg4;}mM{=}l*5pTf{=%h=;8lQ`7(fFsjs0!2x`4*JH9W31dYmU|^*xRg?t#M(59WgAcKC-#ltaZqDV9Un$R>f;d;b3^D%o4fn_`$0S<?IhsxWdKL6m)l<c*^iNvFaP}6?!zm8o;{wX?L%wlA6G+UdQg$;%vn?WF6@XGW99z0{u*(9cT1yd#}K1VIXS0RobJS4)$L|vqbk0Hp-mWn*i6v!Y9sDBbfLvzPXk)>(SZL#9iCi0?|Y*daaiVkd)!DZu6_N(HOT)4<xVnDD0ft2z6$l3?Z_(BytljlNJ1r&ZHjBYnyJ7dgZ&#4w<@qbK5%|?wHyzT?d3J-E=R$O^Wo$3<w#@IK<UU;fu)W$VxH+07;8_ixGGnP!B?_79??{xg0^F!%J(Yb`#uB$fj}S-2m}IwKp+qZ1OkCTAP@)y0)apv5C{YUfj}S-2m}IwKp+qZ1OkCTAP@)y0)apv5C{YUfj}S-2m}IwKp+qZ1OkCTAP@)y0)apv5C{YUfk61b;@-`pl)Jw3;Cr2qWf7`e*gu^7n4^IcL;k2z*xlj4>LzjP|K79X-AaD<wh%TvLz)%ZtacjplFy1!Bru}?mqUFly!+9(X4di^{C!J0A3A=$bo#Bn9qY5Np`WfQlu-ZoO?wx(w_Y^6oVXiSy_Z?Vqe<XE;!7R2mn3j(yN1p|pA?QVEghLRAcNiBQVIV5QNxc42A;ZTTB!RYI5V=fAIQ0x^9GgogHNv7hqlzcu+1j%i@BW&b`GEAo0Ovh%(z)?o=6LuZ6zXq#1Fv5Yb7<Er8L0lmo%>YksKV17Djh}p#-aYj7C9v2cc_IjZH%7D8!8Df6L=O0axCVH0}yz1_zam>7Z~nSPxfVsTAV?KY>206cQfT<kP-?!2J>|(m(hfy5}+!rUkou+wcPeJf=^w5QHbo&wuKQ3d7z+7E_W#q7ePJ#^SS;7_cAScZ&S<75KS$F`_E<3S2oKu>AYi6*wZjc(|SPDp37adUUz%3N$@nzV-gP7@X99nn`n71eBS=e3)W|ASya7&+Dfkuq{gs|2!uIxolyI3zkAKJM!4C-CGFqy3D?+y%2(>D^}K3uZ5wcNcwg(g(w`o#WKv_E&@v%lQi1zgu$W>+&*#$!^W^?O=g!6{9-hzdvfSse^Lh>zmW@q<%gFS8s!Bbc&~n1hdw_fMx-uuQ}KhldbOCr5I-D^$o($NCIGu{9mz-)<%i{+SHCtbE`wub<_Fk+5o~%t)N4hbgTx*FS0(pYp=-_2E7b55MEv*aSD^MW@Uz*=`tZyFD0}+*?GqMq@SYZ8W~|w|`qLnAD#>c>D(Sa{twXT(HZfYMDMy<W#rC(TJfx$=j<BKb$@)V$gz4*{e;Cll^di%BMi!J}x}mbDa0cmh`x(Z#IWRi!)f44uF0{<f^Vqb#geEsb_5J7t@B!VQzp`XvC~3JT4$8z)PxEWwN{A$W`edYdkW3mKD*j&W8IeZ)iqmhNbIPEZ%xag3jtoi(xk)8t%b;ZO_w>zC8FWlaJkESf7LS^|(NbcS!zTuR&wBC8qrZ0e`N7#6So*uX(sfG_4~+>qd05>+{`SWaY9E!cCh!Q!0=+8I9DL!Z$gGKTHEEAp{%WIbt-Z)pkuH)6sLhqXGQhBReuf?`#(4chRi~1oDFS8pj^$xX{3E}|C!ga99#A}cMOpnR;@i?MN7J66g0`$e#Th$%xIN5FGvR>bLxTIvTb)sy-TwyPwkv|^pQ<C(FHrKF(Rtb^51f>d7~_fZ!mzTruI4Hq3<}Jd+q>h3BtdkPRMr95$HI2piYgetF;0^1HV?r(XGdeN|3dKv({oYDXW_VKbm#fvfk=!%1DgBQQAjQ?OY+n@8ZWEg3MUbZK`Nbx6kiX-p%Ym&qrz%D9=0|RSvOC_lTtst_n9Q2XLj%C7n5X^u1U+Ov`9g+YdiLVHmSHr)8tu`QyPj=zKAgKNyp^_J*ReuX5e+7mhRV4nfO!D`RUi!S!j_{m>m(5jlK8ZvByN`pp}qBmUC1tPOz<flMl~BHUIy@w*vC<(V6Oo1or~G?(*)oqHrOeEeqh;omYqf#pe!QI$4Ci^cIxHzC}nSr^b7DwFv)Hy?@x@ZZUd{d6IrBDMpJhhFd24N|5r`d8%x^5}X{NJ*`?=f_sMC#)@`I@bB@Ch#=`w3_f7O*8HLrbt6rJcdAQ~J8U(X<Z~&Ge5kbe?`Rn^S$?}`DO!dZerKIz)ynYB$KcaK=4BX8waBXMScWB;pGQR9%P>N(nlA8Pe3Q$V80J)lM%$*&epY2ja=#$0@<AC|+)VnuAXSDmV#-OoPnTg#xR=8AdMOShy4bC>mg1nmD*LIRQnYq;Q^->+Me14ec>{`4R2L9?bh@eppN-4CI%ZIU{w;U7eQ8S2w%K%uGrt(6b^gw8$rq!#yN?~&ToIPE(zFI!6yd1M2>p|dLIk(*^n1>Q_(kQIQSw#+=6j6%(fgT?rWri*1B-chlF6;%dv`8A?0>~#RhWZKpNzTueY3H-bjjeHMi%CoH5q<l%|sSXva<h{(y{n%fE`(48eY|SrgP<PDz=$Nr3mgxLEr1@R6GSq$R~Bv<(_IH9&ziFY}<^(lgrXpQF1Yu>!EEF75)n8A_Hf4%|+nIna#dYwlK8ooHVY`3&uo|Pb_Q_FOhGr=)UPnPjq_7q!4u36|Z#6*FQ10!76S4=YjMF*b8qm$jtY{L6u}L#W@2o5!=`w=3)b~zYQk^W?W!?&!{HSc!F+i*=K{}ey|br>%Z1RL69z+G<IPp1pIk&mkS<;L;RVua(P{ma3HQS&y6h_Do8|Vc3H>3@HOFm1?6$Dz_OZha3ukD^h<2sP$WTr_FvMieaR50CmK|_F9mwc^(f@1QlY0m@togID$u3!XheQVgV>A@x%Y?DA=FFbdR$os3|`_QEf35D$A_m+az4z0o5$ZTC7sU(qbDRM=09b_QNDoYmq|G=YJHccMIje9be=T%ea(gWp7+m8z4Aaz&h4rneLnp1xfWCSDj!}XTblYGE`Tw~%D?w~3*i3yy|lZ37C`T7q$%Jkgm)CyDfcxBA#f@5^N$yWAl=*Vt(8#-Ii?p>Bijl=ap(F?rR72p>$E;EPf`Rg(^>;}A1MM?-<6XjXN#awSydtJauK+B)o6!_6~QRU>H$;9BCw?;O?8wmg1f(WZ}tBxpS=`XUV60%?9?Jn9fXTO^Zs3CN!}tb=)U!L>t8=X%F#i|3`MXz=Jh2;sv=18zror6y$~+HFz^_kD1`7R{?EK`3*iF$H-Wneg+N`+s#5P%2onu&G&KH=U*1pGx=NrBHhX#toDLSkmhn&3iopWd7H2*G)aBoKEAu4&$5sGAM{PGBHRprpjj!pW5A#96KG?qCcOJZlGtAB2c_8PQA0JGY2ZEj<w+=?-g6wn?9mA1a$njA>-yf0#+6=2!_S8Ax@nUxHkxMq5l|L^p{w)iJu2uNvXl23i>r69+9hpFJqTcZ*e<t*ctQHF<WWbU86}fEW84!_QH85<P4i92t@3*$4L0qxfCEIgpQ1esAfX_P>ZaFKxp;}A<8<H>;0Z4(9f+;hiQOO|b;c0#EV-jcvBwOhVCqb##yDodLM5sF~TZX9#@P)Sf9*J-~*!4U=>(?3!7bGq(hu?_-c@zHss79mV&3eh~IgM8^-?^W9WjYe}`yHu2WE27K&#9~&{u2f|tVP|qA)(;lr5e$4HUw<^`N$601_4G^{R(;Q4+pbC^^R8fg4$<Zy02Yc&=Sh}_S-vm$Q|c7rc3DtK4)aZ^-Ue2mhDhCQ|mLh`9S@y`*ABEIr;h+9~uH}DSH&Ty$Y0dD0BLV@4|(cAK$3mRq?^>b-~@@M(EdF`>apN3jG)D4h1hiLsJr}PYJq?*kf?^=dG`<IG^#(>PwD0w)09#+%@vTNA~k9ZYO+^=F6G<|L`TczX^M%s}hKR2NVxpvJOW6eW&K-t3vR_wq0P?-%zw{$j_P<3CDP0iN}hz5!lA{HS2h3B)+0G>?Qpgh1{!ej?b}0<Ep${`;O9UJffDF$nFq>x?Sp_X|b`W`@&U7y(SJz?@DS^PR66i!EcSNe-cnB-`;D(FA+178@afTCZTQ6VVq4$LXzc2)A1b1$S&oS*qN7%xg{%u(>y6SPd*Y}Sd@a6Lemlkms3$c{6Afz;#Ax#_&}BFLK<#c%$mQ-NW-<UDn3)DbR0gQdpkNb9UtsjHakI<feB16$b_F|;0C8}ZvDFqEZG&aOII-y-Oh~!%T;7zp1(;G>!~bkOwT`Z)H4g!Q`k=rEM?)(zwi7+CA0CQXN-JXY&Pzl>`edgGaEg7zdt&mkb_P~sOrs<a&VkF>|p0w4#qm9ZZSya;=W<ZME~Gil#zwm<mp^w2MOCb_B^DhUfK7>BoA4S+wK<4$;03#b!OZ5d3b{%WN?NxA6GAH)cn-W$E0wnQ~WXc_=WXKY~4^kvYGFCoJ3tf{GXFRAP@)y0)apv5C{YUfj}S-2m}IwKp+qZ1OkCTAP@)y0)apv5C{YUfj}S-2!#I+c|6xSu5nlhm=Be!wQCAOD_4$hE{7m|KSTB;W=;^UF6fHReGmfEPps;{3q(LomEU_#?g~^yGka9Ai-WW()5Vw-39t)#u^Ou+1v0hUbZ7QU!=doH{+j~QAXd#s_nKZBl>N0>pD{|qwf4~U%5f<uB|owXrc&TgpvdCEAq8$Dg2sveNx-7S*?kv+0igDPrjCdh_@yb09?uYl{$DTU{nP~^^!=HHzk>V_WhD9~vWOShX<PA}Cl3g^)C9%_aKH~$Qt?Wi(;%f^<Hpr-5E!iJEAKq4efy_s=uAfLK4dSRr>wigj1(8=j<<H4!yVGDbLz$y(O``J<)N+1X!Xg;;NqA7KB4e<q~9rmZ|#iEs@DUy@{4_EHkL$=G9kI=(bCv(ELxKDuq=xF)>@3Mmc`q8J=@A|vdCN4{ivlu26t|`Ig*me;2kYZ2Obw`tpD-3g*Hwa3)?T>WE+sinu@r-ONXT~GF@;k*-{coKT<ur#V>(k(N3lZ=f#nB(9ZDFV{!C3*ZY-HN*o_ZhB-!fi6cqj&;ZlFb`|kYS4!I@&<1Jls|`yaML3)7bddylzpKsTm5{*nfxT8nG2(~@k2*acisPuiMR|0zIPU7QmlHZCfnP63O7icO#E@m?((+14+!nL%u4j_Mlr?`6l$OHz=!=^69a8w>M83w1oHW|1M(C!fNn>Ml#=Rn4X$(&d`$FR+jZYc>o|lP`Mvj7KsgFvf@xIls!*5HZ@$SgG6#9SB%lnCTgn|r~73A;H`z3=<#O@w^#x9HgWOZJ%D>B$Tok@AZSq7En3!A9!$Y7Vqfa~FVGWh!LOzt&N8Jy*folFyx!FzZ`x5Q5dllSq(e19p6>7(_uCcCcTtNiB*9VYTvd5)WX=lTttRg#ml`>udzoW=j%3Av3`VL{rC*OYMQrrWZ^urgW<DxUjgdk;P3x6jJhsUk;Vta5R?I=1xp*nCgCj|+9wbNQLtxbIR&m67~IEc?Xs+N4t#U2POvNd)w97o)ucJ~F`VGX>LY#)h~LE)6`TFv5Yn)!WXpMtJe=ovUWNMkptoe%bG>A@X&EMU?j&pk#U10}~y6%x7rMf0Uq$ocz<|!KFILq+GupdO!<5+g6(UlBi*p?18BqS|y}D9&0x*Acup2FTZdf;KQz7afbkP7SN5SqhKwRf<K1@86{Tl0PDdsbPC!U@X+{#XQbgnc-`!`BpPJ^iR4xWN0%P~7uAwl*DZ6%AA@bL>sEk!MNBT-wFZe(;enLbZJ?JmjQ72`Eu<>tM&ygxK?YB84+=PdX_JLD<7Fo>zb$>$lg9;$->XjU3vz|vy}J3e-fnPgwTXJk>jm^?eoc68?+*2v9sU;D9x$4a=q)MW2`8UFPm84R0-L^+m$wJKKtFJCMK9PJuI#Fb9=hoRr^}XZZ|(R1sX<e!TZ%6%v`eonO8J3zN^XS3vLATL#nC<XcnRObx+~fb`2&1#f3F(r4?e3>0auR&!2L&VKD?d*u*WRDhVs7vAU~t~b3rN)Ug;h6WQYg^HJ)a<yEB0hmE$^ZdnyQaW6@5Wau9sL&K)+pAfO0S6W@sqf~?OMXXf4pf#}Tc1IEKaFn^nOb7>+7@`nS=R)&JWUG!5zPiqiJXVi(86$HWZ)6Da1!9l>Rar)SrK@d=;vpM_n1i_*4MxUZ@fk1b^c2zDd5RB96i>2iQp*HCHM~3A9@HQC{Q?dUy?lF6HX|e#wUp*#U@8b{UrIjqVHeW)HPmisL>PsMXA>G52;s-Ch&iGAy@P(GlKY?@Se1WBsUXMfD2i%|c|GeSv4V~eXxSit#mh^ijE?0Yk>QSeZ)?yDZWjaoo5#SEj6vZpqAG`pg7W%1+XWbxu_KWCbkqh{?z5YeH?FfotFXZ;$wTDH!Pb&23wt#8({&r72g*|?cjY@x50sJ{XS=ej_fz;-Tik?Owyy<^?iSHr&?3RhD)4T`G>ON`G!r~x%yZcYf3qIsw5qF4lR6_OZSCT%^=%AxTxmezcA%5ca*?iS+hH4_+YYB!=uyayZ;MorwoZuT`4Dho<QJ?rGJ{Birkdt|0sN{-od+AOT?R$Y0hKvf59qvfQlGV)?<cT88=l-s$dt*Y=Cxb#(U%c+5&9SoPhs9f^@$4o3`1N4t%#cnXw#Hd+dyog?JFDT_k(0rwq#yXEvM>a%+zn{&aSz2Ib4jfSx5Mz@<-pUV4B;4<db0V?yKs!rhz~uH8G$#(`|Nu4BJt(Cy-)D*D14OtBKvYj6ej4!eOrC<3Qud6Kf6U2jVL7;w3Qc)AAWDfFG{_}u$R?m!+T%j&0<I0o;xw99xAZv(iMXq<lnTDuf(F3XG3ygdMqv<3$!t%jl(NLyEjjm$6?J0{++?*IJ}dh@kIJ)Jg(<#e+hdSk9?kSk(H_O=#uxbyKg=o&6DMr#TgQ?>T=Km+l>VL`2I1+SK9=<Ds=29X>tNeFdF;(Zc4x-W_LX<&Lp6|*Vu8vl>~I14(N$kOTbI=sR9E3V#&diyq5p!<EJQ@j=f93blx;N_u&MLyl|PK{NMgu#v7Ii1qqlZ_B?o3R07i8-seE?kbo)$a_v-_38<GEb>t6U0*W7NQ`_E?fcw0kQ#}}n$KK<D#X<@3=zMJKyt7d}HV7DK!nt_t7G>NrosYxMPukx+c@>9(G8La@Zp2}7;jS~L8?l&otDpH*QY@Cg5i!z{h(&iL&u=#-WANo`3HR?dF&Lb>Dp_|h28H8~DA8xXMzgZolIK@m<6!DZ-Kp+qT=&eDiB^k7{qFEP5654j+U>s~Val&?k!H|fcUKgS=5s$Okcz_XLWavZ6_M!9ul&NEBN9c+wcgSuMxgkOninBQB9P<vs}gUYaQtC^$FA*X7;4OT3kPV1VYd}qr*usyk_$Em!l6*SPkwo8*f0bST>BKHQWA`Lb`=f`Ye6VJ@`+8>ArMaooEY4{<ByUw9Qw|-FHvcMt3hPR7t5)v@4db5gBG>yxkI^Ls8ew7O42@06#7Yikl(}|70lR6Xq(*dej(+>2W+mmb^6cunNv=v;_VSkM`@4M?-y%kR-d5<=`a8MUTdsmjC1f$vBclS)Q${mkMWn`t62I9ee^vR{wIo63ztuPSrKiyjo-+MXRqp9!jB_@Y<C}vLsoPk=kemZprvtA>#vm#P#w4(RqbO4e24RvGUiO-#iPpS!8ukCzi<7?;`^tN@?(F;7Q=J+xXZzMhTj2}Q&xm4Z#cteql$dCbXS0$s53%zFF@OC!$I|_JM5fxD$ASp0K>}K5Mc!`m}8J;m8$WE^Zg7o8fSf>?atmV-7r58)jr?ky!{er{jHn$v;*Lg$6++C4utTxmr|<t2Se`-;cIP8!SHT{Cx*u+1PbZNQx{H$LhXj+Ugv>OVCfot<meFwwEHi`2n&aU^hyR(;7T}nE0pc_kB$KC$0dGUQjzd!N!*!gEfV4`9U0<JiUL|8<`1*CUO^F+#_X=0S0L2yxlcAb8gQ&aDEH25a9?vw2>$gNR{o%=YitZm4eMj0bS(7mvR7POiG_jv{5j5%aj>6De}>^|JPhc+Qf`@z2eOab>nB|kz}$qT+~{B;bSyW$Q_)L=+e4-wXG;@7kLATvc8Vm>m)*@nubBk1hx;Uk@{>ShuNrv|X);K@>OAD9k_-)|VOO=XlA(<1YK_+4WT+i}lFV>B1=56hIxeK9fVHWY0sHS1h(7sf*FCvZpx_|CXcL(V*Zekq1udt-@NF5d7``-+f5sQ+;F<;-Y%H8{9ci$YKEG#%E*&04dYP}Pr^CUCcU<!c>7X+}9!<NL4uw`}CKPNLkhHhjYx_Y4tnJrm{T!76P2IA8YWg#P<xD^6Z>mh_zeBtCx_BnY{eH4RWs?a~Z{mZ3GcrM?=62=ISSDE1km`z2WdV&Aca4ic7Kjerq+flI1*K0Zs9QX<pyu+&I?Ty}htlE9ecf4bhWGk_&`K8g>^V#6Oq~tqeSF3z&Srx?dqDN1cs8UxrhR-!H5*9oMmt2BW&^Wwi1r!hY_NVoMz-zyFV^z>Ne<42F(I8?+JDi?Bl$&|cQ!mdJfNoZEE{y!N&A(xvH{QTu0D7*8v<J{ywyLJ4I80@_iq0??)b;AZ_TT-U|-PSy}e#pKyl=Gc=N3+X#Gunzw>AoER8Q6o}I~rH6zacrzx3`2fCkF9%aIQ0f+B`yqWNO&z@ZR?F@*DmRmpEodGrt#m&hH8F2OK(KI#d42Vn>{~RHo0h9a*E;Ik~d;F7IFvIQ)DBkY3yS@A`|5Mbr&-SK+xxAZ{O>H{RHp><rDNctk?6kt0IqAT&pY7P5>~#3MbW(iqUpY%&o@4jF?M>b{_h?n8!{oAKI5eh1bcGb7VRt%cKCI-`8BYhjMcXgG-lv1bZ#@;RjdXYrVA%7HCIb>>RH9N?GT?5tys4*f1{}FTpR#o~158Du#8WIY;Im!o_*K6QXx)8t#WFDi^bC~xO-nPNy76XALvsei##&nW4Q2qJVq*X8=?oYgrD1>bE(1oKDc5M1Gk{%tFy+663`l)=g|l=b1L~us!XvseATTyLVyHR;9<4YO^QLEjeD~$&)Bzc=M?P11-Xa64{JMFs-pYV`q1=c1&u2jIyeZkno`2&V+AI{BNC!%XjZBWbbjaaLnf`5`4vCJfn(7kiP}TBLtePwxA}d|giaOJv+|uZEQg9l)xcTSx`#Wi18eP!Ea3BqwWiH&U>rRCvX|KYHm#OfgfaLCxYpL*P-$jKRJ1GzaXNHcHra&RzryUpL6fkuy<$QiD1?Iwwy4aeN;b#BPnt@d^<ZCChQ=Ci&_b>E@;w?$=={MCnxp@*4U+1`1ekch}v({gzuS|r_!p!eOs)=xP<er(vQUY)bW_>%xm;muf-A{e*#zWNtl^0oP9LVh}mp1N?g~Y0r(!B>_;f}VtYVyq(@IC8iQ0Ms?{$IZzAP@)y0)apv5C{YUfj}S-2m}IwKp+qZ1OkCTAP@)y0)apv5C{YUfj}S-2m}IwKp+qZ1OkCTAP@)y0)apv5C{YUfj}S-2m}IwKp+qZ1OkCTAP@)y!vCL=UON2cU5`L-$#rk)od}emw@*#wjl}XYS8x%EL?!dNEuGm26g@X8WxyYS!c8j;ZEE3o?yOpZCr=pKjz5v%{1}41BaU>rNx@hoCqdo76olSKm28AY1CX|X^o*#wKdyHq1sqxM#ewsjj-2A&n9<O9CgrsUKBzGc3G#45<}M|j?SoEeeY)`HJ5F0X?JJiYwr+)N!6WM&T_&jh`rKYVIz#*ktY3u!_3@Wx(O}eo0aClxXOsUlLiYk!gVS8b_`pNY3e*g6|IS~|w-Q<?@oT-4t3wvOPMbU!T08@M7K)9+b21P+p6t9~a}!utQXELH$ilNbeuDfS$Ki^z<;S!q68Ms{v(sQh8Gmyuiazeu!szXf&y_fJ@cYA8g$`ftqe?w(+19!m`rhd{OkH#v_i32j<>otqZ4QoQmDcj`gPJq?aGVBoQ{=2`vFpR!Op_~_@*{YD&Sq$<+zdXPKecju%MvK;0zZ4dwg!u^{;36KTiE<upw1@b2)`+(>O%S5V4+LySx>nK$i#UL1S<K!0)uR3mFr8Oef2V&L@yBhrFcU{eS$&NdtwjWXbABBZMMsn41?|k@qqNHaQHS?&5-623AM6w)GAw1P#oE@t)&?aZ<US|pPzpX73`V~`cbj49yXbp{5T#m&mKGfjwumZ_g1!!P$WTphRaW*o+P0BZY+~%nhfiLamxvNQb4+6M)77q3W&rcHhNO0LUj(K{f0^^RPAF|z^GIhD;F=WUQPu%r9<yDY0`k}#NXueXVZY)Wp@daN*WydLd)|uHVx*lp33<+n+DX*AG&fl)8VWCN6r6&(xLh3AvfV4>Ckq@L+A`g2Gl)#r)kEW0lur<@x13VV2xVQw*OuRu&xhDdX{8>muk|FtzDVG;M6M+Es+WK7rFJ`xMYGZ+uORa=u8+oD@AhOFB3krD9|jcW&+=ppe6>IOxRTN&%IZb0S>))uChpFfVb#kgh+5YbY*>r;_6I;-)3ZE4tvs|bt2K#o+}kDKm75R%pnCBrqnhfkEcMl#T9MtqbYEOf%54kvlO`e<xNrWw-gw>)0H+6lL`eDBPV2R)8JRB+3WDAbT}a>TQGMr6ILrN(Q-Nqe*ai_s$7->yANe|yVT{uLicPR<wOC9RJmBOQ5Hehn`4T~BE{fq6J_oHv;^j+96DQyOF=Da@MP(yGElrK%6?X$0#a@~XzS0f1S-bohwEjlf#YaG+QG#dU^UI@a|o}6k|y5JAw7gM!uMX<rPM)LpyU^&xO$L$)%3J4r~w|vXp&xVZ3G9c^58pGP4I|3cFjh&8B%nZi&%_WAcMKcdfKrS)`D2sd;;6xYVU?UWlTHh*hufmDeQnJJQw4(hdP1WT7TAotQ#UiY{zrNd!YNi2K|wjy>OQ8k+Z8^A6y6wX1RH#AGU88Bo&bkfC1y;p4HX?I9%@_5Fa!MB2y$b3(7-~Tc$AaVBawK3!MuoelrY~DjnPfMk4@1Odk{Xje@AA%pc0EQTP~rQf6In3_`+b92aZG;OpM9XTJr;fvIzsfJx3cOjZ6I2Zjll=KbP*$YTOFt7C6+{Wk$#0%GkN*C#<V`rLo?X_N3c*QLJx$0YQ4$nyOsJO%y{!WR~wPJyR|#V+Q=DY&;o`!S?;3PkUUB)4==LB7J_KYqPaP#5>cR-kPP($}0wol2*G(lpR)B5Vo@O_eMQ^`@Yo_nRRZ(-d?jog+7EorGqkc`G^HNq9igdZz!w1lalAZu_o10rX5Atbc~b;f0f(uYlAzJj$onvWy*rJ!M~i{9PUe^Kuo=mUE+U(wnBn{NV`D<o`G(8!!xF=HKr4qz{3<dKrmT(ICiCCnX*z9sqgDvG-lM{ooj@H+m?m52P1XwA^m?!fnP8`{y^h;RM~<BW9US2-4_q?7G|zM^kqn%%^V!lBzdWHx~ZQf9mx0L-`HB+q~5nYEuXOFB@A59=`!6`ueV;8x?T(eQdBuNeKvtaOuR56+oAmNuhX3I-LI~ye}jo7QQL?b<`JogEZCuV(%`$s`{e6frEmgprWKAjiMj|3I?DPTQC3#2}MZ}`6&{TN`rJucXxM8y6cbvQYs(<k}48%{Ue?`o-yv&pEutz=UHc*J=WT5pL5>q{o#%gxd~piOJ1a4`+(+hTe4SU<1ldR&g|(O>8MI(|9P1)7hmv(2wgi=h`|ght$D*G$U*sXIPOgaN{+^UA$O`mF{gDORjC@ZHWjQnzgde9|2!#l4y{MlyR7Vq{EcY%<<3*n+$J>f<vzf8tOdK+^y{?UTCvtJBg6Dd8?K4-_zduLpa#{ghtcw#$g=NG#J96u*eYm0zLL>}zg~z>QnHe;P;&adrXvZjv*fJAe<tCjD9RZH(Qb_CxwjZ;-;G)FYNSVP-I#KL(doyo9t`U6Ob-$5L7l1OyN@(`kT<hh(#*RD%Q9?vpJETbSiEQT;#&`D@gMZ+-QA1B_A+12@%G}aQq#c8*Lra(SFZGzS}%r#kd79;?#0VzMdanId+|mPS6}+S1UR=fW8RDHyzi2qz3j!xg9-vos=dg1y}jzw)m~(I8QElcsuyQ>x@l}3>cySKzm=JmdaxjfR=lpQ2cI$7W?qZ$K?;qbh+mdH7(4&!%f5R(I5n&lCCb=?4msMaTEpGA>GZktLUcFgtKVBl)9uEkl2T9I)7|)Z^6a+iPZC!B8ZXl-BB9~3lX#jb36C!f(X5^!ArsX7VVdv4ez|+$bFp2hC{D5zQRu??-yGp=e>#!Ynzd8%VkeR$=3F|=JCMrYrcwd6W0P7=*I$Zu>>M=udH7lzMlk0!(Ob4+zghc<Cj~9|?5e1x^-427@)mE~eWDq)CfKfTKWM_1pudqN=8bq0Y)!_(8}MDI<lmj;^~l38ccx;n4i6V?sjsfrqNm<lw?)Q(=gD5ak|~KAq|DHdVt-YQ_oQvM7-Oq2`{SIyaWdkmv)t#C_f;a30$W#`NjcJ1Ju>AlEyX)Nr^@-JigEdBvd=I}5mNi*${)O6fQp$G+L{`9Se-3b7NnSiE%bECMnRcqDqGCPQJ#wJX==;o*Avm-z}-TcFAkqhxF?@Ch(vXM+n=@fKcHHpq5zYkC!ROq+0mf;4wbU&*4StO#aaJr{xE6@>UB%^s$1P5Q+6Zc>v#ZYKR=ai=Mx4Jv-yi45CaU~2J&vk#lsf|v114KCxcYX!PM){X)x2dUaz*1397S~GMrE4z?NFhw?Biq5MZchGw+=bG`o(vpTArPS`p&?zuJo6Dhs>bz`YV+C|~ln7%GLCm?wg<TIFz%`<dgD<_dU|7N?+p7{MjF|K|hoDme4_kc`QnDj0>e(S)*U&<gL)kbYeQ5#=9mAy+Mk{53DD?W=_-%IkR_-Rhvn=anC?SUqIW@@+pEs0Yj9O%86I2Dte4O_0fa18lWt+bF(kget*!dDlZt@W*L6WGt!)(w7IVYB`&Mt|VJq+@~3MFH)K{uQbE%2V!a7YAx_lkM@~%dkZM+njrJK(h6c`(b0W5t&lg78FKbm8)!EqZ*#}CLHIROU_Q|fclSEpKa|@JYzqgs=7c-oTgDmvrqm8_oZ8shd7=|aWbb}&jOzsFNnw4Y>H@KB`v=~lT`>7ayoz463$FfXRCs6B1(u8%hdhG1K$*|XQ?<GaD!qo(<d(bOvV5IkBnJsRD!ry7q)1@-j%VKIB?;UWifdgwNpR~mqvXX55|kXOThHhu!JR?M#mof~BwCC)UD(?VE;AQjoIcVG9B#iKl<;)Jfxd)ijr`qUcy+J(C4p`@?pSf~6kj*YYEs?V;{Mkcy)v)B)(yF#H#}SzyCK^7u68(8H{2?JKS8%f0tTIoMaS<XILf2)<K-v`a-<uV9s5Zz@q2S5pF{$4Ek)M7?f;JR+>*0S3kjP2WQBE`N$`QDjXA%b1X<?X??sRV^1161K9wYh>2MYsuOdP6o7@*iYX2So?5rl!zw!N%f!uYCBoODm`#GtU1do5tuAIpv!O};A$kbR8Fa<Kdx!_L%&$`>0+qNX&(F%GOrbmLEuXO_1?vOybfvPTupF}*LlRzL42m}IwKp+qZ1OkCTAP@)y0)apv5C{YUfj}S-2m}IwKp+qZ1OkCTAP@+I|AUeu&u%(j48WuBl>})21mH>gZ>@d|AMj?`yd8~05OQ3)bMaAVFh2kJ>b7D@2>#xMnf<k)c;|wjK7~^lMmNb6Q%Hs5E|XU?qp}hBa-NUk;DJaiJ8f`P+&B__B-VU(`bDC1;p%GWqezVKER6cR9)U4EBAvdj5g4_<Kw;<aaP*e7370<@j=Dp}6thdAIJP);dto&M`C`}TmFYu}rS6_;k906LnM}QS7!ib)pI1{E_6OmL(5I$Joey}i<P3+cP$0_j_V-bh`y=Q3cN)WteyCk8va>JL2YX|*xR%d)qq%de+2Ro|Y&^Db)Gpr>3r}(8wzPX9?+;Zy>t#=@NRsjCO!36f9r~RPGae|~#-x3*%^lt7I}ICI-O+`!`=m>i8%~M5q&=nOiXCTt%oBKA@Uz{Jp!+2!yn0TQlfT^oZNv@<4Yt@JpP78?hN=ykT%(|2*0e%t4{zo{KU4g;9?^FBmOk>e{`@7Rb_JC;8s2u6XhDtois^#-8z4E2XGSpC!PA9TJi{f<z#QQctkdoec5V8uHH+RL<+5l=ancWFo#LdLmjd9I!@?Qw;ULf^5B=S36bdDegpBV^hC`{OZ~XH0DCoF%G*r4L8iX7Cx}TTCfOzr35zvf<QGL1jV<+Q4=yB{l5sG+Vva}*AT8xKNo&!Ob+7n=IXy5M}Dv5CGebi&8u0$A6Fj1bnoCGX)^V!)mli*px)oB6MWN2QBC|qMmf%Lqs!riGU5VU>kMhSH)aHa3a^0}A_2L?77WA3Ix>B_yeT`yC?_7%gY+d-*-^sZ_r5>nyOopHSv|MKr^wVeT>sX#$hG+N`B3X26a7c5Lt;cbz@=H2)I##PVdUr$K|J(+yd6KSas8MODy>o2LmOp<CoB9jJt_bGEPC#1p4=M3D=$J4>pRP>MDXgWyhi;XFHWWYwHLrpP7Cd^22Qs~HKg68pJ8S$7*C_ddx!Mm0TK7qUYd#`6fKBFMzbDu2O=`(oeMqCy=USISUP0s@Nl(R;O>sb)*p5u`zo(=Z<DE98K%m#(=k^9}r+3+&FqhPWn8)|3|a5eR0L!9uw*OOh@K*z$@eYiUtnhr{Ae67!hPbzVf8PVC0O|5XP#wi<A^**ls`j>^SO~>!MWJ4B}igx*jY*?h~4_a;b*MF&>9(*VVwkSj5?knfOo9~Nba)CJ@{x(nM>vRsJG@Q^5c$y3DiDe0!>$yN-5>|XHI1d(lvd8SX^5F~r>X?joKAiTtVRD0%4<zN*bHRV}Axw$cdHz-bTt3>cWdET6vi1l4_|sJYJfhkugMSMk56;OS<tT)YLyg*-mkL4Y_jilW_X@!>_+{-q<wD?fRnc!yEQE)JA*t=`g%FmS$<W?b0Nb63ijhwWV9iinX1FjP{EglmV;so?4Z~B%GX`?O;kos>(Ucr`{we(%hjBL8cG&N?&ddaXa}&G=y3>Jg)%HK0nN(<a_=?iJAsLLvaOgx@A{>-1yS3;N51d&UZ_8C<!Fc)k_RBNT@Qh)8(S$A>iatIgYoHB==oRvZU#tV*&^Z4uX?9<DT~u%+z{4H9XH1T9M>#_Hx!YHk)hr?Ih6p`^HoU1=bXMk<!@l`5$*(C4F(h+bg_79{i}L>KxUy)EdRqDieAnKi6L0q4;|1QR$u|}7exE-UC|U1%w;6~m^Lt`4HiFSiCW-0Mr!b7zv2$^{DH1j8-ba&Pia}R3J9ioRIJ|WugZ1vS1e9ABzoq#h2~XX04LiD;f(w(OZKt%;@j-65Y;<ELmiCHWBR9{+!TZ@uLwY&5!((%|(5+m|cr4~U!JLN-EeC&7t>hu;ptaz~*?dg9{;`E&r2r4JUVLlxw-EUk?wwpdP>dH=nC9oUicxBW>|#Vw3Hl!yuoqP>#reA5V$utxDE7QIYV}<iQWQjgN@Fa?nkQ+qGHK;_%`|TG$;k>7y!%!9b6f@9ihp*slBN=QwR3M!Kd3|zLFo}E=SqC6F*7bvT8a14>m+ElD{-Qd&h>{l;w^#Uyk=R%oJMZEdIxb_<!R^M3y6!dKRu}TAxicPzx!KKiBt6At_$jw7<yBKVT7s@>89n5U5>B7C+2L*H_um~0nWL$ek(^e-p6TE)#X@mi9a^hxg1X>d%hH4EyrCve<TX4%aHMx?IB;EGV~2RRr1N847sA6&WhYFL&3;3e^I6~q-UBaeekIi_n!I`cOb45ZwsvJ|1m2?muB~IKlxI8wk@nubEy<-DSNj1E|%i0kR}#a&Qe_Gtc;o7F2Qnk0j1f`CHQvy_a_@H!I$r7$!>d;VBI^WzsK&DAa&-w4)Q}KSXi<D)V_gYY`L9o|0b;%P3}CIH}EY+>eJ`4d5w$FnekdT^P6J)Dj6d5+OHT{Zz}cn1{I?+Z^VaRAByqNn^CFI>|(6kG*%)XD#kvaWiFdv#W?;vb(V^{1ed(!Um4Stptr<{KK0+lDD#w8Tcx-dL#Hfg?*3Pd<vOC1Ke&o<cOJ#xhM^*??Vjo0A6|qMyt%iG^@{MVqO|G3Q$^UA&ObQPUx*_2H(K^M72<ASdChme5VO;Ed~}#8z)$;4j-{s;;Eh14F!HwrIHr7bZS`IO9^gDz$g#HoU5!5p(pThT%eadACyjixvx|82Zaoh}X{6rKC*)z2a!u`Zmpr_%AsNvBuU?qacu>=xhnD=c-)H`1o}0b9d3hdE8<Hio=I3G1>atMNznUR_!=a=u4+ZX=yp@)ohw)C&em-%?!_FS-;I}e)D7-LbZ+SQmO@;I>x{T-Iu}i;-7ZY>w#2YROY3p39Q=6j`(#geq-N`0(i(Hi4D|6{}XfD#qv};J%<|2t=ankQXE~@C?QNFvCgSS*(uy5DrAnVQ$>Y9KY+-!H=`O6^(#k};BguQYwc9$bxNnj4HYy}u!``6c`(`L6f{@4HFLhtfA2k%}<(oomW!DufRsS<-6{N~5w!|$Afq1x6yZ!>bxPfKKFvm*xwtrz@b{^eFd88>%M4vNZ8&{5dtpoYZ8aGP)r_GU;<IF)7Ni_y!0ANORV*4|^1S8XyeOu*=SWM4XV?`zGLs!qj@gEciG?~`$NnHB4k2Z{LfJladn#9`Hmiy~KrVz5E<+nj@KIIi3np}scq0q=yShEyl`;#%!M`i`5f`115f><ZaC{9#N^cC+#+W*yOf<Rha8r#KhPnU`#UwSmEG{JkqIakEo@7V(9nlrf5X+5+LVH-D&LOekFMZeAVpj)LyIh7}v*SjafSWY1`v0C#0d{T7mv!Rx>5TCRWJzlmXIbB=0d!hZ)H)FcbCA(i@eYSV*U__C5d%l9!4{^V=AQJ*V-o6io_-Y_Zz{nlvv5#J&p=ewKmBe57R<%G<qN0&g2M(lohk5X7qj}V}>ECa<H-sr^}<>1~@t6BZG97^JMmq`^>K)%O2p<LZc;F0eq8)iay@YOP~s2{=M3M0MaohsOITBTk%sS31oj@Z8>uZC#{#;t*u)u7#MwCeh|8oKDXvoGe<z_sg#tptr~AvxVMQ0ICbP(SkTa6DfRT_$J3Y3??Fp#pzgk8dOFBONZP-`@=R+pmP#3R}R`z_6&$wGB8qS_3b5bU=!Mz@v>PUC^|<;Na^35|I6we{KG$8;*}LmQ#xNf~!l6s!UQJa54|fiXR?;nl@7Efq;*YB3<#3{>Tv6I!ql4$r*;?^IEB?DkE@NEbi`Kx-rOn`CVbea2#Zw-;{a%XabCP2fVm>`!gJ~``dIx^a~VeUbSN2{t82SQRZ9wCSj}S2aoOQB*>hh;uRU1f{492#_g5U;LWXDTO0NbSp1!AhF{LW%G~NM_DkP^^G<@o#-3TwbG{RIwtE)#I#Eg;3Y>%P5VGN!2lLQ=gVRos>IY<Ysh(fY{{ipBc;9L$FTme47AwK!1<2m3+CG0}5#*NLiXS*G0#kjO&SPAJ2iJ^0*sm@^v7uaq4c8Ks)+v5EE42i|moM+_(_4ZIE8Dh|c1tkssAAmbzXYURGxIm1mjLQZ`WX_JfK}^DwNug(#H7)tX~r*sApiOMG2u&)*d5hp=(7YatC1S?|K9)OgLvw0!zB=tu<5LSyaap5#yGzH`<&L!q37mjm%wVtRW|X+5(wSOsMa7`g4|<iy7H5YFvck@G|;#RJ1*1*zfW2OS;gwL9H&Jnm3yW6L30r#bnM?ni!TD{=y=aD+C>O)q@ns=y8sr;4?D#)7J%p2uX74(Kj7k^-d%6^A5i+l>h=K34-i#LzVDPc57+m3DU1uuLtF<q4`<Io&Vwb(6oEN7sTa=cnLG=x4#*T49h-#<`;!iFBz%XB!H>W0bAN~AtdTDB@EK?n^qXKfI0LkoD|&65zJZE}rR?C^G`NZHU%RFEZ#}==Rto8x0+tT_wHe7NND<Y2WR^M!>Uh@4lztLgbT9emy!#5i9$bT-Q(s_5@!1lAJ6|C3e#jrkoX^maS*-Hz@Ml;!#PrzRZUQ`;*W6ORk3*KSZsiZ9ad>)j&yO~@G5AX3$dsWw3Y76w;%PTW;L<g+rdGaB5a+LRiHd$09%k#i%kCHg-MDk|Gb<lKW%7O|=iC6aot#oK`q2+!htfwB{`5ic-r?2}nm#bfyJbgqvKQtTvN_1_^nmOiC%kFY4RZz!WpuG5D0kz$esH7<Ug}$0sdINhcf-(x)no_M%8Pj=dAGwG-z*WTTWw%;>FeJC`c{b5k=S3l+zbyTTonDfo8a!HdUes<Mu=oM%2OZI0KZ<07M}^P2RgcElW*$kpyuL6zU1#(kQa|jUOZh3Dw#j;tG}rML*2?9lKs{2{B_n*#hcY|I{l@tYfcpi=<SHzqOSst0QzjY`AQ&nJaX2upaRx@2SpW{mP0yY+W8K_GKlFbEcab40Rdm8(qC=GK>ac1^^=Ao*st^>^I%;eqy`6`TS_Z{k!ug*R2}mny^K7^?m-^VM~HnfWzL1XQ8Q=8{%qp@J_G`RKp+qZ1OkCTAP@)y0)apv5C{YUfj}S-2m}IwKp+qZ1OkCTAP@)y0)apv5C{YUfj}S-2m}IwKp+qZ1OkCTAP@)y0)apv5C{YUfj}S-2m}IwK=@x_Y%6hmX1^2iFCW>T9_Wa>bC-5$vN+<wQ7MB7N=LMJ;0kGMcSM6YTY8mDCwyv>|K|0&6AEzoetlHxgh!nCYo70N!nFe>frp+rqN@BH_Iz}}lRQ0^a`cY)IpMJ&pR5z^cyH%pOL9ixcORAZ$i2t<QniKGy{`B{Hu5IRwj2JwSo&W_tp}E~IvzdW?}f{YE!?rS-e~jFE;`uO8|ybdKQdeOM7O11ax0ncSpM<Qp5J_K$YmfQAW!9rav$H{o5^)SLC^G3<0sB2(Dmu^=UGP_q&hp(ujhb!Y>$(#o7-WT?%Bs__BMElE&R$Q4J-6lpxq^EX^Mg)hP^g=|DiHhXj{Xf$7o^37ipd-2n~P6B0m^uz>vsIIpJCZc#!lVgR|HOym~lyXSSL_GoP5`?!VT+Mn`AxThtatl1$(GjM_o?<X`)DA`ajkp}?Zl?*JODV=odeIRYE&oiC<Lj<7iH*LLyW>n>Dcwu2dVAkG!2Gbm>ZhyF$s_xRetSiP!m7?}+uxxVjZZF~pn-Y2KBoor#BSj@KcH+$H)xtg-e&Iy1oh1JvlJtz%s?6;D4haInPhYo9bK>F_^3~a)laBxz`I8omVtPKTsveEf~pb5(tDRp1q95DD&{J{?*?yMQykn@NCHpG9`oBP8lqiU63;{KppVQ7<o*dM+<yBgBp=?5$HKOt<x7ktiGzZ7csfxWyT@_Jid(CE|4w?Xd#eN=yTG6%atlzc8#l!^-+DV4ubndJya?f2>&k#qnF*K5;nXdK{;B3+Z7rvv!9?d+N@cL2Hct?q^`dw9VYVPINs2T$#V-&MBSfX%`+X`|=XunGAQmz6ESoL1wNQh^2Ngsy}dO<6+4bV2D=`nPaKg>fc6_#HGU1&Zr&*}`KN+TdT=c5s}(w}|Jo1DN+SrSsJ}f@|=!&dw2M2uW>u#+LXVcGxtOhuFKpmo4u`u|;<%q8z;DnC=c|e^8Vih<5{htQTOXcY}-4RqS7PxkIeTcq?U<2dIkuP`Ulq6CPjMdoO9s12jK5t3Gyf16{^Qr8RD6_%G9?S@4)G@Wk?3s+yUA!0sbo4$o-9_mQ`eabtiRe4-_k6)#YUik5vg(h~P+F1!Coc0^<P$iVdLo@mx!`OqoCAOB2#+TkD`f&*IBxfC){D8jqHCSpDoJtnggOU)ATzM%7+36@mk9JHAkZOK6YBa<m^W7)_Sj(U~MdAO8draNj>fLGJ!_3ammFyURseMyf})U;Vl*5jzaqW!I8$z_O=M_i4S1gbG4x8Y2$Z4LfC!NoSxUW<A3<AV8A^~fo6!l~|T15!J)TT@Cm;_eUJHeUCdaCokEEbU@5zPwv&e2=~b$K4}dKOSzursofXMXg)WF}U2KouUnohEY&v8n&Uul+jn(!ZwV3RVgSt(T0{ZWjC*`v|$4OsL8&*HgsgL^)-IqhDx~+2BcGMSZ>D~&XC!P`^COn{-tb1%1zzDhZkG$=}w*?b+Kj)p8ue#M%jdy#G)vzf*R0vg3DZCXFd8RdXSP8YO(2Tr<ap+HR{CD43UW;R!KGUyq7G)dGWG8B^!lUSiIBek4i2c6=q^29ZbW3qFg5)?ikcLmR5JR(-q%tt5n;YIzmMI(Pc)6flYfOs|$r0aHwGmnQj+=(fbC*eS6B`<xJ?MI)`d-DX9pS*j*2A>RFPT!<(RAa^a%9avRvR+`fK(Zx?7i>fPcQBf+Snyhdbd513U>396a&LFbjD(LZktK#<V=C>xeRSPXvR&$%!JD#2$Hx6404hHsa`pwlSu7zfjq@QlMQ3Xb!0+vAW~zCCQwH32W5yZ#mp{R}UXKRlFp`URwSY_}Za`U)&8^hf#!zJeT0=(M}-BusKYIN`}P1)HJA*-lkW0ol)o)LN?3U@pfhG&esDTt$1;=-z$<7V5#^xji#r9O`-8&~FBuiX<0p4t@u<H-SxioWFz9e)T5S)$j1xcv#$0br#a3oxVINn*|po$46J_=b(+7qW7=*90aMxj2lGGfpt>k<=Wmk=s6siyF@V$lUa2fYeMsomFm9wS7{zn4d%7Jna;yNP>ElF=R6#r6Fig@GY?z6g#pXi^KhVp@$$a1d64)hMbU!u@HzW@rAhfb{O6H>nl67Hj(LlcXC=?W(yAnNeb78CCT1Ndw3!E{Xa2wVwdSGn_o!6g)p;0wq-bHmF%SJu$!)SX=io+j_2<6sIZ&WXlX(&|2cxu==F$dpP@tc8_>AZr?BVEm=Ce5qXE#;N^Q&j!%K?y6aF_*S1ya<A<SdlkmRgP5_zo;7e1n;J-(jTpa6+%%cc|R2D?WJSJN$7qU-NIAfqR#Sr2=2ifUc+aQ(dMRuw!eK)GGZ3PotwQO{ji@8tZe(vTM@-zd79*1E*o_^zfnJ)6;N#c}b<HZVC!`vRML@r{F=WsgTFaBn0yRxGZEf39Qe$PsHq-1h@Rei0bIC5byhoyixco40&9LqQox{Sgt3meg6vx`pS69j(>)D$Ga?UZ$87EV|<&<fzR-k)wEzDX#(DIP2L`b35aO;O*=X;4ocSk#^(&j;nhWlS3bMP;qtFKxsZ!vAT@Q-L*~gSTxl{4+wC+0<`<KC7*jt1_xyO6QqwT-^~4?5m>L4UT9G5-R6|fq{_w0J-yq!n?6QYP>LY}W*mtMD9Dw=(-uLZM{SdT%pwqIm4_1pAZYJ;OgXrM#VwMZNK;gXT&#v18D4sho@7@i2I&#OD6G;%(r9i1!)dlN`b37h(ogi*G70^xI0Z#@ddybuNgH~3Wf^LfzcwI`K{IaVFNY8Ys_wzIYOYnK#Cl>W!wj9=Zv8NXH<_5m%KT!i3k@Ab4E>&P?7DRIRQUO*weYOu)ltRRd@XZqzMR3QyxI&&cA95Z<*qzwU2LCG$>lw+?!OYbB+_+#OxKpbwO;<-l(S5cQvHybL%4SQ-DOPvzY4&w4O?(Mg#i&x1SKpw;l7N8CBQJdHvpHa+6^zaPz5*40qcJk8kM7|_0!nrN4aqi7#e9u`6^ZFgJTNOGbUrKx^G(OuN5b-OB1^bg%Bld*iR<JAoG-%5eyr-%W5rmk@QpLiz7(sBWBbpsm1Eejxrt6w1u8P<h@l6fd!xTrV^S6BuLK*IT2<rIkD@!S-l)MII|rK1owc}n@LRZiK`nAhYcjEF*C8KGfW>wCdbE0cI#RW+9v^s-@d(>AU|{5S<pO^r4zjMQDit?kSC(9<_?af8nO35d2yenG&&sJZ{x+c$PVR5gZ^l_u<AeMY&3L!CZFR3&3o<XA?j)aV!M9(WA@Xf29t}->!M?u@BQ8{jGX=F_2iw5F3Tr#MNlgF!FRUHg)F!Su9PGf99w%zcw;iZ%m8xtv*MZJrF;V~Nc4D_W?Jv{GPW*YK$Z-917y7i%wcdQzh1p@>E7)VYkS#^Kw0NQme?Cb*TXCF(YDX4V$uvpmmK@Ep&!2>6%58RSSCBAh`ff++R}#v#ShV?4cH^CAD+im|yU|GZHk-udZcMp(^xGrlZuC@8WWM;K8|7Kf9Okg=MvrrcDx(9sk>!)Zm59i04BK~;L>JMGvw5#yOa1HD_{jDiaq7l%W=mC$rrnsH&HCwwRyRJoci2tyPB-p2#CehaOgE<fvKHMz*Nv>&^ej&EBz(V@Vq3PEgkp1dstS`yI2>-}+2%>Yog~V?SKg3N{{-vse~(C*ZzSt#agBs&S@zQ_XGmC78)5UClZ4+)OVTs;laS`rpI;#xUFf`V_rCaS7b@P_P*namAFjUBLKUC7@NngV3jM$A+hZ6M(%XfnS(cf38@kYM_e%+bvMxLqHzVMh+lAt_D)+<Vx^U+slk6G)E{ts<Z%%dXLUR%GfDQXDlyKr;NVe?4kfX=WP#Sfi9JhzK$G<!mz@K{1q6;0^uLjK6c46ZrBSnsL7hdMncE0ew3kyzor?<W9!m^XnSysAT`2YL=4}m}+5C{YUfj}S-2m}IwKp+qZ1OkCTAP@)y0)apv5C{YUfj}S-2m}IwKp+qZg#Y)-uW)RwF}^|Nx8~O*NajeEvYB*!(i&;(y#oEUZIE7Gp8G?h4Kg_M%w#aw;M=QJhxDVYk-O)1RM8c4e4w^hCxzS?*{Z~{L^EFEQ%|msTu1aUIC}pm@AOloiu;=>^7#?QR@(lTn(zca(Nk#ZsA{2EU##z$Z@M_sL$ceKrHk{DREawEn)svEo5`B#5mqIcDttNr01K|hEVal<qv#>wLm@o!D9<`Xo*exYUyjq=CV!}pmSjvfj!zq*X-wbkrZW~eaoXN^t=0zfC}Npb{2b8Rc{RDE*#$quhU`Bz<BsiQf)xGzUTE7n@%YhB9~{3i#_2`lgBzRgZt2naU`A@cSf`>7S`1STR<ruz1D|SFw_AS5&H61UEXp6t?+&xpNClv9-9*_PKY!eJROoibdp~TMuvpDh@kN<Kcf;aTy)h-VsBCz|4JWO#fOEwOw>9SXx*6JG!|5l6_K(f+Y35h5p9;^hR=3@E?28C)G*Gc&m=;Vv)``9_VhkHkmRi{ptsrE!wZu!&4nprOmYGmH!g`k4DsP4(1V1@eE6d>o8*dGS)aIPPi{t6vsWoTV@)*g!nfV^>rf2CVqZ`}}d6mKX#T_iQTv&TG-NA9#^;Ac$J22e8zTH9V1qHu-Y95CBK;d2sANBx0@O8;o({2a=wSg=;x2+H0WJPyp-6|NmKF9iVx`Y5(l)snWcqmjIn&D633x`upQP=OWMS^PT1C8U%(U3P<rZ?Fd4V4n^J9-|*z{rgkuFIP-5P8g>G?ozy9hZ|U7r5i#!KvKc*L>pOd8%0d=3*Q~G7nz2hIlwn8!tijDjrJvbV@Ji#6yODf1Vg$JnYqIlG5^zgW%(Hjmo33z&QWcZscSv+}|!g!f6l#)q$|i(HIR1Yv)woutmdWkojqN5Ct#iJNgPQMM2)98{>Z{qhPD-;m&BrNC+)`t0>tN4jL-td?%X2AWVXT@oIZ0u>2IeeVIK3NWD!g4hMqZ{P*rp7Hom=U^Q=QJkB3_*>AMGc<cw33hg;HXMBNHMRYTh)(3<nfx{X-fo5$eWkkjuuKdZ}n{?R~sN5M9Ls;K~@s)YKz&9?iH~rk-3{e+|esRxKz0L*n4OKLADP2K*ety)0%N>+=`M(Gb^@QOdhPYK8AJ~eWwTye=3(}kdN7p?4pln&^!bOe%aM;HqXifD2s;@j?Ua$*-Cd+8UjWfYullL?9$k$+a6lbCoN)-w>`W>739K%4Fxga-9FannL+A3(oM}qj{PTAG;D2V0|(GEBk4NVuB>s)oBL5f=Jv|3*@T>ZIggj*p7TJl+TJCDVH-I-8w8Od0n{WB_ZN+1?0k1oarH^ji%lMdvYiZM_Z`X7z;Ml{&mGY+h8j)v>%FU!Wmqv6^e@}0Y+qk(#<HBI1TG$gN|Z;GamhEUCz(!Ba8m@TKeb}=IgjLOqmy1qq0CYyRv*u`jIwK551_lXAUJ1Wh!U!%cdx%yK2R5YyHnryg4L<56xZSW7SXo!jqdm6A62{Y_&-M6G7;CtK8#8SmDaNFA~x6%;;GB)amtM!AS$!2LUPss;3PPXT=7)=1E+X=A`i1@-uuu<!l@_>lri8<S!9AN^#(p=ZD1Peut5%~~p$Z%XsGPY30!LwHz+rrGScHtJM*nKBd=xy>0WAMh#FRyhj@C4v+XL`fbn_%>9tQlU74M)0C8eLiQXcRwqs3^5378_QUjwzi_z@b0>70pgbxbiDG<!5{fzFOCPqEwfLDfsE-ed<i)*hlvzhcg?)f@XOtPUm3h`}Dr7Yq@xM$l_(8b{_uuc;)YB!+gwoX)%%cqyY2dX^wlIDa89**Q8l~72*-YgXbSq6(L)?lFuEVVywN|lp$qPf<vkH_x^iRiqv0gmHyl;!`(((M^Dq1<9l<NF5QuGEb=|-O72mCtgOGc6d5Y9I-*YeT6QJA+PaibrimyzcX#&YA4JuSgE}Blh5YRgDn!hxaOk_h!{Wk!^>b!ip{Pch1s?ir`DzS)!BM-@xf<Kn{$7ZQtH#cmPwmXj)kyY2x&7o)HEu0eeDG$eL9vtWKTTx+rGT%ilVc5vnq=i|rq<xl4-I|W@*4c)K(AE!FB5Yf<w!Nv;M0(URlbuoX!S}&BA=`lxwDHh6qsspi)@QVfu|OQm+Pm_bJwDZ@By`~gS99<UuqVyUV}DLhYfPZYS2{Q|7C7b4ce+s9If%FK>?>OnOf5tESS1>IzYDu@7sv(J9NDU`S&u1HgVTrt5Lsk`o4eT(i?}g)~fL#S8?R|*=oG}I+*?IU^TjGoqpBdR*komZ@N`9R%45<j#^7;HEM6~so09B##j6rnuqnPk%<-$Pl{IKxs6xTcXwCg;fGN8s=EqvIG+xbhF9Se`J~sv|5f3s-@d$==c~|l+W1w_CgQTT*^8}6MBNPG_+6I}dnsxBF11$TQ@L%yh&z=y<jFOn>tBIn9;?1t8Rht?P9n{vu?#IY{DOk|O7WPa@n}zP3G(D!i|PAZj5=0IqqIAV(YsD|_vyn$XjUN3mcd$x`=9hw+K3b&kEzkXr@Q&syh^7h|2z-hp5ZyWGb9(eHpsn?7v|tvPF2~i+-$UwcZIB1Sr|x567rnNKw%fD&GT;Qh(48@J1EnT!?)$ht2ZfVzZIIbi!~XSoF=6jdlE75*zfr2>;xP<a(9ZVFCHJ7iT>wwHXiAm%3XcG#$i{rXODM#9BxM>cyqbOq1f@5_2&(-7_y4`w+CWy^Xi{^aiJL8b0En=yd@d~#+Ofc*+k>O`V#urMB)7luhtCqMIu|vjc<q7!;w9_g6xz;IKJPge?jRUhE)edk4Hp>qNC6f!@%Vbtol0UTHG3hvKkWnnIRwW%|w-Ov0ETIts3N!X#}8K;K14^8-G+4yeJ%Tz#q@H+@X9G;)hj#y66mE_+r5YQ|%Xby|MQpo>y~r!<Y}H()Ll#=uIh8x~tC-r5l+WqbD3tFl&=3Qo#W^v&nNTqU`a{tI6%023y=^eO{?n$r5K|{C6|;8e(6Br)hPRGVVD0h}xf836>A3-S-oE1p!%2;h{IophD@?w~JS7;LfkCEWf=D@QU`Mv2}|R6sXa=l)ZfqjN?p~4#m2I2={A$e@Y+7<aMc}x$XyrIcKTp!ver+@HcBtR}iRD>zY?`gu!+~Tm`LWBzzY!**Lo$4e#4>?9cs*h1{Z%9S5iqpvoy{`pJPLn9=0>KE##+=7CbY_HWW4L|K7Lcq9Wps-BJB-p&Fk5%~j*j5(0c-T2F5cP@<HHo4JIlnc$DFU1UK=Rt-1-g(#K`5;R%e=}x0AEM~@*iQKrfMv>KS2pKD7*bnQyI5TW0orZke$&O!AZl|x>1zp4)sOd<50%1k)qvnDb!Bk5Xy;~;OF1YTSo@Sdt$@zr7s2$#mB4+YQ1rShf~7#i5zm`dP~cFrYZj|uVfWNpkw7)bS{a%?46cUF)=*bRsv0;tu@<lBQUlQ*m-GD()<U^KLm+2lE&P4a{MY<U9hla#J=CtMgU8EuIm_4TVbAlQLoN07z<ln>&eq2bFiJC<uR-1jR@(wiuVWj5i7eR3Nva9nym%aVX{iaQMH>5D!<wOo;`zkkn=KH2A&lg_rxnz*Vw(r1TS51sr7Xu#8w@6t&v1Qihu**IE$4rBK-TT-ff33sFn-J0bIzp;5-VotT**mr;zhUPNjnmFlVVepc6LLKX&pD$`)-IbOXR1Z=>gS%=v)h@9+)n;tAoFKAbxAQ;l5Tc%y{xqdA0OH>Aa#4p6LT0ffddxL4EMn{9F#-N+0NKsU01^*AEs&=h{|d`+;r2$oA-LKOA2y_4#yS037;7nY7dgAZ&k*sJP1jNbN|K97!GkhNPF@PS*}Vy~?}X{LTTOO<_!FX#3aKZS4#$9sp)q{#MV>0hms3|D0hp04+fU-})a9fTDl7UiY~H=o;Eh$GY7Q<of$`DysUy=f)Vhu1!DW>Fgx$6Yd9o?d8w(gMCoQUsuj;-Um-IobE5t_JLs7caDkJUhuMIlL{2>g?)9`sk<6_pmCpqG~?4ASbCOd$g|ZAD}_g=jAOeY$v19sRkj;KuCUyEPTmc7#u9Q4=aS%LV4~`r4hgoUt9TZUlAtZU`EGh|7aSezJYDPD1$TsAGgOFnfoL{=SXzB2JPGh=&^y!#m3&K-Y3dzN8{5R_n%@po@2U8OHrqhnz&bgIs|}diV}u5TTj4ni^(l#SE#M>87gs3S3{`#|f*le~pqx{oAtK%gDGs&os82LNsp|0^dk)ltn!C-*7b~?;y_N4JJ6Z#!!|C@*K32o1OSoJ1SQXGw$_jn1L+IvHJhxO(0WMC|aoT}p@FVg=->_K;7+$hDe9of?bYJ|rbGxAc=6_}ITqny1x-<NcdNmhpKRWat2+amjuaBON2Qr~H=f^iDwlsL@Ggfh2B^jCm?XSj1C%|gpGjiFPSZE%YP0i+s2D^-4iIl2PP;`EOn%?l=kL(7nBbU8>;C{qttAzJ^a3*^%Op|I07ISP*2E~lv<IT(l#|3E+>wKM?;GvD8s>kiuX5Zik+TQ624Lckk_b_G@aK^z~J~`n-u9zKbTdDfn6YW*0A2gixMINu(t?DH|Y={$Q)0GcE#Vh<rj(-ls#v3AyVhusKUgPmiVNVF2(dYg$=@NoZ1JvhsJP5_&DI>0ZreWyCq?a^i5RRQXU;8V<Bk&7`3E4mtj?o-R;Jh7;VxAlY-`k@x)a#IXr(+CK4Nj?;^2TDsuMzfshgfvT-$xzwJr)PXBqo2#$KeT^z`Ak&ICOF{mHyEihh)Vva}So{Q0tr3%ZZLSWa?)+Jc@DXbM7()_e2~P(!a^(rHjYY!_W4(--yTNXDT~cbmP(0zDjB{D;{GCsrq8BC*Wn->~FQl5>Z9sz<Wo@B#d*rt52t%jJ(3yy^W7j@W0vyx8-xG#Ql8;1OkCTAP@)y0)apv5C{YUfj}S-2m}IwKp+qZ1OkCTAP@)y0)apv5C{YUfj}S-2m}IwKp+qZ1OkCTAP@)y0)apv5C{YUfj}S-2m}IwKp+qZ1OkEZzk+_P>-gfmr|?{w{?>zaP59YGlU8Y^10xL*J8fE?L9b3KL&Ue|KweDWwH>4n4?gOewtRmMH->6s9<l4eH<|B2>eV_B!nN@=ms1PorrB@N-+2ldulGOZ(s=~mJiHIw8Bzv1@uQmk_Z2~t?1T5dr3aAfs1&v6A_rcf1>tRl_n}g4(#9a-2H0-j_ek&-0X|V(w(C3ifokL4?blCFL*7w|a}mp@f%}H9V((RQDCFiHZ6=ezth(8+iY#}ricZGFkVzi-KGj1Xmpmqy*6pI^kj0%R_@bpP?xP=#(=m}jc|6Lp>n;b`Bdn76kwG4%ffbFhAy+qb@x&{Ita@5QjNP+d6O&|up9hZDEOMHm1%s|+U8gzPu#Nv_K5c>2`^V`TL(R}zYM+6F`5V+Lm)0lOH%0fc?7l{4W6W|{|Ea|P8U>sFqujk^fFGqS2PFe^kt?r=bC07szA$w@x{Pw@FX_JCMRglL#|BpM_TR!4X+~1K)pcAX^L#$yCW-1we$ghrk~nwa{66KX%V_9-R^%&)Vp=S9jnSv8==L3kCHo&>sbnL6;8#WbZ?@(b!<rJBnax^o>Zzdm<LuixHfreQVUxP+lLr1gEo`)!_8fCgee98@G(^?tV#!In*C;%9jXl`S2(d-)Rm2Npbf@?0XR<d&+eW`|!!{#S{&U4VPsj*QzPtXq`2K5Td-%7|mEu42y&=_hV_g@sCKuf<#yrIXR4LDgv*oaxa}U+!F%Dc1{(J16%5A7;9<o_oR0Qffa@JJeRY8oqq4`^-8eD#)Gjcg!9aQAlwy(X_0(-sH@WkThVAPrGHg?4jj;cHFH<UJkdIQg)3$HEUEv@Hu_c9wWzEwbOeZ>LXrAB{oI5@+uQNi5e9#^Qn5G?b;*b{bzr)&*}`hdy)YfLG&eh{`lP5Zu40FWgV$E`Ac0NI&Pg%f%~5PR%_7F$9voY=ohFJUPJu2%m@n|mJy9~cYEn`k1SU9?o*Ts;zoPd6M6EsX*(UM9L^n`k)3LCOA{IR>&il*tpyV_-etva7aaEJ#F_3Qgz6g5cz+ACGt(@J<QF=jO$M{zsCP#lv_|FFBM`sGa~%eSXiRJx_$P?c}|!my_V^>yti)(@Bv2Y^K>rHyKze&d)zvNCvN4zsf*01wODU!LMg2u=~~%-Jdck;PB+2S({V}OfI;(8{A2O9?|)TFSk;F&0x5g?@|hsPcOXw$&~_(_jgH*d`X7!eZ%fY-X+6E;M$S|eKJh@_1Wx?N`lKm7g<bCCPCX@m-p8?6X68)ZQgxm|I*~}x$HBEkR9XJButSAA8B-q-cBXJ3zo$1`}-3h^h?eRcSr(I7VpsKy_Wz*vJ*Mq_!3}GZL9yIe;Hb^UMa$!0KCQ#&zZO4VaFREc2~Lt&|1u267o#|Yr3DqDf)>JyFPfL#~}%1GsG`8_9R1Sz_(S`yQ#oT^J45?OB$SgyW?Tfe;M$>f$7Npb6H@%(yKgVo(($JC>8n@av(-N=AxizE?ltRC-C-Y9%O0kI?q6s54ukV$k+PwVY=xq`^o46(CK6q#QTMiYUay&WT_B#Dc(Ny)TIb2=B%~6X^WwBN?Gl$e=+PHRLVMayaeoJ&A)7uN+5!#w;|N16rP)34PhrMgVL_$5TUnaK>aRZ({8m44)bJ5AG0b4-Zy05)^}FG0L5fum_-GQ-E-Mr{jCB%%K6w|cvK0}Z2Sw&{gu%3^oK7qAapy3ddc}C6wqk?e%6KXp^R~vWgg*l=c-K3FN6xdRfade5mvu<XHIP*FzlBQW?x5ejR`xbvxJ~!hIe=T8^341I@F;LLDE5TaV`|$n(+<EleZCGfBvxl%zPyjPP-k=b*+Rnxqyv4wn`|{{*>)gT>;un%$4+}6_6DdEOzjC1(-@*{z>0Z4%<ZmuXvx7L#Nq0qY0{VV2_s$+o&jmt<<{12W-orc}JMc_d8|4emr<Ek*y3k{){Jv&Xj_m@B5&g;iYi?n#3&;u~Lw$3yXg^Rsx3a)sfP#1Rl?bF5ia|cv0;0i8Nab_WWDZV)4Zg`6}YJnrbl=_XS-r+bjZQN!zhI5k=5(*CyBcauL|<FxjN|w_f@i*W{tD5G-;mL+&LP0xykAa;8rqJY{RRK=r&3Ag3U@o4F9AXAbB6$}52Wo|t~_#|7}>#|?=g+5#vDZT!4anGYQT3IU5&`S9PLRO6bP`CuB|q|3;b4`m^fM_)4LgF)f4?%0WZm{(l#JtdS692+O?@3H5@_W9>=x(D(>R+-XS{@?kf7;8|=@;MJ;PA#3T%FhEa8i_r}ZSo-Fk+hw>N*=^%d}R45lm|-E))o9qxgbCtY${)o3&M|gn`V0Eg3kL1mgon$z!|M+UBi|OdFoX~Uh_H7yd`%=qdEsBUhEc^^UMJW8bKGWdpV%suy0OkR}Mr-P5G^6WdrNF%Gs+1+3+~o&+D96Ht4Fcnou0dhT?opK`+{DcrKn6GA5J_-QELu;Z-)s{ym%#QJf70w?2R6qRj!B*=HZoJ_o2R|1>du%>mXueg3M~a={M<7;e|*!dH^~eAKx-P;k7(wjPiN2VQDiKK?5YE-hMcaVX@&&LDk<49tg6S6cWkQ2=t)ODiub3qVWLWa;3)eI_}|UYj;m2(?}g!lRBw;7U^5W5Hhxyk==TJ!*@A|H_7Zgm4MeeL9}YY*qrMJ7j|c^GYDaKB@ETVhIRrrQ2QLD1`_apO5NSO2KVi@|d4yDHKg*XL7if!bG7f^V#xJ&`~}=nD?g?{+mwPbzQ6syn5V4O!Uj3r-_4d-R<8#m1t33&MAYi(|uXFGiAVeVWvHStsJyQx$e6ul!GKK*+;*ya-hG<d?5RCIeZJ#Hkde30o@+qkp=(u2UFo<+PmTkIA3qtIQpdmPP9`NTsl+<Eytg3i^*2Pt*xb1dH+hNGSXRIX{dzgzG=Ir<w{V>$i2dT1YuCB;%ys00=0ODTag$-tl&xAnKK9?F9LR4r$%5FIk%(YTP2u;ets+5SqbN!Ir}JQR>D_9WlwpVN{CrHjoddYp^`HF`&GtDm~5->7h0?UFVpLRZ0!{g{r2x)@2m>gJSlyq)UyJ9y()DnP_KYEmXorSrz(JFuR?zMzw44M(^sCV#d1(Ptnk=jp&Zt%svh6|S3k;q;2ymCZyu_BTNJ0ufloes+^@DACVl7ZuZEOE*8$<Y(SP&AQl;8+`*#@>T8LeD^(g~#mz%Y-TxF21#UwCYQ3}Uz$Jjpmw?1m&0xA4tr65h7XTLk71P(53Da)NL0m|2J<LpSqAfP<SsqwBDcG>QSK(XR~*J+_M8e2usHvUe$uCNHK9>zEyF#J~=EZFy*D1xDEtCXRZLJ+8o11F_IICDTIb8Dag_Liy~{;E^}?)>DZOZxIbUq4v0S~VXy<x8{TCh|aB*6P%Xavs>49Y6E1CKogfT-TzG<$_x2nIl$qIiR!0xWIEb8~n7R2W=&@;i<udLWOM>GznZ5|DBczB17!4dGi@?J;uUw2Y&{{piJorjda+eWxV7PmIex)0|mH{3bE-)8+?|jV8swGp30aC6W@)l&Q_;@#Km*=e>79T^W;kK=U>T?MH;*J#y1(TIN6wCI|<$dI+~o4O9ECMWzAPniJ)C~ZshG`0<6%*(?xP7z}@lPk2MtHq5SXTh6(35Sbpj>!!{KQpZF;$xOHRU$I(q`x7HZg{UppL`BV(-Vz<o+4~&MN)=ys2Z$`m~4=)8den-MF<{0zN(~;0~{rLR_z6faK|H8Zy5C$Hu5<fFCLco%At1QtY2&^2&tiIg{gmN#vr4}-OsHvHJwI%BV&^)TH7VZWD8e{$C4;|s?+srnZ7AweXFE;c){R(WvN>fH#rC`WK*Lh4?6WI!u<C*rG;PuBFmPzjKFuhaGYLeXv9i<O()V+2`kMHx3{v-3nbS_4z2QmRjrgf)}sxpXpJ|}@dAP@)y0)apv5C{YUfj}S-2m}IwKp+qZ1OkCTAP@)y0)apv5C{YUfj}S-2>%C#*oL)PpY?~MI|F+i4+a2*?Jq5hpa6(*a8jJz3;?RhT_=R@2f{v&t7iTRfza_d((S_U05~iqs5&GQ07Bam0oLyR5ZdNdaV5nM<eNHHDGGg|e%QnQHl;6E%Z>TbD*M0}nVxw?@P^GvSB9)!PiSO*@+<F@2V9~`-94D@1|PKkryV>0szH1I04_pq$e5B0p#dd|qV$A>tHhP5bW0+)47pNNl1_ut+^N$bohHqr({N6)8>9>=mB`RQ5t$N_x_tUCzU#Zz`}NCS`}t+>wfA1n^Q=AC`%&JXt|lkuDP{XpOQQV~Z*PBc_$Bw{>q0*|Ip(?7;;%k*q3_Wj_TM<ve$KhS%+-TjTVvdekDeusH}Tp1Jtyg{T+S?ww-&VJ_R)`berqvp?qKlhM>cHP@#Qi;drq^9!mw7|wQlTC&D$*&5}s^BwQO!=zZY{19p_|S?918`)RZH<{8^{&(hO6DKvrriCSy1unCVs=v2&;oW{O4P2bzpS*w|1hcCs~uo%uKK<{#;y>}7RBnBT)NmXz(XZt55=)1K3GQf^ZO3%=gBKEolBZF4yqux=-hO;|WhIyr~O>c?xXQc2>m@Wgo<@pe4MKPkOoc19$t8LAf_j*VcPYYwK36C#*k;ht5G%DL>Te7WFF0GBxrE>T+-#bwK8|F~UqYb3tpoM>@2m!%{sKW!ciXBz{1jhE+!v3Bp^Y`;H3S)GsZ?WrUC4>-_0YqwJXvq<b*5O~;+?Qf?ZuUp=1`gqL=Mw2~RlB@rNLzZr=y=u_$bgLs<qPAj>Z^lt}cS*^AnX9+5q`i$-kBl;<np=+|_Iuk>r|OcKd%rr6?vus~b51(a!EZkjo7<hJwEMyo&bhO6^_RD~Q~EhlERN5rw00pezWbsx2hY)Ij>(I`x^u*Bd%kH7=R9RLKA3S>^CEqFtFu7;lo#y`ecLm?-<PJPepfwb>`(I+Ua=}IxlD`O{&Xrh7C-~?yQ`i(3!t8mx_dWL11U?cecgVuAPTzM<$O*fn8wc=A7Z#Igu-_2@4BN9O7A}HQqx=%MxN*Qo8^jy)8?TC>eu_i$#>L~;j=wly4^FbxK9v4ZDC#wH$O&_tfkAm-4~+B*tWJcT{VW(`-g@G@5PdxV>gxLUm<(Xw9OMgULk$e;=csiaio(QV5eslM~W*YUd2hrQT;NdL0k7L)Va5C!-d>f3XaR5f3sr9Sx5hweMvN_XD-}w`c)L=XE!u9&W<AKOf$W1Gaead&&@a66G?h^wY>({MbM4a0S9{Bxipvi#o+YQaJn0IBKpnqFzS6_YA7}~jGmo3RK?#ON~bew$IOclq1m|wGU>BI=ysg_hvxS|WNp3BUeYL#{;dg=t!TSUJ=((yRKzdS%sfkNtxx_mTpg08x^*PJ)#YLQ8-My>P}{U(u|NGvvTRH8yhJ7ru9ITc`cRO1>%BDz95V3iQhT<@gI->`_3owBS&A6AG*@(E<o?YMP2qIg(z=M%bKTD$BFU}ZGsMCyN%O0Mv$ymv@*N*}Wx{+_`fJS*tMy!MHmj~XMzMc0yLG2+n$wH*tbR%Um<fH$C~=*B<6y=vQl9>(C4AE%GEGR~nAO=+kWo|b>#9>UE^L+2kLga7e)xX<HE&nSwNc*Em*GJYUTeqNjpk5v!Qy?1&%7v0)Mc+W&X@j)r@trF`O#5}(hZNL0;soEerfOSAX19Zli3j-LK8mr<w^5HDe!=7KiP&6XU(GO;??2w?0Uz?0ZA^IHMnJQs<|}!o$|{G^%0~z$6{l&3Xd`ceEz)PD7x2kUPD1Kh6=iM`rp2frKV#Jg=V+nXpZrdZfUQ0s+RHcwaMbs4CPH#I~)_}=kl-djcSRs+iw21X{(axlc?){%GqSP?~?YVqAZ0hAFIlH7hEG1$K6{C|4gL=Duqi{)~AwZAZsa8Pa}!0rQ)hP)5t8RolD2kDB$<8<3|N)q<e9mQI%FYJ(Wq4@o7${f`?B}2#;jYyD?Wb4t8eHe>GQaiw|Z}hLfbE<>V|fn^O5<{L3sl;;Q(}S(Hsn7BSN`p*i$qaAx1+&|GR<p#9;*(L9=~_1X55N<Jw_=3bcdE}uFI=J@;wDIn!lYCD!1TqgxJkJ2Nf1oY0*u=c!2Kr=m7M6N$qNGC3CmsDR?M9p#6lQbU}k*?uTx}RGyX$jVxm@HF5(?kEL?a3=4hjjxD0zDyl_)Y(|vRX)ArGv`Smx-t#ZrA!ldqm_OuXkA6Ohgl3tRD5&c4R)=Zrz`*BI4GFEfu(nNUrnW`f1K0s?7F#TCr0^!SPWi8y1Ub&|ZA>zS$x=tAAhShMb55BDLFNWJT0)So4cul!#8Zw|wKg5Yj9$eTB7kLUQYrDlva5q^eyXI1%rKRQPAF!l^bPZQW?qJF!YgK|%N2PUH&7^hDO#4?H2&@jA<t9E7CfdT6e?x{&<eypVYHvV^h%UoRMNE}{9KEgr6aQ%rx~zO&26w3t-x3~$t|Dk7^ZB%Zdah^h|;*1ygyBq!NhR*R+=Qi`ZWEb4-Qyi7iyjvOx__YU=*2HthrToCrOPVYLsniE#uI#56c@@2ZAMFn(B+F+l}nF3mrW4`~BMgdvqt!=E4D4>TIgj+VZ<dcN?lLftD`E-QSF-F@cpL+Xif>TE2(|h^;M8n)XD)AfiF0{y_Z8gy=y=C*rb)c)cp)8l|YOFqS>~cwMd*{pmxm@a#+uS9doI};Gr<hM4&ZhQ9Q&({}Ws_z{?Z@b(EZWE^dc!ZzB)deh`P|YBVs7(07Ivmnzz^%k4O7x7MABla+wWAeFr9pHZ0j|O`EZ}(l9NJfU)k==v`8l7RSjM>%aZ7wl$H3=iHTG<Ys^&b-+T&t|5Mrdc|1kxn>4RJ7)Qpt6r~JQV`-U`OIAi~G@ab4<nVKM6fLRL+b-4<N$MJ3C;l1}LA8koI$|D&QL0M1TGXo$(krq#G=l=^aKD`I@#UB3xkGC1==~hJ;^pte?{Xtw!^>j>AD^aoTRJUU9@&t2%&(R^+N-G`;h!swj(gdY2|jW~Zb#UmrqDp~@h6$-5AVMOd`DImXrLBdagJ^9s1(=NcW1@YyyC?@9xPWa@1$RlCp#WbzB{D6S;2<$E!*mSS;*v-le!K3S!L|d9^UN$*20T8n;#a;T3i~(iMXL`lH~8qbWS*H6{?1x_`zlS+s-_^y&#gE5Vx+@m*BCj(MNTxGkI)T#4uT}iDJFFYB^<9QEX#U-t^m+(JUcq%bAyb(JZd*a#o^w4D;%qyfDi@h81QC>mHwqVcNdsG6JO-mbO{PIP-2aQ}59^Y_KVs{f<B1bjc@*-53ZjyZVO5x{CS^rt)|!Hep8H8CM=l)Ea$kD&sK^_qTgaiAS+Thihb<Wuusq`C7|s^P|`<#g(bcM*5dF&iWpQnn-3E-uzxiA(91qnm<&jjbJx9|NLI@If7lxp8x8>j!0HHDTAZ3DU#*dE9E_&8p-}yx4gqqJ(3mdGfQ<&jAVv|pZ?pkg2$dX@Y}|B@|evSiIPXpqnP%k&TRo%(d?{=!i-Y07}i{_a_-lxSa#~8D82e!ER&E>tsk!v$38fH2}wD0l|{zS5Nwu-XP+ldakWc|XQOV6<x0x)*=z;5c7GE-({dVR+Q8?t{k{zY(lZm-lRsHsoqGb?>G#uaX?p_W$+~BVCMB}GjLBZ53lrJTT1QRMnnb4TJ4r)AKasuBv()o5PGluZqsWsIneFVY%%d-XojUIEeY$@FOWAgK_l0!{>;$jL-2W4wnY^g1e0G7)eoPGvjPs0VG1FEkeYh0Ilzwih46Th}B`!Ozb*S*zHMM|miX-o!_Qe-L&(~nKV*AzCs;@4w`ISOP^ZzcgD;lfo-nTk26%U*Hs>4>S@32}zyugB<+sU8pNEzw7EKl<9X?fDji0YNvMt;=Y`#Np6T?joF_j%f^&83`fv4WYyJlZ>4Y$qNROG`Jd@Dv{@lFdu)za`fsP`bTTeNlB1rQQ=tz3fS$7q`qzj_agRk-m(bFE@h<XWL4be#s)Mu<{)otz5cjGFY-OGLLp0XtKF_CZ9H)^SPL7R6ui=Yu;^Jd!0@OMw-j777&HGTJM`%NXj0iGyQuC$=ECQaVfuuG*kn}UfWwt3$zaCuNo?**<Es~u|Xx2KYyC)r=>zlc5Lqb(JiDGD{5bwTp7vpiY?O|wMxlUarUU6H%rORRw_$Jw~YF?865~IE2FFGD#Lx-Z;;wWr8kC6H|VTJo!q{)H>qdHW#D4mO$u`q<#-R>r1tOaWgUAMwVD>V@Uj?H-$~s!FwAI3IFQ+{R8C7@@WYmwmeX_Au;0L`oId^c%X(T+Ihk0P#Q3C@)3Huh<C`Vrr0(`ps;{h^id(PF-%wRfr;ewRu&JEPzWus?_Cq-}u2WgKZE^*DiL2K9qA@a7z2I2wub`G6+ai8@S5VHRF-E+c3fgV6%S*Aof*v0j=o@}rLErBjwL9~vf~1zaezpHxK_}!>wDsOr(7?d=3$9NpD0FMF(KuE?ca4_kpNOrX@O|$~#7<SvrQWrrhQ<}7D_yx(YIy~1SN4^<EK@;Uf8`|~96z$&Q_+3P{_>GLYBP|~UQS13pUH2$T~3@n*(Y|%<#hX-*k^%rIi+RS3p;ensdo%tQ}Y`mW!nm!vVDxazPMbHPq|5pIWKi%$KRx>M<34F<~?%Wd2eP5ewR@|!ltgu<7MR8WqZf1sgy3S$#_3qqm;gDi02IliAcfyv%2+LA${m*_;6oGNKrb=)%|@-$RcVN-{L_rbyUoh-a4<CCZ73M$>>xO?O0K1U0YO0Z#(`P3Yk|(k_DWxT)u$HmCKGf%@mM|`Gf?8i0dRA{r6N)<&ir1?Lc31VFA6;wO=}GO97pEP*r`uHJ=PUWjFL#<kQ6|K1ckT@<?w}h(efV9u4&mc|KT?OA^kj)2)?q=#;S6`{&+lS{eRRt35P}mO1KJ32tPP?B5#v?8h0DVI#4XTc1u%lbc;1XQok>c8!Hna4K={NCqtTxkh{aiv3-3Q)t#7+o#UgN}(OQ#n=9vl0u4t{IL_?C)1L4lL-ofWQsTR?LK`une@NcxT@JD(@~r2yncgZ+Lm|5&S+jT)n8b@$z3v;0<!}~>&;B2|Lgw&00000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000008*^>3>j50|YGq000O8001EX+da!p|NsC0|NsC04geGYbZByAb7*gDWOFWVaCrd$5C8xO000000058y00000005ip7wQ`j$;eQ~P_3SlTAW;@Zl$1ZlV+i=qoAIaUsO_*m=~X4l#&V(cT3DEP6dh=XCxM+0{I$7CI&j1ItsN4WCJe2SGtWerx)6e(u0$paP?3E5BCB4Cl}SG-9EX+Uc!3Iva4r~+b4GQu$4ESvEMxBb)3WGOZJBk?kg8;e{6rvVDY|4pZE3#jt|cB?)qkb-nK;e)O1D%#*^M29+%l1eE-BU=S*gIu(eCrvVIr4LocVr4TbA04y6trU%r0&Zr}TGbIQGs%nn|4rEh%OIUEMs|D*KKrXQ`GU%P$&ar=q>RWW}*9<V=Mn0Bmr(sg^^{eKo__&l;t(Ty_g+4;;~C7&}*-1C+FUYF}~|8Kmrch|U*^2qduy&%(6pLxlQ4*NI`L|J;WIH*Oo7un2UbvV6e%fE^#tPU(o^%G|`GCNr9Um}#Y;<x>p%RPRp{{OJ&sWW>aJNcLW%T|{;oNYhsXO|SD);9jM?>#dqWOv$6d-lvHH#)a{wjb#58Kp-y4FFI}0|YGq000O8001EXw}eyd|NsC0|NsC08UPdkbZByAb7*gDWM5-%WoBk+V`*h>baO6laCrd$5CDK400000004#q00000008aM>rc~X6u|Kg6%ibuQ8SCW0xeZ8Q(Km)bUjRIVhn1B7!;+YND)T{U5ilLf?z8XRMJ{5)zHZlkqK2nP%+AJkVrel8w^E<D{j^)R1J@0496V%7xtzJ{hc?Tob&FR9OH(~>xElCpykli>{O{@hn&5N!RDo{V23l<X<71Yxg=AZB~O+9caN9vmQI`%=@OZAVqMIMTo%rVToTU6WBlLozPO&&&0Rx~a&lR%Dz$Iq6I7ac!4=g6`=Xh9{_Z0<sA&1jYlI;4!f1`t_buveBRjW0BUtI1{*gVN;FQXTM@nuH%wG~Uw0HJfmv>OQr_+y*y)temIKSD+dR|8`!tZ6#%m)OeD|2Mbrvw)yco`Qt2u@qr&Y@cgn%<2zTGa$&{20!j6oPj@kx4Rl5sak8^;nV!e#opn)gUIg!b3O0oY;Fyykj;BT;ECQ-WKaL$)R=gC?}bn?`v~@ugn1DWx1(G-vL2f&V|dj?!n$)bJ~)u-Jt31TBOV9hOk+&!bg2JDD{Z`$@9D(lIW^$2g*BP^fj;Gd2|oBx`Wa^JyOCKb-k|yzS{a%yPypguN`yz`DrH<P5nVQBC>&qrcv&1vBJlpe)awLu0zPato-sjXTUL>f9dqqTA2HJ0{6=`7O({LW%B}0LPpf1F9Z%V4Bx)=TW+xl?n{-0`U(;C-;^%3#(5)G(xK#0PA=zbNT+a@8l!^N{WCS~I6Pk39C&zk8O+!uiRq6!0L21xqL-rxTC=BFez|FYj+#s}Gxji)2jnGwxBMgoidW6tQ?(CV-9hP|CSLiO7AY=7%U9KW`}J>7J15p--KZQ>eZE&W7AxRggYAr&l?R(t+0VusazR#aQw`%l+s*;|Yv1-V+?<(m_E`8qY)ovrIb_hn?eP~Ob>2px4=uf;2{VA_mH`KScpLTKlrHtqvZtjPmSoCFrsbT`^XuMx=ocFD;+qQbL_!rkc<e0tiyAmSQO(FJxEf$?X~M_jthMvJPa*h-mUpT2D2o+5&~uwnG58~^=6ox<x`Wa^eYi!|I_sbox77!A9r7<kV@S-){<Qs=k{((>pKHMT+_)!4wFY$h=(t%=G#Iwv3act28OMVprqllA;H>*Y#fdlpR)zL1EusRh`CbdqRheM0akahoO&J7x@tH!I7+l>+>E8YeP)h>@EdT%j2mk;8Apo{g-0}ba|NsC0|NjsG6aZyzbaHuLVsCYBWOFWVaCrd$5CEV600000003wJ00000005ip7wQ`j$;eQ~P_3SlTAW;@Zl$1ZlV+i=qoAIaUsO_*m=~X4l#&V(cT3DEP6dh=XCxM+0{I$7ItoUbItsN4WCO0G%RNfX9AE5{k`g`sy4<jrJ2H`N>z}vwp;K69$r~N92LMn@0|YGq000O8001EXnud0j|NsC0|NsC02>=uTcWGpFXfAGWc>w?r0Ehqp0000007?J=000000GsR=>KhQr$WX>mt)7xvoLr=CrJ!z;W}&X5pq`drR8o|f7oT60k_r-cOUx-w1&SAEBo?Fs`5Kx!3bhIp0WQW%fkGy;Q|tjyO9KQg000080000X0H%b*1poj4|NsC0{|x{X0CRP4aBp&SUrsJ=aCrd$5CDh(00000002t>00000005ip7wQ`j$;eQ~P_3SlTAW;@Zl$1ZlV+i=qoAIaUsO_*m=~X4l#&V(cT3DEP6dh=XCxM+0{I%6ItsN46alWkXUpH6cq{Dy08mQ-0xbhA000080000X0E~L2kSvS<005W<01yBG00000000000Du7i0001EZ)Ij?X=7<+Zgg`lZg6=}O928c11$gm00;m803iU|J<Cpz0RR912mk;M00000000000001h0d9-{0CZ?_Wpik6Y-DpTZg6=}O928c11$gm00;m803iUkgj4K>0{{Sk9smFu00000000000001h0X~iZ0CZ?_Wpik6Y-C?!Z)Ij?X=7<+Zgg`lZg6=}O928c11$gm00;m803iUjQrz)q0001>0000G00000000000001h0T7Y^0A+4;a(Q23Z*^{Db1rUhc~DCM0xbhA000080000X0Gft&l}Z2r0Ehqp00{s900000000000Du9yk^lgAX=HS0E^csnP)h*<Edwn8000O8001EXri8=<O8@`>hyVZp4FCWD0000000000fB`>~0047!aBy#ObYD&`Zg6=}O9ci10000600sbZ0RRBWlK=n!00'

_R8_EMPTY = 'P)h>@EdT%j2mk;8AplW!_A&qe|NsC0|NjsG6aZsyWoBk+V`*h>baO6laCrd$5C8y}1^@s6002=R00000008aW<yTN!)aYT1W1$|quv-BeTc=$Z7#u}xz-|vmY!p!xu@JFaQ4s^d0C55$DM*J%D1z99U9b0FxZ@r7e)&CL=9p`*z1LcM%`w*4U$*;5%Mlh<#SH#4Sk=PTZkf%}7Tp`R=sB}%3zLQ|W-eO#@6zcDr!HD*YxjTJ2Th;9%<fP7vf0xY+x^K~b~9<%xm~+<txOuOY50GqoB;p;000000000000000!2h!9-A*)HmbFj)PydWFYrRj8)lY3S*2PneTY9Y={%NoJZ(Lfm-e#{_wtSnkJj_E4<3h*n`n*SD-ZtN3{@z`UvqtT_{B*Y-yZ^RT*w0-$Z{W6x%j)e?!<jW5d-dF@ma7LH^L)Bp`>gDiofqY%)qlM6>w0aQ&Knd^vhPAy^{Di_vW?|dop-8WqV?=8`l63z=6Yurb<CMEqtS)U>RqwawzXNC)N*k~^J!0<^+ixfkLF)CYM-*}-$hl~s0ROpbSw6JgF3peZMNsYdQGVobFjFvqgtL1I5^RNjXI7#owIezDowF?|EBBe6>7Y2QOT{I%QdA>*sc3M_G(c2_^&lvmZ@Wv5^GzOTB6>?Ol*cMp05Va?f3QEJ5vn?m2Ww*$uu=Mb8lC?+i3M}?Nr%&LRs~;y}881ezX|8vkE_EJwv?vIz^OiF<*>*on7)DEfs?m!-Ez%E*I}ACl@bzyh^;C%Z1l2=_roTxn@y@8zg1@kxGjmIY~<DmMei9oW;X^#c#ico5jO$^0S*sE;4WD(_4-^w@UTp2W;PW+9n2L49ZrHa+4PUsXHdE-XYW0Z>@Vcb*K0b*_7{9YqvNW{+j7I!ChRBzMj0_caNBxSDKeMV6XIvsBUm1$5Z@s99AE4-Y@g|#`UN;@}M|IxGp@`^00V|QAR?$qhgqNW6=1k$0UB=#XWDbj?28p<%{2Ga#9XH^S7<P;*{8}I8*;n+G+WinI7}0kCzw~8VpRna#pM(B7$eOJ}>PoK6~aJ@RmRq&lcNC`H1~^tHgGTF3Qd`YyEG;Uy=n4ipSgSyDSbLuVq#raz#3()fm+B#}x_mKbDby#aALHRyuXU=Bmv8x2egh>V9JOE~1CUT|Y6N+V5qP#s2cM$J1@~>s*ug=GFeK5_(NctH#WEGyS?a|JpHP{I~0JIH<{^?438{@?6uTkIiq&NAuuo|6IQ*uZ+fg9os8FOx<^XJbybtY#NvEv~)nAES}NTFE=_+a-D`*P9Ank96I+N<Z$<vjCHWS)Vo)Zw2L!2()n_bScJ^_>eleKEM5_L;Ga#m#i_xC<Rg){W%l76?@Gqs7Ng9eowD}amg*}a{pVJ`Eglo|KBx5vl7X2G{xU0fOYTm7|6iwN0kW)F)W);RZ^)3mRKp6juSwSmF4i5V{8{ggX@k#=xhy?gFLY|~>YSV&9a8S3`w5wr`|!)~`5xl<qhfp8|CUMc{pxR?J#|sDZ*ups!x1(3zAw6|+gVMT7jm>pnJap*armqE>Hb=4eTs#}#Q-&qD~OF=s_JDnZB+NaL$zLo>yb~!hv~hcCdGcZhU@Ub$E)7(iqu)3{wbF36|HT9x<>bOxu<JJ#`<m@5~oMwYJ^XEdS9~^HTl)pH(tF=M{WNyIYF;|Thj7CQG#}Kd$GgHK2fheY1^}IPNH5)8T(*J`y`E;5O!eGf+Q{1ZDq+x?n&CJkAwe{V@djT!zVXek0jl@-7Bg6(j<LdZSBC5U6a&o*|w99pCzhe#X04gZBEqdQ`&{6S4`AR{wJ32*pQ%I2KjB9_9R~Wb*>e$wsX8%*)JXu_AX8zy@?JuGd5O#Z0c;#-aSUQ?CEIw+9yiq@9o`puUCZbS(@DWjq_ctYtc8i#E3hZY<Z}p%kN-4T(R_~%RPhi@MmN5lVfh^+44P3?HzDca}QK+>)Pdlp5Eo?(mC|FI@YgI|I)r~>Uis1qsOh*i2a$ga`T%V7LOL+2Um4HCm%bRt|=?N;;wg(#`s<r<6ah%dejb*9iN&<-R>G9FZ-MI`ks47GK@`P3pa$zEyE(4kw#JCsw<u2&c;aPhUfZQwu+OKe!D*O7#T04>uedY(J?`W^jkmN+dEMjnY}h{5St`fvl}-u%1f4@zI|Oszey3F_wB2=yi1iHPvY*_7N*IN56xCMC8bO3l<AS<{4&J#SEsTww>}W-?tc#~HZD^PXWoAHs?tLlX!7sLGa(NprfoOZHZ!xtEppIbZGUEo-{7-$^*3cp`|U#pPcEJ#FUEFgvu8<;+-&=?WV6H^xzjUjU}oi9aVlrq>_^93Sy<!Qmv5tT<<|5P<DKntW#HoP@nQewO37u(+k)5R%8R7A4v~(zQZH;s4OyEjCE7mUBmd>fh|aEuHqXwLp%2ToGP2B-ppS#Dx%AAHwB~&dw{DOt=cZP3Tl_6YzSf%`Zh0|B42xwhuRbwH{wjOGaO{t4nS69;Se!?;bS%4dQm^KJVx#Mad-`Na%Ub>FbT!G6A#o4qMxS~plP5o{99{FFln8x3vhK!AsZ{uOpwaUO61-)?hz7$RNQpId*1b8EA<G6@{@DF3U3T@Y@uA$Xbg}zjwxH6UG<h@SYGIoPsbV-F;rX6=sdDn=hmMn{rbtz*9*vqDO_oJlo$vO1m?S+41|95FGfB3VyVYpx$VBOXV&7Qf4GHpS=D^?1SL4Me{9cM%<bBEMQtSP?)^TE;-8tQN+dZlKy_?IR=h4zA`sS+3{iDR7{#4JVM<c|fa9l;7?|0=<E9=rv#)Zk>c02FyIv*<68($rj@mGk<o-{GQBJsA^Y<cyj=b=C`FP1sKqwNi8mtkXV+R<Nb%<4U)SQTFx7_dhw{qm7Y9`AyizxI-b1u1j)KRzz@dp}he=;tZkWs6LfkJ&8VOaHn(rKq#c8*S7~rnqR#vcMfbA9!kDndM21{f}$AiZ5Jd&-K#$ny)hsR`k)>4iiSaGryvD>%0t_YT&0|zSbPS*z1})oLKU_+Q6IIGT^2`<?KMsJ-*=F&BeEMUBbN%6V?Q4U7Nl=?gfOXv&FWdyWWKA@>!GLY_A`t$#LQ1&knq+)#}@pDrFO{y+Th{s^<`)>xwS*eC`sd)lHlF4)%=F@rN4CA9FlfJ^MDEc<N+~7A1WMPC9%~lZ)*&DB2yXy)4^A?s1CKR-Lw)?_79aV?2u<uWc2t-*0~^Jmnd$V>eCh^W7jps~mkg^TEsn4Lf0%av?fF^ADL+duN=eJKA-#9lJGAGgHDMmlh=IlK;9s-qklr*B^CD7_&b~o#gCi&lgGRyS7@*ZatFKHt)@~ep`}t=-H*Ue<UYsv6{XUgX*QI!<{xE1~XDLrA8Yo=Tj-_+O|}B$)_p$wRx;(hnA`8zx3Z0t7oO^qpOdnw>*`qHybwz8k(J|)BYRT=}nC^y)wbdW7)_wJ#00|>g$Fy{qFFvV3vQH#(S+RH?bg1>t!x)X<98^XIIH>|4!1?(09?<H?z}qXq3m-z-{U3^w+NE<1eJ^=l&mj$3&)U*PT9>mp@9^Q$Jq~{{1;!2X6d5DAO=Q`&izyGpd)Nu^a5h7PZb$kEPpxCUwcsNmUJ_y$57y)d9X^+FN93wS}q0t*kQiY-a4M5>qnt>N%$gC;vq6VI{gpOwQ2AvsrHpCuOM5>beWIPWW^Ebj&8BF&Vn;e5pm@L;vJ!`&hp0lcAF%jdnKdoS_p^R}cTyGDG{i__uFgCqq*bZ|=QX`cLk9x_0^Z=^D9w!LpzS>3X@0?aNZ3>3XN{(eG|v>DqGB;7g;ormK5ELEq8y(>2&>bzy}O>AE`kT%(n(({+gP%hGL1rR$4Rp_X4BrD@8c`|q5B(lmdhxkok6G_@Ol>`t+TY5H-U+wXe)|Gei?BhN3cnWnJ=D^I)dJXI?MeYtTpFjfD4-|nm7wp6ug*0hq_q*V3TKX=9+<5caDHZ92E+n@dXJ)rx?$P}G5HX)$UBSrsecdq-_Kl^>=L;nFm?NYSYp}f%6zml~<#h&%b#w2UToCx<_p2^y3!>>Z;8Oho=t@3&YlVq*fET>VoPf1#$K}y<~+evz2$;$>Gos-nBp!L4vBa(E${ok((swU}do5&Bf(-Sq~oBxmN`x5o}$DW7w{*$OTM=Th2xk{oA9eyAxBj(RJoV2k=vp?sgS$yKS<jx6tWX-fsK85jmIk<9jkG=7_Z1(Pd&-9DeRg**79eZ$Jm#2NXUds5sK5;&0*xeyct4=g4ybv6#5huT{ODP_!tAZLPpB#8khmKoxzU`J6ZPnG^DmErsZ{3;F(79r?UaB1U@zm%jZT8Rm(rzA+TK8=6$a$F&>agBEs(G^r4Ju_EIDb*Nwtr+8JL}qAy=QoPv+3tB{rBkVO})*+)Tl$Pn+@#mXpz1-*6UiR{{Q~|0RR91000000000000000000000RQ_ON6k0>FK(~&S=A@*?~$J3QL(qZV}z&p_it|$&~Kj@L>T+n9NQ;`RSMReE7&K~Ze~uZV!2<ulWO<tcYnW_Uo3vaxBUS*)<X(Qcpi}InUlJh|2QB9ohSMSj5#RgUh9vWtU4%`-Mzc6ntM=ObE7XU=x|W_q&tPB#UBu}|3+EO9CkoFCVh_BeR03IHcrcY`DLFNKWfowVXb}QIO}MsGxa^?qxqerE=~7}>+i|sO11J3*Yg)FHk$5{kJk?Q-f8YG{$FkP*Q&Q$ye&3n1lQXo4Qt-^h_1F%=6#r*Qo7@Iv8-14<h$M5#QVd%L04+IisjhRM-Cm|BF3E#dhKrIBHnSepZe-1>GRj)<3~n1i{-1S%Wvm7iKERox1v=W<i(TxDUI(sie+I!QsVtJVo>`+1G{D`#j#nF%=LHd#c}1QxUlJq#d~;<z9p>f#9;n_Ly51)h(UVM1Dipe)$(-QunLc-s^x0W_e;($R7ca1!HM7g(-fa2Z|ox0>b%Vb*?E1PH06pxt6AMQt7T}5@ne^}YOtMU^{Kf#^hJo<*LSA7wU5>1s=Fud(S}D`I6auLSB=Z0j;c3)pL&N#vDSSLsDJ+Z=liN1QvX?@cFR%^>#<IQZmryPR2w#Ze}B^TW12g?rTK(&$N!8ceP(Vyp~pJUJLI(Lq}m6MdR$}fDK$8I?`KJ?(`px#J7L&}Ga8@e(dT@BFI{|R)0oHsXSG>1HMbdeP6G?~w=O&TyxJXiu=xF-w;H-NzP@+eg+KEys<3RMkJ{gDRR8zRi@JEy_vKrBFR8~%^Mrtfm(`+baGUbQujqtBXNOnccSSoo$e9Lhd^JA%TCXX0ebr$2x++ztUsdz)9?oljUe(iAeNu0E`srA+kpAo2`0Jf_Uc29h`K$T9@<+c;xTZ~B8Fd&?a7`y0b$M}d;dSl0<j(S^Z?EgXS0!5fUU)-o-jDG*{pL?Bf4H#z;+tyU#{YMVXE$|~MbP#O69Y8Rv~}T=@Bkf=V^K4>b)bIOlK#bDQ=ks?&)@SRF;IsL8o1BC;w`P^Fr@GPZnxB8Ol;x7p?~s!v-9ek+|sHJ)wW-`AE+@&PNC0D19eS!^fI5`0cz2zdF7+OZm9W>qK2_^uIu6-ZkaEg{MGtra*tmluIgRWlSACXFYCM;AIdrxc&q>8b8AbVKCOKmLkgOCA68@MHHXeL-LBrx4|nw2&`#zZFnDyo%1*JoTVY~O^J8N8baTP01!ra1l78>n+g=ufq<&lG9rBYB2g-f3$-N=9mfBo=*dRzs=E#amr-Q|_dCrMP7eeLDq=WGWXToH0>E4}ecZW-)%{S}9Ig!$2ll$~e&7;LEGpWJ7@EF+_7NzseW2Ntf&lAJ_;^f7o2g|1ZzAt;1x+MLzCtmux2Bmt{NRWQ(_I9h`k{}}sQ+l~QOOP4QE;_C3mMGSTwyv(~kSI^vPq7<(I#Fgvm0lB|i85z&G22p+iDG6y@~wSPqMY)uxEg#cQ6700|7QJfqBK8c@viKj`&YPEtCaCNK~fh~eXz+RLA;uWZ#{3CAoi#J)aA{1*>SV&*J@qjrIlq&lgMHB<$KoK8h6XYNr_LbelIz9Px6m6nrhZ2MuxsEmDc`1lz2Dx$vIvWA#Pqjtg813mjUzU<b<yZ6Qehq^SYi1l?>l<Jw1bi#mAuK67yq0@^f=`uODY`%Iq3>C&Ksp%Z-o_n|#MBGOxxh`8)i)eD7;qShVPb95ejbY{DmZF<7^xdwkPb@}g(-j}!LWv|&Vt;1;uw>R-b~KPWfqtp2>SWk<-L`ae7{zs35Ce!6C?U(2C+H`K0SUiPVyL7H6m%99gYf_1*r({W?AgsS)Ez$aZdglSe_x2cv(!nNVys^`v*j#RsH!Fygcj@AXAO5ghZBu4Yh!#u4#V|9%~qtcPx?(35#7L~W##p{Ur!M#2<Pta1v9hR8nCTN!HkIcQ>6LrylA2P>xO43I0cWX>fO41f{TK{e{H(8^vyLayVIax=QDVx84Wr{AJS~`DPQHqXgGQ!<*Myh_T=eq2`om92aPPeYqO4BA4#;)sOlcsBItp?9InWjTalqhh_NYhvA!?qiiPS>3aOMS0toUXmHUhg*^l&%)le=km-kgg?Xc1eFTBVF&9I1X{LN!O;8rpy>K@lU^EN$(mDO;^)FNt3R%OV=-^uNN$>l&(f4ukGIRG))&puD$1eElr!PE54$~>NFiY`b^(zUDMQ}-tAEvUZ?8fi8)1Y52fm!E3PKL2c+uDf=x3%Jx$TspWLVK-k73s4K4S$)=5#98{a2Jo=w*H{XgDTX_u^~8Hc)zy^^FUcRrQ~Xq}|30`@H~dn8d$1(n_0y?mnH+Of+m#XdpHjIFR|TV}lW|J%Cp+Me;c#$!_Ax3>4S!P*8PO|oNk-kWDnJ8r(GZ=3J?tGG#wt}9>mYDQd?X8k?&QO1HuHLhn>a$1=P9qsHe@axgL+T~iJ+p*4JdgFB4@SNyStsc9v&BaL}>hsDk?|YW&jk2eWUj4YGcgGEXTBUY?76uh1WOTW%fxF7ApFP%3W9QEDn>qW6uIlZxJoC*3ec`it=kDTOYP4u}hxg@=tB1YArJzxJH8|J$WtDZ#>izm$erU@<;ytHI()r#l;@^A6%^4n^;+VZ;?xKyy#eVpDyUWI2;t|_^bI02k#BRropJuhMNZ_k8L(GG&is|``a{rCECKEOr)>{*MLq5z|(eX~7KxwjgyQ#;iAUSrrkL&G^Ds}tEec4z&L|T+{vrB0ZDlx8Rj&<tXk=$Z~Ym_MwCJX9*SyC}GOlB|oXGNPccf~HI``#Hd!sWu+YL>^#M@VF0i$;}BMo7Lx%Z{^}M~Y|mvNaVCMoPU;z8z;3i;{z7bi@ejC^7u6h2A?CB`Ffn?B3HTSzmwCPV4H?a(BYKI-k2l%i_7ij7N`-mNMU+J`J;pmToaFH}dC3%lPx{lUB@(mPQMo9WMS)w8R8@UNq_*Eju@T`(3tiv<#2we}B-gD9Qa=eYj0>l*GTR@}RU=lx*p}YG~k!C|QtKeNL_6QL@HmlV=m7D5>U?;_&T7q*Plx{%T}!q<n}um~F5>QoaYYs@BLpQfjVue_m#Eq}YX)N!i~vQcT}veo3kxDUJ)DTz*|LQi^%6ZM)z{gxL8U>*w@6LTume?e(HWq|C3k&h2>3KmGe!UZ34HQhpbftMbJ*QeL!sop<ri^E)ka?+}z0DU*-v^IO^^N_MVy{kQbmC~0O;ZBx6#D4FHg{p8?j(Ne5mFT2eJ(K31HiMdDC#YmIlZ|XNRy(eXMpQz>W?w+g|oi|}%RICI|-{>Fa9Vf2lqmMTAxGzr~+L~<cA1`Kc=T1BiiI?<4$DVf`lpr6L@0!syF+tW2Gs_=7Gf{5*)3BXQu_Sr$#W~#Zc#@o(eaq!@|73Z(>*#f}XUTH&{w$M~?kSSlbY<a${;BfuV9SJeUs9#0#KO!;erfW1P)Jh9vUIW7-X{K)S%#PxH*7Tidxk`wvCCZ&_CSW%x9rq^N2VC0c3d;h`k`DJ@L}BVmRT~?G4}Jx&skz!$!3CiWVVc*ot?gGe~!2$R-c<<mMixrH0cz7Bv-1xo#DTz^dniZqH0m$!bh_I`mW+@k{?NlhwZ06>hf616sz58=)T8dd#l8jp`RX0?b}-hSdGq;RZn&g3qPAD=1<1iefXXy4WG4;@xz`-pEBi7E#LD*n!Oy+Z%M`zc~SR3>kl>a<$|NZn6no7;*#-WQg7#cS@L+m{>lFN(rikT=S4aBQuo8vI^F&%kh&#9PNz35kZJQKY^&J2K&<ktx{VuMAY}$$Y1MH`frK2jXxrYlKnyyWl%HW!AQwWnEF3kdK-%qH*C2XSf!xZgWVNzqfvhj*V0EZbfdtu2Y}weLKz>)+Hr_TZUmQ&0zHjx)msRmKtK43a|EGQe_r;p!ONE-fwru_IL_!h=-(7Y7iQKpE@XmbV6FKCP?e1RUiCk)4Y}$6OJgKy4MA49edGhaj`-2XdkEO}4hjXgVdn_BZv-9VoM{;QHgdYza9*OnXqM7YJ<%;X<iZ$vi%oX`qqUMIM961<zXh7vN+2Xv(_TAidSyIL$I3!`zLoxka>eYoBnPOG^Q+V%`3@P#EiAj66bg58no@qd@H0d&LW0!%2DPkX0>A?uMWN|C-oAcHrN!q#Q^_~@(ARk{w_53|GUW^l3*xbpFm2OwQ8if>#ky}?X>ffsxDPO~em0MN!t}H01yFaaNsFbOE&9+DT+u~BO#>L9bZb+|}m4gRV_Lac7(Y2Fbc}ev<p`A>vJjJ`ir8QNY#;D`>XS<$l*slZsIr*k_&RI2$>sIxggRfpKKhz=D@rF+DwX|)v;kMeodwh9#w@}Tm=yzvMi@WMA8;q8hi`1$$JYB!%MC+H6|I~;&6RQ;hH#baf6R$b-?6zdCO;8WxT?5X9CF<be8?GA|B<rQoLtL-6Nl~M{!_V!tNY$~uxA?cWO;eY`AI6Te()HQYVd)>NGPK$Kjy*4Rf1pkm4IlWF&D5(uJ58+@k*Oof^-E~9^r0TFsv|#@%hKvG?|rB2%u?6RAuB_RWvhRcO9_vrW^1)+4{9Wz&(@@Vh2!2n&eqMRx*W?dlcTE^+!!&SZjP?sIKtqbQI3vtyqeLgbdJ8VkDhxZCtC~mTV-XP&ej`;U!C}SVzwHUi(Nk9XO;#f9l2BS&p7cO2S@a-lBJmiCabPHJ=Cg(Uq+UElBuf-mfWh-IaAHT{I+MWexR?OYMB4^40S8&Q8p((U27d^T7F5{bnO(}&fTVInp(Gy+iKP=RlPE2UD`i1MKiV?F&zF+vX*SHv5nu<B;9#y{pxcw6SdFMkt3(fPtd8khkxcRjn^g9yX~I-IZjI)Ydr5*g;;e7TX&#y>liH>P=Ds`aZzerqp<I=oe^qcSb2Y|hj;ao@B1b`UBk3inFaQB&W7rT&MyZnHwn?)8AkV;Y*TGw)u>DS>s$J9Umv$VTLN^#`Ka>WC*9D@ZxuIgopnu5o;hKC{k)%69UWQHs?k+Vng3wfo`@^@w%vj?DHkqlW|hRP*5@y&!N$y&!xvvv;|g_mpE~EG)@H{GuUY%(_mp3ix4Zjj;FZOnPMTlT__lYx_j0(TxfLu-{G2Xpy?Vua6)d}=hpktf8s+Z$rw*na%1OP-|9t=e00000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000030Py7#1O9KQg000080000X00Qxuz5oCJ|NsC0{|*2Y0CZ?_Wpik6Y-DpTZg6=401yBG2mk;800014000000001+>=)`A5Xs0;#!#)El3JWxq;934Zj)xAuA`uymS0p-l$aNvUzCyx5_e0?DNY577iT0EqyqUGMkWS2nmP)#3S<K=V=%aEKT40%qx2{}N{`Z`H2r7*P)h>@EdT%j2mk;8ApmD*hsyu||NsC0|Nj~Q6aaK+a%FR9Z){{=V{c_<W@%$-Wo~qHE^csn0RRvHfF1w<00000YybcN00000?Z`0-gFqAj(Y5syzxxrk5V4SuI$gxZ!b%ohNfBb$l{Dfha$(ICQigfm%%^T|O_!LrS#|Vr2znO1OqE*EX%0(p&wdW0|Ng6cM1TBvcQ1eRsVH+RWoBiQAEgEW00000000000Pq)fP)h>@EdT%j2mk;8AprVbKn(x?|NsC0|NjsG6aZyzbaHuLVsCYBWOFWVaCrd$5CEV600000003D400000005ip7wQ`j$;eQ~P_3SlTAW;@Zl$1ZlV+i=qoAIaUsO_*m=~X4l#&V(cT3DEP6dh=XCxM+0{I$7ItoUbItsN4WCJd)>x&C)xvty)s>|9_GP%PZoeuy|O9KQg000080000X0Gft&mH+?$|NsC0{|NvT0C#C*bZ9PaaCrd$5CDh(00000002q=00000005ip7wQ`j$;eQ~P_3SlTAW;@Zl$1ZlV+i=qoAIaUsO_*m=~X4l#&V(cT3DEP6dh=XCxM+0{I%6ItsN46ag;AOMyZrvs3H=P)h>@EdT%j2mk;8Apm5i_}Tye|NsC0|NjjD6aaH|aBy#ObYD&`Zg6=401yC(000000000-000000001+>=)`A5Xs0;#!#)El3JWxq;934Zj)xAuA`uymS0p-l$aNvUzCyx5_e0?DNY577iT0EqyqUGnmP)#3KRh@1~3qD002-+0Rk-pEdT%j2mk;8AplW!_AyZ(0001(1^^HM0000000000004ji00000V{c_<W@%$-Wo~qHE^csnP)h*<Edwn8000O8001EX0`Zx>SpWb400;m84gdfE0000000000fB};q004Aoa%FR9Z){|9E^csnP)h*<Edwn8000O8001EXXJ?1XYybcNfF1w<8UO$Q0000000000fB`HZ004Aoa%FR9Z){{=V{c_<W@%$-Wo~qHE^csnP)h*<Edwn8000O8001EX`d&Z`SpWb4pa1{>5C8xG0000000000fC1zn003ofbaHuLVsCYBWOFWVaCuNm0Rk-pEdT%j2mk;8Apn|&c9lv1004*p000R90000000000004jie<1(>cWGpFXfAGWc~DCM0xbhA000080000X0A!~4*+&2X0Ehqp01W^D00000000000Du7mA^-q$b#QQRa&%u#E^csnP)h{{000001^@;CasdDUiXs32000'


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
        self.r8_extra = False
        self.r8_base_detect = False

    def update(self, t, th, thd, tau_m, tau_s, grip):
        residual, base = super().update(t, th, thd, tau_m, tau_s, grip)
        self.r8_base_detect = bool(base)
        extra = False
        if self._r8_member is not None:
            extra, _, _, _ = self._r8_member.update(
                t, th, thd, tau_s, bool(grip),
                max(0., self._shared.last_wrench[2]), bool(base))
        self.r8_extra = bool(extra)
        return residual, bool(base or extra)
