"""Load-transfer-aware elastic-joint command generator (servo architecture)."""
import numpy as np
from sim import config as C
from sim import trajectory as T

MODE = "servo"


class Controller:
    MODE = MODE
    USE_C1 = True
    USE_C2 = True
    USE_C3 = True
    LOAD_SOURCE = "sensor"
    UNLOAD_START = T.T_RELEASE - 0.06
    UNLOAD_AMOUNT = 0.10
    ENDPOINT_DELAY = -0.005

    def reset(self):
        self.dt = float(C.DT)
        self.elastic = np.asarray(self.nom.elastic, bool)
        self.inv_k = np.zeros(6)
        self.inv_k[self.elastic] = 1.0 / self.nom.K[self.elastic]
        self.v_est = np.zeros(6)
        self.q_prev = None
        self.sensor_bias = np.zeros(6)
        self.bias_ready = False
        self.load_fraction = 0.0
        self.tau_link_prev = None
        self.tau_dot = np.zeros(6)

    def _reference(self, t, q_d, qd_d, qdd_d):
        """Compensate the simulator's documented one-cycle command delay."""
        ta = t + self.dt
        if not self.USE_C3 or ta < T.T_RELEASE - 0.15 or ta >= T.T_RELEASE + 0.10:
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

    def _estimate_load(self, t, th, tau_s, grip, qdd_d):
        q = th - self.inv_k * tau_s
        qd = np.zeros(6) if self.q_prev is None else (q - self.q_prev) / self.dt
        self.q_prev = q.copy()
        self.v_est += 0.08 * (qd - self.v_est)

        tau_empty = self.nom.inverse_dynamics(q, self.v_est, qdd_d)
        residual = tau_s - tau_empty
        if not self.bias_ready:
            self.sensor_bias = residual.copy()
            self.bias_ready = True
        elif t < T.T_GRASP - 0.08:
            self.sensor_bias += 0.002 * (residual - self.sensor_bias)

        trajectory_load = (np.clip((t - T.T_GRASP) / 0.025, 0.0, 1.0)
                           if grip else 0.0)
        if self.LOAD_SOURCE == "ramp":
            target = trajectory_load
            sensor_load = trajectory_load
        elif not grip:
            target = sensor_load = 0.0
        else:
            payload = (self.nom_box.inverse_dynamics(q, self.v_est, qdd_d)
                       - self.nom.inverse_dynamics(q, self.v_est, qdd_d))
            centered = residual - self.sensor_bias
            scale = float(payload @ payload) + 1e-9
            sensor_load = np.clip(float(centered @ payload) / scale, 0.0, 1.0)
            # The projection is informative but dynamically noisy.  Bound it
            # around the documented 25 ms trajectory comparator rather than
            # allowing a single JTS transient to command a motor-torque spike.
            if t < T.T_GRASP + 0.20:
                band = 0.12
            elif t >= self.UNLOAD_START:
                band = 0.02
            else:
                band = 0.01
            target = np.clip(sensor_load, max(0.0, trajectory_load - band),
                             min(1.0, trajectory_load + band))
        self.lambda_sensor = float(sensor_load)
        self.lambda_trajectory = float(trajectory_load)
        self.load_fraction += 0.14 * (target - self.load_fraction)
        if not grip and self.load_fraction < 1e-4:
            self.load_fraction = 0.0
        return self.load_fraction

    def compute(self, t, th, thd, tau_s, grip, q_d, qd_d, qdd_d):
        q_ref, qd_ref, qdd_ref = self._reference(t, q_d, qd_d, qdd_d)
        load = self._estimate_load(t, th, tau_s, grip, qdd_ref) if self.USE_C2 else float(grip)
        if self.USE_C3 and self.UNLOAD_START < t < T.T_RELEASE:
            s = np.clip((t - self.UNLOAD_START) / (T.T_RELEASE - self.UNLOAD_START), 0.0, 1.0)
            unload = 1.0 - self.UNLOAD_AMOUNT * (10 * s**3 - 15 * s**4 + 6 * s**5)
            load = min(load, unload)
        tau_empty = self.nom.inverse_dynamics(q_ref, qd_ref, qdd_ref)
        tau_full = self.nom_box.inverse_dynamics(q_ref, qd_ref, qdd_ref)
        tau_link = tau_empty + load * (tau_full - tau_empty)

        if not self.USE_C1:
            return q_ref, qd_ref, qd_ref, self.nom.B * qdd_ref + tau_link

        theta = q_ref + self.inv_k * tau_link
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
        tau_ff = self.nom.B * theta_ddot + tau_link + friction
        return theta, theta_dot, theta_dot, tau_ff
