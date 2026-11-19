"""R1 plus one-sided Cartesian damping of upward lower-face rebound."""
import mujoco
import numpy as np

import r1_servo
import r1_torque
from sim import config as C, trajectory as T


HEIGHT_BAND_M = 0.015
ARREST_TIME_S = 0.020
COMMAND_DELAY_S = C.DT


def smooth_window(distance, width):
    """C1 compact support: one at zero distance and zero at ``width``."""
    s = float(np.clip(abs(distance) / width, 0.0, 1.0))
    return 1.0 - (3.0 * s**2 - 2.0 * s**3)


def point_state(nom, q, offset):
    """Public nominal world point and translational Jacobian at box lower center."""
    nom.d.qpos[:] = q
    mujoco.mj_kinematics(nom.m, nom.d)
    mujoco.mj_comPos(nom.m, nom.d)
    body = nom.m.body("link_6").id
    rotation = nom.d.xmat[body].reshape(3, 3)
    point = nom.d.xpos[body] + rotation @ offset
    jacobian = np.zeros((3, 6))
    mujoco.mj_jac(nom.m, nom.d, jacobian, None, point, body)
    return point.copy(), jacobian


def reflected_mass(nom, q, jacobian_row):
    mass = nom.mass_matrix(q)
    inverse = np.linalg.solve(mass, jacobian_row)
    return 1.0 / float(jacobian_row @ inverse)


class _Rebound:
    def reset(self):
        super().reset()
        self._offset = np.array([0.0, 0.0, C.BOX_POS[2] + .5*C.BOX_SIZE[2]])
        rest = T.reference(T.T_RELEASE)[0]
        point, jacobian = point_state(self.nom_box, rest, self._offset)
        self._place_height = float(point[2])
        self.reflected_mass_kg = reflected_mass(self.nom_box, rest, jacobian[2])
        # A command is delivered one cycle later, leaving 19 ms of the stated
        # 20 ms arrest horizon.  This fixes the gain from public mechanics.
        self.damping_ns_m = self.reflected_mass_kg / (ARREST_TIME_S-COMMAND_DELAY_S)
        self._q_previous = None
        self._link_velocity = np.zeros(6)
        self.max_upward_speed_m_s = 0.0
        self.max_downward_force_n = 0.0
        self.max_abs_joint_injection_nm = np.zeros(6)
        self.braking_work_j = 0.0
        self.positive_motor_work_j = 0.0
        self.active_samples = 0

    def _rebound_torque(self, th, thd, tau_s):
        q = np.asarray(th) - self.inv_k*np.asarray(tau_s)
        if self._q_previous is None:
            raw_velocity = np.asarray(thd)
        else:
            raw_velocity = (q-self._q_previous)/self.dt
        self._q_previous = q.copy()
        self._link_velocity += .2*(raw_velocity-self._link_velocity)
        point, jacobian = point_state(self.nom, q, self._offset)
        upward_speed = max(float(jacobian[2] @ self._link_velocity), 0.0)
        window = smooth_window(point[2]-self._place_height, HEIGHT_BAND_M)
        cap = max(float(self.box_mass)*9.81, 0.0)
        downward = min(self.damping_ns_m*upward_speed, cap)*window
        injection = -jacobian[2]*downward
        motor_power = float(injection @ np.asarray(thd))
        self.max_upward_speed_m_s = max(self.max_upward_speed_m_s, upward_speed)
        self.max_downward_force_n = max(self.max_downward_force_n, downward)
        self.max_abs_joint_injection_nm = np.maximum(self.max_abs_joint_injection_nm,
                                                      np.abs(injection))
        self.braking_work_j += max(-motor_power, 0.0)*self.dt
        self.positive_motor_work_j += max(motor_power, 0.0)*self.dt
        self.active_samples += int(downward > 0.0)
        return injection


class Servo(_Rebound, r1_servo.Controller):
    def compute(self, t, th, thd, tau_s, grip, q_d, qd_d, qdd_d):
        position, velocity, measured, feedforward = super().compute(
            t, th, thd, tau_s, grip, q_d, qd_d, qdd_d)
        return position, velocity, measured, feedforward + self._rebound_torque(th, thd, tau_s)


class Torque(_Rebound, r1_torque.Controller):
    def compute(self, t, th, thd, tau_s, grip, q_d, qd_d, qdd_d):
        command = super().compute(t, th, thd, tau_s, grip, q_d, qd_d, qdd_d)
        return command + self._rebound_torque(th, thd, tau_s)
