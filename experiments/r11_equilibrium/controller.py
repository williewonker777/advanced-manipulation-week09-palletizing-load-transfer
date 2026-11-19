"""R1 plus a public-geometry, slew-limited normal compliant equilibrium."""
import mujoco
import numpy as np

import r1_servo
import r1_torque
from sim import config as C, trajectory as T


TRANSFER_FRACTION = 0.55
MAX_FORCE_FRACTION = 0.90
EQUILIBRIUM_DEFLECTION_M = 0.005
APPROACH_BAND_M = 0.020
DAMPING_RATIO = 0.70
TORQUE_STEP_FRACTION = 0.05


def approach_weight(height_error):
    """C1 anticipation: zero 20 mm above support, one at/below support."""
    s = float(np.clip(1.0-height_error/APPROACH_BAND_M, 0.0, 1.0))
    return 3.0*s*s-2.0*s*s*s


def slew(previous, requested, bound):
    return previous + np.clip(requested-previous, -bound, bound)


def point_state(nom, q, offset):
    nom.d.qpos[:] = q
    mujoco.mj_kinematics(nom.m, nom.d)
    mujoco.mj_comPos(nom.m, nom.d)
    body = nom.m.body("link_6").id
    rotation = nom.d.xmat[body].reshape(3, 3)
    point = nom.d.xpos[body] + rotation @ offset
    jacobian = np.zeros((3, 6))
    mujoco.mj_jac(nom.m, nom.d, jacobian, None, point, body)
    return point.copy(), jacobian


def reflected_mass(nom, q, row):
    return 1.0/float(row @ np.linalg.solve(nom.mass_matrix(q), row))


class _Equilibrium:
    def reset(self):
        super().reset()
        self._offset = np.array([0.0, 0.0, C.BOX_POS[2]+.5*C.BOX_SIZE[2]])
        rest = T.reference(T.T_RELEASE)[0]
        point, jacobian = point_state(self.nom_box, rest, self._offset)
        self._support_height = float(point[2])
        self.reflected_mass_kg = reflected_mass(self.nom_box, rest, jacobian[2])
        weight = max(float(self.box_mass)*9.81, 0.0)
        self.equilibrium_force_n = TRANSFER_FRACTION*weight
        self.stiffness_n_m = self.equilibrium_force_n/EQUILIBRIUM_DEFLECTION_M
        self.damping_ns_m = 2.0*DAMPING_RATIO*np.sqrt(
            self.reflected_mass_kg*self.stiffness_n_m)
        self._q_previous = None
        self._link_velocity = np.zeros(6)
        self._injection = np.zeros(6)
        self.delivered_force_n = 0.0
        self.requested_force_n = 0.0
        self.vertical_speed_m_s = 0.0
        self.height_error_m = 0.0
        self.slew_limited = False
        self.active = False
        self.braking_work_j = 0.0
        self.positive_motor_work_j = 0.0
        self.slew_limited_samples = 0
        self.max_abs_injection_nm = np.zeros(6)

    def _equilibrium_torque(self, th, thd, tau_s):
        q = np.asarray(th)-self.inv_k*np.asarray(tau_s)
        raw_velocity = (np.asarray(thd) if self._q_previous is None
                        else (q-self._q_previous)/self.dt)
        self._q_previous = q.copy()
        self._link_velocity += .2*(raw_velocity-self._link_velocity)
        point, jacobian = point_state(self.nom, q, self._offset)
        height_error = float(point[2]-self._support_height)
        speed = float(jacobian[2] @ self._link_velocity)
        activation = approach_weight(height_error)
        # Virtual equilibrium is 5 mm below the taught support.  At taught
        # height and zero speed, spring force is exactly 0.55 known weight.
        spring = self.stiffness_n_m*(height_error+EQUILIBRIUM_DEFLECTION_M)
        requested_force = activation*np.clip(
            spring+self.damping_ns_m*speed, 0.0, MAX_FORCE_FRACTION*self.box_mass*9.81)
        requested_torque = -jacobian[2]*requested_force
        next_injection = slew(self._injection, requested_torque,
                              TORQUE_STEP_FRACTION*np.asarray(self.nom.tau_max))
        limited = not np.array_equal(next_injection, requested_torque)
        self._injection = next_injection
        denominator = float(jacobian[2] @ jacobian[2])
        delivered = max(float(-self._injection @ jacobian[2])/max(denominator, 1e-12), 0.0)
        power = float(self._injection @ np.asarray(thd))
        self.delivered_force_n = delivered
        self.requested_force_n = float(requested_force)
        self.vertical_speed_m_s = speed
        self.height_error_m = height_error
        self.slew_limited = limited
        self.active = bool(activation > 0.0)
        self.slew_limited_samples += int(limited)
        self.max_abs_injection_nm = np.maximum(self.max_abs_injection_nm,
                                                np.abs(self._injection))
        self.braking_work_j += max(-power, 0.0)*self.dt
        self.positive_motor_work_j += max(power, 0.0)*self.dt
        return self._injection.copy()


class Servo(_Equilibrium, r1_servo.Controller):
    def compute(self, t, th, thd, tau_s, grip, q_d, qd_d, qdd_d):
        position, velocity, measured, feedforward = super().compute(
            t, th, thd, tau_s, grip, q_d, qd_d, qdd_d)
        return position, velocity, measured, feedforward+self._equilibrium_torque(th, thd, tau_s)


class Torque(_Equilibrium, r1_torque.Controller):
    def compute(self, t, th, thd, tau_s, grip, q_d, qd_d, qdd_d):
        command = super().compute(t, th, thd, tau_s, grip, q_d, qd_d, qdd_d)
        return command+self._equilibrium_torque(th, thd, tau_s)
