"""Stage A2 C-LAND prototype, isolated from submitted R1 controllers.

One load variable drives link feedforward and elastic motor position. Planned
derivatives use noiseless public reference tables; only the sensor-load change
uses the inherited 0.04 EMA. No contact/true-state/fixture values are read.
"""
from dataclasses import dataclass

import mujoco
import numpy as np

import r1_servo
import r1_torque
from sim import config as C, trajectory as T
from sim.nominal import Nominal


@dataclass(frozen=True)
class Parameters:
    u0: float = .6
    u1: float = 0.
    Tu: float = .04
    end: float = 3.875
    depth_per_kg_m: float = 0.
    delay: float = 0.
    damping: float = 0.


def quintic(t, start, duration):
    if duration <= 0:
        raise ValueError("positive duration required")
    s = np.clip((float(t)-start)/duration, 0., 1.)
    h = 10*s**3-15*s**4+6*s**5
    hd = (30*s**2-60*s**3+30*s**4)/duration
    hdd = (60*s-180*s**2+120*s**3)/duration**2
    return float(h), float(hd), float(hdd)


def schedule(t, mass, parameters):
    u = parameters.u0+parameters.u1*mass
    if not 0 <= u < 1 or parameters.Tu < .025 or parameters.end > 3.88:
        raise ValueError("invalid unloading schedule")
    h, hd, hdd = quintic(t, parameters.end-parameters.Tu, parameters.Tu)
    return 1-u*h, -u*hd, -u*hdd


def pose_and_jacobian(nom, q):
    nom.d.qpos[:] = q
    mujoco.mj_kinematics(nom.m, nom.d)
    mujoco.mj_comPos(nom.m, nom.d)
    site = nom.m.site("tcp").id
    jp, jr = np.zeros((3, 6)), np.zeros((3, 6))
    mujoco.mj_jacSite(nom.m, nom.d, jp, jr, site)
    return (nom.d.site_xpos[site].copy(),
            nom.d.site_xmat[site].reshape(3, 3).copy(), np.vstack((jp, jr)))


def pure_vertical(nom, reference, depth):
    if depth == 0:
        return np.asarray(reference).copy()
    position, rotation, _ = pose_and_jacobian(nom, reference)
    target = position-np.array([0., 0., depth])
    q = np.asarray(reference).copy()
    for _ in range(6):
        point, actual_rotation, jacobian = pose_and_jacobian(nom, q)
        # Small-angle orientation error is iterated to numerical convergence.
        angular = .5*sum((np.cross(actual_rotation[:, j], rotation[:, j])
                          for j in range(3)), np.zeros(3))
        error = np.r_[target-point, angular]
        if np.max(np.abs(error)) < 1e-12:
            return q
        q += np.linalg.solve(jacobian, error)
    raise ValueError("pure-z IK did not converge")


def derivatives(values):
    first = np.gradient(values, C.DT, axis=0)
    second = np.zeros_like(values)
    second[1:-1] = (values[2:]-2*values[1:-1]+values[:-2])/C.DT**2
    second[0], second[-1] = second[1], second[-2]
    return first, second


class _LoadTransfer:
    parameters = Parameters()

    def _base_reference(self, t):
        if self.MODE == "servo":
            return r1_servo.Controller._reference(self, t, None, None, None)
        return r1_torque.Controller._reference(self, t)

    def _build_table(self, delay):
        self.ENDPOINT_DELAY = delay
        empty, full, geometry = Nominal(), Nominal(), Nominal()
        full.set_payload(self.box_mass, self.box_com)
        times = np.arange(3747, 4054)*self.dt
        ref = [self._base_reference(t) for t in times]
        q, qd, qdd = (np.array([row[j] for row in ref]) for j in range(3))
        correction = np.zeros_like(q)
        depth = self.parameters.depth_per_kg_m*self.box_mass
        for k, t in enumerate(times):
            h = quintic(t, 3.75, .125)[0]
            fall = 1-quintic(t, 3.9, .1)[0]
            correction[k] = pure_vertical(geometry, q[k], depth*h*fall)-q[k]
        cd, cdd = derivatives(correction)
        q, qd, qdd = q+correction, qd+cd, qdd+cdd
        te = np.array([empty.inverse_dynamics(*args) for args in zip(q, qd, qdd)])
        payload = np.array([full.inverse_dynamics(*args) for args in zip(q, qd, qdd)])-te
        ted, tedd = derivatives(te)
        pd, pdd = derivatives(payload)
        predicted, offload = [], []
        ld_filtered = 0.
        previous_sensor = 1.
        for k, t in enumerate(times):
            factor, fd, fdd = schedule(t, self.box_mass, self.parameters)
            sensor = 1. if t < 3.9 else .86**(1+round((t-3.9)/self.dt))
            old = ld_filtered
            ld_filtered += .04*(factor*(sensor-previous_sensor)/self.dt-ld_filtered)
            ldd = (ld_filtered-old)/self.dt
            load = factor*sensor
            velocity = qd[k]+self.inv_k*(ted[k]+load*pd[k]+fd*sensor*payload[k]
                                        + ld_filtered*payload[k])
            acceleration = qdd[k]+self.inv_k*(tedd[k]+load*pdd[k]
                + 2*(fd*sensor+ld_filtered)*pd[k]+(fdd*sensor+ldd)*payload[k])
            velocity = np.clip(velocity, qd[k]-.35, qd[k]+.35)
            acceleration = np.clip(acceleration, -25., 25.)
            friction = C.VISCOUS*velocity+C.COULOMB*np.tanh(velocity/C.FRICTION_EPS)
            predicted.append(self.nom.B*acceleration+te[k]+load*payload[k]+friction)
            h, hd, hdd = quintic(t, self.parameters.end-self.parameters.Tu, self.parameters.Tu)
            u = self.parameters.u0+self.parameters.u1*self.box_mass
            offload.append(-u*h*payload[k]+self.nom.B*self.inv_k*
                           (-u*h*pdd[k]-2*u*hd*pd[k]-u*hdd*payload[k]))
            previous_sensor = sensor
        prediction = np.asarray(predicted)
        active = (times >= 3.75) & (times < 4.05)
        ramp = ((times[1:] >= self.parameters.end-self.parameters.Tu-.001)
                & (times[1:] <= self.parameters.end+.001))
        scan = np.max(np.abs(prediction[active, 1:3]), axis=0)
        step = float(np.max(np.abs(np.diff(offload, axis=0))[ramp]))
        return dict(times=times, q=q, qd=qd, qdd=qdd, empty=te, payload=payload,
                    ed=ted, edd=tedd, pd=pd, pdd=pdd, predicted_ff=prediction,
                    scan=scan, scheduled_step_nm=step)

    def reset(self):
        super().reset()
        self._a2_active = self.box_mass > 0
        self._previous_sensor = 0.
        self._sensor_derivative = 0.
        self._handoff = None
        self._deflection_previous = None
        self._deflection_velocity = np.zeros(6)
        if not self._a2_active:
            return
        attempts = []
        for delay in dict.fromkeys([self.parameters.delay, self.parameters.delay/2, 0.]):
            table = self._build_table(delay)
            attempts.append({"delay_s": delay, "J2_J3_ff_abs_Nm": table["scan"].tolist()})
            if np.all(table["scan"] <= .95*C.TAU_MAX[1:3]):
                self.table = table
                self.reset_scan = {"attempts": attempts, "selected_delay_s": delay,
                                   "scheduled_step_nm": table["scheduled_step_nm"]}
                break
        else:
            raise ValueError("no endpoint advance satisfies 95% planned FF bound")

    def compute(self, t, th, thd, tau_s, grip, q_d, qd_d, qdd_d):
        if not self._a2_active or t < 3.75 or t >= 4.05:
            output = super().compute(t, th, thd, tau_s, grip, q_d, qd_d, qdd_d)
            self._previous_sensor = self.load_fraction
            return output
        k = int(round(t/self.dt))-3747
        table = self.table
        q, qv, qa = table["q"][k], table["qd"][k], table["qdd"][k]
        e, payload = table["empty"][k], table["payload"][k]
        ed, edd, pd, pdd = (table[name][k] for name in ("ed", "edd", "pd", "pdd"))
        sensor = self._estimate_load(t, th, thd, tau_s, grip)
        factor, fd, fdd = schedule(t, self.box_mass, self.parameters)
        load = factor*sensor
        link_torque = e+load*payload
        if self._handoff is None:
            self._handoff = self.tau_dot-(table["ed"][k-1]
                + self._previous_sensor*table["pd"][k-1])
        old_handoff = self._handoff.copy()
        self._handoff *= .96
        old_sd = self._sensor_derivative
        self._sensor_derivative += .04*(factor*(sensor-self._previous_sensor)/self.dt
                                        - self._sensor_derivative)
        ldot = fd*sensor+self._sensor_derivative
        lddot = fdd*sensor+(self._sensor_derivative-old_sd)/self.dt
        self.tau_dot = ed+load*pd+ldot*payload+self._handoff
        second = edd+load*pdd+2*ldot*pd+lddot*payload
        second += (self._handoff-old_handoff)/self.dt
        theta = q+self.inv_k*link_torque
        velocity = np.clip(qv+self.inv_k*self.tau_dot, qv-.35, qv+.35)
        acceleration = np.clip(qa+self.inv_k*second, -25., 25.)
        self._previous_sensor = sensor
        self.tau_link_prev = link_torque.copy()
        self.last_commanded_load = load
        friction = C.VISCOUS*velocity+C.COULOMB*np.tanh(velocity/C.FRICTION_EPS)
        ff = self.nom.B*acceleration+link_torque+friction
        if self.MODE == "servo":
            if self.parameters.damping:
                deflection = -self.inv_k*tau_s
                if self._deflection_previous is not None:
                    target = (deflection-self._deflection_previous)/self.dt
                    self._deflection_velocity += (1-np.exp(-30*self.dt))*(target-self._deflection_velocity)
                self._deflection_previous = deflection.copy()
                weight = quintic(t, 3.75, .03)[0]*(1-quintic(t, 3.9, .15)[0])
                velocity = velocity-weight*self.parameters.damping*self._deflection_velocity
            return theta, velocity, velocity, ff
        self.v_motor += .2*(thd-self.v_motor)
        pd_torque = self.nom.B*(self.wn**2*(theta-th)+2*self.wn*(velocity-self.v_motor))
        return ff+pd_torque+self.k_tau*(tau_s-link_torque)


class Servo(_LoadTransfer, r1_servo.Controller):
    pass


class Torque(_LoadTransfer, r1_torque.Controller):
    pass
