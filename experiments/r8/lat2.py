"""Normal-only Stage A2 J1 channels with causal process eligibility gates.

Inputs are documented sensors, public reference/models and the unchanged base
alarm only. Simulator contact, true pose, offset/drift/seed are not inputs.
The post-alarm latch does not disable or alter any original detector branch.
"""
from collections import deque

import mujoco
import numpy as np

from experiments.r8.noise_channel import NoiseEstimator
from sim import config as C, trajectory as T
from sim.nominal import Nominal


WINDOWS = ((1.4, 1.7), (3.6, 4.0))


def critical_step(position, velocity, target, pole, dt=.001):
    error = position-target
    decay = np.exp(-pole*dt)
    return (target+decay*((1+pole*dt)*error+dt*velocity),
            decay*(-pole*pole*dt*error+(1-pole*dt)*velocity))


def regressors(mass, com, variant):
    m = float(mass)
    if variant == 'L3':
        return np.array([1., m, 0., 0., 0.])
    return np.r_[1., m, m*np.asarray(com, float)]


class SignalBank:
    def __init__(self, mass, com, *, include_L3=True):
        self.mass, self.com = float(mass), tuple(com)
        self.empty, self.full, self.geometry = Nominal(), Nominal(), Nominal()
        self.full.set_payload(mass, com)
        self.inv_k = np.zeros(6)
        self.inv_k[self.empty.elastic] = 1/self.empty.K[self.empty.elastic]
        self.l3 = None
        if include_L3:
            # Keep the contrast observer's public-model scratch data independent
            # from L2 inverse dynamics, as in its original standalone runtime.
            l3_empty, l3_full = Nominal(), Nominal()
            l3_full.set_payload(mass, com)
            self.l3 = NoiseEstimator(l3_empty, l3_full, mass, com)
        self.mean = deque(maxlen=5)
        self.position_error = self.velocity_error = None
        self.disturbance, self.disturbance_rate = 0., 0.
        self.bottom_offset = np.array([0., 0., C.BOX_POS[2]+C.BOX_SIZE[2]/2])
        self.table_heights = np.array([self.bottom(T.reference(t)[0]) for t in (1.4, 3.9)])

    def bottom(self, q):
        nom = self.geometry
        nom.d.qpos[:] = q
        mujoco.mj_kinematics(nom.m, nom.d)
        body = nom.m.body('link_6').id
        rotation = nom.d.xmat[body].reshape(3, 3)
        return float((nom.d.xpos[body]+rotation@self.bottom_offset)[2])

    def update(self, t, th, thd, tau_s, grip, *, cached_L3=None):
        q, qd, qdd = T.reference(t)
        measured_q = np.asarray(th)-self.inv_k*tau_s
        error = measured_q-q
        if self.position_error is None:
            self.position_error, self.velocity_error = error.copy(), np.zeros(6)
        self.position_error, self.velocity_error = critical_step(
            self.position_error, self.velocity_error, error, 40.)
        acceleration_error = 40**2*(error-self.position_error)-80*self.velocity_error
        predicted = (self.full if grip else self.empty).inverse_dynamics(
            q+self.position_error, qd+self.velocity_error, qdd+acceleration_error)
        raw = float(tau_s[0]-predicted[0])
        self.disturbance, self.disturbance_rate = critical_step(
            self.disturbance, self.disturbance_rate, raw, 25.)
        self.mean.append(float(tau_s[0]))
        if cached_L3 is not None:
            l3 = float(cached_L3)
        elif self.l3 is not None:
            self.l3.update(t, th, thd, tau_s, grip)
            l3 = float(self.l3.r_joint[0])
        else:
            l3 = 0.
        distance = float(np.min(np.abs(self.table_heights-self.bottom(measured_q))))
        return np.array([np.mean(self.mean), self.disturbance, l3]), distance


def centre_signal(values):
    """Offline version of the same causal prewindow intercept/slope baseline."""
    values = np.asarray(values, float)
    centred = np.zeros_like(values)
    for start, end in WINDOWS:
        lo, hi = round((start-.12)*1000), round((start-.02)*1000)
        x = (np.arange(lo, hi)*.001-start)
        design = np.column_stack((np.ones(len(x)), x))
        coefficients = np.linalg.lstsq(design, values[lo:hi], rcond=None)[0]
        indices = np.arange(hi, round(end*1000))
        centred[indices] = values[indices]-coefficients[0]-coefficients[1]*(indices*.001-start)
    return centred


class Channel:
    def __init__(self, variant, mass, com, coefficients, thresholds, width):
        if variant not in ('L1', 'L2', 'L3'):
            raise ValueError('unknown channel')
        self.variant = variant
        self.coefficients = np.asarray(coefficients, float).copy()
        self.thresholds = np.asarray(thresholds, float).copy()
        self.regressor = regressors(mass, com, variant)
        if (self.coefficients.shape != (5, 6000) or self.thresholds.shape != (240,)
                or not np.all(np.isfinite(self.coefficients))
                or not np.all(np.isfinite(self.thresholds))
                or np.any(self.thresholds <= 0) or not .002 <= width <= .006):
            raise ValueError('invalid normal model')
        self.width = float(width)
        self.coefficients.setflags(write=False)
        self.thresholds.setflags(write=False)
        self.raw_history = []
        self.baselines = {}
        self.prior = deque(maxlen=20)
        self.phase = None
        self.disabled = self.latched = False
        self.previous_gate_open = False
        self.count = self.clear = 0
        self.detected = False
        self.diagnostics = dict(contact_blocked=0, entry_disabled=0,
                                active_samples=0, alarm_latched_samples=0)

    def update(self, t, signal, height_distance, grip, base_alarm):
        k = int(round(float(t)*1000))
        if abs(k*.001-float(t)) > 1e-12 or k != len(self.raw_history):
            raise ValueError('channel requires a causal complete 1 ms stream')
        self.raw_history.append(float(signal))
        phase = next((i for i, (a, b) in enumerate(WINDOWS)
                      if a-.02-1e-12 <= t < b-1e-12), None)
        if phase != self.phase:
            self.phase = phase
            self.disabled = False
            self.previous_gate_open = False
            self.count = self.clear = 0
            self.detected = False
            self.prior.clear()
        if base_alarm:
            self.latched = True
        if phase is None:
            return False
        start, end = WINDOWS[phase]
        if phase not in self.baselines:
            lo, hi = round((start-.12)*1000), round((start-.02)*1000)
            if k < hi:
                return False
            x = np.arange(lo, hi)*.001-start
            self.baselines[phase] = np.linalg.lstsq(
                np.column_stack((np.ones(len(x)), x)),
                np.asarray(self.raw_history[lo:hi]), rcond=None)[0]
        baseline = self.baselines[phase]
        value = float(signal)-baseline[0]-baseline[1]*(t-start)
        error = value-float(self.regressor@self.coefficients[:, k])
        threshold = self.thresholds[k//25]
        ratio = abs(error)/threshold
        in_phase = start-1e-12 <= t < end-1e-12
        if self.variant == 'L3' and phase == 0:
            in_phase = False
        blocked = bool(grip and height_distance <= self.width)
        self.diagnostics['contact_blocked'] += int(in_phase and blocked)
        gate_open = bool(in_phase and not blocked)
        if gate_open and not self.previous_gate_open:
            # Only already observed samples enter the entry test.
            if len(self.prior) < 20 or max(self.prior) > .8:
                self.disabled = True
                self.diagnostics['entry_disabled'] += 1
        self.previous_gate_open = gate_open
        self.prior.append(float(ratio))
        if self.latched:
            self.diagnostics['alarm_latched_samples'] += 1
        if not gate_open or self.disabled or self.latched:
            self.count = self.clear = 0
            self.detected = False
            return False
        self.diagnostics['active_samples'] += 1
        over = ratio > 1.
        self.count = self.count+1 if over else 0
        self.clear = 0 if over else self.clear+1
        if self.count >= 4:
            self.detected = True
        elif self.clear >= 8:
            self.detected = False
        emitted = bool(self.detected)
        if emitted:
            self.latched = True
        return emitted
