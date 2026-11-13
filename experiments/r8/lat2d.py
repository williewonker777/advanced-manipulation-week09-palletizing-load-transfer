"""Development-only set membership on the unchanged L2c normal calibration.

Three stiffness hypotheses affect only encoder/JTS link-angle reconstruction.
Payload-position sensitivity is a centered nominal inverse-dynamics derivative
along the two public world-horizontal axes, followed by the same observer and
constant prewindow baseline. This is a discrete/local uncertainty model, not a
certificate over continuous stiffness or nonlinear 30 mm grasp errors.
"""
import numpy as np
import mujoco
from experiments.r8.lat2 import SignalBank, critical_step
from experiments.r8.lat2c import Channel as L2c, WINDOWS
from sim.nominal import Nominal
from sim import trajectory as T


class HypothesisSignal(SignalBank):
    def __init__(self, mass, com, stiffness_scale=1., *, com_delta=None,
                 epsilon_m=.001):
        super().__init__(mass, com, include_L3=False)
        if stiffness_scale <= 0 or epsilon_m <= 0:
            raise ValueError('positive stiffness and difference step required')
        self.inv_k /= stiffness_scale
        centre = np.asarray(com, float) + (0. if com_delta is None else com_delta)
        self.full.set_payload(mass, centre)
        # The taught grasp rotation is public, not a simulated true pose.
        n = Nominal()
        n.d.qpos[:] = T.reference(1.4)[0]
        mujoco.mj_kinematics(n.m, n.d)
        rotation = n.d.xmat[n.m.body('link_6').id].reshape(3, 3)
        self.axes_local = rotation.T[:, :2].copy()
        self.epsilon = epsilon_m
        self.offset_models = []
        for axis in self.axes_local.T:
            pair = []
            for sign in [-1., 1.]:
                model = Nominal()
                model.set_payload(mass, centre+sign*epsilon_m*axis)
                pair.append(model)
            self.offset_models.append(pair)
        self.sensitivity = np.zeros(2)
        self.sensitivity_rate = np.zeros(2)

    def update(self, t, th, thd, tau_s, grip, *, cached_L3=None):
        values, distance = super().update(t, th, thd, tau_s, grip,
                                          cached_L3=cached_L3)
        q, qd, qdd = T.reference(t)
        error = np.asarray(th)-self.inv_k*tau_s-q
        acceleration = 40**2*(error-self.position_error)-80*self.velocity_error
        args = (q+self.position_error, qd+self.velocity_error, qdd+acceleration)
        raw = np.zeros(2)
        if grip and self.mass > 0:
            for j, (minus, plus) in enumerate(self.offset_models):
                raw[j] = -(plus.inverse_dynamics(*args)[0]
                           - minus.inverse_dynamics(*args)[0])/(2*self.epsilon)
        self.sensitivity, self.sensitivity_rate = critical_step(
            self.sensitivity, self.sensitivity_rate, raw, 25.)
        return values, distance


def membership_statistic(errors, sensitivities):
    errors, sensitivities = np.asarray(errors), np.asarray(sensitivities)
    if errors.shape != (3,) or sensitivities.shape != (3, 2):
        raise ValueError('three stiffness hypotheses and two offset axes required')
    return float(np.min(np.maximum(0., abs(errors)
                    - .03*np.linalg.norm(sensitivities, axis=1))))


class Membership:
    def __init__(self, mass, com, model):
        self.banks = [HypothesisSignal(mass, com, scale)
                      for scale in [.8, 1., 1.2]]
        self.channel = Channel(mass, com, model['coefficients'],
            model['thresholds'], model['threshold_coefficients'],
            model['entry_bounds'], float(model['width']), float(model['support_N']))
        self.signals, self.sensitivities = [], []
        self.baselines = {}

    def update(self, t, th, thd, tau_s, grip, support_N, base_alarm):
        observations = [b.update(t, th, thd, tau_s, grip) for b in self.banks]
        signal = np.array([v[0][1] for v in observations])
        sensitivity = np.array([b.sensitivity.copy() for b in self.banks])
        self.signals.append(signal)
        self.sensitivities.append(sensitivity)
        k = int(round(t*1000))
        phase = next((j for j, (a, b) in enumerate(WINDOWS)
                      if a-.02-1e-12 <= t < b-1e-12), None)
        errors, centred_sensitivity, statistic = np.zeros(3), np.zeros((3, 2)), 0.
        if phase is not None:
            if phase not in self.baselines:
                start = WINDOWS[phase][0]
                lo, hi = round((start-.12)*1000), round((start-.02)*1000)
                self.baselines[phase] = (
                    np.mean(self.signals[lo:hi], axis=0),
                    np.mean(self.sensitivities[lo:hi], axis=0))
            baseline, sb = self.baselines[phase]
            errors = signal-baseline-self.channel.regressor @ self.channel.coefficients[:, k]
            centred_sensitivity = sensitivity-sb
            statistic = membership_statistic(errors, centred_sensitivity)
        self.last_nominal_distance = observations[1][1]
        hit = self.channel.update(t, signal[1], self.last_nominal_distance, support_N,
                                  grip, base_alarm, statistic)
        return hit, errors, centred_sensitivity, statistic


class Channel(L2c):
    def update(self, t, signal, height_distance, support_force_N, grip, base_alarm,
               statistic):
        if not np.isfinite(statistic) or statistic < 0:
            raise ValueError("finite nonnegative membership statistic required")
        k = int(round(t*1000))
        if abs(k*.001-t) > 1e-12 or k != len(self.raw) or not 0 <= k < 6000:
            raise ValueError('complete ordered 1ms stream required')
        self.raw.append(float(signal))
        self.latched = bool(self.latched or base_alarm)
        phase = next((j for j, (a, b) in enumerate(WINDOWS)
                      if a-.02-1e-12 <= t < b-1e-12), None)
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
            lo, hi = round((a-.12)*1000), round((a-.02)*1000)
            if k < hi:
                return False
            self.baselines[phase] = float(np.mean(self.raw[lo:hi]))
        baseline = self.baselines[phase]
        in_phase = a-1e-12 <= t < b-1e-12
        geometric = bool(grip and height_distance <= self.width)
        support = bool(grip and self.mass > 0 and support_force_N > self.support_N)
        self.diagnostics['contact_blocked'] += int(in_phase and geometric)
        self.diagnostics['support_blocked'] += int(in_phase and support)
        opened = bool(in_phase and not (geometric or support))
        if opened and not self.previous_open:
            indices = np.arange(k-100, k)
            error = np.asarray(self.raw[k-100:k])-baseline\
                -self.regressor@self.coefficients[:, indices]
            limit_abs, limit_var = self.entry_bounds[phase]
            prior_ratio = abs(error[-20:])/self.threshold(indices[-20:])
            if (len(error) != 100 or np.max(abs(error)) > limit_abs
                    or np.var(error) > limit_var or np.max(prior_ratio) > .8):
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
        self.count = self.count+1 if abs(error) > self.threshold(k) else 0
        if self.count >= 4:
            self.latched = True
            return True
        return False
