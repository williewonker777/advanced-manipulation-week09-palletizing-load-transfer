"""A3 L2-only normal-calibrated monitors, independent of existing branches."""
import numpy as np

from experiments.r8.contact_v3 import mass_weight


WINDOWS = ((1.4, 1.7), (3.6, 4.0))


def regressors(mass, com, band=None):
    m = float(mass)
    p = m*np.asarray(com, float)
    w = 0. if band is None else mass_weight(m, *band)
    return np.r_[1., m, p, w*m, w*p]


def scale_regressors(mass, com):
    return np.array([1., mass/8., mass*com[0]/.24, mass*com[1]/.16])


def centre_signal(signal):
    signal = np.asarray(signal, float)
    if signal.shape != (6000,):
        raise ValueError('complete normal stream required')
    result = np.zeros(6000)
    for a, b in WINDOWS:
        lo, hi = round((a-.12)*1000), round((a-.02)*1000)
        start, stop = round((a-.1)*1000), round(b*1000)
        result[start:stop] = signal[start:stop]-np.mean(signal[lo:hi])
    return result


class Channel:
    def __init__(self, variant, mass, com, coefficients, thresholds,
                 threshold_coefficients, entry_bounds, width_m, support_N,
                 *, band=None):
        if variant not in ('a', 'b'):
            raise ValueError('only the two registered L2b candidates are allowed')
        self.variant = variant
        self.regressor = regressors(mass, com, band)
        self.scale_regressor = scale_regressors(mass, com)
        self.coefficients = np.asarray(coefficients, float)
        self.thresholds = np.asarray(thresholds, float)
        self.threshold_coefficients = np.asarray(threshold_coefficients, float)
        self.entry_bounds = np.asarray(entry_bounds, float)
        if (self.coefficients.shape != (9, 6000) or self.thresholds.shape != (240,)
                or self.threshold_coefficients.shape != (240, 4)
                or self.entry_bounds.shape != (2, 2)
                or np.any(self.thresholds <= 0) or np.any(self.entry_bounds <= 0)):
            raise ValueError('invalid normal model')
        arrays = (self.coefficients, self.thresholds, self.threshold_coefficients,
                  self.entry_bounds)
        if not all(np.isfinite(a).all() for a in arrays):
            raise ValueError('nonfinite normal model')
        for a in arrays:
            a.setflags(write=False)
        self.mass = float(mass)
        self.width = float(width_m)
        self.support_N = float(support_N)
        if (not .002 <= self.width <= .006 or not np.isfinite(self.support_N)
                or self.support_N <= 0):
            raise ValueError('invalid normal contact calibration')
        self.raw = []
        self.baselines = {}
        self.latched = False
        self.phase = None
        self.disabled = self.previous_open = False
        self.count = 0
        self.diagnostics = dict(active_samples=0, contact_blocked=0,
            support_blocked=0, entry_disabled=0, alarm_latched_samples=0)

    def threshold(self, indices):
        bins = np.asarray(indices)//25
        value = self.thresholds[bins]
        if self.variant == 'b':
            value = value*np.exp(np.clip(self.threshold_coefficients[bins]
                @ self.scale_regressor, -10., 10.))
        return np.maximum(.3, value)

    def update(self, t, signal, height_distance, support_force_N, grip, base_alarm):
        k = int(round(t*1000))
        if abs(k*.001-t) > 1e-12 or k != len(self.raw) or not 0 <= k < 6000:
            raise ValueError('complete ordered 1ms stream required')
        self.raw.append(float(signal))
        self.latched = bool(self.latched or base_alarm)
        # Exact empty-payload fallback: the new normal model is calibrated only
        # on non-D4 positive masses; the existing detector remains authoritative
        # for the discrete no-box plant mode.
        if self.mass == 0:
            return False
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
        error = float(signal)-baseline-float(self.regressor@self.coefficients[:, k])
        self.count = self.count+1 if abs(error) > self.threshold(k) else 0
        if self.count >= 4:
            self.latched = True
            return True
        return False
