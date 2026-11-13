"""A4 L2c uses the L2b_b gates and enables a calibrated no-box channel.

The separate no-box arrays are selected by known mass before construction;
there is no invented mass, true-contact input, or change to prior source files.
"""
import numpy as np

from experiments.r8.lat2b import Channel as PreviousChannel, WINDOWS


class Channel(PreviousChannel):
    def __init__(self, mass, com, coefficients, thresholds, threshold_coefficients,
                 entry_bounds, width_m, support_N, *, band=None):
        super().__init__('b', mass, com, coefficients, thresholds,
                         threshold_coefficients, entry_bounds, width_m,
                         support_N, band=band)

    def update(self, t, signal, height_distance, support_force_N, grip, base_alarm):
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
        error = float(signal)-baseline-float(self.regressor@self.coefficients[:, k])
        self.count = self.count+1 if abs(error) > self.threshold(k) else 0
        if self.count >= 4:
            self.latched = True
            return True
        return False
