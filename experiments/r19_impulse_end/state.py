"""Unsigned normalized-impulse end state over the existing eight-frame window."""
from collections import deque
import numpy as np
from experiments.r18_single_pass.reference_recovery import WINDOW_SAMPLES


class ImpulseEndState:
    def __init__(self): self.reset()
    def reset(self):
        self._ratios=deque(maxlen=WINDOW_SAMPLES);self.active=False;self.end_confirmed=False
    @property
    def samples(self):return len(self._ratios)
    @property
    def axis_sums(self):
        return np.zeros(6) if not self._ratios else np.sum(np.asarray(self._ratios),axis=0)
    def update(self,onset,ratios,armed=True):
        if not armed:self.reset();return False
        ratios=np.asarray(ratios,float)
        if ratios.shape!=(6,) or not np.isfinite(ratios).all() or np.any(ratios<0):raise ValueError('six finite unsigned ratios required')
        onset=bool(onset)
        if onset and not self.active:self.active=True;self._ratios.clear()
        if not self.active:self.end_confirmed=False;return False
        self._ratios.append(ratios.copy())
        self.end_confirmed=(not onset and len(self._ratios)==WINDOW_SAMPLES and bool(np.all(self.axis_sums<=WINDOW_SAMPLES)))
        if self.end_confirmed:self.active=False
        return self.active
