"""Causal event continuity from R18 onset and observable unloading evidence."""
from __future__ import annotations

from collections import deque

from experiments.r18_single_pass.reference_recovery import WINDOW_SAMPLES


class EventState:
    """Keep one event active until a complete R18 window is observably normal.

    This state does not delay or replace the R18 onset.  It has no clock, fixed
    refractory interval, or access to programmed force.  Rearming follows only
    after every sample in the existing eight-sample evidence window is inside
    both the physical floor and the learned normal envelope.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self._unloaded = deque(maxlen=WINDOW_SAMPLES)
        self.active = False
        self.unload_confirmed = False

    @property
    def unload_samples(self):
        return sum(self._unloaded)

    def update(self, onset, unloaded, armed=True):
        if not armed:
            self.reset()
            return False
        onset = bool(onset)
        if onset and not self.active:
            self.active = True
            self._unloaded.clear()
        if not self.active:
            self.unload_confirmed = False
            return False
        self._unloaded.append(bool(unloaded))
        self.unload_confirmed = (
            not onset
            and len(self._unloaded) == self._unloaded.maxlen
            and all(self._unloaded)
        )
        if self.unload_confirmed:
            self.active = False
        return self.active
