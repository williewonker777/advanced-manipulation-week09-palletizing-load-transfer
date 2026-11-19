"""Five-sample advisory view of unchanged released-source evidence."""
from __future__ import annotations

from experiments.r31_held_identity.candidate import Detector as Parent
from experiments.r34_released_source.candidate import ReleasedSourceHook


NOTICE_SAMPLES = 5


class EdgeNotice:
    def __init__(self):
        self.previous = False
        self.remaining = 0

    def feed(self, evidence):
        evidence = bool(evidence)
        if evidence and not self.previous:
            self.remaining = NOTICE_SAMPLES
        self.previous = evidence
        notice = self.remaining > 0
        if notice:
            self.remaining -= 1
        return notice


class Detector(Parent):
    """One unchanged parent update OR a bounded released-evidence notification."""

    def reset(self):
        super().reset()
        self.released_source = ReleasedSourceHook(self)
        self.released_notice_gate = EdgeNotice()
        self.released_evidence = False
        self.released_notice = False

    def update(self, t, th, thd, tau_m, tau_s, grip):
        residual, parent = super().update(t, th, thd, tau_m, tau_s, grip)
        self.released_evidence = bool(
            self.released_source.update(self, t, th, thd, tau_s, grip))
        self.released_notice = bool(
            self.released_notice_gate.feed(self.released_evidence))
        return residual, bool(parent or self.released_notice)
