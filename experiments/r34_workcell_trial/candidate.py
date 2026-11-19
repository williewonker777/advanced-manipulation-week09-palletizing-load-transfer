"""Public-development composition; the unchanged R31 native result always wins."""
import numpy as np

from experiments.r33_vertical_composition.candidate import Detector as VerticalParent
from experiments.r34_precontact_inverse.candidate import MODEL
from experiments.r34_precontact_inverse.inverse import FlangeInverse
from experiments.r34_precontact_inverse.witness import PrecontactInverseWitness
from experiments.r34_released_source.advisory import EdgeNotice
from experiments.r34_released_source.candidate import ReleasedSourceHook
from experiments.r34_transfer_shear.candidate import Route as TransferShear


class FirstLiftOwner:
    """Associate simultaneous axial and planar projections; do not hide the native alarm."""
    def __init__(self):
        self.owner = None
        self.quiet = 0

    def feed(self, shear, grip, lateral, axial):
        if not grip:
            self.owner = None
            self.quiet = 0
            return lateral, axial
        s = shear.latest
        window = shear.event.win.v
        if s is not None and s.mode == 1 and len(window) >= 8:
            block = np.asarray(list(window)[-8:]).mean(0)
            # Full flange and generalized momentum must BOTH return to their
            # own geometry-conditioned healthy fields, including vertical load.
            limit = min(shear.event.limits[0])
            both_quiet = max(np.linalg.norm(block[:3]), np.linalg.norm(block[6:9])) < limit
            self.quiet = self.quiet + 1 if both_quiet else 0
            if self.quiet >= 8:
                self.owner = None
        if axial and self.owner != 'shear':
            self.owner = 'axial'
            return False, True
        if lateral and s is not None and s.mode == 1:
            if self.owner == 'axial':
                return False, False
            self.owner = 'shear'
        return lateral, False


class Detector(VerticalParent):
    def reset(self):
        super().reset()
        self.additions = getattr(self.ctrl, 'MODE', None) == 'torque'
        if self.additions:
            self.shear = TransferShear()
            self.first_owner = FirstLiftOwner()
            self.pre_inverse = FlangeInverse(self.nom, self.nom_box)
            self.pre_source = PrecontactInverseWitness(MODEL['threshold_N'], MODEL['persistence'])
            self.pre_edge = EdgeNotice()
            self.released = ReleasedSourceHook(self)
            self.post_edge = EdgeNotice()
        self.shear_event = self.pre_event = self.post_event = False
        self.pre_evidence = self.post_evidence = False

    def update(self, t, th, thd, tm, ts, grip):
        residual, r33 = super().update(t, th, thd, tm, ts, grip)
        if not self.additions:
            return residual, r33
        native = self.parent_event
        shear = self.shear.feed(self, t, th, thd, tm, ts, grip, native)
        self.shear_event, axial = self.first_owner.feed(self.shear, grip, shear, self.pick_event)
        before = bool(native or axial or self.carry_event or self.shear_event)
        sample = self.pre_inverse.update(t, th, ts, grip)
        self.pre_evidence = self.pre_source.extra(sample, bool(grip), before)
        self.pre_event = self.pre_edge.feed(self.pre_evidence)
        self.post_evidence = self.released.update(self, t, th, thd, ts, grip)
        self.post_event = self.post_edge.feed(self.post_evidence)
        return residual, bool(before or self.pre_event or self.post_event)
