"""Only pre-announce a force when measured yaw has room for the pending signed innovation."""
import math
import numpy as np
from experiments.r29_disjoint_events.candidate import Detector as Composed
from experiments.r29_loaded_rising_certificate.candidate import NATIVE_NOTICE_SAMPLES, rising_certificate, rising_means
from experiments.r30_source_association.candidate import MeasuredCoordinates, prefer_planar_notice


def has_yaw_reserve(yaw, bound, force_changes, jacobian_column):
    """Guard a lead notice, never the native yaw signal.

    Check the present measured moment and both continuations obtained by
    adding the measured planar force change projected on joint 1. Fractions
    of each change lie between those endpoints. This is a first-order screen,
    not a bound on an arbitrary future collision or an exact delay model.
    """
    if yaw is None or bound is None or not all(math.isfinite(x) for x in (yaw,bound)) or bound<=0:
        return False
    if not prefer_planar_notice(force_changes,jacobian_column):
        return False
    projected=np.asarray(force_changes,float)@np.asarray(jacobian_column,float)
    predicted=float(yaw)+projected
    return bool(abs(float(yaw))<bound and np.isfinite(predicted).all() and np.all(np.abs(predicted)<bound))


class Detector(Composed):
    def reset(self):
        super().reset()
        self.r27_coords=MeasuredCoordinates()
        self.pending_owner=None
        self.pending_width=0
        self.planar_notices=self.deferred_notices=0

    def update(self,t,th,thd,tau_m,tau_s,grip):
        residual,reported=super().update(t,th,thd,tau_m,tau_s,grip)
        gate=self.r27_gate
        if gate.state=='quiet':
            self.pending_owner=None
            self.pending_width=0
        native=bool(self.r29_active_routes or self.r29_native_consumed)
        if self.pending_owner=='planar' and gate.state=='settled' and self.r27_certificate and not native:
            # This optional plateau belongs to the earlier notice; native yaw is unchanged.
            reported=False
        if self.pending_owner is None and gate.state=='rising' and not native:
            levels=self._levels(gate)
            if rising_certificate(gate,self.r27_normals,self.r27_coords.last,levels,grip,native,self.r29_routes_complete):
                before,now=rising_means(gate)
                if (has_yaw_reserve(self.yaw_value,self.bounds[int(bool(grip))],now-before,self.r27_projection.jp[:2,0])
                        and self.r29_link.start(gate.axis,gate.reference,gate.peak,grip)):
                    self.pending_owner='planar'
                    self.pending_width=NATIVE_NOTICE_SAMPLES
                    self.planar_notices+=1
                else:
                    self.pending_owner='incumbent'
                    self.deferred_notices+=1
        optional=self.pending_width>0
        if optional:self.pending_width-=1
        return residual,bool(reported or optional)
