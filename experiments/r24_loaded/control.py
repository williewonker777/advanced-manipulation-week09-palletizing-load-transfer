"""Signed support-yaw attribution in the original detector; all quantities are causal sensor estimates."""
import math



from experiments.r22_guarded.candidate import GuardedReleaseBalance
from experiments.r23_detached.free import Detector as Released


def subtract_aligned(value, prediction):
    """Never add a yaw alarm or reverse its sign by subtracting a modeled reaction."""
    if not math.isfinite(prediction) or value * prediction <= 0.:
        return value
    return math.copysign(max(0., abs(value) - abs(prediction)), value)


class LoadedYaw(GuardedReleaseBalance):
    def __init__(self, mass, packet, sliding):
        super().__init__(mass)
        self.packet, self.sliding = packet, sliding
        self.was_away = False
        self.before_projection = None
        self.reaction = 0.

    def feed(self, th, tau, mass_row, grip):
        value = super().feed(th, tau, mass_row, grip)
        s = self.packet
        self.before_projection = value
        self.reaction = 0.
        if not grip:
            self.was_away = False
            return value
        if not s.last_table_allowed:
            self.was_away = True
            return value
        if not (self.was_away and self.loaded and value is not None and s.last_table_wrench[2] > 0):
            return value
        jac, _, _ = s._point_jacobian(s.q_prev)
        support = s.last_table_wrench
        if self.sliding:
            # From the measured link motion, friction at the box bottom dissipates
            # tangential power; torsional resistance opposes spin around world Z.
            speed = jac @ s.v
            planar_speed = math.hypot(*speed[:2])
            if planar_speed <= 1e-12:
                return value
            fz = float(support[2])
            fx, fy = -s.friction_mu * fz * speed[:2] / planar_speed
            tz = -math.copysign(s.torsion_radius * fz, speed[5]) if speed[5] else 0.
            sliding_yaw = jac[0,0]*fx + jac[1,0]*fy + jac[5,0]*tz
            measured_yaw = float(jac[:,0] @ s.last_wrench)
            if sliding_yaw * measured_yaw > 0 and support[:2] @ speed[:2] < 0:
                self.reaction = math.copysign(min(abs(measured_yaw), abs(sliding_yaw)), sliding_yaw)
        else:
            self.reaction = float(jac[:,0] @ support)
        return subtract_aligned(value, self.reaction)


class Projected(Released):
    sliding = False

    def reset(self):
        super().reset()
        if getattr(self.ctrl, 'MODE', None)=='torque' and self.box_mass > 0:
            self.yaw = LoadedYaw(self.box_mass, self._shared, self.sliding)


class Sliding(Projected):
    sliding = True
