"""Causal destination, native acquisition priority and a signed cross-grip shear ticket."""
from collections import deque

import numpy as np

from experiments.r31_held_identity.candidate import Detector as Parent
from experiments.r34_surface_shear.candidate import Event, parameters
from experiments.r34_surface_shear.field import BLOCK, coordinates, row_vector
from experiments.r34_surface_shear.signal import Signal


def acquiring_yaw(history, bound):
    """Extrapolate only outward motion inside the existing acquisition band."""
    if len(history) != 2 * BLOCK + 1 or history[0] is None or history[-1] is None:
        return False
    y = history[-1]
    dy = y - history[0]
    return abs(y) < bound and y * dy > 0 and abs(y + dy * 18 / (2 * BLOCK)) >= bound


class Destination:
    def __init__(self):
        self.yaw = deque(maxlen=2 * BLOCK + 1)
        self.start = None
        self.fastest = 0.
        self.arrived = False

    def feed(self, q, xyz, grip):
        if not grip:
            if self.start is not None:
                self.__init__()
            return False
        if self.start is None:
            self.start = xyz[:2].copy()
        self.yaw.append(float(q[0]))
        if len(self.yaw) == self.yaw.maxlen:
            travel = abs(self.yaw[-1] - self.yaw[0])
            self.fastest = max(travel, self.fastest)
            if (np.linalg.norm(xyz[:2] - self.start) >= 2 * .16 and self.fastest > 0
                    and travel <= .25 * self.fastest):
                self.arrived = True
        return self.arrived


class Route:
    def __init__(self):
        self.sensor = Signal()
        self.event = Event(parameters())
        self.yaw = deque(maxlen=2 * BLOCK + 1)
        self.destination = Destination()
        self.latest = None
        self.last_grip = False
        self.held_ticket = self.pending = self.confirmed = self.count = self.width = 0
        self.block_packet = False

    def feed(self, d, t, th, thd, tm, ts, grip, parent):
        pkt = d._shared
        s = self.sensor.feed(d, t, th, ts, grip)
        q = pkt.q_prev
        arrived = (q is not None and self.destination.feed(q, d.nom.tcp_position(q), grip))
        self.yaw.append(d.yaw_value)
        if self.last_grip and not grip:
            self.pending = self.held_ticket
            self.confirmed = self.count = self.width = 0
        if grip:
            self.pending = self.confirmed = self.count = self.width = 0
            self.held_ticket = 0
        self.last_grip = bool(grip)
        if s is not None and (
                (s.mode == 1 and self.sensor.stage.max_z - self.sensor.stage.start[2] > .060)
                or (s.mode == 2 and not arrived)):
            s = None
        self.latest = s
        event = self.event
        was_reported = event.reported
        extra = event.feed(s, bool(parent), float(pkt.force_threshold[0]), pkt.persistence)
        if s is None:
            self.block_packet = False
            return False
        center = event.bank.center(s.mode, coordinates(s.mode, s.progress_m, s.z_m))
        if center is None:
            return False
        v = row_vector(s) - center
        if len(event.win.v) >= BLOCK:
            last = np.asarray(list(event.win.v)[-BLOCK:]).mean(0)
            tangent = 1 if last[1] > 0 else -1
            tangential = abs(last[1]) > abs(last[0])
            if s.mode == 2 and tangential and not event.native and not event.reported:
                limit = event.limits[1, 2 if tangent > 0 else 6]
                if min(tangent * last[1], tangent * last[7]) > limit:
                    self.held_ticket = tangent
            if extra and not was_reported:
                self.block_packet = (
                    (s.mode == 1 and acquiring_yaw(self.yaw, float(d.bounds[int(bool(grip))])))
                    or (s.mode == 2 and not tangential))
        elif extra:
            raise AssertionError("incomplete normal-force field window")
        # A supported observation may be confirmed after release, never on a
        # predicted/bracketed missing sample. Failure of fresh signed agreement
        # extinguishes the ticket; energy cannot be carried across unrelated events.
        if s.mode == 3 and self.pending:
            sign = self.pending
            threshold = event.limits[2, 2 if sign > 0 else 6]
            agreement = sign * v[[1, 4, 7]]
            if parent or np.min(agreement) <= 0:
                self.pending = self.count = 0
            elif (np.min(agreement) > threshold and abs(v[9]) / .16 < np.min(agreement)
                  and np.linalg.norm(v[:3]) < pkt.force_threshold[0] / 2):
                self.count += 1
                if self.count >= pkt.persistence:
                    self.confirmed = sign
                    self.width = pkt.persistence
                    self.pending = 0
            else:
                self.count = 0
        if self.confirmed and s.mode == 3:
            if self.width:
                self.width -= 1
                return not parent
            # Defer the residual tail of the same agreed direction. A later
            # genuine source is allowed after independent measured quiet.
            if len(event.win.v) == 2 * BLOCK:
                signs = self.confirmed * np.asarray(event.win.v)[:, [1, 4, 7]]
                if np.all(np.max(signs[-BLOCK:], axis=1) < .45):
                    self.confirmed = 0
                else:
                    return False
        return bool(extra and not self.block_packet)


class Detector(Parent):
    def reset(self):
        super().reset()
        self.shear = Route() if getattr(self.ctrl, 'MODE', None) == 'torque' else None
        self.shear_event = self.parent_event = False

    def update(self, t, th, thd, tm, ts, grip):
        residual, base = super().update(t, th, thd, tm, ts, grip)
        self.parent_event = bool(base)
        self.shear_event = bool(self.shear and self.shear.feed(self, t, th, thd, tm, ts, grip, base))
        return residual, bool(base or self.shear_event)
