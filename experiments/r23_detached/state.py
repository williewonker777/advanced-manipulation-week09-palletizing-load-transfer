"""Admit a newly rising planar disturbance, not a residual already high at startup."""
import math


class OnsetGate:
    def __init__(self):
        self.quiet = 0
        self.armed = self.accepted = False

    def feed(self, xy, above, incumbent, bound):
        if xy is None:
            self.__init__()
            return False
        if self.accepted:
            if above:
                return True
            self.accepted = self.armed = False
            self.quiet = 0
        if incumbent:
            self.quiet = 0
            self.armed = False
        elif math.hypot(*xy) < 0.8 * bound and not above:
            self.quiet += 1
            if self.quiet >= 3:
                self.armed = True
        else:
            self.quiet = 0
        if above and self.armed:
            self.accepted = True
            self.armed = False
        return self.accepted
