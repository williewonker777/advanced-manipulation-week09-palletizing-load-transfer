"""Measured grip-relative ascent and precontact return geometry."""
from sim import config as C


class Approach:
    minimum_excursion = 0.5 * float(C.BOX_SIZE[2])

    def __init__(self):
        self.reset()

    def reset(self):
        self.anchor = self.peak = self.previous = None

    def feed(self, bottom_z, grip):
        if not grip or bottom_z is None:
            self.reset()
            return False
        value = float(bottom_z)
        if self.anchor is None:
            self.anchor = self.peak = self.previous = value
            return False
        self.peak = max(self.peak, value)
        excursion = self.peak - self.anchor
        descent = self.peak - value
        descending = value < self.previous
        self.previous = value
        return bool(excursion >= self.minimum_excursion and descending
                    and 2.0 * descent >= excursion and value > self.anchor)
