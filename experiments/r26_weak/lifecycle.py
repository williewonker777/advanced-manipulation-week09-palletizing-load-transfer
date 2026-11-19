"""Finite signed-pulse lifecycle with incumbent ownership."""
import numpy as np


class PulseLifecycle:
    """Release one shot after an unowned, measured pulse has physically ended."""

    def __init__(self, *, onset=8.394415779390961, onset_samples=2,
                 release_fraction=.45, release_score_fraction=.65,
                 release_samples=4, maximum_episode=200, hold=8, rearm=8):
        self.onset = float(onset)
        self.onset_samples = int(onset_samples)
        self.release_fraction = float(release_fraction)
        self.release_score_fraction = float(release_score_fraction)
        self.release_samples = int(release_samples)
        self.maximum_episode = int(maximum_episode)
        self.hold_samples = int(hold)
        self.rearm_samples = int(rearm)
        self.reset()

    def reset(self):
        self.phase = "quiet"
        self.rise = self.fall = self.age = self.hold = self.quiet = 0
        self.owned = False
        self.direction = None
        self.peak = 0.0

    def feed(self, vector, score, eligible, incumbent):
        vector = np.asarray(vector, float)
        finite = vector.shape == (2,) and np.isfinite(vector).all() and np.isfinite(score)
        eligible = bool(eligible and finite)
        score = float(score) if finite else 0.0

        if self.phase == "quiet":
            over = eligible and score > self.onset
            self.rise = self.rise + 1 if over else 0
            self.quiet = 0 if over else self.quiet + 1
            if self.rise >= self.onset_samples:
                magnitude = float(np.linalg.norm(vector))
                self.direction = vector / magnitude if magnitude else np.zeros(2)
                self.peak = max(score, 1e-12)
                self.owned = bool(incumbent)
                self.phase = "active"
                self.age = self.fall = 0
            return False

        if self.phase == "active":
            self.age += 1
            self.owned = self.owned or bool(incumbent)
            self.peak = max(self.peak, score)
            projection = float(np.dot(vector, self.direction)) if finite else 0.0
            ended = (not eligible or projection <= self.release_fraction * self.peak or
                     score <= self.release_score_fraction * self.peak)
            self.fall = self.fall + 1 if ended else 0
            if self.fall >= self.release_samples:
                if not self.owned:
                    self.hold = self.hold_samples
                self.phase = "rearm"
                self.quiet = 0
            elif self.age >= self.maximum_episode:
                # No confirmed physical end: discard rather than time-release.
                self.phase = "rearm"
                self.quiet = 0

        if self.phase == "rearm":
            low = not eligible or score <= self.onset
            self.quiet = self.quiet + 1 if low else 0
            if self.quiet >= self.rearm_samples and self.hold == 0:
                self.phase = "quiet"
                self.rise = self.fall = self.age = 0
                self.owned = False
                self.direction = None
                self.peak = 0.0

        active = self.hold > 0
        if self.hold:
            self.hold -= 1
        return active
