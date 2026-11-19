"""Causal deferred admission for the frozen signed world-horizontal witness."""
from experiments.r25_weak.world_horizontal import WorldHorizontalMixin


class DeferredAdmission:
    """Let the incumbent consume approaching evidence before releasing an extra."""

    def __init__(self, *, ceiling=100, hold=8):
        self.ceiling = int(ceiling)
        self.hold_samples = int(hold)
        self.pending = False
        self.age = self.hold = 0

    def feed(self, candidate, incumbent, approaching):
        if incumbent:
            self.pending = False
            self.age = self.hold = 0
            return False
        if candidate and approaching and not self.hold:
            self.pending = True
        if self.pending:
            self.age += 1
            if not approaching or self.age >= self.ceiling:
                self.pending = False
                self.age = 0
                self.hold = self.hold_samples
        elif candidate and not approaching:
            return True
        active = self.hold > 0
        if self.hold:
            self.hold -= 1
        return active


class DeferredWorldHorizontalMixin(WorldHorizontalMixin):
    approaching_ratio = .8
    pending_ceiling_samples = 100

    def reset(self):
        super().reset()
        self.world_horizontal_admission = DeferredAdmission(
            ceiling=self.pending_ceiling_samples,
            hold=self.world_horizontal.hold_samples)

    def update(self, t, th, thd, tau_m, tau_s, grip):
        # Call below WorldHorizontalMixin once so its frozen observer state and
        # incumbent update remain unchanged; replace only optional OR admission.
        residual, incumbent = super(WorldHorizontalMixin, self).update(
            t, th, thd, tau_m, tau_s, grip)
        candidate = self.world_horizontal(t, th, tau_s, grip, self._shared)
        normal = getattr(self, "normal_model", None)
        ratio = (0.0 if normal is None or normal.threshold <= 0 else
                 float(normal.ema / normal.threshold))
        approaching = ratio >= self.approaching_ratio
        extra = self.world_horizontal_admission.feed(candidate, bool(incumbent), approaching)
        return residual, bool(incumbent or extra)
