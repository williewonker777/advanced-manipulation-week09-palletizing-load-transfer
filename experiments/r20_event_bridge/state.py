"""Source-aware continuity between an auxiliary event and the incumbent alarm."""


class HandoffState:
    """Bridge only evidence that overlaps an already active auxiliary event.

    The state owns no clock or refractory interval.  Its clearance is the
    incumbent detector's existing consecutive-clear counter and threshold.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.active = False

    def update(self, event_active, incumbent_evidence, incumbent_detected,
               clear_samples, release_samples, armed=True):
        if not armed:
            self.reset()
            return False
        if bool(event_active) and bool(incumbent_evidence):
            self.active = True
        if self.active and (bool(incumbent_detected)
                            or int(clear_samples) >= int(release_samples)):
            self.active = False
        return self.active
