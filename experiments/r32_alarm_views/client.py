"""Use the historical new-event Boolean while keeping the complete R32 joint state."""
from experiments.r32_joint_state.candidate import Detector as SustainedAlarm


class NoticeClient(SustainedAlarm):
    """One physical observer update. The companion state is never passed off as a new rise."""
    def update(self, *args):
        residual, _ = super().update(*args)
        return residual, self.decision.event_output
