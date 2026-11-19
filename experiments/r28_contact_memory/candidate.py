"""Disclosed control: physical eligibility gates reports, not evidence that force was low."""
from experiments.r28_finite_contact.candidate import Detector as AirContact
from experiments.r28_finite_contact.pulse_diagnostic import ContiguousOwnedEpisode


class Detector(AirContact):
    def reset(self):
        super().reset()
        frozen = self.grip_episode
        # Identical frozen thresholds and report length; replace state lifetimes only.
        episode = ContiguousOwnedEpisode(frozen.bound, 1.0, frozen.persistence, frozen.report_samples)
        episode.release = frozen.release
        self.grip_episode = episode
