"""One physical simulation update: preserve live sensor routes while fixing two different event boundaries."""
from experiments.r25_fragment.stream import ContinuationMixin
from experiments.r25_pickup.candidate import Detector as Pickup


class Detector(ContinuationMixin, Pickup):
    """Apply measured load continuation after the causal pickup ownership decision."""
