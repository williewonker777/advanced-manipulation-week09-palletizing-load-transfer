"""Frozen standalone R25 development detector: R24 OR signed world-horizontal force."""
from experiments.r24_composed.detector import Detector as R24
from experiments.r25_weak.world_horizontal import WorldHorizontalMixin


class Detector(WorldHorizontalMixin, R24):
    world_horizontal_threshold = 8.394415779390961
    world_horizontal_support_uncertainty = .2
