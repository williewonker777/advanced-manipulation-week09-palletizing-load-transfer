"""Frozen v2: R24 OR causally deferred signed world-horizontal evidence."""
from experiments.r24_composed.detector import Detector as R24
from experiments.r25_weak.world_horizontal_v2 import DeferredWorldHorizontalMixin


class Detector(DeferredWorldHorizontalMixin, R24):
    world_horizontal_threshold = 8.394415779390961
    world_horizontal_support_uncertainty = .2
