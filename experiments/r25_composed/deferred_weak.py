"""Three-component development trial using the separately frozen deferred admission."""
from experiments.r25_composed.detector import Detector as Joint
from experiments.r25_weak.normal_force_detector_v2 import Detector as FrozenWeak
from experiments.r25_weak.world_horizontal_v2 import DeferredWorldHorizontalMixin


class Detector(DeferredWorldHorizontalMixin, Joint):
    """Keep both existing boundary repairs and never suppress their alarms."""
    world_horizontal_threshold = FrozenWeak.world_horizontal_threshold
    world_horizontal_support_uncertainty = FrozenWeak.world_horizontal_support_uncertainty
