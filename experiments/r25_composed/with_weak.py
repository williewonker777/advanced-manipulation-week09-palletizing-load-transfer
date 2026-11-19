"""Pre-independent trial: measured weak-force onset OR the unchanged two-repair detector."""
from experiments.r25_composed.detector import Detector as Joint
from experiments.r25_weak.normal_force_detector import Detector as FrozenWeak
from experiments.r25_weak.world_horizontal import WorldHorizontalMixin


class Detector(WorldHorizontalMixin, Joint):
    """Reuse exactly the normal-only calibration from the standalone observation."""
    world_horizontal_threshold = FrozenWeak.world_horizontal_threshold
    world_horizontal_support_uncertainty = FrozenWeak.world_horizontal_support_uncertainty
