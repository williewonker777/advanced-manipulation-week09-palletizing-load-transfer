"""D3 ablation: velocity, bias, and wrench decomposition without persistence."""
from experiments.d3_detector import Detector as _Detector


class Detector(_Detector):
    USE_CALIBRATED_THRESHOLDS = False
