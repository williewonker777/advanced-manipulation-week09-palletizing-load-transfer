"""D3 ablation: velocity estimator only."""
from experiments.d3_detector import Detector as _Detector


class Detector(_Detector):
    USE_BIAS = False
    USE_WRENCH = False
    USE_CALIBRATED_THRESHOLDS = False
