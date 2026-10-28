"""D3 ablation: velocity estimator plus online JTS bias."""
from experiments.d3_detector import Detector as _Detector


class Detector(_Detector):
    USE_WRENCH = False
    USE_CALIBRATED_THRESHOLDS = False
