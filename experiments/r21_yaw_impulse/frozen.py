"""Normal-only torque bounds for the global yaw-balance ablation."""
from experiments.r21_yaw_impulse.detector import Detector as GlobalBalance


class Detector(GlobalBalance):
    bounds = (3.23779270273884, 5.111942835466052)
