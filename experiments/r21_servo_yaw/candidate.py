"""Servo R10 plus the frozen, servo-normal-calibrated base-yaw witness."""
import json
from pathlib import Path
import numpy as np
from experiments.r21_servo_yaw.detector import Detector as Base
from experiments.r21_yaw_impulse.coarse_calibration import BINS,bin_index
class Detector(Base):
 def reset(self):
  super().reset();doc=json.loads(Path(__file__).with_name('normal_bins.json').read_bytes());table=np.asarray(doc['threshold_Nm'],float)
  if doc.get('schema')!='r21-servo-yaw-normal-bins.v1' or table.shape!=(2,BINS) or not np.all(np.isfinite(table)) or np.any(table<=0):raise ValueError('frozen servo normal bounds required')
  self._bins=table
 def update(self,t,*args):
  k=bin_index(t);self.bounds=self._bins[:,k];return super().update(t,*args)
