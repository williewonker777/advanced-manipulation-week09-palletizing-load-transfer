import hashlib,json,unittest
from pathlib import Path
import numpy as np
from experiments.r21_servo_yaw import alias,candidate
class TestServoYaw(unittest.TestCase):
 def test_frozen_model(self):
  p=Path(candidate.__file__).with_name('normal_bins.json');d=json.loads(p.read_bytes())
  self.assertEqual(hashlib.sha256(p.read_bytes()).hexdigest(),'32c8144f0768f199f98981e8b5ea3dfb3cdc3481a034aaaa940e55ab4c7d3e4e')
  self.assertEqual(d['schema'],'r21-servo-yaw-normal-bins.v1');self.assertEqual(d['normal_episodes'],16);self.assertEqual(np.asarray(d['threshold_Nm']).shape,(2,60))
 def test_alias(self):self.assertIs(alias.Detector,candidate.Detector)
 def test_bounds_are_positive_finite_and_oof_reserved(self):
  d=json.loads(Path(candidate.__file__).with_name('normal_bins.json').read_bytes());a=np.asarray(d['threshold_Nm']);r=np.asarray(d['held_com_reserve_Nm']);self.assertTrue(np.all(np.isfinite(a)) and np.all(a>0));self.assertTrue(np.all(r>0))
if __name__=='__main__':unittest.main()
