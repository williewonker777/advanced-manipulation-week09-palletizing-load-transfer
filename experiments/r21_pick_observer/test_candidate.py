import hashlib,json,unittest
from pathlib import Path
import numpy as np
from experiments.r21_pick_observer import candidate,detector

class CandidateTest(unittest.TestCase):
 def test_feature_contract_and_invalid_payload(self):
  self.assertTrue(np.array_equal(candidate.features(2,(1,2,3)),[1,2,4,1,2,3,2,4,6]))
  for value in ((1,(1,2)),(float('nan'),(0,0,0))):
   with self.assertRaises(ValueError):candidate.features(*value)
 def test_frozen_normal_model_shape_and_oof_bound(self):
  p=Path(candidate.MODEL);doc=json.loads(p.read_bytes())
  self.assertEqual(hashlib.sha256(p.read_bytes()).hexdigest(),'499b2577f659dff86bcc665fffc385adddadfe21824ef0e06c7d848caa49c6bf')
  self.assertEqual(doc['schema'],'r21-pick-normal-model.v1');self.assertTrue(doc['normal_only'])
  self.assertEqual(np.asarray(doc['coefficients']).shape,(480,9,2));self.assertEqual(len(doc['scale']),9)
  self.assertEqual(doc['grouping'],'leave-one-COM-group-out')
  self.assertGreater(doc['threshold_N'],doc['oof_max_abs_smoothed_N'])
 def test_stable_alias_exports_only_frozen_candidate(self):
  self.assertIs(detector.Detector,candidate.Detector);self.assertEqual(detector.__all__,['Detector'])
 def test_domain_is_finite_interpolation_not_extrapolation(self):
  doc=json.loads(Path(candidate.MODEL).read_bytes());m=np.asarray(doc['masses']);c=np.asarray(doc['coms'])
  self.assertEqual((m.min(),m.max()),(0.,8.));self.assertTrue(np.array_equal(c.min(0),[-.025,-.02,-.01]));self.assertTrue(np.array_equal(c.max(0),[.025,.02,.03]))
if __name__=='__main__':unittest.main()
