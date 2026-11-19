import os,unittest
from pathlib import Path
import numpy as np
from experiments.r15_early_com.stream import Stream,DEADLINE_INDEX,PREFIX_CENTRES
from tools.r13_grasp_estimator import public_features

DATA=Path(os.environ.get('R15_EARLY_COM_TEST_DATA','missing.npz'))
class Test(unittest.TestCase):
 def setUp(self):
  if not DATA.exists():self.skipTest('set R15_EARLY_COM_TEST_DATA')
 def test_prefix_deadline_and_full_parity(self):
  with np.load(DATA) as z:
   s=Stream(float(z['mass']),z['com'])
   for k in range(DEADLINE_INDEX):s.observe(float(z['t'][k]),z['encoder'][k],z['jts'][k],bool(z['grip'][k]))
   with self.assertRaises(ValueError):s.finish_prefix()
   s.observe(float(z['t'][DEADLINE_INDEX]),z['encoder'][DEADLINE_INDEX],z['jts'][DEADLINE_INDEX],bool(z['grip'][DEADLINE_INDEX]));p=s.finish_prefix();self.assertEqual(p['completed_at_s'],2.463);self.assertEqual(p['y'].shape,(len(PREFIX_CENTRES),6))
   for k in range(DEADLINE_INDEX+1,3550):s.observe(float(z['t'][k]),z['encoder'][k],z['jts'][k],bool(z['grip'][k]))
   full=s.finish();expected=public_features(z['t'],z['encoder'],z['jts'],z['grip'],float(z['mass']),z['com'])
  for key in expected:np.testing.assert_allclose(full[key],expected[key],rtol=0,atol=0)
if __name__=='__main__':unittest.main()
