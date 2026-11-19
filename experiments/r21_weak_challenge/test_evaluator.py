import unittest
import numpy as np
from experiments.r21_weak_challenge import evaluator as e

class EvaluateTest(unittest.TestCase):
 def test_unmatched_is_symmetric_tolerance_not_order_or_count_cheat(self):
  self.assertEqual(e.unmatched([1.,2.,2.002],[1.0005,2.0002]),[2.002])
 def test_expected_force_rejects_signed_or_scalar_change(self):
  t=np.arange(6000)*.001;impact=(1.5,(6.,-8.,0.));signed=e.runner.expected_signed(t,(impact,None))
  log={'t':t,'f_ext_xyz_native':signed.copy(),'f_ext_xyz_second':np.zeros_like(signed),
       'f_ext_xyz_committed':signed.copy(),
       'f_ext':np.fromiter((np.linalg.norm(x) for x in signed),float,count=len(signed))}
  self.assertTrue(all(e.expected_force(log,impact).values()))
  log['f_ext_xyz_native'][1550,0]=np.nextafter(log['f_ext_xyz_native'][1550,0],np.inf)
  self.assertFalse(e.expected_force(log,impact)['native'])
 def test_prefix_check_does_not_compare_the_future_force(self):
  a={'x':np.array([0.,0.,0.])};b={'x':np.array([0.,1.,1.])}
  self.assertEqual(e.prefix_map(a,b,('x',),1),{'x':True})
  self.assertEqual(e.prefix_map(a,b,('x',),2),{'x':False})
 def test_candidate_registry_never_accepts_arbitrary_module_name(self):
  with self.assertRaisesRegex(ValueError,'unknown arm'):e.resolve(['r10','arbitrary'],None)
if __name__=='__main__':unittest.main()
