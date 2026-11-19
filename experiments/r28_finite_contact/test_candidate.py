from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

import numpy as np

from experiments.r26_delivery_pair.candidate import Detector as C1
from experiments.r28_finite_contact.candidate import Detector, admissible_normal, support_owner
from experiments.r28_landing_contact.candidate import OwnedHorizontalEpisode


class FiniteContactTests(unittest.TestCase):
    def test_geometrically_absent_is_not_unknown_admissible(self):
        self.assertEqual(admissible_normal(NS(last_table_allowed=False)), 0)
        for v in (None, [], [0, 0, np.nan], [0, 0, -np.inf], [0, 0, np.inf], ['0', '0', '0']):
            with self.subTest(v=v):
                self.assertIsNone(admissible_normal(NS(last_table_allowed=True, last_table_wrench=v)))
        for v in (None, 0, np.nan):
            self.assertIsNone(admissible_normal(NS(last_table_allowed=v)))
        self.assertEqual(admissible_normal(NS(last_table_allowed=np.bool_(True),last_table_wrench=[0,0,-2])),0)
        self.assertEqual(admissible_normal(NS(last_table_allowed=True,last_table_wrench=[0,0,27.3])),27.3)

    def test_finite_valid_support(self):
        self.assertTrue(support_owner(0.0,0.5,None)[0])
        self.assertFalse(support_owner(np.nan,0.5,None)[0])
        for validity, fraction in ((None,1.),(np.nan,1.),(True,np.nan),(False,1.),(True,.49)):
            self.assertFalse(support_owner(8.,.5,NS(lambda_support_valid=validity,lambda_support=fraction))[0])
        self.assertTrue(support_owner(8.,.5,NS(lambda_support_valid=np.bool_(True),lambda_support=.5))[0])

    def test_missing_sample_cancels_only_optional_report(self):
        item=object.__new__(Detector)
        item.box_mass=5.;item.normal_resolution=.5
        item._shared=NS(last_table_allowed=True,last_table_wrench=[0,0,0.])
        item.ctrl=NS(shared_estimator=NS(lambda_support_valid=True,lambda_support=.9))
        item.grip_episode=OwnedHorizontalEpisode(10.,.5,2,8)
        z=np.zeros(3);h=np.array([12.,0,0])
        def tick(value,incumbent=False):
            item.grip_force=NS(update=lambda *a: NS(force_n=value))
            with patch.object(C1,'update',return_value=(z,incumbent)):
                residual,flag=item.update(0,None,None,None,None,True)
                self.assertIs(residual,z)
                return flag
        self.assertFalse(tick(z));self.assertFalse(tick(h));self.assertTrue(tick(h))
        self.assertTrue(tick([np.nan,0,0],True));self.assertFalse(item.grip_force_extra)
        self.assertEqual(item.grip_episode.pulse,0)
        self.assertFalse(tick(h))
        self.assertFalse(tick(z));self.assertFalse(tick(h));self.assertTrue(tick(h))
        item._shared.last_table_wrench=[0,0,np.nan]
        self.assertFalse(tick(h));self.assertFalse(item.load_ownership)
