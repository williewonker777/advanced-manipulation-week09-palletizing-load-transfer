import unittest

import numpy as np

from experiments.r27_load_change.manifold_event import Coordinates,FeasibleNormals
from experiments.r27_load_change.train_manifold import cover,envelopes


class NormalSetTests(unittest.TestCase):
    def setUp(self):
        self.model={'location':[0.]*12,'scale':[1.]*12,'nodes':[[0.]*12,[1.]*12,[-1.]*12],
                    'upper_bounds_N':np.asarray([np.full((2,8),1.),np.full((2,8),3.),np.full((2,8),2.)]).tolist(),
                    'radius_squared':[15.]*3}

    def test_both_sources_must_exclude_both_feasible_neighbors(self):
        f=FeasibleNormals(self.model)
        self.assertFalse(f.unexpected(np.zeros(12),[[4,0],[2,0]],0))
        self.assertTrue(f.unexpected(np.zeros(12),[[4,0],[4,0]],0))
        self.assertFalse(f.unexpected(np.ones(12)*20,[[100,0],[100,0]],0))
        self.assertFalse(f.unexpected(None,[[100,0],[100,0]],0))

    def test_coordinates_use_only_contiguous_encoder_positions(self):
        c=Coordinates()
        for k in range(40): self.assertIsNone(c.feed(np.ones(6)*.001*k,True))
        np.testing.assert_allclose(c.feed(np.ones(6)*.04,True),np.r_[[.02]*6,[1.]*6])
        self.assertIsNone(c.feed(np.zeros(6),False))
        self.assertIsNone(c.feed(np.zeros(6),True))

    def test_training_coverage_contains_its_own_normal_samples(self):
        rng=np.random.default_rng(923);x=rng.normal(size=(550,16))
        a,b,n=cover(x[:,:12],5);caps,rad,_=envelopes(x,a,b,n)
        f=FeasibleNormals(dict(location=a,scale=b,nodes=n,upper_bounds_N=caps,radius_squared=rad))
        for row in x[::9]:
            self.assertFalse(any(f.unexpected(row[:12],row[12:].reshape(2,2),axis)for axis in range(8)))
