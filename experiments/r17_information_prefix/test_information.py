from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np

from experiments.r17_information_prefix.estimate import past,select
from experiments.r17_information_prefix.fit_normal import fit_files
from experiments.r17_information_prefix.stream import feature
from sim import trajectory as T


class InformationTests(unittest.TestCase):
    def test_two_adjacent_passing_normal_folds_required(self):
        def idx(bits):
            return select([{'passes':b} for b in bits])
        self.assertIsNone(idx([True, False, True]))
        self.assertIsNone(idx([True]))
        self.assertEqual(idx([False,True,True,True]),1)

    def test_slice_fails_on_off_grid_or_missing_data(self):
        with self.assertRaises(ValueError):
            past({},2441)
        z = {k:np.zeros((3,6)) for k in ('y','jac','stiffness')}
        with self.assertRaises(ValueError):
            past(z,2440)

    def test_fit_refuses_reused_records(self):
        with self.assertRaisesRegex(ValueError,'distinct'):
            fit_files([Path('same.npz')]*24,{'centres_ms':[2920]})

    def test_observation_independent_of_suffix_and_oracle(self):
        n=2944
        t=np.arange(n)*.001
        q=np.array([T.reference(float(v))[0] for v in t])
        data=dict(t=t,encoder=q,jts=np.zeros((n,6)),grip=t>=T.T_GRASP)
        with TemporaryDirectory() as name:
            folder=Path(name)
            def save(path,signals):
                np.savez(path,mass=np.array(5.4),com=[.01,-.02,.01],**signals)
            save(folder/'prefix.npz',data)
            extended={k:np.concatenate((x,np.full((7,)+x.shape[1:],np.nan))) for k,x in data.items()}
            extended['grip']=np.r_[data['grip'],[False]*7]
            extended['oracle_force']=np.array([1e50])
            save(folder/'longer.npz',extended)
            first=feature(folder/'prefix.npz',2920);second=feature(folder/'longer.npz',2920)
            self.assertEqual(first.keys(),second.keys())
            for key,value in first.items():
                self.assertTrue(np.array_equal(value,second[key]),key)
            save(folder/'missing.npz',{k:x[:-1] for k,x in data.items()})
            with self.assertRaisesRegex(ValueError,'before frozen decision'):
                feature(folder/'missing.npz',2920)
