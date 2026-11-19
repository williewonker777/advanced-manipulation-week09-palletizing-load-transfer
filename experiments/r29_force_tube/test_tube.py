import unittest
import numpy as np
from experiments.r29_force_tube.tube import Notice,Windows,batch_features


class ForceTubeTests(unittest.TestCase):
    def test_online_stencil_matches_offline_over_missing_and_grip_switch(self):
        random=np.random.default_rng(9);data=random.normal(size=(80,16));data[31]=np.nan
        phase=np.zeros(80,int);phase[57:]=1;w=np.array([2.,0.,0.,.1])
        ticks,x,y=batch_features(data,phase,w);out={int(k):(a,b)for k,a,b in zip(ticks,x,y)}
        online=Windows()
        for k,d in enumerate(data):
            actual=online.feed(phase[k],d,w)
            if k not in out:self.assertIsNone(actual)
            else:
                np.testing.assert_allclose(actual[0],out[k][0],atol=1e-12)
                np.testing.assert_allclose(actual[1],out[k][1],atol=1e-12)

    def test_reports_once_then_rearms_on_measured_return(self):
        d=Notice();a=-np.ones((2,8));a[:,2]=2
        active=[d.feed(a,False)for _ in range(20)]
        self.assertEqual(np.flatnonzero(active).tolist(),[3,4,5,6])
        self.assertFalse(d.feed(-np.ones((2,8)),False))
        b=-np.ones((2,8));b[:,5]=1
        self.assertEqual([d.feed(b,False)for _ in range(4)],[False,False,False,True])

    def test_two_channels_must_agree_and_native_always_passes(self):
        a=-np.ones((2,8));a[0,0]=2;a[1,3]=2;d=Notice()
        self.assertFalse(any(d.feed(a,False)for _ in range(12)))
        self.assertTrue(d.feed(None,True))
        a[:,4]=3
        self.assertEqual([d.feed(a,k==2)for k in range(10)],[False,False,True]+[False]*7)
