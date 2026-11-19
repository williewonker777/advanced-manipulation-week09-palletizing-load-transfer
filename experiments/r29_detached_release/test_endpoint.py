import unittest
from experiments.r29_detached_release.candidate import Endpoint


def send(e, **kw):
    values = dict(report=True, detached=True, vector=(0., 0.), bound=4., mean_quiet=True, sole=True)
    return e.update(**{**values, **kw})


class EndpointTest(unittest.TestCase):
    def test_consumes_only_continuation_on_agreement(self):
        e = Endpoint()
        self.assertEqual([send(e) for _ in range(3)], [True, True, False])
        self.assertTrue(send(e, sole=False))
        self.assertTrue(send(e, mean_quiet=False))
        self.assertTrue(send(e, vector=(-6., 0.)))
        self.assertEqual([send(e) for _ in range(3)], [True, True, False])

    def test_invalid_fail_open_then_rearm(self):
        for bad in (None, (float('nan'), 0.), (float('inf'), 0.)):
            e = Endpoint()
            for _ in range(5): send(e)
            self.assertTrue(send(e, vector=bad))
            self.assertTrue(send(e))
            self.assertTrue(send(e))
            self.assertFalse(send(e))
            self.assertFalse(send(e, report=False, detached=False))
            self.assertTrue(send(e))

    def test_onset_and_absence_are_preserved(self):
        e = Endpoint()
        for _ in range(5):
            self.assertFalse(send(e, report=False))
        self.assertTrue(send(e))
        self.assertFalse(send(e, report=False, sole=False))
        self.assertTrue(send(e, vector=(0., -3.21)))
