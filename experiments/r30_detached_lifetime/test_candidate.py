import unittest
from experiments.r30_detached_lifetime.candidate import AcquisitionEndpoint


class EpisodeBoundary(unittest.TestCase):
    def setUp(self):
        self.g=AcquisitionEndpoint()

    def feed(self, vector, quiet=False, *, detached=True, sole=True):
        return self.g.update(True,detached,vector,4.,quiet,sole)

    def retire(self):
        self.assertTrue(self.feed((8.,0.)))
        for value in (True,True,False):
            self.assertIs(self.feed((0.,1.),True),value)

    def test_old_witness_cannot_republish_on_one_high_sample(self):
        self.retire()
        self.assertFalse(self.feed((4.,1.)))
        self.assertFalse(self.feed((8.,0.)))

    def test_new_acquisition_even_one_tick_later_is_delivered(self):
        self.retire()
        self.assertFalse(self.g.update(False,False,None,4.,False,False))
        self.assertTrue(self.feed((8.,0.)))

    def test_nonexclusive_or_uncertain_keeps_output(self):
        self.retire()
        self.assertTrue(self.feed((8.,0.),sole=False))
        self.assertTrue(self.feed(None))
        self.assertTrue(self.feed((float('nan'),0.)))
