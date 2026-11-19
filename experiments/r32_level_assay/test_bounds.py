import unittest
from .bounds import lower_in_window


class WindowFeasibility(unittest.TestCase):
    def test_266_neither_a_long_first_nor_a_sparse_total_can_fake_both(self):
        self.assertEqual(lower_in_window(89,2734,3020),89)
        self.assertGreater(lower_in_window(89,2734,3020),4+5)

    def test_window_edge_is_half_open(self):
        self.assertEqual(lower_in_window(89,3019,3020),1)
        self.assertEqual(lower_in_window(89,3020,3020),0)
        self.assertEqual(lower_in_window(0,2734,3020),0)

    def test_any_earlier_legal_start_can_only_increase_in_window_occupancy(self):
        for start in range(1000,1099):
            actual=len(set(range(start,start+28)) & set(range(1000,1100)))
            self.assertGreaterEqual(actual,lower_in_window(28,1098,1100))
