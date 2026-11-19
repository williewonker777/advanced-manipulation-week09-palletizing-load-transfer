import unittest

from experiments.r17_staged_realtime.contract import loaded_class
from experiments.r17_staged_realtime.controller import Controller
from experiments.r17_staged_realtime.staged import Stream


class StagedTest(unittest.TestCase):
    def test_contract(self):
        self.assertIs(loaded_class, Controller)

    def test_incomplete_height_fails_closed(self):
        stream = Stream.__new__(Stream)
        stream._height_result = None; stream._height_stage = 0
        with self.assertRaisesRegex(ValueError, "incomplete"):
            stream.finish_height_decision({})

    def test_stale_height_fails_closed(self):
        stream = Stream.__new__(Stream)
        stream._height_result = {"accept": False}; stream._height_stage = 15
        stream._y = [1]; stream._jac = [2]; stream._stiffness = [3]
        with self.assertRaisesRegex(ValueError, "stale"):
            stream.finish_height_decision({"y": [9], "jac": [2], "stiffness": [3]})


if __name__ == "__main__":
    unittest.main()
