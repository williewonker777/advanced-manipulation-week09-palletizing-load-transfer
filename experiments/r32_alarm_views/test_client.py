import unittest
from types import SimpleNamespace
from unittest.mock import patch
from .client import NoticeClient, SustainedAlarm


class ExplicitConsumerTest(unittest.TestCase):
    def test_native_only_response_does_not_invent_a_notice(self):
        det=NoticeClient.__new__(NoticeClient)
        def update(obj, *args):
            obj.decision=SimpleNamespace(event_output=False, linked_native=True, alarm_level=True)
            return 7., True
        with patch.object(SustainedAlarm,'update',autospec=True,side_effect=update) as run:
            self.assertEqual(det.update(1,2,3,4,5,False),(7.,False))
            run.assert_called_once_with(det,1,2,3,4,5,False)
        self.assertTrue(det.decision.alarm_level)
