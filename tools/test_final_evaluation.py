import contextlib
import io
import unittest
from unittest.mock import patch
from tools import final_evaluation


class FinalProvenanceTests(unittest.TestCase):
    def test_collect_only_is_rejected_before_reading_logs_or_hashing_source(self):
        with patch('sys.argv',['final_evaluation.py']), \
                patch.object(final_evaluation,'collect') as collect, \
                patch.object(final_evaluation,'hashes') as hashes, \
                contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit) as raised:
                final_evaluation.main()
        self.assertEqual(raised.exception.code,2)
        collect.assert_not_called()
        hashes.assert_not_called()


if __name__=='__main__':
    unittest.main()
