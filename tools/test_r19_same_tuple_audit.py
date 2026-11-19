from pathlib import Path
import tempfile
import unittest
from unittest import mock

from tools import r19_same_tuple_audit as audit


class SameTupleInputs(unittest.TestCase):
    def test_cannot_replace_full_reference_with_partial_success(self):
        with tempfile.TemporaryDirectory() as folder:
            base=Path(folder);c=base/'kit';c.mkdir();(c/'sim').mkdir();(c/'model').mkdir();(c/'evaluate.py').touch()
            h=base/'cal.json';h.write_text('{}');o=base/'out.json';s=base/'ref.json';s.write_text('{"passed":true}')
            with mock.patch.object(audit.imp.setup,'snapshots') as build,self.assertRaises(SystemExit):
                audit.main(['--kit',str(c),'--height-calibration',str(h),'--impulse-audit',str(s),'--out',str(o)])
            build.assert_not_called();self.assertFalse(o.exists())

    def test_comparator_has_no_machine_specific_cpu_and_requires_fixed_grid(self):
        source=(audit.ROOT/audit.PROGRAM).read_text()
        self.assertNotIn('sched_setaffinity(',source)
        self.assertIn('scoring.finite_grid(',source)
        self.assertIn('r18_diverged',source)
