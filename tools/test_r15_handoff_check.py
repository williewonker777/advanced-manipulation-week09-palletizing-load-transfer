import os
from pathlib import Path
import sys
import tempfile
import unittest
from experiments.r15_com_model.controller import START, END, weight
from experiments.r15_com_model.handoff import ratios
from tools.r15_handoff_check import (
    DEPENDENCIES, decode_height_calibration, run_checked, summarize,
    validate_paths, verify_unchanged,
)

class Tests(unittest.TestCase):
    def test_exact_negative_counts_are_successful_reproduction(self):
        rows=[{'id':str(i),'passed':i<3,'gates':{'g4':i<3},'r14_g4':{'all_four_g4':i<2}} for i in range(4)]
        result=summarize(rows)
        self.assertEqual((result['r15_g4_cases'],result['r14_g4_cases']),(3,2))
        self.assertFalse(result['qualified']);self.assertEqual(result['failed'],['3'])
    def test_missing_ids_fail_closed(self):
        with self.assertRaises(ValueError):summarize([{'id':'','passed':False}]*4)
    def test_public_dependencies_and_causal_weight(self):
        self.assertEqual(len(DEPENDENCIES),len(set(DEPENDENCIES)))
        self.assertFalse(any('fixture' in value or 'record.py' in value for value in DEPENDENCIES))
        self.assertEqual((weight(START),weight(END)),(0.,1.))
        self.assertEqual(ratios({'cost':{'x':0.}}, {'cost':{'x':0.}}),{'x':0.})
    def test_malformed_calibration_fails_closed(self):
        with self.assertRaises((ValueError,KeyError)):decode_height_calibration(b'{}')
    def test_snapshot_mutation_detected(self):
        with tempfile.TemporaryDirectory() as folder:
            p=Path(folder)/'x';p.write_bytes(b'a')
            p.write_bytes(b'b')
            with self.assertRaises(ValueError):verify_unchanged(p,'ca978112ca1bbdcafac231b39a23dc4da786eff8147c4e72b9807785afee48bb')
    def test_symlink_and_existing_output_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);kit=root/'kit';kit.mkdir();(kit/'sim').mkdir();(kit/'model').mkdir();(kit/'evaluate.py').touch();cal=root/'cal';cal.write_text('{}');out=root/'out';out.write_text('x')
            with self.assertRaises(ValueError):validate_paths(kit,cal,out)
            out.unlink();link=root/'link';link.symlink_to(cal)
            with self.assertRaises(ValueError):validate_paths(kit,link,out)
    def test_negative_subprocess_return_fails(self):
        with self.assertRaises(RuntimeError):run_checked([sys.executable,'-c','raise SystemExit(7)'],cwd=Path.cwd(),env=os.environ.copy(),label='negative')

if __name__=='__main__':unittest.main()
