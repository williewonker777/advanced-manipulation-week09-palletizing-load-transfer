import hashlib
import json
from pathlib import Path
import sys
import unittest
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.dont_write_bytecode=True
import v2_detector_guarded as G
from tools import v2_domain_study as S
from tools.production_audit import audit_text


class DomainTests(unittest.TestCase):
    def test_bounds_are_training_only(self):
        p=json.loads((ROOT/'results/v2_protocol/PROTOCOL.json').read_text())
        train=p['train'];d=G.INFERENCE_DOMAIN
        self.assertEqual(d['mass_min'],min(c['mass'] for c in train))
        self.assertEqual(d['mass_max'],max(c['mass'] for c in train))
        self.assertEqual(d['com_min'],[min(c['com'][i] for c in train) for i in range(3)])
        self.assertEqual(d['com_max'],[max(c['com'][i] for c in train) for i in range(3)])
        for c in train:self.assertTrue(G.within_training_domain(c['mass'],c['com']))

    def test_generic_fallback_not_a_zero_only_rule(self):
        c=(.01,.00,.02)
        for m in (.0,.15,.35,7.8,8.):self.assertFalse(G.within_training_domain(m,c))
        self.assertTrue(G.within_training_domain(3.7,c))
        self.assertFalse(G.within_training_domain(3.7,(.05,0.,.02)))
        self.assertFalse(G.within_training_domain(np.nan,c))
        self.assertFalse(G.within_training_domain(3.7,(np.nan,0.,.02)))

    def test_runtime_input_audit(self):
        self.assertEqual(audit_text((ROOT/'v2_detector_guarded.py').read_text())['errors'],[])

    def test_all_directions_cross_both_masses(self):
        d=json.loads((ROOT/'results/v2_protocol/PROTOCOL_OOD.json').read_text())
        rows=S.cases(d,'development','L2','servo')
        self.assertEqual(len(rows),66)
        for c in d['development']['conditions']:
            group=[r for r in rows if r['kind']=='collision' and r['mass']==c['mass']]
            self.assertEqual(len(group),32)
            self.assertEqual({r['direction'] for r in group},set(d['development']['directions']))


if __name__=='__main__':unittest.main()
