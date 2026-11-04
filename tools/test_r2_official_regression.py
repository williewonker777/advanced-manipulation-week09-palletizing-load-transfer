import copy
import json
from pathlib import Path
import unittest
from tools.r2_official_regression import compare


class OfficialRegressionTests(unittest.TestCase):
    def setUp(self):
        self.old = json.loads((Path(__file__).resolve().parents[1]/'results/v2_protocol/S4_OFFICIAL.json').read_text())
        self.rows = {}
        for mode in ('servo', 'torque'):
            old = self.old['combinations'][mode+'_v2_v2']
            control = {}
            for mass, b in old['control'].items():
                control[mass] = [str(b[k]) for k in ('rms_mm','max_mm','saturation_pct','pick_nm','place_nm','table_n')]
                control[mass] += [str(v) for v in b['false_alarms']]+[b['status']]
            collisions = [dict(mass=r['mass'],time=r['time'],
                               cells=[r[k] for k in ('force10','force15','force40','force100','false_max','minimum')])
                          for r in old['collisions']]
            self.rows[mode] = dict(control_and_collisions=dict(control=control, collisions=collisions))

    def test_exact_physics_and_decisions_ignore_header_and_timing(self):
        self.assertEqual(compare(self.rows, self.old), [])

    def test_physical_or_detection_regression_rejected(self):
        changed = copy.deepcopy(self.rows)
        changed['servo']['control_and_collisions']['control']['8.0'][0] = '999.00'
        self.assertEqual(len(compare(changed,self.old)), 1)
        changed = copy.deepcopy(self.rows)
        changed['torque']['control_and_collisions']['collisions'][0]['cells'][0] = 'changed'
        self.assertEqual(len(compare(changed,self.old)), 1)


if __name__ == '__main__':
    unittest.main()
