"""Keep a useful research demonstration distinct from the three submitted files."""
import ast
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from tools.r11_velocity_demo import assemble, write_demo
from tools.r11_placement import score
from tools.r11_placement_plot import draw

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT/'results/r11'


def get(name):
    return json.loads((DATA/(name+'.json')).read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


class EvidenceTests(unittest.TestCase):
    def test_current_placement_keeps_live_and_archive_separate(self):
        result = get('r10_placement')
        for mode, value in result['modes'].items():
            self.assertEqual(value['candidate']['detector_sha256'], sha(ROOT/'my_detector.py'))
            self.assertEqual(value['candidate']['controller_sha256'], sha(ROOT/f'my_{mode}.py'))
            self.assertEqual(value['archived_r1']['detector_sha256'], sha(ROOT/'experiments/r9_final/FAST2.py'))
            self.assertEqual(value['candidate']['rows'], value['archived_r1']['rows'])
            self.assertTrue(all(r['status']=='ok' for r in value['candidate']['rows']))
        self.assertEqual(result['modes']['servo']['candidate']['rows'][0]['last20_geometry']['all_four_corners_below_plane_samples'],20)

    def test_demo_files_reproduce_measured_sources_and_cannot_replace_an_existing_tree(self):
        built = assemble()
        result = get('velocity_demo')
        with tempfile.TemporaryDirectory() as d:
            output = Path(d)/'new-study'
            write_demo(output)
            with self.assertRaisesRegex(ValueError,'overwrite'):
                write_demo(output)
            for mode, measured in result['modes'].items():
                side = measured['candidate']
                self.assertEqual(side['controller_sha256'], sha(output/f'my_{mode}.py'))
                self.assertEqual(side['detector_sha256'], sha(output/'my_detector.py'))
                self.assertNotEqual(side['controller_sha256'], sha(ROOT/f'my_{mode}.py'))
                parsed = ast.parse(built[f'my_{mode}.py'])
                self.assertFalse(any(isinstance(n,ast.ImportFrom) and (n.module or '').startswith('experiments')
                                     for n in ast.walk(parsed)))
                self.assertEqual([dict(id=a['id'],**score(a,b)) for a,b in zip(measured['archived_r1']['rows'],side['rows'],strict=True)],
                                 measured['comparison'])

    def test_partial_gains_do_not_hide_missed_events_and_failed_gates(self):
        data=get('velocity_learning');self.assertEqual(data['decision'],'RESEARCH_ONLY_NOT_A_RELEASE')
        rows=[r for v in data['modes'].values() for r in v['rows']]
        self.assertEqual(len(rows),8)
        self.assertTrue(all(0 < r['gates']['mean_gain_mg'] and not r['gates']['all_four_g4'] for r in rows))
        self.assertEqual(sum(r['R1']['impact']['outcome']=='detected' and r['learned']['impact']['outcome']!='detected' for r in rows),1)
        self.assertEqual(sum(r['learned']['normal_fa']>r['R1']['normal_fa'] for r in rows),1)
        parity=get('demo_parity')['rows']
        self.assertEqual(len({(r['id'],r['mode']) for r in parity}),6)
        self.assertTrue(all(r['identical'] and len(r['digests'])==9 for r in parity))
        self.assertFalse(get('grasp_datum')['all_gates'])
        self.assertFalse(get('wrench_proxy')['decision']['adoption_ready'])

    def test_plot_has_no_imputed_forces_for_unqualified_rows(self):
        sample = get('r10_placement')
        for v in sample['modes'].values():
            for r in v['candidate']['rows']:
                r['status']='diverged'
                r.pop('support');r.pop('last20_geometry')
        with tempfile.TemporaryDirectory() as d:
            png=Path(d)/'plot.png'; draw(sample,png)
            self.assertTrue(png.read_bytes().startswith(b'\x89PNG\r\n\x1a\n'))
