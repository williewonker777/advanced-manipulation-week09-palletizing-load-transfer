from pathlib import Path
import tempfile
import unittest
from tools import r32_full_census as census


class FullCensusTest(unittest.TestCase):
    def test_requires_the_actual_complete_reference(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'result.json';path.write_text('{}')
            with self.assertRaisesRegex(ValueError,'complete authenticated'):
                census.verified_reference(path)
            sym=Path(tmp)/'linked';sym.symlink_to(path)
            with self.assertRaisesRegex(ValueError,'regular'):
                census.verified_reference(sym)

    def test_raw_rows_publish_as_one_regular_file_in_numeric_order(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            rows=root/'rows';rows.mkdir()
            for i in (1,0):
                (rows/f'{i:03d}.json').write_text('{"id":%d,"source":"unit"}\n' % i)
            report=root/'report';report.mkdir()
            census.collect_rows(rows,report/'episodes.jsonl','unit',2)
            census.shared.publish(report,root/'out')
            self.assertEqual((root/'out/episodes.jsonl').read_text().splitlines(),
                             ['{"id":0,"source":"unit"}','{"id":1,"source":"unit"}'])

    def test_only_open_methods_and_authored_overlay(self):
        manifest,blobs,pin=census.sources()
        self.assertEqual(len(pin),64)
        self.assertEqual(len(census.shared.manifest_entries(manifest)),20)
        for method,(name,digest) in census.METHODS.items():
            _,specific,_=census.sources(method)
            self.assertEqual(census.shared.digest(specific[f'experiments/{name}/candidate.py']),digest)
        self.assertFalse(any(n.startswith(('sim/','model/')) for n in blobs))
