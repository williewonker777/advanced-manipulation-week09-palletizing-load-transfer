import hashlib
import json
import unittest
from tools.build_public import portable_json_bytes, rebind_public_report_manifest


class PortableStudyTests(unittest.TestCase):
    def test_identifiers_removed_without_dropping_cases_or_outliers(self):
        value={'metadata':{'stage':'d4'},'results':[{'case_id':0,'score':3.2,
            'host_before':{'foreign_heavy_python_pids':[121,122],'load_average':[2.1,2.2,2.3]},
            'timing':{'outliers':[{'combined':{'wall_ns':1200000,'cpu_ns':1100000}}]}}]}
        raw=json.dumps(value).encode();result=json.loads(portable_json_bytes(raw))
        self.assertEqual(len(result['results']),1)
        row=result['results'][0]
        self.assertEqual(row['score'],3.2)
        self.assertEqual(row['timing'],value['results'][0]['timing'])
        self.assertEqual(row['host_before']['foreign_heavy_python_count'],2)
        self.assertNotIn('foreign_heavy_python_pids',row['host_before'])
        self.assertEqual(result['private_raw_sha256'],hashlib.sha256(raw).hexdigest())

    def test_freeze_and_model_bytes_are_not_reformatted(self):
        raw=b'{"source_files":{"my_detector.py":"abc"},"value": 3.25}\n'
        self.assertEqual(portable_json_bytes(raw),raw)

    def test_large_json_compacts_without_losing_poles_or_outliers(self):
        value = {'cases': [{'poles': [[0.99, -0.001], [1.001173, 0.0]],
                            'outliers': [{'wall_ns': 6016548}]}]}
        raw = json.dumps(value, indent=2).encode() + b' ' * (4*1024**2)
        published = portable_json_bytes(raw)
        result = json.loads(published)
        self.assertEqual(result['cases'], value['cases'])
        self.assertLess(len(published), len(raw))
        self.assertEqual(result['private_raw_sha256'], hashlib.sha256(raw).hexdigest())
        self.assertNotIn('Raw process IDs omitted', result['publication_note'])

    def test_small_array_document_preserves_original_bytes(self):
        raw = b'[ {"case": 1}, {"case": 2} ]\n'
        self.assertEqual(portable_json_bytes(raw), raw)

    def test_report_manifest_rebinds_public_bytes_and_keeps_work_hash(self):
        raw=json.dumps({'results':[{'score':3.2,'host_before':{'foreign_heavy_python_pids':[121]},
                                  'timing':{'outliers':[{'wall_ns':1200000}]}}]}).encode()
        published=portable_json_bytes(raw)
        old=hashlib.sha256(raw).hexdigest()
        manifest=json.dumps({'artifacts':{'study':{'path':'results/study.json',
                                                  'sha256':old,'provenance':'fresh'}}}).encode()
        public=json.loads(rebind_public_report_manifest(manifest,{'results/study.json':published}))
        entry=public['artifacts']['study']
        self.assertEqual(entry['sha256'],hashlib.sha256(published).hexdigest())
        self.assertEqual(entry['prepublication_work_sha256'],old)
        self.assertEqual(json.loads(published)['results'][0]['timing'],json.loads(raw)['results'][0]['timing'])

    def test_report_manifest_refuses_missing_artifact(self):
        manifest=b'{"artifacts":{"study":{"path":"missing.json","sha256":"old"}}}'
        with self.assertRaises(ValueError):
            rebind_public_report_manifest(manifest,{})


if __name__=='__main__':unittest.main()
