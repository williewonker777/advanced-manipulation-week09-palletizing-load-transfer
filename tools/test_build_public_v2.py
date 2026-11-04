import hashlib
import json
import unittest
from tools.build_public import portable_json_bytes


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


if __name__=='__main__':unittest.main()
