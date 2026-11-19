"""Contract tests for the public R17 information-prefix payload audit."""
import contextlib,io,json,os
from pathlib import Path
import subprocess,sys,tempfile,unittest
from unittest.mock import patch
from tools import r17_payload_audit as audit
from experiments.r17_information_prefix import portable

class Tests(unittest.TestCase):
 @classmethod
 def setUpClass(cls):
  cls.plan=json.loads((audit.ROOT/audit.SOURCE_PATHS['plan']).read_text())
  cls.cal=json.loads((audit.ROOT/audit.SOURCE_PATHS['calibration']).read_text())
  cls.ref=json.loads((audit.ROOT/audit.SOURCE_PATHS['reference']).read_text())
 def test_frozen_provenance_plan_and_reference(self):
  portable.validate(self.cal,self.ref,self.plan)
  self.assertEqual(self.cal['portable_provenance']['source_normal_count'],24)
  self.assertFalse(self.cal['portable_provenance']['truth_used_in_fit'])
  self.assertIn('cannot independently regenerate',self.cal['portable_provenance']['stock_regeneration'])
  self.assertEqual(self.ref['metrics']['vector_rmse_mm'],0.12634724358985133)
  self.assertEqual(self.ref['metrics']['accepted'],9); self.assertEqual(self.ref['metrics']['big_signed'],9)
 def test_offset_distinguishes_none_and_explicit_zero(self):
  case=self.plan['holdout'][0]; x,y,z=case['shift']
  self.assertIsNone(portable.offset(case,'none')); self.assertEqual(portable.offset(case,'explicit_zero'),(0.,0.,0.))
  self.assertEqual(portable.offset(case,'xy'),(x,y,0.)); self.assertEqual(portable.offset(case,'up'),(x,y,z)); self.assertEqual(portable.offset(case,'down'),(x,y,-z))
 def test_metrics_distinguish_point_and_action_error(self):
  rows=json.loads(json.dumps(self.ref['rows'])); self.assertEqual(portable.metrics(rows),self.ref['metrics'])
  rows[2]['action_error_mm']=[9.,9.]; self.assertFalse(portable.metrics(rows)['qualified'])
 def test_source_has_no_private_fixture_raw_output_or_absolute_path(self):
  for path in (Path(audit.__file__),Path(portable.__file__)):
   source=path.read_text(); self.assertNotIn('r5_fixture',source); self.assertNotIn('savez',source)
   self.assertNotIn(str(Path.home().resolve())+os.sep,source); self.assertNotIn(str(audit.ROOT.resolve())+os.sep,source)
  self.assertIn('predictions_sealed_before_offline_score',Path(portable.__file__).read_text())
 def test_snapshot_is_immutable_after_source_change(self):
  with tempfile.TemporaryDirectory() as folder:
   root=Path(folder)
   for key,relative in audit.SOURCE_PATHS.items():
    p=root/relative;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes((key+'-old').encode())
   fake=root/'tools/r17_payload_audit.py';fake.parent.mkdir(parents=True,exist_ok=True);fake.write_bytes(b'audit-old')
   with patch.object(audit,'ROOT',root),patch.object(audit,'__file__',str(fake)): blobs,pins=audit.inputs()
   (root/audit.SOURCE_PATHS['robust']).write_bytes(b'new')
   self.assertEqual(blobs['robust'],b'robust-old');self.assertEqual(pins['robust'],audit.sha(b'robust-old'))
 def test_outer_validation_does_not_import_the_external_simulator(self):
  # A clean distribution intentionally does not ship sim/. Run the outer
  # validation helper in a fresh isolated interpreter before any kit is loaded.
  inputs=[str(audit.ROOT/audit.SOURCE_PATHS[k]) for k in ('calibration','reference','plan')]
  with tempfile.TemporaryDirectory() as folder:
   script=("import json, runpy, sys; "
           "fn = runpy.run_path(sys.argv[1])['validate']; "
           "fn(*(json.loads(open(name, encoding='utf-8').read()) for name in sys.argv[2:])); "
           "assert 'sim' not in sys.modules")
   result=subprocess.run([sys.executable,'-I','-B','-c',script,portable.__file__,*inputs],cwd=folder,
                         text=True,capture_output=True)
   self.assertEqual(result.returncode,0,result.stderr)
 def test_help_works_from_unrelated_cwd_and_existing_output_fails(self):
  with tempfile.TemporaryDirectory() as folder:
   result=subprocess.run([sys.executable,'-B',str(Path(audit.__file__).resolve()),'--help'],cwd=folder,env={**os.environ,'PYTHONDONTWRITEBYTECODE':'1'},text=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
   self.assertEqual(result.returncode,0,result.stderr);self.assertIn('--kit',result.stdout)
   existing=Path(folder)/'existing';existing.mkdir();argv=['r17_payload_audit.py','--kit',folder,'--out',str(existing)]
   with patch.object(sys,'argv',argv),contextlib.redirect_stderr(io.StringIO()),self.assertRaises(SystemExit): audit.main()

if __name__=='__main__':unittest.main()
