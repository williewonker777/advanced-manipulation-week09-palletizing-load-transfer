import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import tools.r12_coupled_demo as demo
from tools.r12_coupled_demo import SOURCES,build_sources,verify_sources,write_sources

class CoupledDemoTest(unittest.TestCase):
 def test_exactly_three_self_contained_sources(self):
  sources,_=build_sources();self.assertEqual(set(sources),{'my_servo.py','my_torque.py','my_detector.py'});self.assertNotIn('experiments.r12_',sources['my_torque.py']);self.assertNotIn('experiments.r12_',sources['my_detector.py']);self.assertIn('MODE",None)!="torque":return residual,original',sources['my_detector.py'])
 def test_overwrite_rejected(self):
  sources,_=build_sources()
  with tempfile.TemporaryDirectory() as d:
   out=Path(d)/'export';write_sources(out,sources)
   with self.assertRaisesRegex(ValueError,'new'):write_sources(out,sources)
 def test_tamper_rejected(self):
  sources,_=build_sources()
  with tempfile.TemporaryDirectory() as d:
   out=Path(d)/'export';pins=write_sources(out,sources);(out/'my_torque.py').write_text('tampered')
   with self.assertRaisesRegex(ValueError,'tamper'):verify_sources(out,pins)
 def test_broken_symlink_output_rejected(self):
  sources,_=build_sources()
  with tempfile.TemporaryDirectory() as d:
   out=Path(d)/'export';out.symlink_to(Path(d)/'missing',target_is_directory=True)
   with self.assertRaisesRegex(ValueError,'symlink'):write_sources(out,sources)
 def test_export_only_direct_script_from_unrelated_cwd(self):
  script=Path(__file__).with_name('r12_coupled_demo.py').resolve()
  with tempfile.TemporaryDirectory() as d:
   out=Path(d)/'export';env=os.environ.copy();env.pop('PYTHONPATH',None);env['PYTHONDONTWRITEBYTECODE']='1'
   done=subprocess.run([sys.executable,str(script),'--out-dir',str(out)],cwd='/',env=env,text=True,capture_output=True)
   self.assertEqual(done.returncode,0,done.stderr);self.assertEqual(sorted(p.name for p in out.iterdir()),['my_detector.py','my_servo.py','my_torque.py'])
 def test_public_coefficient_tamper_rejected(self):
  with tempfile.TemporaryDirectory() as d:
   bad=Path(d)/'coefficients.json';bad.write_bytes(SOURCES['coefficients'].read_bytes()+b' ')
   with patch.dict(SOURCES,{'coefficients':bad}):
    with self.assertRaisesRegex(ValueError,'pinned'):build_sources()
 def test_export_does_not_read_private_npz(self):
  with tempfile.TemporaryDirectory() as d,patch.object(demo,'RESEARCH_MODEL',Path(d)/'absent.npz'):
   result=demo.export(Path(d)/'export')
   self.assertEqual(result['status'],'EXPORTED')
if __name__=='__main__':unittest.main()
