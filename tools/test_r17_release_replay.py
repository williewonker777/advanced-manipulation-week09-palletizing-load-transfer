"""Contract tests for the opt-in R17 negative sentinel replay."""
import contextlib,hashlib,io,os
from pathlib import Path
import subprocess,sys,tempfile,unittest
from unittest.mock import patch
from tools import r17_release_replay as audit
from experiments.r17_release_coincidence import replay

class Tests(unittest.TestCase):
 def test_case_is_single_already_exposed_sentinel(self):
  self.assertEqual(replay.CASE,{'id':'new_surface_normal_release','mass':5.38,'com':[.021,-.016,.011],'offset':[0.,0.,0.],'collision':[3.94,[0.,0.,16.]],'role':'ALREADY_EXPOSED_R16_SENTINEL'})
 def test_model_is_explicit_nondefault_and_pinned(self):
  path=audit.ROOT/audit.MODEL;self.assertTrue(path.is_file());self.assertEqual(audit.digest(path.read_bytes()),'42ee9d2eebc85a424f67d2ab3e45431d4a42ea3cbd89b9e78532fa000aba4b81')
  self.assertNotIn('my_detector.py',audit.MODEL);self.assertNotIn('collision_detector.py',audit.MODEL)
 def test_dependencies_are_neutral_complete_and_snapshot_once(self):
  blobs,pins=audit.snapshots();self.assertEqual(len(audit.DEPENDENCIES),len(set(audit.DEPENDENCIES)))
  self.assertFalse(any(Path(x).name=='run.py' for x in audit.DEPENDENCIES));self.assertEqual(pins[audit.MODEL],audit.digest(blobs[audit.MODEL]))
 def test_public_sources_have_no_affinity_private_fixture_or_absolute_user_path(self):
  for path in (Path(audit.__file__),Path(replay.__file__)):
   source=path.read_text();self.assertNotIn('sched_setaffinity',source);self.assertNotIn('r5_fixture',source)
   self.assertNotIn(str(Path.home().resolve())+os.sep,source);self.assertNotIn(str(audit.ROOT.resolve())+os.sep,source)
  self.assertIn("'candidate_passed':False",Path(replay.__file__).read_text())
  self.assertIn("'promotion_eligible':False",Path(replay.__file__).read_text())
 def test_negative_disposition_fails_closed_on_integrity_or_reference_error(self):
  event=lambda outcome,delay,false:{'outcome':outcome,'delay_ms':delay,'false_alarms':len(false),
                                    'false_alarm_times':list(false),'prealarm':False}
  arms={'r10':event('detected',40.,[]),'raw_r15':event('missed',None,[]),
        'D_coupled':event('missed',None,[]),
        'R17C_prediction':event('detected',21.,replay.EXPECTED_FALSE_TIMES['R17C_prediction']),
        'R17D_coincidence':event('detected',29.,replay.EXPECTED_FALSE_TIMES['R17D_coincidence'])}
  signature=replay.exact_negative_signature(arms);self.assertTrue(all(signature.values()))
  integrity=dict.fromkeys(replay.INTEGRITY_FIELDS,True)
  self.assertEqual(replay.disposition({},signature),(False,'INVALID_REPLAY'))
  self.assertEqual(replay.disposition(integrity,{}),(False,'INVALID_REPLAY'))
  valid,status=replay.disposition(integrity,signature);self.assertTrue(valid);self.assertEqual(status,'NEGATIVE_CONTROL_REPRODUCED')
  valid,status=replay.disposition({**integrity,'identical_D_physics':False},signature);self.assertFalse(valid);self.assertEqual(status,'INVALID_REPLAY')
  arms['r10']['delay_ms']=39.;bad=replay.exact_negative_signature(arms);self.assertFalse(bad['reference_delays_exact'])
  self.assertEqual(replay.disposition(integrity,bad),(False,'INVALID_REPLAY'))
 def test_public_observer_matches_the_frozen_research_algorithm(self):
  path=audit.ROOT/'experiments/r17_release_coincidence/portable_detector.py'
  # Restoring only the time/return annotations must reproduce the exact frozen file.
  source=path.read_bytes()
  for typed,untyped in [(b't: float, th,', b't, th,'), (b'grip) -> tuple[np.ndarray, bool]:', b'grip):')]:
   self.assertEqual(source.count(typed),1);source=source.replace(typed,untyped)
  self.assertEqual(hashlib.sha256(source).hexdigest(),'f7d94c84943a6c4ac95337671cc4235c169461888ddfb638d6698fdd132f9d6c')
 def test_help_from_unrelated_cwd_and_existing_output_rejected(self):
  with tempfile.TemporaryDirectory() as folder:
   result=subprocess.run([sys.executable,'-B',str(Path(audit.__file__).resolve()),'--help'],cwd=folder,env={**os.environ,'PYTHONDONTWRITEBYTECODE':'1'},text=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
   self.assertEqual(result.returncode,0,result.stderr);self.assertIn('--height-calibration',result.stdout)
   existing=Path(folder)/'existing.json';existing.write_text('{}');argv=['r17_release_replay.py','--kit',folder,'--height-calibration',str(existing),'--out',str(existing)]
   with patch.object(sys,'argv',argv),contextlib.redirect_stderr(io.StringIO()),self.assertRaises(SystemExit):audit.main()

if __name__=='__main__':unittest.main()
