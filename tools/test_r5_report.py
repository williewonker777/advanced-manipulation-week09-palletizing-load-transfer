"""Synthetic tests for the R5 public report renderer."""
import json,tempfile,unittest
from unittest import mock
from pathlib import Path
from tools import r5_report as R

class ReportTests(unittest.TestCase):
 def test_curation_hashes_only_existing_inputs_and_marks_pending(self):
  with tempfile.TemporaryDirectory() as n:
   root=Path(n);p=root/R.REQUIRED[0];p.parent.mkdir(parents=True);p.write_text('{}')
   doc=R.build_curation(root)
   self.assertEqual(R.sha(p),doc['sources'][R.REQUIRED[0]]['sha256'])
   self.assertIn(R.REQUIRED[1],doc['pending_required'])
 def test_markdown_table_keeps_all_rows(self):
  text=R.md_table(['a','b'],[{'a':'1','b':'2'},{'a':'3','b':'4'}])
  self.assertEqual(4,len(text.splitlines()))
 def test_privacy_rejects_machine_path(self):
  with self.assertRaises(ValueError):R.safe('/ho'+'me/example','x')
 def test_public_markdown_link_scan_covers_docs_and_report(self):
  with tempfile.TemporaryDirectory() as n:
   root=Path(n);(root/'docs').mkdir();(root/'report').mkdir()
   (root/'README.md').write_text('[doc](docs/a.md)');(root/'docs/a.md').write_text('[notes](../report/b.md)');(root/'report/b.md').write_text('ok')
   paths=R.public_markdown_paths(root);self.assertEqual(3,len(paths));R.check_links(root,paths)
   (root/'report/b.md').write_text('[missing](no.md)')
   with self.assertRaises(ValueError):R.check_links(root,paths)

 def test_default_link_scan_also_covers_results_and_tool_documents(self):
  with tempfile.TemporaryDirectory() as n:
   root=Path(n);(root/'results').mkdir();(root/'tools').mkdir()
   (root/'README.md').write_text('Public summary')
   (root/'results/study.md').write_text('[input](missing.json)')
   (root/'tools/reproduce.md').write_text('Instructions')
   self.assertEqual(3,len(R.public_markdown_paths(root)))
   with self.assertRaises(ValueError):R.check_links(root)
   (root/'results/missing.json').write_text('{}')
   R.check_links(root)

 def test_existing_link_parser_handles_encoded_targets_and_code_examples(self):
  with tempfile.TemporaryDirectory() as n:
   root=Path(n);(root/'asset name.txt').write_text('data')
   (root/'README.md').write_text('[asset](asset%20name.txt)\n```md\n[example](not-a-file)\n```\n')
   R.check_links(root)
 def test_media_block_has_all_four_gifs_and_mp4_links(self):
  text=R.media_block()
  for prefix in ('05','06','07','08'):
   self.assertIn(prefix,text);self.assertIn('.gif',text);self.assertIn('.mp4',text)
 def test_official_rows_preserve_exact_strings(self):
  ref={'official_no_collision_0_4_8kg':{'R1':{m:{k:{'rms_mm_exact':'1.17','max_mm_exact':'3.91','saturation_pct_exact':'0.0','pick_nm_exact':'9.7','place_nm_exact':'14.6','false_alarms_exact':['0','0','0'],'status_exact':'정상'} for k in ('0.0','4.0','8.0')} for m in ('servo','torque')}}}
  control={k:{'rms_mm':1.17,'max_mm':3.91,'saturation_pct':0.0,'pick_nm':9.7,'place_nm':14.6,'table_n':79.0,'false_alarms':[0,0,0],'status':'정상'} for k in ('0.0','4.0','8.0')}
  source={'combinations':{m+'_v2_v2':{'control':control} for m in ('servo','torque')}}
  rows=R.official_rows({'docs':{'S4_OFFICIAL':source},'ref':ref})
  self.assertEqual('1.17 / 3.91',rows[0]['TCP RMS/최대 [mm]']);self.assertNotIn('1.170',rows[0]['TCP RMS/최대 [mm]'])
 def test_slides_cover_required_sections(self):
  state=[{'variant':'SCALE_STATE','mode':mode,'validation':'v','development':'d','D4':'r',
          'perturb':'p','status':'채택','val_phase':'v','dev_phase':'d','d4_phase':'r',
          'normal':'fit 1/2; calibration 1/2; audit 1/2 N; 추가 표본/시작 0/0','timing':'t',
          'params':263,'threshold':1.0,'mode_id':mode_id} for mode,mode_id in (('A 서보','servo'),('B 토크','torque'))]
  contract={'status':'FAIL','tests_run':526,'errors':8}
  fake={'effective':{'servo':'R1','torque':'R1'},'splits':'정상 fit 24행','rows':state, 'ref':{},'docs':{'R5_FINAL_CONTRACT_REVIEW':contract},'adoption':{}}
  tr=[{'항목':'x','기준':'A 1; B 1','v1':'A 1; B 1','R1':'A 1; B 1','R5':'A 1; B 1'}]*6
  simple=[{'방식':'A','소스':'x','충돌 검출':'1','정상 FA':'0','판정':'통과'}]
  with mock.patch.object(R,'t_rows',return_value=tr),mock.patch.object(R,'cause_rows',return_value=[]),\
       mock.patch.object(R,'timing_rows',return_value=[]),mock.patch.object(R,'holdout_rows',return_value=simple),\
       mock.patch.object(R,'official_rows',return_value=[]),mock.patch.object(R,'collision_rows',return_value=[]):
   with mock.patch.object(R,'candidate_official_comparison',return_value=[]):
    titles=[x['title'] for x in R.slides(fake)]
  self.assertIn('R5 결론',titles);self.assertIn('남은 미달',titles)

if __name__=='__main__':unittest.main(verbosity=2)
