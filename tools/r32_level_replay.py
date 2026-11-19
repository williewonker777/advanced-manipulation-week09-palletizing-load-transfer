"""Build or execute the kit-authenticated R32 level/event research replay."""
import argparse,json,os,sys,tempfile,zipfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from tools import r21_replay as shared
from tools import r31_event_replay as parent

DIR='experiments/r32_level_replay/';CLOSURE=DIR+'closure.json';PREFIX='week09-r32-level-events'
MEMBERS={DIR+x for x in ('__init__.py','channels.py','replay.py','test_channels.py')}|{'experiments/r32_measured_level/candidate.py','experiments/r32_native_handoff/candidate.py','experiments/r32_native_handoff/protocol.json','tools/r32_level_replay.py'}
README='''# R32 event/level research replay\n\nThis is not a safety detector or enrolled submission. It needs MuJoCo 3.17.0 and your original course kit, which is never copied into this ZIP.\n\n    python -B tools/r32_level_replay.py compare --kit /path/to/icpbl_sim --scope both --out /fresh/output\n\nThe replay separates the actual returned Boolean from R31 optional notice spans and measured native/continuation telemetry. Spans are half-open 1 ms sample intervals. Telemetry is not physical force truth. Frozen A is included for research comparison despite known historical guard failures and is not enrollable. C2 separately reports event_output/event_started, measured_load, linked_native and the actual returned alarm_level. NoticeCompatible is source-only compatibility behavior and must never be described as a duration or level score.\n'''
def read(name):return (ROOT/name).read_bytes()
def load():
 raw,files,overlay=parent.load();data=read(CLOSURE);reg=json.loads(data)
 if reg.get('schema')!='r32-level-replay-closure.v1' or reg.get('kit_included') is not False or reg.get('r31_closure_sha256')!=shared.digest(raw) or set(reg['files'])!=MEMBERS:raise ValueError('closure contract differs')
 files={**files,parent.CLOSURE:raw}
 for name,pin in reg['files'].items():
  blob=read(name)
  if shared.digest(blob)!=pin:raise ValueError('changed member '+name)
  files[name]=blob
 return data,files,sorted(set(overlay)|(MEMBERS-{'tools/r32_level_replay.py'}))
def bundle(out):
 if os.path.lexists(out) or not out.parent.is_dir() or shared.has_symlink_component(out.parent):raise ValueError('fresh nonsymlink ZIP path required')
 raw,files,_=load();files[CLOSURE]=raw;files['README.md']=README.encode();files['bundle.json']=(json.dumps({'kit_included':False,'recorded_outputs_included':False,'members':{n:shared.digest(b) for n,b in files.items()}},indent=2,sort_keys=True)+'\n').encode()
 with zipfile.ZipFile(out,'x') as z:
  for n,b in sorted(files.items()):
   i=zipfile.ZipInfo(PREFIX+'/'+n,(1980,1,1,0,0,0));i.create_system=3;i.external_attr=0o100644<<16;i.compress_type=zipfile.ZIP_DEFLATED;z.writestr(i,b,compresslevel=9)
 return {'archive_sha256':shared.digest(out.read_bytes()),'entries':len(files)}
def compare(a):
 import mujoco
 kit,out=shared.validate_paths(a.kit,a.out);raw,files,overlay=load();manifest=files[shared.KIT_MANIFEST];pins=shared.authenticate_kit(kit,manifest)
 if mujoco.__version__!='3.17.0':raise ValueError('MuJoCo 3.17.0 required')
 env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',PYTHONNOUSERSITE='1',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',MUJOCO_GL='disable');env.pop('PYTHONPATH',None);env.pop('PYTHONHOME',None)
 with tempfile.TemporaryDirectory(prefix='r32-level-',dir=out.parent) as td:
  work,report=Path(td)/'work',Path(td)/'report';work.mkdir();report.mkdir();src={n:files[n] for n in overlay}
  shared.install_overlay(kit,work,manifest,src,{n:shared.digest(b) for n,b in src.items()})
  shared.run([sys.executable,'-B','-m','experiments.r32_level_replay.replay',a.scope,'--out','result.json'],work,env,report/'run.log')
  data=(work/'result.json').read_bytes();obj=json.loads(data);expected={'examples':32,'place':20,'both':52}[a.scope]
  if obj['schema']!='r32-level-replay.v2' or obj['episodes']!=expected or obj['arms']!=['c3','identity','level_a','handoff_c2'] or obj['candidate_a_enrollable'] or shared.authenticate_kit(work,manifest)!=pins or shared.authenticate_kit(kit,manifest)!=pins:raise ValueError('wrong replay')
  (report/'result.json').write_bytes(data);receipt={'schema':'r32-level-replay-receipt.v1','scope':a.scope,'episodes':expected,'result_sha256':shared.digest(data),'kit_files_sha256':pins,'closure_sha256':shared.digest(raw)};(report/'receipt.json').write_text(json.dumps(receipt,indent=2,sort_keys=True)+'\n');shared.publish(report,out)
 return receipt
def main(argv=None):
 p=argparse.ArgumentParser();s=p.add_subparsers(dest='cmd',required=True);s.add_parser('describe');z=s.add_parser('build-zip');z.add_argument('--out',type=Path,required=True);c=s.add_parser('compare');c.add_argument('--kit',type=Path,required=True);c.add_argument('--scope',choices=('examples','place','both'),default='both');c.add_argument('--out',type=Path,required=True);a=p.parse_args(argv)
 try:o=compare(a) if a.cmd=='compare' else bundle(a.out) if a.cmd=='build-zip' else {'scopes':{'examples':32,'place':20,'both':52},'closure_sha256':shared.digest(load()[0])}
 except (ValueError,RuntimeError,OSError,KeyError) as e:p.exit(2,str(e)+'\n')
 print(json.dumps(o,sort_keys=True))
if __name__=='__main__':main()
