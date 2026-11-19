"""Re-run all 385 open conditions in a fresh overlay of the authenticated stock course kit."""
import argparse
import json
import os
from pathlib import Path
import sys
import tempfile

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0,str(ROOT))
from tools import r21_replay as shared
from tools import r31_event_replay as parent

REG='experiments/r32_level_assay/replay-inputs.json'
REG_SHA='861b6c391217ac47a125533849a9c66b864e12ec2543b1b3fa4690bf3f86d040'
EXT_REG='experiments/r32_level_assay/census-extensions.json'
EXT_SHA='7edb777ca48af6f8c4e710b1ce23bf5c16f36823ad876abec322853037629f2c'
OLD='7070f5f1651e07faa8701f28421082b7eaf32a16cb866dbf8e5d9ffa07018d43'
METHODS={'load':('r32_measured_level','e08c0622c0fb6947f80b2c535a5ef0506f58a6fe8b44c8526323e5d405ebc25c'),
         'handoff':('r32_native_handoff','d3949f6452b6d3573a9b61a63b77900d0ce78e8603dba2912a7b2e8c034dd837'),
         'readback':('r32_detached_readback','a417c78fff0a2da124038921172b102ec426678e9b9a4a43a6a738fbdccbd4b1'),
         'joint':('r32_joint_state','633b1d49229ed07764d925bb84a5248d7db183aef2a2e4e4d835512752033d1a')}


def verified_reference(path):
    if not path.is_file() or shared.has_symlink_component(path):
        raise ValueError('regular full R31 reference required')
    data=path.read_bytes()
    if shared.digest(data)!=OLD:
        raise ValueError('expected the complete authenticated R31 full-d4 result.json')
    return data


def sources(method='load'):
    raw,files,overlay=parent.load()
    reg=(ROOT/REG).read_bytes()
    if shared.digest(reg)!=REG_SHA:
        raise ValueError('census source registration differs')
    pins=json.loads(reg)
    if method in ('readback','joint'):
        extra=(ROOT/EXT_REG).read_bytes()
        if shared.digest(extra)!=EXT_SHA or set(json.loads(extra))&set(pins):
            raise ValueError('changed extension or replacement of registered source')
        pins.update(json.loads(extra))
    if set(pins)&set(overlay):
        raise ValueError('cannot replace a frozen parent module')
    blobs={n:files[n] for n in overlay}
    for n,digest in pins.items():
        path=ROOT/n
        if shared.has_symlink_component(path) or shared.digest(path)!=digest:
            raise ValueError('changed source: '+n)
        blobs[n]=path.read_bytes()
    return files[shared.KIT_MANIFEST],blobs,shared.digest(raw)


def collect_rows(rows,out,pin,count=385):
    paths=sorted(rows.iterdir())
    if [p.name for p in paths]!=[f'{i:03d}.json' for i in range(count)]:
        raise ValueError('missing or unexpected raw episode')
    with out.open('xb') as dest:
        for i,path in enumerate(paths):
            if path.is_symlink() or not path.is_file():
                raise ValueError('irregular raw episode')
            raw=path.read_bytes()
            obj=json.loads(raw)
            if obj.get('id')!=i or obj.get('source')!=pin or len(raw.splitlines())!=1:
                raise ValueError('raw episode/source differs')
            dest.write(raw.rstrip(b'\n')+b'\n')


def run(a):
    import mujoco
    kit,out=shared.validate_paths(a.kit,a.out)
    ref=verified_reference(a.reference)
    manifest,blobs,parent_pin=sources(a.method);kit_pins=shared.authenticate_kit(kit,manifest)
    if mujoco.__version__!='3.17.0':raise ValueError('requires course MuJoCo 3.17.0')
    name,pin=METHODS[a.method];entry='experiments.'+name+'.candidate:Detector'
    env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',PYTHONNOUSERSITE='1',OPENBLAS_NUM_THREADS='1',
             OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',MUJOCO_GL='disable')
    for key in ('PYTHONPATH','PYTHONHOME','PYTHONOPTIMIZE'):env.pop(key,None)
    with tempfile.TemporaryDirectory(prefix='r32-open-census-',dir=out.parent) as tmp:
        work=Path(tmp)/'work'; report=Path(tmp)/'report';work.mkdir();report.mkdir()
        shared.install_overlay(kit,work,manifest,blobs,{n:shared.digest(v) for n,v in blobs.items()})
        refpath=work/'r31-open-matrix.json';refpath.write_bytes(ref)
        rows=work/'episode-rows'; rows.mkdir();result=report/'result.json'
        shared.run([sys.executable,'-B','-m','experiments.r32_level_assay.check','d4',
                    '--entry',entry,'--sha',pin,'--previous',str(refpath),'--rows',str(rows),
                    '--out',str(result),'--jobs',str(a.jobs)],work,env,report/'execution.log')
        obj=json.loads(result.read_bytes())
        if (obj.get('candidate_sha256')!=pin or obj.get('candidate')!=entry
                or obj.get('parent_event_channel_exact')!=(0 if a.method=='readback' else 385) or obj.get('matched_force_physics_prefix')!=385
                or len(list(rows.glob('*.json')))!=385 or obj.get('episodes')!=385
                or shared.authenticate_kit(kit,manifest)!=kit_pins or shared.authenticate_kit(work,manifest)!=kit_pins):
            raise ValueError('incomplete or changed replay')
        collect_rows(rows,report/'episodes.jsonl',pin)
        receipt=dict(raw_jsonl_sha256=shared.digest(report/'episodes.jsonl'),schema='r32-full-census-receipt.v1',authored_inputs=REG_SHA,parent_closure_sha256=parent_pin,
            candidate_sha256=pin,prior_sha256=OLD,kit_manifest_sha256=shared.digest(manifest),
            protected_files=len(kit_pins),result_sha256=shared.digest(result),episodes=385,method=a.method,
            kit_unchanged=True,classification='disclosed sample-matched research; not independent qualification')
        if a.method in ('readback','joint'):
            receipt['extension_inputs_sha256']=EXT_SHA
            receipt['separate_identity_event_rows']=obj['parent_event_channel_exact']
        (report/'receipt.json').write_text(json.dumps(receipt,indent=2,sort_keys=True)+'\n')
        shared.publish(report,out)
    return receipt


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--kit',type=Path,required=True);p.add_argument('--reference',type=Path,required=True)
    p.add_argument('--method',choices=tuple(METHODS),required=True)
    p.add_argument('--out',type=Path,required=True);p.add_argument('--jobs',type=int,choices=range(1,9),default=2)
    try:print(json.dumps(run(p.parse_args()),sort_keys=True))
    except (ValueError,OSError,RuntimeError) as error:p.exit(2,str(error)+'\n')
