"""Build a kit-free bundle or run the authentic official table on four disclosed research arms."""
import argparse
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
import zipfile

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0,str(ROOT))
from tools import r27_g4_replay as g4
from tools.r27_g4_replay import shared

INDEX='experiments/r28_contact_memory/export.json'
PREFIX='week09-r28-contact-memory'
README="""# R28: contact versus observation memory / 착지 제어 연구

**이 패키지는 수업 원본 서보 데모가 아닙니다.** 사용자 원본 키트와 MuJoCo 3.17.0 환경이 필요합니다.
원본 평가기에서 동일 착지 제어로 C1, V3, 접촉 중 재준비를 지우는 방식, 관측 기억을 보존하는 방식을
각각 29회 실행합니다. 이미 알려진 시험이고, 제출본·로봇·독립 시험을 변경하지 않습니다.

```bash
OUT=$(mktemp -d)
python -B tools/r28_contact_memory.py official --kit /path/to/icpbl_sim --out "$OUT/official"
```
"""
DIRS=('r28_contact_memory','r28_finite_contact','r28_landing_contact')


def sources():
    raw,base,blobs=g4.load()
    if shared.digest(raw)!='94513c903f47e81380e12a28d056166199fa013d343ad623a65b7932a09e6a4c':
        raise ValueError('authentic earlier G4 dependency closure required')
    ip=ROOT/INDEX
    if not ip.is_file() or shared.has_symlink_component(ip):raise ValueError('regular index required')
    ib=ip.read_bytes();spec=json.loads(ib)
    if spec.get('schema')!='r28-memory-four-arm-package.v1' or spec.get('kit_included') is not False:
        raise ValueError('incorrect export index')
    members=dict(blobs);members[g4.CLOSURE]=raw;members[INDEX]=ib
    overlay=set(base['overlay'])
    for name,wanted in spec['files'].items():
        path=Path(name); src=ROOT/path
        if (name!='tools/r28_contact_memory.py' and not (len(path.parts)==3 and path.parts[0]=='experiments' and
                path.parts[1] in DIRS and path.name in ('__init__.py','candidate.py','official.py','contract.json','plan.json','protocol.json','pulse_diagnostic.py'))):
            raise ValueError('not an authored memory-study member')
        if (path.is_absolute() or path.as_posix()!=name or not src.is_file() or shared.has_symlink_component(src)):
            raise ValueError('non-canonical export member')
        data=src.read_bytes()
        if shared.digest(data)!=wanted:raise ValueError('source pin changed: '+name)
        members[name]=data
        if name.startswith('experiments/'):overlay.add(name)
    return members,overlay


def bundle(out):
    if os.path.lexists(out) or shared.has_symlink_component(out.parent) or out.resolve().is_relative_to(ROOT):
        raise ValueError('fresh external archive required')
    members,_=sources();members['README.md']=README.encode()
    members['REPLAY-PINS.json']=(json.dumps({k:shared.digest(v)for k,v in sorted(members.items())},sort_keys=True,indent=2)+'\n').encode()
    with zipfile.ZipFile(out,'x')as z:
        for name,data in sorted(members.items()):
            info=zipfile.ZipInfo(PREFIX+'/'+name,(1980,1,1,0,0,0))
            info.create_system=3;info.external_attr=0o100644<<16;info.compress_type=zipfile.ZIP_DEFLATED
            z.writestr(info,data,compresslevel=9)
    return {'archive_sha256':shared.digest(out.read_bytes()),'entries':len(members)}


def official(kit,out,expected):
    import mujoco
    kit,out=shared.validate_paths(kit,out); req=kit/'requirements.txt';external=kit/'collision_detector.py'
    if (mujoco.__version__!='3.17.0' or shared.has_symlink_component(req) or not req.is_file() or 'mujoco==3.17.0' not in req.read_text().splitlines() or
            shared.has_symlink_component(external) or not external.is_file() or shared.digest(external.read_bytes())!=g4.COURSE_DETECTOR_SHA):
        raise ValueError('authenticated original physics and external course detector required')
    members,overlay=sources();pins={k:shared.digest(v)for k,v in members.items()}
    manifest=members[shared.KIT_MANIFEST];shared.authenticate_kit(kit,manifest)
    env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',PYTHONNOUSERSITE='1',MUJOCO_GL='disable')
    env.pop('PYTHONPATH',None)
    with tempfile.TemporaryDirectory(dir=out.parent)as tmp:
        work=Path(tmp)/'work';res=Path(tmp)/'report';work.mkdir();res.mkdir()
        selected={k:members[k]for k in overlay}
        shared.install_overlay(kit,work,manifest,selected,{k:pins[k]for k in selected})
        shutil.copyfile(external,work/external.name);shared.authenticate_kit(work,manifest)
        shared.run([sys.executable,'-B','-m','experiments.r28_contact_memory.official','--out','result.json'],work,env,res/'run.log')
        body=(work/'result.json').read_bytes();doc=json.loads(body);arms={'c1','v3','contact-reset','observation-memory'}
        if doc.get('episodes')!=116 or set(doc['records'])!=arms or doc['grader_sha256']!=shared.EVALUATOR_SHA256:
            raise ValueError('wrong original evaluator or incomplete comparison')
        changes=[]
        if expected:
            if not expected.is_file() or shared.has_symlink_component(expected):raise ValueError('regular reference required')
            changes=g4.result_differences(json.loads(expected.read_bytes()),doc)
            if set(changes)-{'/records/'+arm+'/stdout_sha256'for arm in arms}:raise ValueError('scientific result changed')
        if pins!={k:shared.digest(v)for k,v in sources()[0].items()}:raise ValueError('authored source changed')
        shared.authenticate_kit(kit,manifest);shared.authenticate_kit(work,manifest)
        if any(shared.digest(f.read_bytes())!=g4.COURSE_DETECTOR_SHA for f in (external,work/external.name)):raise ValueError('external source changed')
        (res/'result.json').write_bytes(body)
        note=dict(schema='r28-memory-official-receipt.v1',episodes=116,scientific_results_checked=bool(expected),differences=changes,
            result_sha256=shared.digest(body),source_index_sha256=pins[INDEX],source_files=len(pins),simulation_only=True,default_promoted=False)
        (res/'receipt.json').write_text(json.dumps(note,indent=2,sort_keys=True)+'\n');shared.publish(res,out)
    return note


def main():
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='mode',required=True)
    z=sub.add_parser('build-zip');z.add_argument('--out',required=True,type=Path)
    a=sub.add_parser('official');a.add_argument('--out',required=True,type=Path);a.add_argument('--kit',required=True,type=Path);a.add_argument('--expected',type=Path)
    x=p.parse_args()
    try:result=bundle(x.out) if x.mode=='build-zip' else official(x.kit,x.out,x.expected)
    except (ValueError,RuntimeError,OSError)as exc:p.exit(2,str(exc)+'\n')
    print(json.dumps(result,sort_keys=True))


if __name__=='__main__':
    main()
