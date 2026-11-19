"""Replay disclosed 6–8 N conditions, or build a deterministic student-only source bundle."""
import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import sys
import tempfile
import zipfile

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0,str(ROOT))
from tools import r21_replay as shared

DIR='experiments/r27_portable/'
CLOSURE=DIR+'closure.json'
PREFIX='week09-r27-loaded'
STOCK={'evaluate.py','run.py','servo_controller.py','torque_controller.py','collision_detector.py'}
HELP="""# R27: 알려진 두 약충격을 재생하기

수업 원본 데모가 아니다. 같은 학생 토크 제어에서 이전 감지기와 선택한 **연구 후보**를 비교한다.
원본 키트와 그림은 포함하지 않는다. 이전 R26 대조값 일부는 입력·기존 경보를 확인하는 용도로 포함한다. MuJoCo 3.17.0이 설치된 수업 환경에서 ZIP을 풀고 이 폴더로 이동한다.

```bash
python -B tools/r27_force_replay.py describe
KIT=/path/to/original/icpbl_sim
OUT=$(mktemp -d)
python -B tools/r27_force_replay.py compare --kit "$KIT" --arm signed-change --scope registered --out "$OUT/test"
```

`registered`는 공개된 이동 중·놓기 직전 충격 및 무충격 대조를 두 방식으로 총 8회 실행한다.
`directions`는 같은 두 자세에서 방향과 물성을 바꾸어 60회 실행한다.
`result.json`의 사건, 오경보, 구간과 `receipt.json`의 실행 무결성을 따로 본다.
알려진 입력에서의 검출을 미공개 시험 합격이나 실기 안전성으로 해석하지 않는다. 출력 폴더는 새 경로여야 한다.
"""


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def legal(name):
    p=PurePosixPath(name)
    return (isinstance(name,str) and name==p.as_posix() and name not in ('','.','README.md','bundle.json')
            and not p.is_absolute() and not any(x in ('..','.git','sim','model','__pycache__') for x in p.parts)
            and p.name not in STOCK and p.suffix not in ('.pyc','.xml','.stl') and '\\' not in name)


def file(name):
    if not legal(name):raise ValueError('not a student-source member: '+str(name))
    p=ROOT/name
    if not p.is_file() or shared.has_symlink_component(p):raise ValueError('regular source required: '+name)
    return p.read_bytes()


def closure():
    blob=file(CLOSURE);j=json.loads(blob)
    if j.get('schema')!='r27-known-force-authored-release.v1' or j.get('kit_included') is not False or j.get('unopened_conditions_included') is not False:
        raise ValueError('wrong source contract')
    pins=j['files'];overlay=j['overlay']
    if not isinstance(pins,dict) or not set(overlay)<=set(pins) or CLOSURE in pins or len(overlay)!=len(set(overlay)):
        raise ValueError('invalid release membership')
    source={n:file(n) for n in pins}
    if any(sha(source[n])!=h for n,h in pins.items()):raise ValueError('authored source dependency changed')
    if 'tools/r27_force_replay.py' not in pins or shared.KIT_MANIFEST not in pins or DIR+'runner.py' not in overlay:
        raise ValueError('incomplete R27 source closure')
    if any('experiments/r27_loaded_holdout/' in n for n in source):
        raise ValueError('independent challenge is not part of this diagnostic')
    reg=json.loads(source[DIR+'registry.json'])
    return blob,j,source,reg


def build(out):
    if (os.path.lexists(out) or not out.parent.is_dir() or shared.has_symlink_component(out.parent)
            or out.resolve().is_relative_to(ROOT)):
        raise ValueError('specify a fresh archive outside the source tree')
    blob,j,source,_=closure()
    members=dict(source)
    members[CLOSURE]=blob;members['README.md']=HELP.encode()
    members['bundle.json']=(json.dumps(dict(schema=j['schema'],kit_included=False,new_candidate_results_included=False,r26_reference_excerpt_included=True,
        files={n:sha(v)for n,v in sorted(members.items())}),indent=2,sort_keys=True)+'\n').encode()
    with zipfile.ZipFile(out,'x')as z:
        for n,v in sorted(members.items()):
            meta=zipfile.ZipInfo(PREFIX+'/'+n,(1980,1,1,0,0,0));meta.compress_type=zipfile.ZIP_DEFLATED
            meta.create_system=3;meta.external_attr=0o100644<<16
            z.writestr(meta,v,compresslevel=9)
    return dict(archive_sha256=sha(out.read_bytes()),entries=len(members))


def replay(args):
    import mujoco
    kit,out=shared.validate_paths(args.kit,args.out)
    req=kit/'requirements.txt'
    if (mujoco.__version__!='3.17.0' or not req.is_file() or shared.has_symlink_component(req)
            or 'mujoco==3.17.0' not in req.read_text().splitlines()):
        raise ValueError('authentic MuJoCo 3.17.0 course environment required')
    blob,j,source,reg=closure()
    if args.arm not in reg['arms']:raise ValueError('unknown opt-in arm; see describe')
    mani=source[shared.KIT_MANIFEST];kp=shared.authenticate_kit(kit,mani)
    env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',PYTHONNOUSERSITE='1',OPENBLAS_NUM_THREADS='1',
             OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',MUJOCO_GL='disable')
    env.pop('PYTHONPATH',None)
    with tempfile.TemporaryDirectory(prefix='week09-r27-',dir=out.parent)as tmp:
        t=Path(tmp);stage=t/'work';result=t/'result';stage.mkdir();result.mkdir()
        bodies={n:source[n]for n in j['overlay']}
        shared.install_overlay(kit,stage,mani,bodies,{n:j['files'][n]for n in bodies})
        shared.authenticate_kit(stage,mani)
        cmd=[sys.executable,'-B','-m','experiments.r27_portable.runner','--arm',args.arm,'--scope',args.scope,
             '--jobs',str(args.jobs),'--out','result.json']
        shared.run(cmd,stage,env,result/'run.log');shutil.copyfile(stage/'result.json',result/'result.json')
        raw=(result/'result.json').read_bytes();doc=json.loads(raw)
        if (doc['episodes']!={'registered':8,'directions':60}[args.scope] or doc['default_promoted']
                or doc['entrypoint']!=reg['arms'][args.arm]['entrypoint']):
            raise ValueError('incomplete disclosed comparison')
        if args.expected:
            if not args.expected.is_file() or shared.has_symlink_component(args.expected) or args.expected.read_bytes()!=raw:
                raise ValueError('expected physical result did not reproduce')
        shared.authenticate_kit(kit,mani);shared.authenticate_kit(stage,mani)
        if closure()[0]!=blob:raise ValueError('source closure changed while replaying')
        receipt=dict(schema='r27-known-force-receipt.v1',kit_manifest_sha256=sha(mani),kit_files=kp,runtime=mujoco.__version__,
                     closure_sha256=sha(blob),result_sha256=sha(raw),episodes=doc['episodes'],simulation_only=True,
                     optional_not_promoted=True,expected_result_checked=bool(args.expected))
        (result/'receipt.json').write_text(json.dumps(receipt,indent=2,sort_keys=True)+'\n')
        shared.publish(result,out)
    return receipt


def main():
    p=argparse.ArgumentParser(description=__doc__);cmd=p.add_subparsers(dest='command',required=True)
    cmd.add_parser('describe');b=cmd.add_parser('build-zip');b.add_argument('--out',type=Path,required=True)
    a=cmd.add_parser('compare');a.add_argument('--arm',required=True);a.add_argument('--kit',type=Path,required=True)
    a.add_argument('--out',type=Path,required=True);a.add_argument('--scope',choices=('registered','directions'),default='registered')
    a.add_argument('--jobs',type=int,choices=(1,2),default=1);a.add_argument('--expected',type=Path)
    args=p.parse_args()
    try:
        result=closure()[3] if args.command=='describe' else build(args.out) if args.command=='build-zip' else replay(args)
    except (ValueError,OSError,RuntimeError) as exc:
        p.exit(2,str(exc)+'\n')
    print(json.dumps(result,sort_keys=True,ensure_ascii=False))


if __name__=='__main__':
    main()
