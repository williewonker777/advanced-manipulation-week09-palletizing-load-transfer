"""Re-run the official table for two research detectors on the same G4 landing controller."""
import argparse
import json
import os
from pathlib import Path,PurePosixPath
import shutil
import sys
import tempfile
import zipfile

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT)not in sys.path:
    sys.path.insert(0,str(ROOT))
from tools import r21_replay as shared

CLOSURE='experiments/r27_g4_export/closure.json'
PREFIX='week09-r27-g4'
COURSE_DETECTOR_SHA='fe4b1101ba41ffa46e04500f7153a0c569125b20083d8cd3a9fe6d8d69bda884'
README="""# R27: 착지 제어와 충돌 감지를 같이 실행하기

원본 수업 데모가 아니다. 원본 코드는 제공하지 않는다. 수업에서 받은 원본 키트와 MuJoCo 3.17.0을 사용한다.
같은 하중 이양용 토크 제어에 두 감지기를 번갈아 연결해, **원본 evaluate.py**로 각각 29회씩 평가한다.

```bash
KIT=/path/to/original/icpbl_sim
OUT=$(mktemp -d)
python -B tools/r27_g4_replay.py official --kit "$KIT" --out "$OUT/landing"
```

`result.json`은 공식 비시간 셀과 놓친 입력까지 보존한다. 이것은 알려진 유한한 시뮬레이션이다.
기본 R10 제출, 제공 서보 예제, 실기 안전을 의미하지 않는다.
"""

NONDETERMINISTIC_PATHS={
    '/records/c1/stdout_sha256',
    '/records/r27/stdout_sha256',
}


def result_differences(left,right,path=''):
    """Return exact JSON paths which differ; never normalize scientific values."""
    if type(left) is not type(right):
        return [path or '/']
    if isinstance(left,dict):
        out=[]
        for key in sorted(set(left)|set(right)):
            child=path+'/'+str(key)
            if key not in left or key not in right:out.append(child)
            else:out.extend(result_differences(left[key],right[key],child))
        return out
    if isinstance(left,list):
        if len(left)!=len(right):return [path or '/']
        out=[]
        for index,(a,b) in enumerate(zip(left,right)):
            out.extend(result_differences(a,b,path+'/'+str(index)))
        return out
    return [] if left==right else [path or '/']


def compare_result(actual,expected,actual_bytes=None,expected_bytes=None):
    """Accept only evaluator-stdout hash drift; all semantic fields stay exact."""
    paths=result_differences(actual,expected)
    scientific=[path for path in paths if path not in NONDETERMINISTIC_PATHS]
    cells=all(actual['records'][arm]['cells']==expected['records'][arm]['cells']
              for arm in ('c1','r27'))
    return {
        'byte_exact':actual_bytes==expected_bytes if actual_bytes is not None and expected_bytes is not None else not paths,
        'semantic_exact':not scientific,
        'all_non_time_cells_exact':cells,
        'difference_paths':paths,
        'scientific_difference_paths':scientific,
        'stdout_hash_only_difference':bool(paths) and not scientific,
    }


def load():
    src=ROOT/CLOSURE
    if shared.has_symlink_component(src):raise ValueError('regular closure required')
    raw=src.read_bytes();j=json.loads(raw)
    if j.get('schema')!='r27-frozen-g4-export.v1' or j.get('kit_included') is not False:raise ValueError('wrong closure')
    blobs={}
    for name,wanted in j['files'].items():
        rel=PurePosixPath(name);p=ROOT/name
        if (not name or '\\' in name or name!=rel.as_posix() or rel.is_absolute() or '..' in rel.parts
                or any(k in rel.parts for k in ('sim','model','.git','__pycache__','r27_loaded_holdout'))
                or rel.name in ('run.py','evaluate.py','servo_controller.py','collision_detector.py','torque_controller.py')
                or not p.is_file() or shared.has_symlink_component(p)):raise ValueError('forbidden source member: '+name)
        b=p.read_bytes()
        if shared.digest(b)!=wanted:raise ValueError('locked source changed: '+name)
        blobs[name]=b
    if not set(j['overlay'])<=set(blobs) or shared.digest(blobs[shared.KIT_MANIFEST])!=shared.MANIFEST_SHA256:
        raise ValueError('incomplete overlay')
    return raw,j,blobs


def bundle(out):
    if os.path.lexists(out) or not out.parent.is_dir() or shared.has_symlink_component(out.parent) or out.resolve().is_relative_to(ROOT):
        raise ValueError('fresh archive outside research root required')
    raw,_,blobs=load();blobs[CLOSURE]=raw;blobs['README.md']=README.encode()
    blobs['bundle.json']=(json.dumps({'kit_included':False,'new_results_included':False,'members_sha256':{n:shared.digest(b) for n,b in blobs.items()}},
                                     sort_keys=True,indent=2)+'\n').encode()
    with zipfile.ZipFile(out,'x')as z:
        for n,b in sorted(blobs.items()):
            i=zipfile.ZipInfo(PREFIX+'/'+n,(1980,1,1,0,0,0));i.create_system=3;i.external_attr=0o100644<<16;i.compress_type=zipfile.ZIP_DEFLATED
            z.writestr(i,b,compresslevel=9)
    return {'zip_sha256':shared.digest(out.read_bytes()),'entries':len(blobs)}


def official(args):
    import mujoco
    kit,out=shared.validate_paths(args.kit,args.out);req=kit/'requirements.txt'
    if mujoco.__version__!='3.17.0' or shared.has_symlink_component(req) or 'mujoco==3.17.0' not in req.read_text().splitlines():
        raise ValueError('original MuJoCo 3.17.0 course environment required')
    external=kit/'collision_detector.py'
    if not external.is_file() or shared.has_symlink_component(external) or shared.digest(external.read_bytes())!=COURSE_DETECTOR_SHA:
        raise ValueError('authentic external course detector source required by upstream parser imports')
    raw,j,blobs=load();mani=blobs[shared.KIT_MANIFEST];kp=shared.authenticate_kit(kit,mani)
    env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',PYTHONNOUSERSITE='1',OPENBLAS_NUM_THREADS='1',
             OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',MUJOCO_GL='disable')
    env.pop('PYTHONPATH',None)
    with tempfile.TemporaryDirectory(prefix='r27-g4-',dir=out.parent)as tmp:
        t=Path(tmp);work=t/'work';result=t/'result';work.mkdir();result.mkdir()
        source={n:blobs[n]for n in j['overlay']}
        shared.install_overlay(kit,work,mani,source,{n:j['files'][n]for n in source})
        shutil.copyfile(external,work/external.name)
        shared.authenticate_kit(work,mani)
        shared.run([sys.executable,'-B','-m','experiments.r27_landing_integration.unloaded_official','--out','official'],work,env,result/'run.log')
        science=work/'official/result.json';b=science.read_bytes();doc=json.loads(b)
        if (doc.get('episodes')!=58 or set(doc['records'])!={'c1','r27'}
                or doc['runtime']['evaluator_sha256']!=shared.EVALUATOR_SHA256):raise ValueError('incorrect original grader or incomplete results')
        comparison=None
        if args.expected:
            if not args.expected.is_file() or shared.has_symlink_component(args.expected):
                raise ValueError('regular authoritative result required')
            expected_bytes=args.expected.read_bytes();expected=json.loads(expected_bytes)
            comparison=compare_result(doc,expected,b,expected_bytes)
            if not comparison['semantic_exact'] or not comparison['all_non_time_cells_exact']:
                raise ValueError('clean scientific result differs from authoritative run: '
                                 +','.join(comparison['scientific_difference_paths']))
        shared.authenticate_kit(kit,mani);shared.authenticate_kit(work,mani)
        if shared.digest((work/external.name).read_bytes())!=COURSE_DETECTOR_SHA or shared.digest(external.read_bytes())!=COURSE_DETECTOR_SHA:
            raise ValueError('external course dependency changed')
        if load()[0]!=raw:raise ValueError('source dependencies changed')
        shutil.copyfile(science,result/'result.json')
        note=dict(schema='r27-g4-official-physical-receipt.v1',runtime=mujoco.__version__,kit_source_sha256=kp,
                  closure_sha256=shared.digest(raw),result_sha256=shared.digest(b),episodes=58,default_promoted=False,
                  original_servo_demo=False,expected_result_checked=bool(args.expected),
                  result_comparison=comparison)
        (result/'receipt.json').write_text(json.dumps(note,sort_keys=True,indent=2)+'\n')
        shared.publish(result,out)
    return note


def main():
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='action',required=True)
    z=sub.add_parser('build-zip');z.add_argument('--out',type=Path,required=True);sub.add_parser('describe')
    a=sub.add_parser('official');a.add_argument('--kit',type=Path,required=True);a.add_argument('--out',type=Path,required=True);a.add_argument('--expected',type=Path)
    args=p.parse_args()
    try:
        value=official(args) if args.action=='official' else bundle(args.out) if args.action=='build-zip' else load()[1]['description']
    except (ValueError,RuntimeError,OSError)as exc:p.exit(2,str(exc)+'\n')
    print(json.dumps(value,sort_keys=True,ensure_ascii=False))


if __name__=='__main__':
    main()
