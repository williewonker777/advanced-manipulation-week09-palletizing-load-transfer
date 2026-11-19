"""Replay only the original three enrolled G4 arms after disclosure, with a separately supplied kit."""
import argparse
import json
import os
from pathlib import Path, PurePosixPath
import sys
import tempfile
import zipfile

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from tools import r21_replay as shared

CLOSURE='experiments/r29_g4_matrix_replay/closure.json'
PREFIX='week09-r29-g4-matrix'
# Membership observed in a real three-arm run and verified against prior authenticated authored sources.
MEMBERS_SHA256='e35b3244ed06aa1e6862d4f0d055226c4705c534ba7b98630deb3c22d56d7b4e'
OVERLAY_SHA256='3692a64a8715aa8405013bda7d94786bdbb16f217a1457b3ff88fe33f7e7bcca'
HISTORIC_SCIENTIFIC_SHA256='ab9df9b9ed74d519bde4981db6eb2edef6f0d65e41ea14882dfee199bf566cb0'


def names_sha(names):
    return shared.digest(('\n'.join(sorted(names))+'\n').encode())
README="""# R29 공개 후 착지 시험 재생 (90회)

원본 데모가 아니며, 새 독립 판정이 아닙니다. 이전에 고정했던 C1, V3, 관측 기억 방식의
30개 동일 물리 입력만 실행합니다. 원래 판정 FAIL을 그대로 재현합니다.
수업 키트, 내장 그림, 결과 JSON, 미개봉 조건을 포함하지 않습니다. 인증된 MuJoCo 3.17.0 키트를 따로 준비하세요.

    python -B tools/r29_g4_matrix.py replay --kit /path/to/icpbl_sim --out /new/output

원래 등록한 물리/채점 함수 본문은 바이트 단위로 같습니다. 설치 위치를 찾는 부분과
무결성 검사만 이식했습니다. 이에 따라 runner_sha256만 달라지며, 다른 모든 필드는 대조합니다.
"""


def file(name):
    p=PurePosixPath(name)
    if (not name or name!=p.as_posix() or p.is_absolute() or '\\' in name
            or set(p.parts)&{'results','result','evidence','..','.git','.omx','sim','model','__pycache__','r27_loaded_holdout','r28_landing_prospect'}
            or p.name in ('result.json','run.py','evaluate.py','servo_controller.py','torque_controller.py','collision_detector.py')
            or p.suffix in ('.png','.npz','.xml','.stl','.pyc','.zip')):
        raise ValueError('not an authorized student source: '+name)
    f=ROOT/name
    if not f.is_file() or shared.has_symlink_component(f):raise ValueError('regular source required')
    return f.read_bytes()


def load():
    raw=file(CLOSURE);c=json.loads(raw)
    if (c.get('schema')!='r29-fixed-three-arm-matrix-disclosed-replay.v1' or c['kit_included'] is not False
            or c['new_arms_permitted'] is not False or c['locked_verdict'] is not False
            or c['historic_result_sha256']!='09c224f2b8b32ab72d3f4e796911311f3d260697de40de9075887c86e1c0dca4'):
        raise ValueError('wrong three-arm replay identity')
    overlay=c['overlay'];pins=c['files']
    if (names_sha(pins)!=MEMBERS_SHA256 or names_sha(overlay)!=OVERLAY_SHA256
            or c['historic_scientific_sha256']!=HISTORIC_SCIENTIFIC_SHA256 or not set(overlay)<=pins.keys() or len(set(overlay))!=len(overlay) or CLOSURE in pins
            or not {'tools/r29_g4_matrix.py','LICENSE','THIRD_PARTY_NOTICES',shared.KIT_MANIFEST}<=pins.keys()):
        raise ValueError('invalid membership')
    blobs={n:file(n)for n in pins}
    if any(shared.digest(b)!=pins[n]for n,b in blobs.items()):raise ValueError('a replay dependency changed')
    return raw,c,blobs


def bundle(out):
    if (os.path.lexists(out) or not out.parent.is_dir() or shared.has_symlink_component(out.parent)
            or out.resolve().is_relative_to(ROOT)):
        raise ValueError('fresh external archive required')
    raw,c,blobs=load(); files=dict(blobs);files[CLOSURE]=raw;files['README.md']=README.encode()
    files['bundle.json']=(json.dumps(dict(schema=c['schema'],kit_included=False,prior_results_included=False,
            members={n:shared.digest(b)for n,b in files.items()}),sort_keys=True,indent=2)+'\n').encode()
    with zipfile.ZipFile(out,'x')as z:
        for n,b in sorted(files.items()):
            info=zipfile.ZipInfo(PREFIX+'/'+n,(1980,1,1,0,0,0));info.create_system=3;info.external_attr=0o100644<<16
            info.compress_type=zipfile.ZIP_DEFLATED;z.writestr(info,b,compresslevel=9)
    return dict(zip_sha256=shared.digest(out.read_bytes()),entries=len(files))


def scientific(doc):
    # No metric, threshold, timing, duration, source closure, support, edge or event result is ignored.
    if 'runner_sha256' not in doc:raise ValueError('runner identity missing')
    return shared.digest(json.dumps({k:v for k,v in doc.items()if k!='runner_sha256'},sort_keys=True,separators=(',',':'),allow_nan=False).encode())


def run(kit,out):
    import mujoco
    kit,out=shared.validate_paths(kit,out);raw,c,blobs=load();manifest=blobs[shared.KIT_MANIFEST]
    pins=shared.authenticate_kit(kit,manifest)
    req=kit/'requirements.txt'
    if (mujoco.__version__!='3.17.0' or shared.has_symlink_component(req)
            or 'mujoco==3.17.0' not in req.read_text().splitlines()):raise ValueError('MuJoCo 3.17.0 kit required')
    env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',PYTHONNOUSERSITE='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',
             OPENBLAS_NUM_THREADS='1',MUJOCO_GL='disable')
    for k in ('PYTHONPATH','PYTHONHOME'):env.pop(k,None)
    with tempfile.TemporaryDirectory(prefix='r29-fixed-g4-',dir=out.parent)as tmp:
        work=Path(tmp)/'work';result=Path(tmp)/'out';work.mkdir();result.mkdir()
        overlay={n:blobs[n]for n in c['overlay']}
        shared.install_overlay(kit,work,manifest,overlay,{n:shared.digest(b)for n,b in overlay.items()})
        shared.authenticate_kit(work,manifest)
        shared.run([sys.executable,'-B','-m','experiments.r29_g4_matrix_replay.matrix','--out','matrix-run'],work,env,result/'run.log')
        data=(work/'matrix-run'/'result.json').read_bytes();doc=json.loads(data)
        if (doc.get('episodes')!=90 or doc['primary_protection_passed'] is not False
                or doc['runner_sha256']!=c['files']['experiments/r29_g4_matrix_replay/matrix.py']
                or scientific(doc)!=c['historic_scientific_sha256']):raise ValueError('a scientific field or registered verdict changed')
        if shared.authenticate_kit(kit,manifest)!=pins or shared.authenticate_kit(work,manifest)!=pins or load()[0]!=raw:
            raise ValueError('source or kit changed during replay')
        (result/'result.json').write_bytes(data)
        receipt=dict(schema='r29-fixed-g4-matrix-receipt.v1',episodes=90,locked_verdict=False,
            scientific_equal=True,metadata_exception=['runner_sha256'],historic_result_sha256=c['historic_result_sha256'],
            actual_result_sha256=shared.digest(data),scientific_sha256=scientific(doc),kit_files_sha256=pins,kit_manifest_sha256=shared.digest(manifest),closure_sha256=shared.digest(raw))
        (result/'receipt.json').write_text(json.dumps(receipt,indent=2,sort_keys=True)+'\n')
        shared.publish(result,out)
    return receipt


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);s=p.add_subparsers(dest='cmd',required=True)
    s.add_parser('describe');z=s.add_parser('build-zip');z.add_argument('--out',required=True,type=Path)
    a=s.add_parser('replay');a.add_argument('--kit',type=Path,required=True);a.add_argument('--out',type=Path,required=True);v=p.parse_args(argv)
    try:doc=run(v.kit,v.out) if v.cmd=='replay' else bundle(v.out) if v.cmd=='build-zip' else load()[1]
    except (ValueError,RuntimeError,OSError,KeyError)as exc:p.exit(2,str(exc)+'\n')
    print(json.dumps(doc,sort_keys=True))


if __name__=='__main__':main()
