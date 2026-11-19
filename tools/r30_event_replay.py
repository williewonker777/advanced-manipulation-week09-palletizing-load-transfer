"""Run five explicitly labeled research arms on disclosed torque inputs, or bundle the sources."""
import argparse
import json
import os
from pathlib import Path
import sys
import tempfile
import zipfile

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0,str(ROOT))
from tools import r29_event_replay as previous
from tools import r21_replay as shared

DIR = 'experiments/r30_event_replay/'
CLOSURE = DIR + 'closure.json'
PREFIX = 'week09-r30-torque-events'
SCOPES = {'signed-stock':50, 'paired-directions':60, 'place':25}
ARMS = ('c3','composed','rising','geometry','headroom')
MEMBERS = {DIR + f for f in ('__init__.py','runner.py','anchors.json')} | {
    'experiments/r29_native_event_link/evaluator.py', 'tools/r30_event_replay.py',
    *('experiments/r30_source_association/'+f for f in ('__init__.py','candidate.py','registration.json')),
    *('experiments/r30_yaw_headroom/'+f for f in ('__init__.py','candidate.py','protocol.json')),
}
README = """# 토크 충돌 경보의 중복 재현 (R30 연구)

원본 수업 데모나 제출용 R10이 아닙니다. 제공한 키트는 복사본에서 읽기만 하며 ZIP에 포함하지 않습니다.
MuJoCo 3.17.0 환경에서 이 ZIP을 풀고 다음처럼 실행합니다. 출력 경로는 새 폴더여야 합니다.

    python -B tools/r30_event_replay.py compare --kit /path/to/icpbl_sim --scope paired-directions --out /new/output

같은 my_torque에서 C3, R29 조합, R29 조기, R30 방향 전용, R30 yaw 여유 방식의 경보를 직접 비교합니다.
signed-stock은 두 자세의 정·역·두 직교 방향, paired-directions는 같은 방향·반대 방향·교차 충격,
place는 놓기 직전 무충격·첫째만·둘째만·두 번·긴 힘 한 번입니다.
result.json 안 delays_ms와 rows에서 검출 지연·중복·무충격·실제 힘·충격 전 일치를 확인하세요.
공개된 조건의 재생이지 독립 시험이나 실기 안전 인증이 아닙니다. 미사용 평가 조건/결과물은 포함하지 않습니다.
"""


def load():
    old, blobs, overlay = previous.load()
    raw = previous.base.file(CLOSURE)
    extra = json.loads(raw)
    if (extra.get('schema') != 'r30-disclosed-torque-replay-source.v1' or
            extra.get('r29_closure_sha256') != shared.digest(old) or extra.get('kit_included') is not False or
            set(extra['files']) != MEMBERS or set(blobs) & MEMBERS):
        raise ValueError('unexpected source membership or changed parent')
    blobs = {**blobs,previous.CLOSURE:old}
    for name, pin in extra['files'].items():
        data = previous.base.file(name)
        if pin != shared.digest(data):
            raise ValueError('changed replay source: ' + name)
        blobs[name] = data
    return raw, blobs, sorted(set(overlay) | {n for n in MEMBERS if not n.startswith('tools/')})


def bundle(out):
    if (os.path.lexists(out) or not out.parent.is_dir() or shared.has_symlink_component(out.parent)
            or out.resolve().is_relative_to(ROOT)):
        raise ValueError('fresh ZIP outside the source tree required')
    raw, files, _ = load()
    files[CLOSURE] = raw
    files['README.md'] = README.encode()
    files['bundle.json'] = (json.dumps(dict(kit_included=False,unopened_conditions_included=False,
        recorded_candidate_outputs_included=False,members={n:shared.digest(b) for n,b in files.items()}),
        indent=2,sort_keys=True)+'\n').encode()
    with zipfile.ZipFile(out,'x') as z:
        for n,b in sorted(files.items()):
            info=zipfile.ZipInfo(PREFIX+'/'+n,(1980,1,1,0,0,0)); info.create_system=3
            info.external_attr=0o100644<<16; info.compress_type=zipfile.ZIP_DEFLATED
            z.writestr(info,b,compresslevel=9)
    return dict(archive_sha256=shared.digest(out.read_bytes()),entries=len(files))


def compare(a):
    import mujoco
    kit, out = shared.validate_paths(a.kit,a.out)
    req = kit/'requirements.txt'
    if (mujoco.__version__!='3.17.0' or shared.has_symlink_component(req) or
            'mujoco==3.17.0' not in req.read_text().splitlines()):
        raise ValueError('original kit and MuJoCo 3.17.0 required')
    raw, blobs, overlay = load()
    manifest = blobs[shared.KIT_MANIFEST]
    pins = shared.authenticate_kit(kit,manifest)
    env = dict(os.environ,PYTHONDONTWRITEBYTECODE='1',PYTHONNOUSERSITE='1',OPENBLAS_NUM_THREADS='1',
               OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',MUJOCO_GL='disable')
    for k in ('PYTHONPATH','PYTHONHOME'):
        env.pop(k,None)
    with tempfile.TemporaryDirectory(prefix='r30-disclosed-',dir=out.parent) as tmp:
        work, report = (Path(tmp)/n for n in ('work','report'))
        work.mkdir(); report.mkdir()
        sources = {n:blobs[n] for n in overlay}
        shared.install_overlay(kit,work,manifest,sources,{n:shared.digest(b) for n,b in sources.items()})
        if shared.authenticate_kit(work,manifest)!=pins:
            raise ValueError('kit overlay identity failed')
        shared.run([sys.executable,'-B','-m','experiments.r30_event_replay.runner',a.scope,'--out','result.json'],
                   work,env,report/'run.log')
        data=(work/'result.json').read_bytes(); j=json.loads(data)
        if (j['schema']!='r30-disclosed-event-replay.v1' or j['scope']!=a.scope or j['arms']!=list(ARMS)
                or j['episodes']!=SCOPES[a.scope] or j['course_original'] or j['default_changed']):
            raise ValueError('wrong or incomplete replay')
        if (shared.authenticate_kit(kit,manifest)!=pins or shared.authenticate_kit(work,manifest)!=pins
                or load()[0]!=raw):
            raise ValueError('sources or course files changed during run')
        (report/'result.json').write_bytes(data)
        receipt=dict(schema='r30-replay-receipt.v1',runtime=mujoco.__version__,scope=a.scope,episodes=j['episodes'],
            result_sha256=shared.digest(data),kit_files_sha256=pins,kit_manifest_sha256=shared.digest(manifest),
            closure_sha256=shared.digest(raw))
        (report/'receipt.json').write_text(json.dumps(receipt,sort_keys=True,indent=2)+'\n')
        shared.publish(report,out)
    return receipt


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__); sub=p.add_subparsers(dest='cmd',required=True)
    sub.add_parser('describe'); z=sub.add_parser('build-zip'); z.add_argument('--out',type=Path,required=True)
    c=sub.add_parser('compare'); c.add_argument('--kit',type=Path,required=True); c.add_argument('--out',type=Path,required=True)
    c.add_argument('--scope',choices=SCOPES,default='paired-directions'); a=p.parse_args(argv)
    try:
        obj = (compare(a) if a.cmd=='compare' else bundle(a.out) if a.cmd=='build-zip' else
               dict(course_original=False,arms=ARMS,scope_episodes=SCOPES,closure_sha256=shared.digest(load()[0])))
    except (ValueError,RuntimeError,OSError,KeyError) as exc:
        p.exit(2,str(exc)+'\n')
    print(json.dumps(obj,sort_keys=True))


if __name__=='__main__':
    main()
