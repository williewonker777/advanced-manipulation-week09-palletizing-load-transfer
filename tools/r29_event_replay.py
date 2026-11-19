"""Build a source-only R29 event replay, or run it with an authenticated original kit."""
import argparse
import json
import os
from pathlib import Path
import sys
import tempfile
import zipfile

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from tools import r21_replay as shared
from tools import r27_force_replay as base

CLOSURE = 'experiments/r29_event_replay/closure.json'
PREFIX = 'week09-r29-torque-events'
SCOPES = {'known-detached':30, 'known-place':30, 'known-full':156, 'prospective-detached':150}
ARMS = ('released','linked','composed','rising-audit')
README = """# R29 토크 충격 연구 재현

수업 원본 데모/제출본이 아닙니다. 원본 키트·그림·미공개 시험·후보 결과를 포함하지 않습니다.
MuJoCo 3.17.0과 원본 키트를 준비한 뒤 이 ZIP을 풀고 실행하세요.

    python -B tools/r29_event_replay.py compare --kit /path/to/icpbl_sim --scope known-detached --arm released --out /new/path
    python -B tools/r29_event_replay.py compare --kit /path/to/icpbl_sim --scope known-full --arm composed --out /another/path

각 힘에 대해 무충격/첫째만/둘째만/모두/힘 한 번 지속을 구분하고 모든 시간·물리 검증 값을 남깁니다.
rising-audit는 방향 보호 기준에서 탈락한 방식이며 공개된 입력에만 쓸 수 있습니다.
prospective-detached는 사전 고정했던 150회의 공개 후 재생이며 released 방식만 허용합니다. 재생은 새 독립 시험이 아닙니다.
"""


def load():
    raw, old, blobs, _ = base.closure()
    extra_raw = base.file(CLOSURE)
    extra = json.loads(extra_raw)
    if (extra.get('schema') != 'r29-authored-torque-delivery.v1' or
            extra['kit_included'] is not False or extra['base_closure_sha256'] != shared.digest(raw)):
        raise ValueError('wrong R29 closure or changed base')
    blobs = dict(blobs); blobs[base.CLOSURE] = raw
    files = extra['files']; overlay = set(extra['overlay'])
    if (not overlay <= files.keys() or len(overlay) != len(extra['overlay'])
            or set(files) & set(blobs) or CLOSURE in files
            or not {'tools/r29_event_replay.py', 'experiments/r29_event_replay/runner.py'} <= files.keys()):
        raise ValueError('invalid R29 membership')
    for n, pin in files.items():
        if n != 'tools/r29_event_replay.py' and not n.startswith((
                'experiments/r29_detached_release/', 'experiments/r29_event_replay/',
                'experiments/r29_native_event_link/', 'experiments/r29_disjoint_events/', 'experiments/r29_loaded_rising_certificate/')):
            raise ValueError('not a published R29 dependency')
        b = base.file(n)
        if shared.digest(b) != pin: raise ValueError('changed source: ' + n)
        blobs[n] = b
    return extra_raw, blobs, sorted(set(old['overlay']) | overlay)


def bundle(out):
    if (os.path.lexists(out) or not out.parent.is_dir() or shared.has_symlink_component(out.parent)
            or out.resolve().is_relative_to(ROOT)):
        raise ValueError('fresh external archive required')
    raw, blobs, _ = load()
    files = dict(blobs); files[CLOSURE] = raw; files['README.md'] = README.encode()
    files['bundle.json'] = (json.dumps(dict(kit_included=False, candidate_results_included=False,
        unopened_conditions_included=False, members={n: shared.digest(b) for n,b in files.items()}),
        indent=2, sort_keys=True) + '\n').encode()
    with zipfile.ZipFile(out,'x') as z:
        for n,b in sorted(files.items()):
            info = zipfile.ZipInfo(PREFIX + '/' + n, (1980,1,1,0,0,0))
            info.create_system=3; info.external_attr=0o100644<<16; info.compress_type=zipfile.ZIP_DEFLATED
            z.writestr(info,b,compresslevel=9)
    return dict(zip_sha256=shared.digest(out.read_bytes()), entries=len(files))


def compare(a):
    import mujoco
    if a.scope == 'prospective-detached' and a.arm != 'released':
        raise ValueError('this later-disclosed prospective panel is locked to its enrolled arm')
    kit,out=shared.validate_paths(a.kit,a.out)
    req=kit/'requirements.txt'
    if (mujoco.__version__!='3.17.0' or shared.has_symlink_component(req)
            or 'mujoco==3.17.0' not in req.read_text().splitlines()):
        raise ValueError('MuJoCo 3.17.0 and authorized original kit required')
    raw,blobs,overlay=load(); manifest=blobs[shared.KIT_MANIFEST]
    kit_pins=shared.authenticate_kit(kit,manifest)
    env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',PYTHONNOUSERSITE='1',OPENBLAS_NUM_THREADS='1',
             OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',MUJOCO_GL='disable')
    for key in ('PYTHONPATH','PYTHONHOME'):env.pop(key,None)
    with tempfile.TemporaryDirectory(prefix='r29-events-',dir=out.parent)as tmp:
        work=Path(tmp)/'work'; report=Path(tmp)/'report';work.mkdir();report.mkdir()
        sources={n:blobs[n] for n in overlay}
        shared.install_overlay(kit,work,manifest,sources,{n:shared.digest(b)for n,b in sources.items()})
        shared.authenticate_kit(work,manifest)
        shared.run([sys.executable,'-B','-m','experiments.r29_event_replay.runner','--arm',a.arm,'--scope',a.scope,
                    '--jobs',str(a.jobs),'--out','result.json'],work,env,report/'run.log')
        data=(work/'result.json').read_bytes();j=json.loads(data)
        if (j.get('schema')!='r29-disclosed-torque-physical-replay.v1' or j['arm']!=a.arm or
                j['scope']!=a.scope or j['episodes']!=SCOPES[a.scope] or j['default_changed'] or j['course_demo']):
            raise ValueError('incomplete study replay')
        if shared.authenticate_kit(kit,manifest)!=kit_pins or shared.authenticate_kit(work,manifest)!=kit_pins or load()[0]!=raw:
            raise ValueError('source/kit changed during replay')
        (report/'result.json').write_bytes(data)
        receipt=dict(schema='r29-torque-receipt.v1',runtime=mujoco.__version__,episodes=j['episodes'],
                     result_sha256=shared.digest(data),kit_files_sha256=kit_pins,
                     kit_manifest_sha256=shared.digest(manifest),closure_sha256=shared.digest(raw))
        (report/'receipt.json').write_text(json.dumps(receipt,sort_keys=True,indent=2)+'\n')
        shared.publish(report,out)
    return receipt


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);s=p.add_subparsers(dest='cmd',required=True)
    s.add_parser('describe');z=s.add_parser('build-zip');z.add_argument('--out',required=True,type=Path)
    c=s.add_parser('compare');c.add_argument('--arm',choices=ARMS,default='released')
    c.add_argument('--scope',choices=SCOPES,default='known-detached');c.add_argument('--jobs',choices=(1,2),default=1,type=int)
    c.add_argument('--kit',type=Path,required=True);c.add_argument('--out',type=Path,required=True);a=p.parse_args(argv)
    try:
        result=(bundle(a.out) if a.cmd=='build-zip' else compare(a) if a.cmd=='compare' else
                dict(arms=ARMS,scope_episodes=SCOPES,closure_sha256=shared.digest(load()[0]),course_demo=False))
    except (ValueError,RuntimeError,OSError,KeyError)as exc:p.exit(2,str(exc)+'\n')
    print(json.dumps(result,sort_keys=True))


if __name__=='__main__':main()
