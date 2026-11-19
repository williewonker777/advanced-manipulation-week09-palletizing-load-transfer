"""Reproduce an R31 research comparison on a copy of your original course kit."""
import argparse
import json
import os
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from types import FunctionType

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from tools import r21_replay as shared
from tools import r30_event_replay as parent

DIR = 'experiments/r31_event_replay/'
CLOSURE = DIR + 'closure.json'
PREFIX = 'week09-r31-held-events'
SCOPES = {'examples': 4, 'all-duplicates': 14, 'full-d4': 180}
MEMBERS = {DIR+'__init__.py', DIR+'runner.py', 'tools/r31_event_replay.py',
           *('tools/'+x+'.py' for x in ('check_integrity', 'd4_robustness', 'sweep')),
           *('experiments/r21_full_protection/'+x+'.py' for x in ('__init__', 'force_audit', 'runner_v2')),
           *('experiments/r22_census/'+x+'.py' for x in ('__init__', 'check')),
           *('experiments/r30_detached_lifetime/'+x+'.py' for x in ('__init__', 'candidate')),
           *('experiments/r31_held_identity/'+x for x in ('__init__.py', 'candidate.py', 'protocol.json'))}
README = """# R31 토크 충돌 경보: 같은 힘이 다시 울리는지 확인하기

수업 원본도 제출본도 아닌, 공개한 과거 입력의 재실행입니다. ZIP에 수업 자료나 새 감지기의 미리 계산한 결과는 없습니다.
MuJoCo 3.17.0과 본인이 받은 원본 키트가 필요합니다. 새 출력 디렉터리를 지정하세요.

    python -B tools/r31_event_replay.py compare --kit /path/to/icpbl_sim --scope examples --out /new/result

4개 대표 충격 및 물성이 일치하는 무충격 대조를 세 감지기로 직접 실행합니다.
all-duplicates는 기존에 늘었던 중복 14조건 전체, full-d4는 385조건(1,155회) 전체입니다.
필요하면 --jobs 1~6을 지정할 수 있습니다. result.json의 alarm_intervals는 1 ms 표본 [시작, 끝)이며
strict는 진짜 첫 상승이 있다는 뜻입니다. 중복 여부는 별도 duplicate 열을 확인하세요.
"""


def load():
    raw, files, overlay = parent.load()
    read = parent.previous.base.file
    data = read(CLOSURE)
    reg = json.loads(data)
    if (reg.get('schema') != 'r31-held-event-authored-sources.v1' or reg.get('kit_included') is not False
            or reg.get('r30_closure_sha256') != shared.digest(raw) or set(reg['files']) != MEMBERS
            or set(files) & MEMBERS):
        raise ValueError('changed source contract')
    files = {**files, parent.CLOSURE: raw}
    for name, pin in reg['files'].items():
        source = read(name)
        if shared.digest(source) != pin:
            raise ValueError('changed member: '+name)
        files[name] = source
    return data, files, sorted(set(overlay) | (MEMBERS-{'tools/r31_event_replay.py'}))


def bundle(out):
    # Use the unchanged parent archive writer with our fixed membership, metadata and title.
    fn = parent.bundle
    return FunctionType(fn.__code__, {**fn.__globals__, 'load': load, 'CLOSURE': CLOSURE,
                                      'PREFIX': PREFIX, 'README': README, 'ROOT': ROOT})(out)


def compare(a):
    import mujoco
    kit, out = shared.validate_paths(a.kit, a.out)
    raw, files, overlay = load()
    manifest = files[shared.KIT_MANIFEST]
    pins = shared.authenticate_kit(kit, manifest)
    if mujoco.__version__ != '3.17.0':
        raise ValueError('MuJoCo 3.17.0 required')
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', PYTHONNOUSERSITE='1', OPENBLAS_NUM_THREADS='1',
               OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', MUJOCO_GL='disable')
    for key in ('PYTHONPATH', 'PYTHONHOME'):
        env.pop(key, None)
    with TemporaryDirectory(prefix='r31-physical-replay-', dir=out.parent) as td:
        work, report = (Path(td)/n for n in ('work', 'report'))
        work.mkdir()
        report.mkdir()
        src = {n: files[n] for n in overlay}
        shared.install_overlay(kit, work, manifest, src, {n: shared.digest(b) for n, b in src.items()})
        try:
            shared.run([sys.executable, '-B', '-m', 'experiments.r31_event_replay.runner', a.scope, '--jobs', str(a.jobs), '--out', 'out.json'],
                       work, env, report/'run.log')
        except RuntimeError as exc:
            lines = (report/'run.log').read_text().splitlines()
            raise RuntimeError(str(exc) + ('; ' + lines[-1] if lines else '')) from exc
        data = (work/'out.json').read_bytes()
        obj = json.loads(data)
        if (obj['schema'] != 'r31-disclosed-physical-replay.v1' or obj['scope'] != a.scope or obj['course_original']
                or obj['default_changed'] or obj['arms'] != ['c3', 'headroom', 'identity']
                or sum(r['input']['kind'] == 'collision' for r in obj['rows']) != SCOPES[a.scope]
                or (a.scope == 'full-d4' and len(obj['rows']) != 385)
                or obj['episodes'] != len(obj['rows'])*3 or load()[0] != raw
                or shared.authenticate_kit(work, manifest) != pins or shared.authenticate_kit(kit, manifest) != pins):
            raise ValueError('wrong replay or changed inputs')
        (report/'result.json').write_bytes(data)
        receipt = dict(schema='r31-replay-receipt.v1', scope=a.scope, episodes=obj['episodes'],
                       result_sha256=shared.digest(data), kit_files_sha256=pins,
                       kit_manifest_sha256=shared.digest(manifest), closure_sha256=shared.digest(raw))
        (report/'receipt.json').write_text(json.dumps(receipt, indent=2, sort_keys=True)+'\n')
        shared.publish(report, out)
    return receipt


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='cmd', required=True)
    sub.add_parser('describe')
    z = sub.add_parser('build-zip')
    z.add_argument('--out', type=Path, required=True)
    c = sub.add_parser('compare')
    c.add_argument('--kit', type=Path, required=True)
    c.add_argument('--scope', choices=SCOPES, default='examples')
    c.add_argument('--out', type=Path, required=True)
    c.add_argument('--jobs', type=int, choices=range(1, 7), default=1)
    a = p.parse_args()
    try:
        obj = (compare(a) if a.cmd == 'compare' else bundle(a.out) if a.cmd == 'build-zip'
               else dict(scopes=SCOPES, course_original=False, closure_sha256=shared.digest(load()[0])))
    except (KeyError, RuntimeError, ValueError, OSError) as exc:
        p.exit(2, str(exc)+'\n')
    print(json.dumps(obj, sort_keys=True))


if __name__ == '__main__':
    main()
