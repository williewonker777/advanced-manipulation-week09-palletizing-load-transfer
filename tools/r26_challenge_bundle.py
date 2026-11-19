"""Export only the validated student independent replay, not the original kit or outcomes."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import zipfile

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0,str(ROOT))
from experiments.r26_challenge_portable import support

PREFIX='week09-r26-yaw'
PATHS=('results/r26/exposed_choice.json','results/r26/candidate_lock.json','results/r26/independent_closure.json')
HELP="""# R26 토크 감지 독립 조건 재생

수업 원본 데모가 아니다. R25 연구 기준과 선택 후 고정한 회전 경보 방식을
같은 외력과 무외력 대조에서 비교한다. 수업 소스·모델은 포함하지 않았다.
수업 키트의 MuJoCo 3.17.0 환경에서 이 폴더로 이동해 아래를 실행한다.

```bash
KIT=/path/to/original/icpbl_sim
OUT=$(mktemp -d)
python -B tools/r26_challenge_replay.py --kit "$KIT" --jobs 2 \\
  --selection results/r26/exposed_choice.json --lock results/r26/candidate_lock.json \\
  --closure results/r26/independent_closure.json --out "$OUT/replay"
```

한 번에 실제 물리 156에피소드를 수행한다. `first.json`의 개별 결과·실패 및
`verification.json`의 무결성 검사를 구분한다. 기존 결과를 덮어쓰지 않는다.
검증 프로그램이 정상 종료해도 과학적 합격·기본 제출 승격·실기 안전을 뜻하지 않는다.
"""


def build(out):
    if os.path.lexists(out) or not out.parent.is_dir() or any(x.is_symlink()for x in (out.parent,*out.parent.parents)) or out.resolve().is_relative_to(ROOT):
        raise ValueError('fresh output ZIP outside the source tree required')
    release=support.read_release(*(ROOT/p for p in PATHS));members={}
    required=set(release['files']) | set(PATHS) | {'tools/r26_challenge_bundle.py','LICENSE','THIRD_PARTY_NOTICES'}
    for name in sorted(required):
        p=ROOT/name; rel=Path(name)
        if (rel.is_absolute() or '..' in rel.parts or rel.parts[0]in ('sim','model')
                or rel.name in ('evaluate.py','run.py','servo_controller.py','torque_controller.py','collision_detector.py')
                or not p.is_file() or any(part.is_symlink() for part in (p,*p.parents))):
            raise ValueError('not a course-free authored member: '+name)
        members[name]=p.read_bytes()
        if name in release['files'] and support.digest(members[name])!=release['files'][name]:
            raise ValueError('source changed: '+name)
    members['README.md']=HELP.encode()
    members['bundle.json']=(json.dumps({'schema':'r26-independent-replay-bundle.v1','course_kit_included':False,'result_included':False,
        'default_promoted':False,'selected':release['entrypoint'],'members_sha256':{n:support.digest(v)for n,v in sorted(members.items())}},
        indent=2,sort_keys=True)+'\n').encode()
    with zipfile.ZipFile(out,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=9)as z:
        for n,v in sorted(members.items()):
            info=zipfile.ZipInfo(PREFIX+'/'+n,(1980,1,1,0,0,0));info.compress_type=zipfile.ZIP_DEFLATED
            info.create_system=3;info.external_attr=0o100644<<16
            z.writestr(info,v,compresslevel=9)
    return hashlib.sha256(out.read_bytes()).hexdigest()


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,required=True)
    print(build(p.parse_args().out))
