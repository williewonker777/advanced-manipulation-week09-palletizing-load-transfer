"""Build a deterministic, course-free, external-kit replay bundle."""
import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import sys
import zipfile

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0,str(ROOT))
from tools import r26_landing_replay as replay

ROOT_NAME='week09-r26-landing'
CLOSURE_SHA256='811ed20938dbd588f1d1cacb6f0b274365440512395234450fd898cbfe1a8976'
EXTRA=('LICENSE','THIRD_PARTY_NOTICES','tools/r26_landing_bundle.py',
       'docs/R26_REPLAY_KO.md','results/r26/landing.json','results/r26/course_identity.json',
       'results/r26/landing.png')


def digest(blob):
    return hashlib.sha256(blob).hexdigest()


def read_member(name):
    path=ROOT/name;p=PurePosixPath(name)
    if (not name or p.is_absolute() or '..' in p.parts or str(p)!=name or '\\' in name
            or p.parts[0]in ('sim','model') or p.name in ('evaluate.py','run.py','servo_controller.py','torque_controller.py','collision_detector.py')
            or not path.is_file() or any(x.is_symlink() for x in (path,*path.parents))):
        raise ValueError('unexpected bundle member: '+name)
    return path.read_bytes()


def build(out):
    if os.path.lexists(out) or not out.parent.is_dir() or any(x.is_symlink() for x in (out.parent,*out.parent.parents)) or out.resolve().is_relative_to(ROOT):
        raise ValueError('fresh ZIP outside the source and a regular directory required')
    data=replay.closure()
    if digest(read_member(replay.CLOSURE))!=CLOSURE_SHA256:
        raise ValueError('not the registered landing replay closure')
    named={n:read_member(n) for n in sorted(set(data['files'])|set(EXTRA)|{replay.CLOSURE})}
    if any(digest(named[n])!=h for n,h in data['files'].items()):raise ValueError('source changed while assembling course-free bundle')
    named['README.md']=named['docs/R26_REPLAY_KO.md']
    named['receipt.json']=(json.dumps({'schema':'r26-course-free-landing-bundle.v1','kit_or_original_image_included':False,
        'default_promoted':False,'execution':'optional original or separate custom replay with your authenticated course kit',
        'files_sha256':{n:digest(v)for n,v in sorted(named.items())}},indent=2,sort_keys=True)+'\n').encode()
    with zipfile.ZipFile(out,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=9) as dest:
        for n,v in sorted(named.items()):
            info=zipfile.ZipInfo(ROOT_NAME+'/'+n,(1980,1,1,0,0,0));info.compress_type=zipfile.ZIP_DEFLATED
            info.create_system=3;info.external_attr=0o100644<<16
            dest.writestr(info,v,compresslevel=9)
    return digest(out.read_bytes())


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--out',required=True,type=Path)
    print(build(parser.parse_args().out))
