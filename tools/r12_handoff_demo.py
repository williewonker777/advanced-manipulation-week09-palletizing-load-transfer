"""Build a self-contained guarded torque handoff for *simulation*, never overwrite a release."""
import argparse
import ast
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
FILES = {'servo':'r1_servo.py','torque':'r1_torque.py','detector':'experiments/r10_scratch/CANDIDATE.py',
         'mixin':'experiments/r12_guarded_handoff/controller.py'}


def assemble(root=ROOT):
    config = json.loads((root/'results/r12/guarded_handoff.json').read_text())
    if config['disposition'] != 'simulation_only_do_not_replace_r10':
        raise ValueError('a frozen research-only verdict is required')
    source = {}
    for key,name in FILES.items():
        data = (root/name).read_bytes()
        if hashlib.sha256(data).hexdigest() != config['bundle_pins'][key]:
            raise ValueError('pinned source differs: '+name)
        source[key] = data.decode()
    # The release base already imports numpy and the public trajectory as T.
    # Retain exactly the measured method bodies, replacing only the base name.
    tree = ast.parse(source['mixin'])
    defs = [n for n in tree.body if isinstance(n,(ast.ClassDef,ast.FunctionDef))]
    if [n.name for n in defs] != ['curve','transfer_weight','normal_part','Controller']:
        raise ValueError('unexpected handoff dependencies')
    base = defs[-1].bases
    if len(base) != 1 or ast.unparse(base[0]) != 'r1_torque.Controller':
        raise ValueError('unexpected base')
    parts = [ast.get_source_segment(source['mixin'],n) for n in defs]
    parts[-1] = parts[-1].replace('class Controller(r1_torque.Controller):','class Controller(_R12Baseline):',1)
    torque = source['torque'] + '\n\n# Simulation-only: not the release controller.\n_R12Baseline = Controller\n\n'+'\n\n'.join(parts)+'\n'
    compile(torque,'r12_demo_torque','exec')
    return {'my_torque.py':torque,'my_servo.py':source['servo'],'my_detector.py':source['detector']}


def write_demo(out, root=ROOT):
    if out.exists() or out.is_symlink() or not out.parent.is_dir():
        raise ValueError('choose a new directory in an existing parent')
    result = assemble(root)
    out.mkdir()
    for name,data in result.items():
        with (out/name).open('x',encoding='utf-8',newline='\n') as f:
            f.write(data)


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',required=True,type=Path)
    args=parser.parse_args()
    try:
        write_demo(args.out)
    except (OSError,ValueError) as err:
        parser.error(str(err))
    print('Wrote simulation-only guarded torque example (non-release):',args.out)
