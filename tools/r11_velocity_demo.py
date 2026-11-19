"""Assemble the frozen experimental velocity-ILC example, never the live entries.

Writes three model-free, self-contained *source* files for simulation to a
new directory. A candidate that loses reference detections is not a release.
"""
import argparse
import ast
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FILES = {'r1_servo':'r1_servo.py', 'r1_torque':'r1_torque.py',
         'r10_detector':'experiments/r10_scratch/CANDIDATE.py',
         'frame':'experiments/r11_datum/controller.py',
         'profile':'experiments/r11_velocity_ilc/controller.py',
         'plan':'experiments/r11_velocity_ilc/plan.json'}


def definition(source, name):
    choices = [n for n in ast.parse(source).body if isinstance(n,(ast.FunctionDef, ast.ClassDef)) and n.name == name]
    if len(choices) != 1:
        raise ValueError('expected one definition: '+name)
    return ast.get_source_segment(source, choices[0])


def assemble():
    data = json.loads((ROOT/'results/r11/velocity_learning.json').read_text())
    if data['decision'] != 'RESEARCH_ONLY_NOT_A_RELEASE':
        raise ValueError('expected a frozen research-only example')
    checked = {}
    for key, path in FILES.items():
        b = (ROOT/path).read_bytes()
        if hashlib.sha256(b).hexdigest() != data['sources'][key]:
            raise ValueError('demo source changed: '+path)
        checked[key] = b.decode()
    core = '\n\n'.join([definition(checked['frame'],'frame'),definition(checked['profile'],'basis'),
                        definition(checked['profile'],'_NormalVelocity')])
    result = {'my_detector.py':checked['r10_detector']}
    for mode in ('servo','torque'):
        record = data['modes'][mode]['model']
        amps = tuple(record['amplitudes_mm'])
        if (record['mode'] != mode or record['controller_sha256'] != data['sources']['profile']
                or record['plan_sha256'] != data['sources']['plan'] or len(amps) != 4
                or any(not math.isfinite(x) or abs(x) > 1.60000001 for x in amps)):
            raise ValueError('invalid frozen coefficients')
        text = (checked['r1_'+mode]+'\n\n# R11 research comparison only (not the submitted controller).\n'
                +'_FrozenR1 = Controller\n\n'+core+'\n\nclass Controller(_NormalVelocity, _FrozenR1):\n'
                +'    amplitudes_mm = '+repr(amps)+'\n')
        compile(text,'experimental_'+mode,'exec')
        result['my_'+mode+'.py'] = text
    return result


def write_demo(out):
    if out.is_symlink() or out.exists() or not out.parent.is_dir():
        raise ValueError('choose a new directory in an existing parent; refusing to overwrite')
    result = assemble()
    out.mkdir()
    for name, text in result.items():
        with (out/name).open('x',encoding='utf-8',newline='\n') as f:
            f.write(text)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',type=Path,required=True); args = p.parse_args()
    try:
        write_demo(args.out)
    except (OSError, ValueError) as exc:
        p.error(str(exc))
    print('Wrote simulation-only, non-release R11 velocity candidate:',args.out)
