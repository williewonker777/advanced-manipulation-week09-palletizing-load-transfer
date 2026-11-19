"""Make a three-file course submission from the frozen R10 servo and R31 torque paths.

The builder links ordinary Python definitions, not pickles, eval/exec, import
hooks, or a second estimator. Every numerical function is retained verbatim.
Only authored module imports and reads of seven fixed normal-model tables
are replaced. The existing R10 release and all historical sources stay intact.
"""
from __future__ import annotations

import argparse
import ast
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import sys
import textwrap
import zipfile

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from tools import r31_event_replay as frozen

ENTRY = 'experiments.r31_held_identity.candidate'
R31 = '1e33b4698ef3678afe72cc4e286fcab5d27b2232e7bf087fdef7500ccc674764'
CLOSURE = 'bf2cc4d8281e0256bd5a8988dd5bbdd27f11ff72a77ddee92b5a293855520c3b'
RELEASE = {
    'my_servo.py': '40706968fb72ed8d7a864e44e828787ae8db0365c2927e5166e4a5b2d83bba8b',
    'my_torque.py': '8f66f8f03cde65fcc6d8ef2f11643691cd8fe2e96dad6eaeaa17a0c7aa3db371',
    'my_detector.py': '09f9a8573814a31b335b36ec5a78e0dedf262daa9039954568abd6bedd8dbc67',
}
ZIP_NAME = 'week09-final-r10-servo-r31-torque.zip'

# Only fields used at inference. No fit logs, validation examples, force labels
# or scored outputs are embedded. Numerical values are encoded losslessly by JSON.
TABLES = {
    'r21_yaw_impulse/normal_bins': ('schema', 'threshold_Nm'),
    'r22_release/model': ('schema', 'threshold_Nm'),
    'r22_tangent/model': ('bounds_N',),
    'r24_loaded/normal_response': ('schema', 'coefficients_Nm_per_N'),
    'r24_loaded/dc_drive': ('schema', 'low', 'high', 'eligible'),
    'r27_load_change/settled_model': ('joint_bounds_N', 'separate_bounds_N'),
    'r27_load_change/manifold_model': ('location', 'scale', 'nodes', 'upper_bounds_N', 'radius_squared'),
}
# Original Path-valued module constants that are just serialised calibrations.
BINDINGS = {
    ('r22_combined.core', 'BOUNDS'): 'r22_release/model',
    ('r22_guarded.candidate', 'NORMAL_BOUNDS'): 'r21_yaw_impulse/normal_bins',
    ('r24_loaded.memory', 'MODEL'): 'r24_loaded/normal_response',
    ('r24_loaded.dc', 'MODEL'): 'r24_loaded/dc_drive',
}
READS = {
    ('r21_yaw_impulse.coarse', "Path(__file__).with_name('normal_bins.json')"): 'r21_yaw_impulse/normal_bins',
    ('r22_tangent.frozen', "Path(__file__).with_name('model.json')"): 'r22_tangent/model',
    ('r27_load_change.manifold', "folder / 'settled_model.json'"): 'r27_load_change/settled_model',
    ('r27_load_change.manifold', "folder / 'manifold_model.json'"): 'r27_load_change/manifold_model',
}

def digest(b):
    return hashlib.sha256(b).hexdigest()

def path_for(module):
    return module.replace('.', '/') + '.py'

def symbol(module):
    return '_' + module.removeprefix('experiments.').replace('.', '_')

def table_name(key):
    return '_normal_table_' + key.replace('/', '_')

def inside(node):
    return isinstance(node, ast.ImportFrom) and str(node.module).startswith('experiments.')

def module_edges(node, files):
    """Yield (bound local name, source module, selected symbol or whole module)."""
    for member in node.names:
        if member.name == '*' or node.level:
            raise ValueError('non-explicit import')
        local = member.asname or member.name
        if path_for(node.module) in files:
            yield local, node.module, member.name
        else:
            target = node.module + '.' + member.name
            if path_for(target) not in files:
                raise ValueError('undeclared dependency: ' + target)
            yield local, target, None

def graph(files):
    done, visiting, order = set(), set(), []
    exports = defaultdict(set)
    exports[ENTRY].add('Detector')
    def visit(name):
        if name in done:return
        if name in visiting:raise ValueError('dependency cycle')
        visiting.add(name)
        tree=ast.parse(files[path_for(name)])
        for node in ast.walk(tree):
            if inside(node):
                for local, source, item in module_edges(node, files):
                    if item is None:
                        # The only whole-module import uses explicit named attributes.
                        attrs={v.attr for v in ast.walk(tree) if isinstance(v,ast.Attribute)
                               and isinstance(v.value,ast.Name) and v.value.id==local}
                        if not attrs:raise ValueError('unscoped module import: '+source)
                        exports[source].update(attrs)
                    else:
                        exports[source].add(item)
                    visit(source)
        visiting.remove(name);done.add(name);order.append(name)
    visit(ENTRY)
    return order,exports

def section(name, files, exports):
    """Splice at AST byte ranges, leaving original algorithm text and comments intact."""
    short=name.removeprefix('experiments.')
    raw=files[path_for(name)];tree=ast.parse(raw);starts=[0]
    for line in raw.splitlines(keepends=True):starts.append(starts[-1]+len(line))
    edits=[]
    def replace(node,text):
        lo=starts[node.lineno-1]+node.col_offset;hi=starts[node.end_lineno-1]+node.end_col_offset
        edits.append((lo,hi,text.encode()))
    class Link(ast.NodeVisitor):
        def visit_ImportFrom(self,node):
            if inside(node):
                bound=[f'{local} = {symbol(mod)}'+('.'+item if item else '')
                       for local,mod,item in module_edges(node,files)]
                replace(node,'; '.join(bound))
            elif node.module in ('__future__','pathlib'):
                # Runtime annotations in these pinned files are only float/np.ndarray.
                replace(node,'')
        def visit_Import(self,node):
            if any(x.name=='my_detector' for x in node.names):
                if len(node.names)!=1 or node.names[0].asname:raise ValueError('unexpected base import')
                replace(node,'my_detector = _submission_r10')
        def visit_Assign(self,node):
            target=node.targets[0] if len(node.targets)==1 else None
            if isinstance(target,ast.Name):
                key=(short,target.id)
                if key in BINDINGS:
                    replace(node.value,table_name(BINDINGS[key]));return
                if key==('r27_load_change.manifold','folder'):
                    replace(node,'# The two calibrated fields below were embedded at packaging.');return
            self.generic_visit(node)
        def visit_Call(self,node):
            f=node.func
            if isinstance(f,ast.Attribute) and f.attr=='read_bytes':
                src=ast.unparse(f.value);key=(short,src)
                if key in READS:replace(node,table_name(READS[key]))
                elif key in BINDINGS or key==('r24_loaded.dc','SUPPORT'):replace(node,src)
                else:raise ValueError('unexpected data access: '+str(key))
                return
            self.generic_visit(node)
    Link().visit(tree)
    end=len(raw)
    for a,b,text in sorted(edits,reverse=True):
        if b>end:raise ValueError('overlapping edit')
        raw=raw[:a]+text+raw[b:];end=a
    # No loader/hidden filesystem fallback is allowed in any numerical method.
    linked=raw.decode().strip()
    for node in ast.walk(ast.parse(linked)):
        if isinstance(node,ast.Name) and node.id in {'Path','open','__file__','__import__','exec','eval'}:
            raise ValueError('unexpected dynamic facility in '+name)
        if inside(node):raise ValueError('incomplete source linkage')
    make='_make'+symbol(name)
    result=', '.join(f'{v}={v}' for v in sorted(exports))
    return (f'# Source: {path_for(name)}\ndef {make}():\n'+textwrap.indent(linked,'    ')
            +f'\n    return _SubmissionNamespace({result})\n{symbol(name)} = {make}()\n')

DISPATCH = '''
class Detector:
    """Course adapter: use R10 servo or R31 with the supplied torque controller.

    The simulator supplies the controller and nominal models before reset.
    Reset selects one method; each physical sample updates that method once.
    Unknown/stock controllers without the shared torque contract retain R10.
    """
    SOURCE = _submission_r10.Detector.SOURCE
    API = _submission_r10.Detector.API

    def __init__(self, nom=None, nom_box=None, box_mass=0.0, box_com=(0.0, 0.0, 0.0)):
        self._impl = None
        self.box_mass = float(box_mass)
        self.box_com = tuple(box_com)
        if nom is not None and nom_box is not None:
            self.nom, self.nom_box = nom, nom_box
            self.reset()

    def reset(self):
        ctrl = vars(self).get('ctrl')
        shared = getattr(ctrl, 'shared_estimator', None)
        use_torque = (getattr(ctrl, 'MODE', None) == 'torque'
                      and getattr(shared, 'API', None) == self.API
                      and getattr(shared, 'SOURCE', None) == self.SOURCE
                      and callable(getattr(shared, 'update', None))
                      and callable(getattr(shared, 'reset', None)))
        cls = _submission_torque_class() if use_torque else _submission_r10.Detector
        engine = cls()
        engine.nom, engine.nom_box, engine.ctrl = self.nom, self.nom_box, ctrl
        engine.box_mass, engine.box_com = self.box_mass, self.box_com
        engine.reset()
        self._impl = engine
        self.selected_method = 'r31-torque' if use_torque else 'r10'
        # No routing, wrapper allocation or second update inside the 1 ms loop.
        self.update = engine.update

    def update(self, t, th, thd, tau_m, tau_s, grip):
        raise RuntimeError('attach course models and call reset() before the first sample')

    def __getattr__(self, name):
        impl = vars(self).get('_impl')
        if impl is None:
            raise AttributeError(name)
        return getattr(impl, name)
'''

def materialize():
    closure,files,_ = frozen.load()
    if digest(closure)!=CLOSURE or digest(files[path_for(ENTRY)])!=R31:raise ValueError('changed R31 closure')
    release={n:(ROOT/n).read_bytes() for n in RELEASE}
    if {n:digest(b) for n,b in release.items()}!=RELEASE or files['my_detector.py']!=release['my_detector.py']:
        raise ValueError('changed baseline controller or detector')
    order,exports=graph(files)
    tables={}
    for key,fields in TABLES.items():
        data=json.loads(files['experiments/'+key+'.json'])
        tables[key]=json.dumps({f:data[f] for f in fields},sort_keys=True,allow_nan=False,separators=(',',':'))
    sections=[section(m,files,exports[m]) for m in order]
    head=('# Week09 combined hand-in: unchanged R10 body below; R31 linkage and mode adapter follow.\n'
          '# Original course simulator, models and score sheets are not distributed in this file.\n')
    glue=("\n\n# --- Fixed, authored R31 torque inference (loaded only in torque mode). ---\n"
          "from functools import lru_cache as _submission_cache\n"
          "from types import SimpleNamespace as _SubmissionNamespace\n"
          "_submission_r10 = _SubmissionNamespace(Detector=Detector)\n\n"
          "@_submission_cache(maxsize=1)\ndef _submission_torque_class():\n")
    body='\n'.join(f'{table_name(k)} = {v!r}' for k,v in tables.items())+'\n\n'+'\n'.join(sections)
    body+='\nreturn '+symbol(ENTRY)+'.Detector\n'
    detector=head+release['my_detector.py'].decode()+glue+textwrap.indent(body,'    ')+'\n'+DISPATCH
    compile(detector,'my_detector.py','exec')
    names={path_for(m) for m in order}|{'experiments/'+k+'.json' for k in tables}
    record=dict(schema='week09-static-combination.v1',controllers=RELEASE,r31_entry=ENTRY+':Detector',
                r31_source=R31,r31_closure=CLOSURE,modules=len(order),normal_tables=len(tables),
                embedded_runtime_keys=TABLES,inputs={n:digest(files[n]) for n in sorted(names)},
                detector_sha256=digest(detector.encode()),new_detection_rules=False)
    return {**release,'my_detector.py':detector.encode()},record

def write_zip(path,blobs):
    with zipfile.ZipFile(path,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=9) as archive:
        for name,b in sorted(blobs.items()):
            info=zipfile.ZipInfo(name,(2026,1,1,0,0,0));info.create_system=3;info.external_attr=0o100644<<16
            archive.writestr(info,b,compress_type=zipfile.ZIP_DEFLATED,compresslevel=9)

def build(out):
    if out.exists() and (out.is_symlink() or not out.is_dir() or any(out.iterdir())):
        raise ValueError('choose an empty or new directory')
    blobs,receipt=materialize();out.mkdir(parents=True,exist_ok=True)
    for name,b in blobs.items():(out/name).write_bytes(b)
    z=out/ZIP_NAME;write_zip(z,blobs)
    with zipfile.ZipFile(z) as a:
        if a.testzip() or sorted(a.namelist())!=sorted(blobs) or any(a.read(n)!=v for n,v in blobs.items()):
            raise ValueError('archive differs from built sources')
    receipt.update(zip=z.name,zip_sha256=digest(z.read_bytes()),members={n:digest(v) for n,v in blobs.items()})
    (out/'build.json').write_text(json.dumps(receipt,indent=2,sort_keys=True)+'\n')
    return receipt

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();rec=build(a.out)
    print(rec['zip_sha256'],rec['zip'],rec['modules'],'modules',rec['normal_tables'],'model tables')
