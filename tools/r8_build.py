"""Assemble the fixed servo set-membership channel into one submitted file."""
import argparse
import ast
import base64
import hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FINAL = ROOT / 'experiments/r8_final'
SOURCES = {
    'contact_v3': ('mass_weight',),
    'lat2': ('critical_step', 'SignalBank'),
    'lat2b': ('regressors', 'scale_regressors', 'Channel'),
    'lat2c': ('Channel',),
    'lat2d': ('HypothesisSignal', 'membership_statistic', 'Membership', 'Channel'),
}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def selected(module, names):
    nodes = ast.parse((ROOT / f'experiments/r8/{module}.py').read_text()).body
    return [n for n in nodes if isinstance(n, (ast.FunctionDef, ast.ClassDef))
            and n.name in names]


def render():
    chunks = [(FINAL / 'FALLBACK.py').read_text(), '\n\n_R8BaseDetector = Detector\n',
              'import io as _r8_io\nimport base64 as _r8_b64\n',
              'from collections import deque\nimport mujoco\n',
              'from sim.nominal import Nominal\nfrom sim import config as C, trajectory as T\n',
              'WINDOWS = ((1.4, 1.7), (3.6, 4.0))\n']
    for module, names in SOURCES.items():
        for node in selected(module, names):
            text = ast.unparse(node)
            if module == 'lat2':
                # Membership always constructs include_L3=False. Remove only
                # the unreachable contrast-observer dependency, not its math.
                for child in ast.walk(node):
                    if isinstance(child, ast.If) and ast.unparse(child.test) == 'include_L3':
                        child.body = [ast.Pass()]
                    if isinstance(child, ast.If) and ast.unparse(child.test) == 'self.l3 is not None':
                        child.body = [ast.Assign(targets=[ast.Name(id='l3', ctx=ast.Store())],
                                                 value=ast.Constant(0.))]
                text = ast.unparse(ast.fix_missing_locations(node))
            if module == 'lat2b':
                text = text.replace('class Channel:', 'class _R8CalibratedChannel:')
            elif module == 'lat2c':
                text = text.replace('class Channel(PreviousChannel):',
                                    'class _R8L2c(_R8CalibratedChannel):')
            elif module == 'lat2d':
                text = text.replace('class Channel(L2c):', 'class _R8Channel(_R8L2c):')
                text = text.replace('self.channel = Channel(', 'self.channel = _R8Channel(')
            chunks.append('\n\n' + text + '\n')
    for kind in ('positive', 'empty'):
        blob = base64.b85encode((FINAL / f'servo_{kind}.npz').read_bytes()).decode('ascii')
        chunks.append(f'\n_R8_{kind.upper()} = {blob!r}\n')
    chunks.append('''

def _r8_model(empty):
    encoded = _R8_EMPTY if empty else _R8_POSITIVE
    with np.load(_r8_io.BytesIO(_r8_b64.b85decode(encoded)), allow_pickle=False) as z:
        return {key: z[key] for key in z.files}


class Detector(_R8BaseDetector):
    """R7 unchanged OR fixed L2d for the exact servo mode only."""

    def reset(self):
        super().reset()
        mode = getattr(getattr(self, 'ctrl', None), 'MODE', None)
        self._r8_member = (Membership(self.box_mass, self.box_com,
                                    _r8_model(self.box_mass == 0.))
                           if mode == 'servo' else None)
        self.r8_extra = False
        self.r8_base_detect = False

    def update(self, t, th, thd, tau_m, tau_s, grip):
        residual, base = super().update(t, th, thd, tau_m, tau_s, grip)
        self.r8_base_detect = bool(base)
        extra = False
        if self._r8_member is not None:
            extra, _, _, _ = self._r8_member.update(
                t, th, thd, tau_s, bool(grip),
                max(0., self._shared.last_wrench[2]), bool(base))
        self.r8_extra = bool(extra)
        return residual, bool(base or extra)
''')
    return ''.join(chunks).encode()


def build(check=False):
    content = render()
    path = FINAL / 'C1.py'
    if check:
        if not path.exists() or path.read_bytes() != content:
            raise ValueError('R8 assembly differs')
    else:
        if path.exists() and path.read_bytes() != content:
            raise ValueError('refusing to replace a different assembly')
        path.write_bytes(content)
    return hashlib.sha256(content).hexdigest()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check', action='store_true')
    print(build(parser.parse_args().check))
