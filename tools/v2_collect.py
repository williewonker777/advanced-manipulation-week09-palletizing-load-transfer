"""Record only allowed-input self-supervised residuals; no oracle targets."""
import argparse
import hashlib
import importlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.dont_write_bytecode = True
from tools.check_integrity import check_integrity
from tools.v2_learned_core import normal_features
from tools.v2_resources import wait_for_capacity, snapshot
from sim.runner import run_episode
import v1_detector


class RecordingDetector(v1_detector.Detector):
    def reset(self):
        super().reset()
        self.samples = []

    def update(self, t, th, thd, tau_m, tau_s, grip):
        result = super().update(t, th, thd, tau_m, tau_s, grip)
        packet = self._shared
        active = bool(grip and not packet.last_table_allowed)
        # No runner log or plant state is present in this sampling function.
        x = normal_features(t, self.box_mass, self.box_com) if active else np.zeros(77)
        y = packet.last_wrench[:2].copy() if active else np.zeros(2)
        self.samples.append((t, x, y, active))
        return result


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()
    check_integrity(ROOT)
    protocol_path = ROOT/'results/v2_protocol/PROTOCOL.json'
    protocol = json.loads(protocol_path.read_text())
    args.out.mkdir(parents=True, exist_ok=True)
    records = []
    for mode in protocol['architectures']:
        cls = importlib.import_module('v1_'+mode).Controller
        for split in ('train', 'validation'):
            for index, condition in enumerate(protocol[split]):
                state = wait_for_capacity()
                controller, detector = cls(), RecordingDetector()
                # Guard outside the measured algorithm, without changing sim.
                class Guarded(cls):
                    def compute(self, t, *signals):
                        if int(round(t*1000)) % 250 == 0:
                            wait_for_capacity()
                        return super().compute(t, *signals)
                # Result/log are deliberately discarded: not training targets.
                run_episode(Guarded(), box_mass=condition['mass'],
                            box_com=condition['com'], detector=detector)
                t, x, y, active = zip(*detector.samples)
                target = args.out/f'{mode}_{split}_{index:02}.npz'
                np.savez_compressed(target, t=t, x=x, y=y, active=active)
                records.append(dict(file=target.name, mode=mode, split=split,
                                    condition=condition, source='allowed-observation self-supervised target',
                                    active_samples=int(np.count_nonzero(active)),
                                    sha256=hashlib.sha256(target.read_bytes()).hexdigest(),
                                    host_before=state, host_after=snapshot()))
                print(mode, split, index, records[-1]['active_samples'], flush=True)
    data = dict(protocol_sha256=hashlib.sha256(protocol_path.read_bytes()).hexdigest(),
                production_sha256={f: hashlib.sha256((ROOT/f).read_bytes()).hexdigest()
                                   for f in ('v1_servo.py','v1_torque.py','v1_detector.py')},
                target_definition='last_wrench[:2] from v1 JTS/encoder/public-model observer ONLY',
                features_definition='normal_features; no hidden state/config/time-case lookup',
                training_seed=protocol['seeds']['training'],
                holdout_opened=False, official_conditions_used=False, records=records)
    (args.out/'manifest.json').write_text(json.dumps(data, indent=2)+'\n')
    check_integrity(ROOT)


if __name__ == '__main__':
    main()
