"""Last-only unchanged-kit default-name regression with an owned process guard."""
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.dont_write_bytecode = True
from tools.r2_study import sha, require_private
from tools.r2_extended_study import verify_freeze
from tools.v2_resources import snapshot, wait_for_capacity
from tools.check_integrity import check_integrity

D = ROOT/'results/v2_protocol'


def compare(records, original):
    errors = []
    for mode, data in records.items():
        old = original['combinations'][mode+'_v2_v2']
        current = data['control_and_collisions']
        for mass, row in current['control'].items():
            b = old['control'][str(float(mass))]
            wanted = [b[k] for k in ('rms_mm', 'max_mm', 'saturation_pct', 'pick_nm', 'place_nm', 'table_n')]
            wanted += b['false_alarms']+[b['status']]
            normalized = [float(v) for v in row[:6]]+[int(v) for v in row[6:9]]+[row[9]]
            if normalized != wanted:
                errors.append(dict(architecture=mode, mass=mass, section='control', actual=row, R1=wanted))
        lookup = {(r['mass'], r['time']): r for r in old['collisions']}
        for row in current['collisions']:
            b = lookup[row['mass'], row['time']]
            wanted = [b[k] for k in ('force10','force15','force40','force100','false_max','minimum')]
            if row['cells'] != wanted:
                errors.append(dict(architecture=mode, mass=row['mass'], time=row['time'],
                                   section='collision', actual=row['cells'], R1=wanted))
    return errors


def main():
    if len(sys.argv) != 2:
        raise SystemExit('usage: r2_official_regression.py PRIVATE_LOG_DIRECTORY')
    private = require_private(sys.argv[1]); private.mkdir(parents=True, exist_ok=True)
    selection = json.loads((D/'R2_HOLDOUT_SELECTION.json').read_text())
    verify_freeze(selection)
    holdout = D/'R2_HOLDOUT.json'
    if not holdout.is_file():
        raise RuntimeError('Official regression must follow the completed new holdout')
    marker = D/'R2_OFFICIAL_STARTED.json'
    with marker.open('x') as stream:
        json.dump(dict(started_utc=datetime.now(timezone.utc).isoformat(),
                       holdout_sha256=sha(holdout), freeze_sha256=sha(D/'R2_FREEZE.json'),
                       tuning_or_repair_after_official=False), stream, indent=2)
    check_integrity(ROOT)
    before = wait_for_capacity()
    command = [sys.executable, 'tools/validate_submission.py', '--prefix', 'my',
               '--out', 'results/v2_protocol/R2_OFFICIAL_DEFAULT']
    env = os.environ.copy(); env.pop('PYTHONPATH', None)
    env.update(PYTHONDONTWRITEBYTECODE='1', OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1')
    states, pauses = [], 0
    with (private/'default_name_regression.log').open('w') as log:
        child = subprocess.Popen(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT,
                                 start_new_session=True)
        stopped = False
        try:
            while child.poll() is None:
                state = snapshot(); states.append(state)
                # The newly created process group is exclusively ours. Neither
                # a foreign PID nor a foreign process group is ever signalled.
                safe = state['mem_available_gib'] >= 6 and len(state['foreign_heavy_python_pids']) <= 4
                if not safe and not stopped:
                    os.killpg(child.pid, signal.SIGSTOP); stopped = True; pauses += 1
                elif safe and stopped:
                    os.killpg(child.pid, signal.SIGCONT); stopped = False
                time.sleep(.25)
            if stopped:
                os.killpg(child.pid, signal.SIGCONT)
            if child.returncode:
                raise RuntimeError('Official evaluation failed; no repair/re-promotion allowed')
        finally:
            if child.poll() is None and stopped:
                os.killpg(child.pid, signal.SIGCONT)
    records = json.loads((D/'R2_OFFICIAL_DEFAULT/summary.json').read_text())
    errors = compare(records, json.loads((D/'S4_OFFICIAL.json').read_text()))
    output = dict(schema='R2_OFFICIAL_REGRESSION.v1', official=True,
        source='unchanged evaluate.py in temporary unchanged-kit overlay under default entry names',
        results=records, regression_errors=errors, R1_compatibility_pass=not errors,
        source_sha256={name: sha(ROOT/name) for name in ('my_servo.py','my_torque.py','my_detector.py')},
        guard=dict(host_before=before, samples=len(states), owned_group_pause_count=pauses,
                   min_available_RAM_GiB=min(s['mem_available_gib'] for s in states),
                   maximum_heavy_python_count=max(len(s['foreign_heavy_python_pids']) for s in states),
                   foreign_processes_signalled=0),
        holdout_sha256=sha(holdout), tuning_after_exposure=False,
        failure_action='reject new method; retain immutable R1 only, never repair/re-promote')
    (D/'R2_OFFICIAL_REGRESSION.json').write_text(json.dumps(output, ensure_ascii=False, indent=2)+'\n')
    check_integrity(ROOT); verify_freeze(selection)
    print('Official exact-R1 compatibility:', not errors, 'errors:', errors)


if __name__ == '__main__':
    main()
