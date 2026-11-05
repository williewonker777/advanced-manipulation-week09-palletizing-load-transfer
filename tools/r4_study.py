"""Registered detector-only studies; unchanged plant streams, explicit replay.

No holdout conditions are generated at import. Cached physical streams may be
replayed only after the fresh full-loop equivalence certificate exists.
"""
from pathlib import Path
from datetime import datetime, timezone
import argparse
import copy
import hashlib
import importlib
import json
import os
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.dont_write_bytecode = True
from tools import r3_study as R, v2_study as V, sweep
from tools.check_integrity import check_integrity
from sim.runner import run_episode, collision_eval
import r1_detector

D = ROOT / 'results/v2_protocol'
P = D / 'PROTOCOL_R4.json'
REGISTRATION = '7562963b1857df7d01afc49065cd62bf17f3160a'
REGISTERED_PROTOCOL_SHA256 = 'f4f5c0cca4617901aa0be71ce513941c8cded35a308bfd7403ac49a52fb45182'
REDESIGN_PROTOCOL = D / 'PROTOCOL_R4_REDESIGN_W40.json'
REGISTERED_REDESIGN_SHA256 = '648f890dceb8de5e5d90aa837fd3525253eca0236b6a413293aef4697ece2b61'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write_new(path, value):
    path = Path(path)
    if path.exists():
        raise RuntimeError('completed evidence must not be overwritten: ' + path.name)
    R.write(path, value)


def protocol():
    # Portable evidence trees do not contain private research-branch objects.
    # Verify the immutable registration bytes here; sealed-stage execution
    # separately checks the actual committed source objects in the work tree.
    if REGISTERED_PROTOCOL_SHA256 != sha(P):
        raise RuntimeError('registered protocol changed')
    return read(P)


def science_hashes():
    paths = [P, ROOT / 'r1_detector.py', ROOT / 'r1_servo.py', ROOT / 'r1_torque.py',
             ROOT / 'tools/r4_study.py', ROOT / 'tools/r4_build.py',
             ROOT / 'experiments/r4_sequential.py']
    return {p.relative_to(ROOT).as_posix(): sha(p) for p in paths}


def redesign_active():
    """True only for the registered, byte-exact W40 causal repair."""
    return REDESIGN_PROTOCOL.is_file() and sha(REDESIGN_PROTOCOL) == REGISTERED_REDESIGN_SHA256


def equivalence_path():
    return D / ('R4_REDESIGN_EQUIVALENCE.json' if redesign_active()
                else 'R4_EQUIVALENCE.json')


def validation_document_path(variant, mode):
    if redesign_active() and variant == 'W40_ACTIVE':
        return D / f'R4_REVALIDATION_{variant}_{mode}.json'
    return D / f'R4_VALIDATION_{variant}_{mode}.json'


def validation_baseline_path(variant, mode):
    if redesign_active() and variant == 'W40_ACTIVE':
        return D / f'R4_BASE_revalidation_{mode}.json'
    return D / f'R4_BASE_validation_{mode}.json'


def redesign_decision_path():
    return D / ('R4_REDESIGN_DECISION_V2.json' if redesign_active()
                else 'R4_REDESIGN_DECISION.json')


def validation_review_path():
    return D / ('R4_REVALIDATION_REVIEW.json' if redesign_active()
                else 'R4_VALIDATION_REVIEW.json')


def literal_store_free(spec):
    """The one revised implementation is eligible only after its new proof."""
    canonical = next((item for item in protocol()['candidate_grid']
                      if item['id'] == 'W40_ACTIVE'), None)
    if not redesign_active() or spec != canonical:
        return False
    needed = [D / 'R4_REDESIGN_BUILD.json', D / 'R4_REDESIGN_NORMAL_AUDIT.json',
              D / 'R4_REDESIGN_EQUIVALENCE.json']
    if not all(path.is_file() for path in needed):
        return False
    return all(bool(read(path).get('passed')) for path in needed)


def runtime_bundle(require_committed=True):
    import subprocess
    hashes = science_hashes()
    paths = [D / 'R4_CALIBRATION.json', D / 'R4_BUILT.json',
             ROOT / 'tools/test_r4_sequential.py', ROOT / 'tools/test_r4_study.py',
             ROOT / 'tools/r4_review.py']
    generated = sorted((ROOT / 'experiments/r4_generated').glob('*.py'))
    paths += generated
    if len(generated) != 4:
        raise RuntimeError('incomplete registered runtime bundle')
    if redesign_active():
        archived = sorted((ROOT / 'experiments/r4_original').glob('*.py'))
        if len(archived) != 4:
            raise RuntimeError('incomplete original R4 source archive')
        paths += [REDESIGN_PROTOCOL, ROOT / 'tools/r4_redesign.py',
                  ROOT / 'tools/test_r4_redesign.py', D / 'R4_REDESIGN_BUILD.json',
                  D / 'R4_REDESIGN_NORMAL_AUDIT.json'] + archived
    for path in paths:
        hashes[path.relative_to(ROOT).as_posix()] = sha(path)
    if require_committed:
        for name, digest in hashes.items():
            data = subprocess.check_output(['git', 'show', 'HEAD:' + name], cwd=ROOT)
            if hashlib.sha256(data).hexdigest() != digest:
                raise RuntimeError('runtime bundle not committed: ' + name)
    return hashes


def require_equivalence():
    bundle = runtime_bundle()
    certificate = read(equivalence_path())
    if not certificate['passed'] or certificate['runtime_bundle_sha256'] != bundle:
        raise RuntimeError('missing, stale or failed full-loop equivalence certificate')
    return bundle


def validate_normal_document(doc, stage, mode):
    expected = base_cases(stage, mode)
    if (doc.get('schema') != 'R4_BASE_STREAMS.v1' or doc.get('stage') != stage
            or doc.get('architecture') != mode or doc.get('protocol_sha256') != sha(P)
            or doc.get('registration') != REGISTRATION or doc.get('source_sha256') != science_hashes()
            or len(doc.get('results', [])) != len(expected)):
        raise RuntimeError('normal data provenance/count differs from registration')
    for row, design in zip(doc['results'], expected):
        if any(row.get(k) != v for k, v in design.items()):
            raise RuntimeError('normal condition/order differs from registered design')


def base_cases(stage, mode, explicit=None):
    p = protocol()
    if explicit is not None:
        design = explicit
    elif stage in ('fit', 'calibration', 'audit'):
        design = [dict(c, kind='reference', collision=None, pick_offset=None)
                  for c in p['calibration']['normal'][stage]['conditions']]
    else:
        design = p[stage]['cases']
    return [R.case_row(R.baseline_spec(mode), c, i, stage='r4_' + stage,
                       collision=c.get('collision'), pick_offset=c.get('pick_offset'),
                       **{k: v for k, v in c.items() if k not in
                          ('mass', 'com', 'collision', 'pick_offset', 'case_id', 'kind')})
            for i, c in enumerate(design)]


def collect(stage, private, explicit=None, harness=None):
    """Fresh R1 full-loop episodes; never emulate or patch the kit."""
    check_integrity(ROOT)
    if stage in ('fit', 'calibration', 'audit') and (explicit is not None or harness is not None):
        raise RuntimeError('normal stages accept only exact registered normal conditions')
    if stage not in ('fit', 'calibration', 'audit'):
        require_equivalence()
    hashes = science_hashes()
    outputs = []
    for mode in ('servo', 'torque'):
        target = D / f'R4_BASE_{stage}_{mode}.json'
        if target.exists():
            raise RuntimeError('stage already complete: ' + target.name)
        rows = []
        for row in base_cases(stage, mode, explicit):
            case_harness = harness if row.get('perturb') is not None else None
            rows.append(R.run_case(row, private, harness=case_harness, perturb=row.get('perturb')))
            print(stage, mode, len(rows), 'FA', rows[-1]['false_alarms'], flush=True)
        # Collision interpretation requires matched reference streams. Noise
        # references contain no collision and need no extra pairing.
        R.classify_group(rows)
        if science_hashes() != hashes:
            raise RuntimeError('science source changed while collecting')
        record = dict(schema='R4_BASE_STREAMS.v1', stage=stage, architecture=mode,
                      protocol_sha256=sha(P), registration=REGISTRATION,
                      source_sha256=hashes, fresh_simulation=True, results=rows)
        write_new(target, record)
        outputs.append(record)
    return outputs


def raw(row, private):
    private = R.require_private(private)
    path = private / row['private_raw_basename']
    if path.resolve().parent != private.resolve():
        raise RuntimeError('unsafe raw basename')
    if sha(path) != row['private_raw_sha256']:
        raise RuntimeError('raw SHA differs: ' + path.name)
    with np.load(path, allow_pickle=False) as data:
        return {k: data[k] for k in data.files}


_PREDICTIONS = {}


def prediction(times, mass, com, mode):
    """Exact scalar runtime arithmetic, cached only by allowed known inputs."""
    times = np.asarray(times)
    if len(times) != 6000 or not np.array_equal(times, np.arange(6000) * .001):
        raise RuntimeError('replay requires the exact full 1 ms grid')
    key = (mode, float(mass), tuple(com))
    if key not in _PREDICTIONS:
        p = r1_detector.MODEL_PARAMETERS[mode]
        mean, scale, weights = np.asarray(p['mean']), np.asarray(p['scale']), np.asarray(p['weights'])
        _PREDICTIONS[key] = np.vstack([
            np.clip((r1_detector.normal_features(float(t), mass, com) - mean) / scale, -8., 8.) @ weights
            for t in times])
    return _PREDICTIONS[key]


def scores(row, arrays):
    errors = arrays['estimated_wrench'][:, :2] - prediction(
        arrays['obs_t'], row['mass'], row['com'], row['architecture'])
    return np.asarray([float(np.linalg.norm(error)) for error in errors])


def monitor_trace(spec, calibration, times, values, active):
    from experiments.r4_sequential import SequentialMonitor
    monitor = SequentialMonitor(spec, calibration)
    extra = np.zeros(len(times), bool)
    statistics = np.zeros(len(times))
    phases = np.zeros(len(times), int)
    for i, (t, value, enabled) in enumerate(zip(times, values, active)):
        extra[i] = monitor.update(float(t), float(value), bool(enabled))
        statistics[i], phases[i] = monitor.statistic, monitor.phase
    return extra, statistics, phases


def calibrate(private):
    """Fit/freeze only new no-collision signals; no collision files opened."""
    from tools.r4_build import materialize
    p = protocol()
    hashes = science_hashes()
    if (D / 'R4_CALIBRATION.json').exists():
        raise RuntimeError('calibration already frozen')
    output = dict(schema='R4_CALIBRATION.v1', protocol_sha256=sha(P),
                  source_sha256=science_hashes(), collision_episodes_read=0,
                  official_episodes_read=0, holdout_episodes_read=0, parameters={}, audit=[])
    for mode in ('servo', 'torque'):
        stages = {}
        for stage in ('fit', 'calibration', 'audit'):
            doc = read(D / f'R4_BASE_{stage}_{mode}.json')
            validate_normal_document(doc, stage, mode)
            stages[stage] = [(row, raw(row, private)) for row in doc['results']]
            if any(r['kind'] != 'reference' or r['collision'] is not None for r, _ in stages[stage]):
                raise RuntimeError('normal-only calibration firewall violated')
        for spec in p['candidate_grid']:
            n = 5 if spec['phase_conditioning'] else 1
            buckets = [[] for _ in range(n)]
            for row, a in stages['fit']:
                value = scores(row, a)
                phase = np.searchsorted(p['monitor']['phase_boundaries_s'][1:-1], a['obs_t'], side='right')
                for j in range(n):
                    mask = phase == j if n == 5 else a['active'].astype(bool)
                    buckets[j].extend(value[mask].tolist())
            location = [float(np.median(b)) for b in buckets]
            scale = [max(.25, float(1.4826 * np.median(np.abs(np.asarray(b) - loc))))
                     for b, loc in zip(buckets, location)]
            if spec['id'] == 'W40_ACTIVE':
                location, scale = [0.], [1.]
            cal = dict(location=location, scale=scale, threshold=[float('inf')] * n)
            maxima = np.zeros(n)
            for row, a in stages['calibration']:
                _, stat, phase = monitor_trace(spec, cal, a['obs_t'], scores(row, a), a['active'])
                for j in range(n):
                    maxima[j] = max(maxima[j], float(np.max(stat[phase == j] if n == 5 else stat)))
            cal['threshold'] = (1.05 * maxima + 1e-9).tolist()
            key = mode + ':' + spec['id']
            output['parameters'][key] = dict(spec=spec, calibration=cal)
            for row, a in stages['audit']:
                extra, _, _ = monitor_trace(spec, cal, a['obs_t'], scores(row, a), a['active'])
                output['audit'].append(dict(architecture=mode, variant=spec['id'], case_id=row['case_id'],
                    R1_FA=row['false_alarms'], extra_true_samples=int(extra.sum()),
                    extra_rises=int(np.sum(extra & ~np.r_[False, extra[:-1]])), passed=not bool(extra.any())))
    output['audit_passed'] = all(row['passed'] for row in output['audit'])
    paths = []
    for spec in p['candidate_grid']:
        params = {mode: output['parameters'][mode + ':' + spec['id']] for mode in ('servo', 'torque')}
        paths.append(materialize(params, ROOT / 'experiments/r4_generated' / (spec['id'] + '.py')))
    if science_hashes() != hashes:
        raise RuntimeError('science source changed during calibration; no calibration published')
    # Deterministic materialization is validated before the completion record.
    # Partial generated files can be resumed only when their bytes are exact.
    write_new(D / 'R4_CALIBRATION.json', output)
    write_new(D / 'R4_BUILT.json', dict(source_sha256=science_hashes(), generated=paths,
                                      calibration_sha256=sha(D / 'R4_CALIBRATION.json')))
    print('normal-only audit', output['audit_passed'], flush=True)
    return output


def replay_document(document, private, spec, calibration):
    """Only the new monitor is replayed; physical metrics remain original bytes."""
    rows, flags = [], {}
    for original in document['results']:
        row = copy.deepcopy(original)
        a = raw(original, private)
        extra, _, _ = monitor_trace(spec, calibration, a['obs_t'], scores(original, a), a['active'])
        detected = np.asarray(a['detect'], bool) | extra
        e_log = dict(a, detect=detected)
        e_log['r'] = a['r'] if 'r' in a else a['joint_residual']
        evaluated = collision_eval(e_log, len(a['obs_t']), original['collision'])
        row.update({k: R.json_safe(v) for k, v in evaluated.items()})
        row.update(variant=spec['id'], physical_stream_reused=True,
                   R1_false_alarms=original['false_alarms'], extra_true_samples=int(extra.sum()),
                   _detect=detected, physical_bit_identity='unchanged R1 stream; fresh equivalence certificate and no-feedback argument')
        flags[row['case_id']] = detected
        rows.append(row)
    # The perturbation matrix has no matched no-collision trajectory for its
    # six collision rows. Preserve evaluator definition and mark that scope.
    refs = {R.reference_key(r): r['_detect'] for r in rows if r['kind'] == 'reference'}
    for row in rows:
        if row['collision'] is not None:
            ref = refs.get(R.reference_key(row))
            if ref is not None:
                row.update(sweep.classify_collision(row, ref, row['collision'][0]))
            else:
                row['collision_outcome'] = ('masked' if row['alarm_before_collision'] else
                    'detected' if row['detect_delay_ms'] is not None else 'missed')
                row['classification_note'] = 'evaluator only; no matched unforced perturbation reference'
        row.pop('_detect', None)
    return dict(schema='R4_MONITOR_REPLAY.v1', architecture=document.get('architecture'),
                variant=spec['id'], baseline_document_stage=document.get('stage'),
                protocol_sha256=sha(P), calibration_sha256=sha(D / 'R4_CALIBRATION.json'),
                source_sha256=science_hashes(), fresh_simulation=False, results=rows), flags


def replay_stage(stage, private, documents=None):
    bundle = require_equivalence()
    calibration = read(D / 'R4_CALIBRATION.json')
    for mode in ('servo', 'torque'):
        doc = documents[mode] if documents else read(D / f'R4_BASE_{stage}_{mode}.json')
        for spec in protocol()['candidate_grid']:
            result, _ = replay_document(doc, private, spec,
                        calibration['parameters'][mode + ':' + spec['id']]['calibration'])
            write_new(D / f'R4_{stage.upper()}_{spec["id"]}_{mode}.json', result)
            print('replay', stage, mode, spec['id'], len(result['results']), flush=True)
    if runtime_bundle() != bundle:
        raise RuntimeError('runtime bundle drift during replay')


def equivalence(private):
    """Fresh full-loop timing and bit-equivalence for every configuration/mode."""
    check_integrity(ROOT)
    p = protocol(); cal = read(D / 'R4_CALIBRATION.json')
    if not cal['audit_passed']:
        # Data are still measured diagnostically, but audit failure can never
        # become eligible simply because this parity test passes.
        print('normal audit failed: parity remains diagnostic', flush=True)
    hashes = science_hashes()
    bundle = runtime_bundle()
    # Calibration and generated source must be committed before first collision.
    import subprocess
    committed = [D / 'R4_CALIBRATION.json'] + list((ROOT / 'experiments/r4_generated').glob('*.py'))
    for path in committed:
        data = subprocess.check_output(['git', 'show', 'HEAD:' + path.relative_to(ROOT).as_posix()], cwd=ROOT)
        if hashlib.sha256(data).hexdigest() != sha(path):
            raise RuntimeError('pre-collision artifacts not committed')
    records = []
    for mode in ('servo', 'torque'):
        # One new reference and one new collision, defined by validation.
        design = [p['validation']['cases'][0], p['validation']['cases'][4]]
        base = []
        for c in design:
            row = base_cases('equivalence', mode, [c])[0]
            row['case_id'] = c['case_id']
            base.append(R.run_case(row, private))
        baseline = dict(results=[])
        for row in base:
            row.pop('_detect', None); baseline['results'].append(row)
        for spec in p['candidate_grid']:
            module = importlib.import_module('experiments.r4_generated.' + spec['id'])
            param = cal['parameters'][mode + ':' + spec['id']]['calibration']
            expected, expected_flags = replay_document(baseline, private, spec, param)
            for row in baseline['results']:
                TC, TD = V.timed_classes(importlib.import_module('r1_' + mode).Controller, module.Detector)
                class Guarded(TC):
                    def compute(self, t, *args):
                        if int(round(t * 1000)) % 250 == 0:
                            R.resources.wait_for_capacity()
                        return super().compute(t, *args)
                ctrl, det = Guarded(), TD()
                before = R.resources.wait_for_capacity()
                result, log = run_episode(ctrl, box_mass=row['mass'], box_com=row['com'], detector=det,
                                          collision=row['collision'], pick_offset=row['pick_offset'])
                arrays = raw(row, private)
                keys = p['detector_gates']['physical_arrays']
                bit = {k: bool(np.array_equal(log[k], arrays[k])) for k in keys}
                decision = bool(np.array_equal(log['detect'], expected_flags[row['case_id']]))
                timing = V.timing_summary(ctrl, det)
                records.append(dict(architecture=mode, variant=spec['id'], case_id=row['case_id'],
                    physical_arrays_bit_equal=bit, online_offline_decisions_equal=decision,
                    passed=all(bit.values()) and decision, timing=timing, host_before=before))
                # Keep all measured timing outliers; actual validation stream
                # and model inputs already reside in the immutable raw record.
                target = Path(private) / f'equivalence_{spec["id"]}_{mode}_{row["case_id"]}.npz'
                if target.exists():
                    raise RuntimeError('equivalence attempt exists')
                np.savez_compressed(target, **{k: v for k, v in log.items() if isinstance(v, np.ndarray)})
                records[-1]['raw_sha256'] = sha(target)
                print('equivalence', mode, spec['id'], row['case_id'], records[-1]['passed'], flush=True)
    if science_hashes() != hashes:
        raise RuntimeError('science source changed during parity')
    if runtime_bundle() != bundle:
        raise RuntimeError('runtime bundle changed during equivalence')
    output = dict(schema='R4_EQUIVALENCE.v1', source_sha256=hashes, runtime_bundle_sha256=bundle, results=records,
                  passed=all(r['passed'] for r in records), fresh_full_loop_simulations=len(records) + 4,
                  argument='Detector return is logged only, never feeds unchanged controller. Wrapper calls unchanged parent once and mutates only owned monitor fields. Finite full-loop physical/decision equivalence checks supplement that structural argument; no global or real safety claim.')
    write_new(D / 'R4_EQUIVALENCE.json', output)
    if not output['passed']:
        raise RuntimeError('online/offline or physical parity failed')


def validation_online(private):
    """All 38 registered validation cases, all configurations, serial timing."""
    bundle = require_equivalence()
    private = R.require_private(private)
    cal = read(D / 'R4_CALIBRATION.json')
    for mode in ('servo', 'torque'):
        baseline = read(D / f'R4_BASE_validation_{mode}.json')
        for spec in protocol()['candidate_grid']:
            target = D / f'R4_VALIDATION_{spec["id"]}_{mode}.json'
            if target.exists():
                raise RuntimeError('validation already complete')
            module = importlib.import_module('experiments.r4_generated.' + spec['id'])
            _, expected = replay_document(baseline, private, spec,
                            cal['parameters'][mode + ':' + spec['id']]['calibration'])
            rows = []
            for original in baseline['results']:
                TC, TD = V.timed_classes(importlib.import_module('r1_' + mode).Controller, module.Detector)
                class Guarded(TC):
                    def compute(self, t, *args):
                        if int(round(t * 1000)) % 250 == 0:
                            R.resources.wait_for_capacity()
                        return super().compute(t, *args)
                ctrl, det = Guarded(), TD()
                before = R.resources.wait_for_capacity()
                result, log = run_episode(ctrl, box_mass=original['mass'], box_com=original['com'],
                    detector=det, collision=original['collision'], pick_offset=original['pick_offset'])
                a = raw(original, private)
                bit = {k: bool(np.array_equal(log[k], a[k])) for k in protocol()['detector_gates']['physical_arrays']}
                decision = bool(np.array_equal(log['detect'], expected[original['case_id']]))
                row = copy.deepcopy(original)
                row.update(variant=spec['id'], fresh_simulation=True, physical_stream_reused=False,
                    physical_arrays_bit_equal=bit, online_offline_decisions_equal=decision,
                    equivalence_passed=all(bit.values()) and decision,
                    timing=V.timing_summary(ctrl, det), host_before=before,
                    false_alarms=int(result['false_alarms']), false_alarm_times=result['false_alarm_times'],
                    detect_delay_ms=result['detect_delay_ms'], alarm_before_collision=result['alarm_before_collision'],
                    tool_contact_delay_ms=result['tool_contact_delay_ms'],
                    g4=V.g4_window_metrics(log, original['mass']), _detect=log['detect'])
                row.update(sweep.evaluate_control_metrics(result, log))
                name = f'validation_{spec["id"]}_{mode}_{original["case_id"]:04}.npz'
                path = private / name
                if path.exists():
                    raise RuntimeError('retain prior validation attempt, do not overwrite')
                np.savez_compressed(path, **{k: v for k, v in log.items() if isinstance(v, np.ndarray)})
                row.update(private_raw_basename=name, private_raw_sha256=sha(path))
                rows.append(row)
                print('online validation', mode, spec['id'], len(rows), row['equivalence_passed'], flush=True)
            R.classify_group(rows)
            if runtime_bundle() != bundle:
                raise RuntimeError('runtime bundle changed during validation')
            write_new(target, dict(schema='R4_ONLINE_VALIDATION.v1', architecture=mode, variant=spec['id'],
                runtime_bundle_sha256=bundle, all_physical_bits_equal=all(r['equivalence_passed'] for r in rows),
                fresh_simulation=True, source_sha256=science_hashes(), results=rows))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('action', choices=('collect', 'calibrate', 'equivalence', 'replay', 'validation-online'))
    ap.add_argument('--stage', choices=('fit', 'calibration', 'audit', 'validation', 'development'))
    ap.add_argument('--private-dir', type=Path, required=True)
    args = ap.parse_args()
    if args.action == 'collect': collect(args.stage, args.private_dir)
    elif args.action == 'calibrate': calibrate(args.private_dir)
    elif args.action == 'equivalence': equivalence(args.private_dir)
    elif args.action == 'validation-online': validation_online(args.private_dir)
    else: replay_stage(args.stage, args.private_dir)


if __name__ == '__main__':
    main()
