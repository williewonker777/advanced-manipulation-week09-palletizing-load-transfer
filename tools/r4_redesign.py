"""One registered W40 causal repair and its sealed revalidation adapter.

``prepare`` performs source archiving, deterministic materialization and a
normal-only cached replay.  ``validate`` is the separate, post-commit command
that performs the preregistered fresh full-loop cases.  It must not be called
before the prepare artifacts and revised source are committed.
"""
from pathlib import Path
import argparse
import ast
import copy
import hashlib
import importlib
import json
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.dont_write_bytecode = True

from tools import r3_study as R, r4_build as B, r4_study as S, v2_study as V, sweep
from tools.production_audit import audit_text
from sim.runner import run_episode

D = S.D
ADDENDUM = D / 'PROTOCOL_R4_REDESIGN_W40.json'
BUILD = D / 'R4_REDESIGN_BUILD.json'
AUDIT = D / 'R4_REDESIGN_NORMAL_AUDIT.json'
CERTIFICATE = D / 'R4_REDESIGN_EQUIVALENCE.json'
ARCHIVE = ROOT / 'experiments/r4_original'
GENERATED = ROOT / 'experiments/r4_generated'
REGISTERED_COMMIT = '069f855680aa5747e63efc9b9254db7bbf2ccb1c'
REGISTERED_SHA256 = S.REGISTERED_REDESIGN_SHA256
OLD_GATE = '        active = outside_table_height_band(self._shared, grip)\n'
NEW_GATE = ('        active = bool(grip and np.min(np.abs(self._shared.table_heights - '
            '(self._shared.nom.d.xpos[self._shared.nom.m.body("link_6").id] + '
            'self._shared.nom.d.xmat[self._shared.nom.m.body("link_6").id].reshape(3, 3) '
            '@ self._shared.bottom_offset)[2])) > self._shared.height_band)\n')


def _registered():
    if S.sha(ADDENDUM) != REGISTERED_SHA256:
        raise RuntimeError('registered W40 redesign bytes changed')
    value = S.read(ADDENDUM)
    if value.get('schema') != 'R4_REDESIGN_W40.v1' or value.get('variant') != 'W40_ACTIVE':
        raise RuntimeError('wrong redesign registration')
    return value


def pure_read_gate(packet, grip):
    """Read the current parent-produced nominal workspace without mutating it."""
    if not grip:
        return False
    body = packet.nom.m.body('link_6').id
    point = packet.nom.d.xpos[body] + packet.nom.d.xmat[body].reshape(3, 3) @ packet.bottom_offset
    return bool(np.min(np.abs(packet.table_heights - point[2])) > packet.height_band)


def _parameters(item):
    return item['parameters']


def _original_sources():
    built = S.read(D / 'R4_BUILT.json')
    return {Path(item['path']).name: item for item in built['generated']}


def revised_source(original, parameters):
    """Recreate registered bytes, then make the sole allowed source edit."""
    expected = (B.R1.read_text(encoding='utf-8') + '\n\n'
                + B.KERNEL.read_text(encoding='utf-8') + B._wrapper(parameters))
    if original != expected:
        raise RuntimeError('generated source no longer matches deterministic R4 build')
    if original.count(OLD_GATE) != 1:
        raise RuntimeError('expected exactly one redundant geometry-helper call')
    return original.replace(OLD_GATE, NEW_GATE, 1)


def literal_audit(text):
    tree = ast.parse(text)
    # Locate the adapter update, not the copied R1 detector update.
    adapter = next(node for node in tree.body
                   if isinstance(node, ast.ClassDef) and node.name == 'R4Detector')
    update = next(node for node in adapter.body
                  if isinstance(node, ast.FunctionDef) and node.name == 'update')
    calls = [node for node in ast.walk(update) if isinstance(node, ast.Call)]
    forbidden = []
    for call in calls:
        name = ''
        if isinstance(call.func, ast.Name):
            name = call.func.id
        elif isinstance(call.func, ast.Attribute):
            name = call.func.attr
        if name in ('outside_table_height_band', '_point_jacobian', 'mj_kinematics', 'mj_comPos'):
            forbidden.append(name)
    super_updates = sum(isinstance(call.func, ast.Attribute) and call.func.attr == 'update'
                        and isinstance(call.func.value, ast.Call)
                        and isinstance(call.func.value.func, ast.Name)
                        and call.func.value.func.id == 'super' for call in calls)
    own = {'_r4_baseline_flag', '_r4_last_extra', '_r4_last_score'}
    illegal_stores = []
    for node in ast.walk(update):
        if isinstance(node, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            for target in targets:
                if isinstance(target, ast.Attribute) and isinstance(target.value, ast.Name) \
                        and target.value.id == 'self' and target.attr not in own:
                    illegal_stores.append(target.attr)
    passed = super_updates == 1 and not forbidden and not illegal_stores and NEW_GATE in text
    return dict(passed=passed, super_update_calls=super_updates,
                forbidden_mutator_calls=forbidden, nonowned_self_stores=illegal_stores,
                gate='current nominal xpos/xmat read only')


def _normal_audit(private, spec, parameters):
    rows = []
    for mode in ('servo', 'torque'):
        baseline = S.read(D / f'R4_BASE_audit_{mode}.json')
        replay, _ = S.replay_document(baseline, private, spec,
                                      parameters[mode]['calibration'])
        for row in replay['results']:
            rows.append(dict(architecture=mode, case_id=row['case_id'],
                             extra_true_samples=row['extra_true_samples'],
                             passed=row['extra_true_samples'] == 0))
    return rows


def prepare(private):
    """No-simulation source repair and cached normal-only audit."""
    registration = _registered()
    if BUILD.exists() or AUDIT.exists():
        raise RuntimeError('redesign prepare evidence already exists')
    entries = _original_sources()
    if set(entries) != {'W40_ACTIVE.py', 'CUSUM_ACTIVE_K05.py',
                        'CUSUM_ACTIVE_K10.py', 'CUSUM_PHASE_K05.py'}:
        raise RuntimeError('unexpected original generated set')
    ARCHIVE.mkdir(parents=True, exist_ok=True)
    archive_hashes = {}
    for name, item in entries.items():
        source = GENERATED / name
        expected = (B.R1.read_text(encoding='utf-8') + '\n\n'
                    + B.KERNEL.read_text(encoding='utf-8') + B._wrapper(_parameters(item)))
        data = expected.encode()
        if hashlib.sha256(data).hexdigest() != item['sha256']:
            raise RuntimeError('original source is not deterministic: ' + name)
        target = ARCHIVE / name
        if target.exists() and target.read_bytes() != data:
            raise RuntimeError('archive collision: ' + name)
        target.write_bytes(data)
        archive_hashes[target.relative_to(ROOT).as_posix()] = S.sha(target)
        current = source.read_text(encoding='utf-8')
        if name != 'W40_ACTIVE.py' and current != expected:
            raise RuntimeError('non-W40 generated source changed: ' + name)
        if name == 'W40_ACTIVE.py' and current not in (expected, revised_source(expected, _parameters(item))):
            raise RuntimeError('W40 generated source has an unregistered edit')

    item = entries['W40_ACTIVE.py']
    target = GENERATED / 'W40_ACTIVE.py'
    original_text = (ARCHIVE / 'W40_ACTIVE.py').read_text(encoding='utf-8')
    new_text = revised_source(original_text, _parameters(item))
    static = audit_text(new_text, target.relative_to(ROOT).as_posix())
    literal = literal_audit(new_text)
    if static['errors'] or not literal['passed']:
        raise RuntimeError('revised W40 source audit failed')
    target.write_text(new_text, encoding='utf-8')
    normal_rows = _normal_audit(private, item['parameters']['servo']['spec'], item['parameters'])
    normal_pass = len(normal_rows) == 32 and all(row['passed'] for row in normal_rows)
    build = dict(schema='R4_REDESIGN_BUILD.v1', passed=True,
        redesign_protocol_sha256=S.sha(ADDENDUM), registration_commit=REGISTERED_COMMIT,
        original_generated_sha256={name: item['sha256'] for name, item in entries.items()},
        archived_sha256=archive_hashes, revised_path=target.relative_to(ROOT).as_posix(),
        revised_sha256=S.sha(target), original_w40_sha256=item['sha256'],
        calibration_sha256=S.sha(D / 'R4_CALIBRATION.json'), parameters=item['parameters'],
        thresholds_unchanged=True, exact_replacement_count=1,
        production_audit=static, literal_store_audit=literal)
    audit = dict(schema='R4_REDESIGN_NORMAL_AUDIT.v1', passed=normal_pass,
        redesign_protocol_sha256=S.sha(ADDENDUM), calibration_sha256=S.sha(D / 'R4_CALIBRATION.json'),
        revised_source_sha256=S.sha(target), collision_episodes_read=0,
        official_episodes_read=0, holdout_episodes_read=0,
        cached_normal_rows=len(normal_rows), extra_true_samples=sum(r['extra_true_samples'] for r in normal_rows),
        rows=normal_rows, thresholds_unchanged=True)
    if not normal_pass:
        raise RuntimeError('revised W40 normal audit failed')
    S.write_new(BUILD, build)
    S.write_new(AUDIT, audit)
    return dict(build=build, audit=audit)


def _fresh_baseline(mode, private, cases, bundle):
    target = S.validation_baseline_path('W40_ACTIVE', mode)
    if target.exists():
        raise RuntimeError('revalidation baseline already exists')
    rows = []
    for design in cases:
        S.check_integrity(ROOT)
        if S.runtime_bundle() != bundle:
            raise RuntimeError('runtime bundle changed before a registered revalidation case')
        row = S.base_cases('revalidation', mode, [design])[0]
        row['case_id'] = design['case_id']
        rows.append(R.run_case(row, private))
        print('revalidation baseline', mode, len(rows), flush=True)
    R.classify_group(rows)
    result = dict(schema='R4_BASE_STREAMS.v1', stage='revalidation', architecture=mode,
        protocol_sha256=S.sha(S.P), redesign_protocol_sha256=S.sha(ADDENDUM),
        registration=REGISTERED_COMMIT, source_sha256=S.science_hashes(),
        fresh_simulation=True, results=rows)
    S.write_new(target, result)
    return result


def _online(mode, private, baseline, spec, calibration, bundle):
    target = S.validation_document_path(spec['id'], mode)
    if target.exists():
        raise RuntimeError('W40 revalidation already exists')
    module = importlib.import_module('experiments.r4_generated.W40_ACTIVE')
    _, expected = S.replay_document(baseline, private, spec, calibration)
    rows = []
    for original in baseline['results']:
        S.check_integrity(ROOT)
        if S.runtime_bundle() != bundle:
            raise RuntimeError('runtime bundle changed before a fresh W40 episode')
        TC, TD = V.timed_classes(importlib.import_module('r1_' + mode).Controller, module.Detector)
        class Guarded(TC):
            def compute(self, t, *args):
                if int(round(float(t) * 1000)) % 250 == 0:
                    R.resources.wait_for_capacity()
                return super().compute(t, *args)
        ctrl, det = Guarded(), TD()
        before = R.resources.wait_for_capacity()
        result, log = run_episode(ctrl, box_mass=original['mass'], box_com=original['com'],
            detector=det, collision=original['collision'], pick_offset=original['pick_offset'])
        arrays = S.raw(original, private)
        bits = {key: bool(np.array_equal(log[key], arrays[key]))
                for key in S.protocol()['detector_gates']['physical_arrays']}
        decision = bool(np.array_equal(log['detect'], expected[original['case_id']]))
        row = copy.deepcopy(original)
        row.update(variant=spec['id'], fresh_simulation=True, physical_stream_reused=False,
            physical_arrays_bit_equal=bits, online_offline_decisions_equal=decision,
            equivalence_passed=all(bits.values()) and decision,
            timing=V.timing_summary(ctrl, det), host_before=before,
            false_alarms=int(result['false_alarms']), false_alarm_times=result['false_alarm_times'],
            detect_delay_ms=result['detect_delay_ms'], alarm_before_collision=result['alarm_before_collision'],
            tool_contact_delay_ms=result['tool_contact_delay_ms'],
            g4=V.g4_window_metrics(log, original['mass']), _detect=log['detect'])
        row.update(sweep.evaluate_control_metrics(result, log))
        name = f'revalidation_W40_ACTIVE_{mode}_{original["case_id"]:04}.npz'
        path = Path(private) / name
        if path.exists():
            raise RuntimeError('retain prior revalidation attempt')
        np.savez_compressed(path, **{k: v for k, v in log.items() if isinstance(v, np.ndarray)})
        row.update(private_raw_basename=name, private_raw_sha256=S.sha(path))
        rows.append(row)
        print('online revalidation', mode, len(rows), row['equivalence_passed'], flush=True)
    R.classify_group(rows)
    S.check_integrity(ROOT)
    if S.runtime_bundle() != bundle:
        raise RuntimeError('runtime bundle changed during revalidation')
    doc = dict(schema='R4_ONLINE_REVALIDATION.v1', architecture=mode, variant=spec['id'],
        redesign_protocol_sha256=S.sha(ADDENDUM), runtime_bundle_sha256=bundle,
        all_physical_bits_equal=all(r['equivalence_passed'] for r in rows),
        fresh_simulation=True, source_sha256=S.science_hashes(), results=rows)
    S.write_new(target, doc)
    return doc


def validate(private):
    """Run the exact stored 38-case design once per mode after prepare commit."""
    S.check_integrity(ROOT)
    registration = _registered()
    bundle = S.runtime_bundle()
    if not S.read(BUILD).get('passed') or not S.read(AUDIT).get('passed'):
        raise RuntimeError('prepare evidence is missing or failed')
    if CERTIFICATE.exists():
        raise RuntimeError('redesign equivalence already exists')
    cases = registration['revalidation']['cases']
    if len(cases) != 38 or [c['case_id'] for c in cases] != list(range(38)):
        raise RuntimeError('registered revalidation design changed')
    built = _original_sources()['W40_ACTIVE.py']
    spec = built['parameters']['servo']['spec']
    documents = {}
    for mode in ('servo', 'torque'):
        baseline = _fresh_baseline(mode, private, cases, bundle)
        documents[mode] = _online(mode, private, baseline, spec,
                                  built['parameters'][mode]['calibration'], bundle)
    if not all(doc['all_physical_bits_equal'] and len(doc['results']) == 38
               for doc in documents.values()):
        raise RuntimeError('full W40 revalidation parity failed')
    old_path = D / 'R4_EQUIVALENCE.json'
    old = S.read(old_path)
    original_sources = _original_sources()
    current_hashes = {name: S.sha(GENERATED / name) for name in original_sources}
    archived_hashes = {name: S.sha(ARCHIVE / name) for name in original_sources}
    if current_hashes['W40_ACTIVE.py'] == archived_hashes['W40_ACTIVE.py']:
        raise RuntimeError('W40 repair was not materialized')
    if any(current_hashes[name] != archived_hashes[name]
           for name in original_sources if name != 'W40_ACTIVE.py'):
        raise RuntimeError('an unchanged configuration source drifted')
    reused = []
    for row in old['results']:
        if row['variant'] == 'W40_ACTIVE':
            continue
        item = copy.deepcopy(row)
        item.update(source_reused=True, source_certificate_sha256=S.sha(old_path),
                    literal_store_free=False, adoption_eligible=False,
                    reuse_reason='generated source byte-identical to original; diagnostic only')
        reused.append(item)
    fresh = []
    for mode, doc in documents.items():
        for case_id in (0, 4):
            row = next(r for r in doc['results'] if r['case_id'] == case_id)
            fresh.append(dict(architecture=mode, variant='W40_ACTIVE', case_id=case_id,
                physical_arrays_bit_equal=row['physical_arrays_bit_equal'],
                online_offline_decisions_equal=row['online_offline_decisions_equal'],
                passed=row['equivalence_passed'], timing=row['timing'],
                raw_sha256=row['private_raw_sha256'], source_reused=False,
                literal_store_free=True, adoption_eligible=True))
    results = fresh + reused
    if S.runtime_bundle() != bundle:
        raise RuntimeError('runtime bundle changed before redesign certificate')
    S.check_integrity(ROOT)
    certificate = dict(schema='R4_REDESIGN_EQUIVALENCE.v1', passed=len(results) == 16
        and all(r['passed'] for r in results), redesign_protocol_sha256=S.sha(ADDENDUM),
        runtime_bundle_sha256=bundle, revised_w40_sha256=S.sha(GENERATED / 'W40_ACTIVE.py'),
        original_equivalence_sha256=S.sha(old_path), results=results,
        generated_source_sha256=current_hashes, archived_source_sha256=archived_hashes,
        fresh_full_loop_W40_rows=76, fresh_representative_rows=4,
        unchanged_source_reused_rows=12,
        revalidation_documents_sha256={mode: S.sha(S.validation_document_path('W40_ACTIVE', mode))
                                       for mode in ('servo', 'torque')},
        argument=('W40 calls its unchanged R1 parent exactly once, then reads the parent-produced nominal '
                  'xpos/xmat workspace without a helper call or store. All 76 revised W40 rows preserve '
                  'registered physical arrays and offline decisions. The other three byte-identical sources '
                  'reuse their original parity rows but remain adoption-ineligible for the literal store rule.'))
    if not certificate['passed']:
        raise RuntimeError('redesign equivalence certificate failed')
    S.write_new(CERTIFICATE, certificate)
    S.check_integrity(ROOT)
    return certificate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('prepare', 'validate'))
    parser.add_argument('--private-dir', type=Path, required=True)
    args = parser.parse_args()
    result = prepare(args.private_dir) if args.action == 'prepare' else validate(args.private_dir)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
