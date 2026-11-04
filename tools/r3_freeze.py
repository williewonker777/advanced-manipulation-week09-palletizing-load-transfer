"""Prepare and commit-bind R3 decisions before the single sealed holdout."""
from pathlib import Path
from datetime import datetime, timezone
import argparse
import hashlib
import json
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools import r3_study as R
from tools import r3_select as S

D = ROOT / 'results/v2_protocol'
DECISION = D / 'R3_PREFREEZE_DECISION.json'
SELECTION = D / 'R3_HOLDOUT_SELECTION.json'
FREEZE = D / 'R3_FREEZE.json'


def read(path):
    return json.loads(Path(path).read_text())


def prepare():
    if DECISION.exists() or SELECTION.exists() or FREEZE.exists():
        raise RuntimeError('pre-holdout decision already recorded')
    if (D / 'R3_HOLDOUT_OPENED.json').exists() or (D / 'R3_OFFICIAL_OPENED.json').exists():
        raise RuntimeError('no selection/tuning after exposure')
    names = ['PROTOCOL_R3.json', 'R3_BOUND_STABILITY_REVIEW.json', 'R3_CHALLENGERS.json',
             'R3_REDESIGN_DECISION.json', 'R3_DEVELOPMENT_REVIEW.json',
             'R3_D4_REVIEW.json', 'R3_PERTURB_REVIEW.json', 'R3_SIGNAL_ANALYSIS.json',
             'R3_PRNE_MATH.json', 'R3_PERTURB_PARITY.json', 'R3_PERTURBATIONS.json']
    names += ['R3_D4_SCOPE_PARITY.json', 'R3_REPLAY_TIMING.json', 'R3_PREFREEZE_INTEGRITY.json']
    evidence = {name: read(D / name) for name in names}
    if not all(row['passed'] for row in evidence['R3_PERTURB_PARITY.json']['results']):
        raise RuntimeError('analysis harness parity is not qualified')
    if evidence['R3_PRNE_MATH.json']['passed'] is not True:
        raise RuntimeError('pRNE mathematical qualification failed')
    if evidence['R3_PREFREEZE_INTEGRITY.json']['passed'] is not True:
        raise RuntimeError('pre-freeze integrity qualification failed')
    if evidence['R3_BOUND_STABILITY_REVIEW.json']['protocol_gate_check']['status'] != 'PASS':
        raise RuntimeError('R2 gate identity is not qualified')
    challengers = evidence['R3_CHALLENGERS.json']['architectures']
    selection, modes = [], {}
    for mode in ('servo', 'torque'):
        challenger = challengers[mode]
        stage_status = {stage: evidence[file]['architectures'][mode]['review']['status']
                        for stage, file in (
                            ('development', 'R3_DEVELOPMENT_REVIEW.json'),
                            ('d4', 'R3_D4_REVIEW.json'), ('perturbations', 'R3_PERTURB_REVIEW.json'))}
        # This round's five settings all failed before development.  A green
        # diagnostic result must never become a second chance at adoption.
        if (challenger.get('irreversibly_ineligible_for_adoption') is not True
                or challenger.get('validation_eligible') is not False):
            raise RuntimeError('R3 negative-result freeze requires prior irreversible rejection')
        chosen = R.baseline_spec(mode)
        selection.append(chosen)
        modes[mode] = dict(challenger=challenger['selected'],
                           irreversibly_ineligible_for_adoption=True,
                           preholdout_adoption_eligible=False,
                           effective_selection=chosen, stage_status=stage_status,
                           reason='validation/forced-point stability rejection is irreversible; no diagnostic revival')
    record = dict(schema='R3_PREFREEZE_DECISION.v1',
                  recorded_utc=datetime.now(timezone.utc).isoformat(),
                  chosen_specs=selection, architectures=modes,
                  source_evidence_sha256={f'results/v2_protocol/{name}': R.sha(D / name)
                                          for name in names},
                  no_tuning_after_holdout=True, no_runner_up_or_diagnostic_revival=True,
                  new_video_pairs_required=False,
                  video_reason='neither architecture adopts a new method; existing comparison media unchanged')
    S._atomic_write(DECISION, record)
    S._atomic_write(SELECTION, dict(schema='R3_HOLDOUT_SELECTION.v1',
                                    chosen_specs=selection, final_preholdout_selection=True,
                                    decision_sha256=R.sha(DECISION)))
    return record


def source_paths():
    # The entire R2 scientific freeze remains protected, then all R3 runtime,
    # analysis, selection and tests are added.  Publication-only renderers are
    # excluded explicitly: they may still format already-frozen observations.
    names = set(read(D / 'R2_FREEZE.json')['source_files'])
    for spec in R.specifications() + [R.baseline_spec(mode) for mode in ('servo', 'torque')]:
        names.update(R.source_hashes(spec['architecture'], spec))
    for directory in ('sim', 'model'):
        names.update(path.relative_to(ROOT).as_posix()
                     for path in (ROOT / directory).rglob('*') if path.is_file())
    names.update(path.relative_to(ROOT).as_posix() for path in (ROOT / 'models').glob('*.json'))
    names.update(path.relative_to(ROOT).as_posix()
                 for path in (ROOT / 'experiments').rglob('r3*.py'))
    names.update(path.relative_to(ROOT).as_posix()
                 for path in (ROOT / 'experiments/r3_generated').glob('*.py'))
    for pattern in ('r3_*.py', 'test_r3_*.py'):
        names.update(path.relative_to(ROOT).as_posix() for path in (ROOT / 'tools').glob(pattern)
                     if not path.name.startswith(('r3_report', 'test_r3_report')))
    names.update(('my_servo.py', 'my_torque.py', 'my_detector.py', 'evaluate.py',
                  'tools/validate_submission.py', 'tools/compare_baseline.py',
                  'tools/check_integrity.py', 'tools/kit_manifest.sha256',
                  'tools/production_audit.py', 'tools/sweep.py',
                  'results/v2_protocol/PROTOCOL_R3.json',
                  'results/v2_protocol/R3_REDESIGN_DECISION.json',
                  'results/v2_protocol/R3_PREFREEZE_DECISION.json',
                  'results/v2_protocol/R3_HOLDOUT_SELECTION.json',
                  'results/v2_protocol/S4_OFFICIAL.json'))
    return sorted(names)


def freeze(source_commit):
    if FREEZE.exists() or (D / 'R3_HOLDOUT_OPENED.json').exists():
        raise RuntimeError('freeze already recorded or holdout exposed')
    decision = read(DECISION)
    if decision['chosen_specs'] != [R.baseline_spec(mode) for mode in ('servo', 'torque')]:
        raise RuntimeError('no revival of the rejected diagnostic sources')
    files = {name: R.sha(ROOT / name) for name in source_paths()}
    source_commit = subprocess.check_output(
        ['git', 'rev-parse', source_commit + '^{commit}'], cwd=ROOT, text=True).strip()
    for name, expected in files.items():
        recorded = subprocess.check_output(['git', 'show', source_commit + ':' + name], cwd=ROOT)
        if hashlib.sha256(recorded).hexdigest() != expected:
            raise RuntimeError('source commit does not contain current frozen bytes: ' + name)
    record = dict(schema='R3_FREEZE.v1', frozen_utc=datetime.now(timezone.utc).isoformat(),
                  local_commit=source_commit, protocol_sha256=R.sha(R.PROTOCOL),
                  chosen_specs=decision['chosen_specs'],
                  selection_evidence=dict(path=DECISION.relative_to(ROOT).as_posix(),
                                           sha256=R.sha(DECISION)),
                  fallback_R1_sha256=R.protocol()['fallback_R1_sha256'], source_files=files,
                  scope='all runtime/experiment/analysis/selection/protocol/weights/integrity; publication-only postprocessors excluded',
                  external_analysis_source_sha256=read(D / 'R3_PERTURBATIONS.json')['harness_sha256'],
                  tuning_after_holdout_or_official=False)
    S._atomic_write(FREEZE, record)
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepare', action='store_true')
    parser.add_argument('--source-commit')
    args = parser.parse_args()
    if args.prepare == bool(args.source_commit):
        parser.error('choose --prepare OR --source-commit')
    result = prepare() if args.prepare else freeze(args.source_commit)
    print(result['schema'], len(result.get('source_files', ())), 'frozen source files')


if __name__ == '__main__':
    main()
