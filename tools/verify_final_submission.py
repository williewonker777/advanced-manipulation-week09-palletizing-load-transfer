"""Run the authentic course evaluator on the final hand-in ZIP (R32 servo / R31 torque).

Reuses ``tools/verify_course_submission_zip.py`` unchanged: a fresh copy of the
licensed kit per run, the exact three-file ZIP, R1 controller pins, no research
imports and a kit-unchanged check.  Expected non-timing cells: servo = the recorded
official run of ``experiments.r32_servo_ownership.candidate``, torque = the pinned
R31 official record.  Usage::

    python -B tools/verify_final_submission.py --zip <final zip> --kit <icpbl_sim> --out <new dir>
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from tools import verify_course_submission_zip as v  # noqa: E402

OFFICIAL = ROOT / 'results/final/official'


def expected_cells():
    pinned, pins, _ = v.reference_cells()  # pinned R10 servo + R31 torque records
    r32 = v.normalized((OFFICIAL / 'servo_r32.txt').read_text(encoding='utf-8'))
    r31 = v.normalized((OFFICIAL / 'torque_r31.txt').read_text(encoding='utf-8'))
    r10 = v.normalized((OFFICIAL / 'servo_r10.txt').read_text(encoding='utf-8'))
    if r31 != pinned['torque'] or r10 != pinned['servo']:
        raise ValueError('recorded module runs differ from the pinned R10/R31 official records')
    if r32['control'] != r10['control']:
        raise ValueError('R32 servo changed control cells')
    return {'servo': r32, 'torque': pinned['torque']}, pins


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--zip', type=Path, required=True)
    ap.add_argument('--kit', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    a = ap.parse_args()
    kit, out = a.kit.resolve(), a.out.resolve()
    if out.exists() or out.is_relative_to(kit):
        raise SystemExit('fresh output outside the kit required')
    before = v.authenticate_kit(kit)
    files, zip_sha = v.inspect_zip(a.zip.resolve(), None)
    expected, pins = expected_cells()
    out.mkdir(parents=True)
    records = []
    for naming in ('explicit-student-names', 'course-default-aliases'):
        for mode in ('servo', 'torque'):
            rec = v.one_run(kit, files, naming, mode, expected, out)
            rec.pop('wall_seconds', None)
            records.append(rec)
            print(naming, mode, 'detected', rec['detected_of_24'], '/24', flush=True)
    if v.authenticate_kit(kit) != before:
        raise SystemExit('input kit changed')
    for mode in ('servo', 'torque'):
        same = [r['cells'] for r in records if r['mode'] == mode]
        if any(x != same[0] for x in same[1:]):
            raise SystemExit(mode + ' differs between explicit and alias names')
    report = dict(
        schema='week09-final-submission-official-verification.v1', status='pass',
        submission_zip=a.zip.name, submission_sha256=zip_sha,
        files={n: v.sha256(b) for n, b in files.items()},
        authentic_evaluate_sha256=v.EVALUATE_SHA256, kit_manifest_sha256=v.KIT_MANIFEST_SHA256,
        input_kit_unchanged=True, repository_experiments_or_tools_required_by_submission=False,
        selected={'servo_detector': 'R32 servo ownership', 'torque_detector': 'R31 held identity',
                  'controllers': 'R1 my_servo.py / my_torque.py (unchanged)', 'fallback': 'R10'},
        official_scores={'servo': records[0]['detected_of_24'], 'torque': records[1]['detected_of_24']},
        previous_submission_r10={'servo': 23, 'torque': 18},
        reference_pins=pins, executions=len(records) * 29, records=records)
    (out / 'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({'status': 'pass', 'scores': report['official_scores'], 'zip_sha256': zip_sha}))


if __name__ == '__main__':
    main()
