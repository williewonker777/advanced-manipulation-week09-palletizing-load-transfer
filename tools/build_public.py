"""Curate our code/results only. No Git initialization or network operations."""
from pathlib import Path
import hashlib
import json
import subprocess

ROOT=Path(__file__).resolve().parents[1]
DEST=ROOT/'publish/repo'


def portable_json_bytes(data):
    """Omit process identities and compact large JSON without losing values."""
    def walk(value):
        if isinstance(value, dict):
            result = {}
            for key, item in value.items():
                if key == "foreign_heavy_python_pids":
                    result["foreign_heavy_python_count"] = len(item)
                else:
                    result[key] = walk(item)
            return result
        if isinstance(value, list):
            return [walk(item) for item in value]
        return value
    original = json.loads(data)
    portable = walk(original)
    identifiers_changed = portable != original
    compact = (isinstance(portable, dict) and
               portable.get("metadata", {}).get("stage") == "d4") or len(data) > 4*1024**2
    if not identifiers_changed and not compact:
        return data
    if isinstance(portable, dict):
        portable["publication_note"] = (
            ("Raw process IDs omitted. " if identifiers_changed else "") +
            ("Large JSON whitespace compacted losslessly. " if compact else "") +
            "Every case, numeric metric, eigenvalue, load/RAM summary and timing outlier is retained. "
            "Evidence hashes elsewhere may name the original work record; "
            "published-byte hashes are in artifacts/SHA256SUMS.")
        portable["private_raw_sha256"] = hashlib.sha256(data).hexdigest()
    return (json.dumps(portable, ensure_ascii=False, indent=None if compact else 2,
                       separators=(",", ":") if compact else None, allow_nan=False)+"\n").encode()


def rebind_public_report_manifest(data, selected):
    """Bind report entries after PID removal or lossless JSON compaction.

    Work/raw artifact hashes remain explicit provenance. No score, case,
    outlier, model, or source code is removed or numerically changed here.
    """
    manifest = json.loads(data)
    changed = False
    for group in manifest.get('artifacts', {}).values():
        for item in group if isinstance(group, list) else [group]:
            name = item['path']
            if name not in selected:
                raise ValueError('Report artifact missing from curated output: '+name)
            digest = hashlib.sha256(selected[name]).hexdigest()
            if item['sha256'] != digest:
                item['prepublication_work_sha256'] = item['sha256']
                item['sha256'] = digest
                changed = True
    if not changed:
        return data
    manifest['prepublication_work_manifest_sha256'] = manifest.get(
        'private_raw_sha256', hashlib.sha256(data).hexdigest())
    manifest['publication_byte_binding'] = ('Artifact SHA256 entries name published bytes after '
        'process-identifier sanitization or lossless whitespace compaction; prepublication work '
        'hashes are retained separately. Every case, score, eigenvalue and timing outlier is preserved.')
    return (json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False)+'\n').encode()


def main():
    DEST.mkdir(parents=True,exist_ok=True)
    selected={}
    def copy(name, dest=None):
        data = (ROOT/name).read_bytes()
        if str(name).endswith('.json'):
            data = portable_json_bytes(data)
        selected[dest or name] = data
    for name in ('my_servo.py','my_torque.py','my_detector.py'):
        copy(name)
    tools=('check_integrity','evaluate_checked','sweep','compare_baseline','production_audit',
           'd4_robustness','final_evaluation','presentation_figures','final_report','shared_ablation',
           'shared_estimator_core','sync_shared_core','timing_distribution','audit_public',
           'detector_fast_calibrate','test_audit_public','test_d4_robustness','test_final_evaluation',
           'test_detector','test_production_audit','test_production_contract','test_shared_estimator',
           'test_sweep','test_integrity','test_control_d2',
           'render_comparison','test_render_comparison','validate_media','test_validate_media')
    for tool in tools:copy('tools/'+tool+'.py')
    for name in ('v1_servo.py','v1_torque.py','v1_detector.py',
                 'v2_detector_candidate.py','v2_detector_guarded.py','v2_detector_geometry.py'):
        copy(name)
    for name in ('r1_servo.py','r1_torque.py','r1_detector.py'):
        copy(name)
    v2_tools = ('build_public','validate_submission','v2_resources','v2_learned_core','v2_collect',
                'v2_train','v2_build_candidate','v2_geometry_audit','v2_study',
                'v2_select','v2_select_final','v2_domain_study','v2_official','v2_trace','v2_report',
                'v2_timing_diagnostics','test_v2_timing_diagnostics',
                'render_v2_comparison','test_render_v2_comparison',
                'test_v2_learning','test_v2_normal','test_v2_study',
                'test_v2_domain','test_v2_gate_fix','test_v2_report','test_build_public_v2')
    for name in v2_tools:
        copy('tools/'+name+'.py')
    r2_tools = ('r2_p0','r2_linear','r2_plant_fd','r2_stability','r2_study',
                'r2_extended_study','r2_select','r2_decide','r2_matched_ridge','r2_report',
                'r2_report_manifest',
                'r2_official_regression','test_r2_official_regression',
                'render_r2_comparison','test_r2_p0','test_r2_linear',
                'test_r2_plant_fd','test_r2_stability','test_r2_study',
                'test_r2_extended_study','test_r2_contact','test_r2_prne',
                'test_r2_robust','test_r2_matched_ridge','test_r2_report',
                'test_render_r2_comparison')
    for name in r2_tools:
        copy('tools/'+name+'.py')
    for path in (ROOT/'models').glob('*.json'):
        copy(path.relative_to(ROOT).as_posix())
    for path in (ROOT/'results/v2_protocol').rglob('*'):
        if path.is_file() and path.suffix in {'.json','.txt'}:
            copy(path.relative_to(ROOT).as_posix())
    copy('report/PRESENTATION_V2.md')
    copy('report/PRESENTATION_R2.md')
    copy('results/presentation_R1.pdf')
    copy('tools/kit_manifest.sha256')
    copy('tools/SWEEP.md')
    for path in (ROOT/'experiments').glob('*.py'):
        copy(path.relative_to(ROOT).as_posix())
    for path in (ROOT/'docs').glob('*.md'):
        copy(path.relative_to(ROOT).as_posix())
    results=('FINAL_OFFICIAL.json','FINAL_FREEZE.json','FINAL_GOALS.json','FINAL_INPUT_AUDIT.json',
             'FINAL_SERVO_PARITY.json','FINAL_TORQUE_PARITY.json','FINAL_SUMMARY.md',
             'FINAL_TIMING_DISTRIBUTION.json','D4_ROBUSTNESS_PROTOCOL.json','D4_ROBUST_my_servo.json',
             'D4_ROBUST_my_torque.json','D4_SHARED_ABLATION.json','D4_SUPPORT_FEASIBILITY.json',
             'D4_CONTROL_FAILURES.csv','D4_PERTURBATIONS.json','D3_FAST_PROTOCOL.json',
             'D3_FAST_CALIBRATION.json','D3_STATE_PROTOCOL.json','D3_STATE_CALIBRATION.json',
             'D3_velocity_oracle_analysis.json','D5_TRACE_METRICS.json','ENV.md','presentation.pdf')
    for name in results:copy('results/'+name)
    # Remove the private raw-log location note, not measurements/method/limits.
    content=(ROOT/'results/D4_PERTURBATIONS.md').read_text().split('\nRaw evidence:')[0].rstrip()+'\n'
    selected['results/D4_PERTURBATIONS.md']=content.encode()
    for path in (ROOT/'results').glob('evaluation_*.txt'):
        copy(path.relative_to(ROOT).as_posix())
    for path in (ROOT/'results/figures').glob('*.png'):
        copy(path.relative_to(ROOT).as_posix())
    for name in ('d4-robust-servo','d4-robust-torque','final-servo-parity','final-torque-parity',
                 'final-baseline-tool-servo','final-baseline-tool-torque'):
        for basename in ('results.json','results.csv','SUMMARY.md'):
            copy('results/sweeps/'+name+'/'+basename)
    selected['report/RESULTS.md']=(ROOT/'results/FINAL_SUMMARY.md').read_bytes()
    selected['requirements.txt']=b'numpy==2.5.4\nmujoco==3.17.0\nmatplotlib==3.11.2\n'
    selected['.gitignore']=b'.omx/\n.venv-train/\n.venv/\n__pycache__/\n*.py[cod]\n/result.png\n/snapshot_*.png\n/*.csv\n'
    author=subprocess.check_output(['git','config','--local','--get','user.name'],cwd=ROOT,text=True).strip()
    license_text=(ROOT/'model/LICENSE-doosan-robot2.txt').read_text()
    license_text=license_text.replace('Copyright (c) 2024, Doosan Robotics Inc.',f'Copyright (c) 2026, {author} and contributors')
    selected['LICENSE']=license_text.encode()
    for name in ('README.md', 'THIRD_PARTY_NOTICES', 'report/REPRODUCE.md'):
        selected[name]=(ROOT/name).read_bytes()
    for stem in ('01_servo_no_collision','02_torque_no_collision',
                 '03_servo_place_collision','04_servo_transport_collision'):
        for suffix in ('mp4','gif'):
            copy(f'artifacts/media/{stem}.{suffix}')
    copy('artifacts/media/media_manifest.json')
    for stem in ('05_v2_servo_transport_collision','06_v2_torque_transport_collision'):
        for suffix in ('mp4','gif'):
            copy(f'artifacts/media/{stem}.{suffix}')
    copy('artifacts/media/v2_manifest.json')
    for stem in ('07_r2_servo_comparison','08_r2_torque_comparison'):
        for suffix in ('mp4','gif'):
            copy(f'artifacts/media/{stem}.{suffix}')
    copy('artifacts/media/r2_manifest.json')
    report_manifest = 'results/v2_protocol/R2_REPORT_MANIFEST.json'
    selected[report_manifest] = rebind_public_report_manifest(selected[report_manifest], selected)
    # Validate the destination BEFORE writing, including on reruns.
    existing={p.relative_to(DEST).as_posix() for p in DEST.rglob('*')
              if p.is_file() and '.git' not in p.relative_to(DEST).parts}
    unexpected=existing-set(selected)-{'artifacts/SHA256SUMS'}
    if unexpected:
        raise ValueError(f'Unexpected staging files: {sorted(unexpected)}')
    for name,data in selected.items():
        path=DEST/name
        if path.is_symlink():raise ValueError('Unsafe staging symlink')
        path.parent.mkdir(parents=True,exist_ok=True)
        path.write_bytes(data)
    manifest=DEST/'artifacts/SHA256SUMS'
    manifest.parent.mkdir(exist_ok=True)
    # No hidden additions silently survive a rebuild.
    actual={p.relative_to(DEST).as_posix() for p in DEST.rglob('*') if p.is_file() and '.git' not in p.relative_to(DEST).parts}
    unexpected=actual-set(selected)-{'artifacts/SHA256SUMS'}
    if unexpected:raise ValueError(f'Unexpected staging files: {sorted(unexpected)}')
    manifest.write_text(''.join(f'{hashlib.sha256(data).hexdigest()}  {name}\n' for name,data in sorted(selected.items())))
    print('Curated',len(selected)+1,'files;',sum(len(v) for v in selected.values())+manifest.stat().st_size,'bytes; no Git/network operation')


if __name__=='__main__':
    main()
