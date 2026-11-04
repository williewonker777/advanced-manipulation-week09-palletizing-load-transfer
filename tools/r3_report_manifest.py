"""Derive R3 report curation from complete frozen evidence, never simulate."""
from pathlib import Path
import argparse
import hashlib
import json
import sys
from collections import Counter

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools import r3_report as P
D = ROOT / 'results/v2_protocol'


def read(name):
    return json.loads((D / name).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def aggregate_episode_timings(documents, root=ROOT):
    """Publication-only aggregation of explicit measured episode documents.

    The frozen research helper's broad perturb glob also picks up nominal
    parity records, which contain no paired timing.  Do not modify that frozen
    source or filter individual measured rows: require every declared episode.
    """
    groups, hashes = {}, {}
    for path, expected_rows in documents:
        data = json.loads(path.read_text())
        rows = data.get('results')
        if not isinstance(rows, list) or len(rows) != expected_rows:
            raise ValueError('incomplete measured episode document: ' + path.name)
        hashes[path.relative_to(root).as_posix()] = sha(path)
        for row in rows:
            # Missing timing in an actual episode is an error, never a row
            # deletion.  Parity is not one of the declared measured documents.
            row['timing']['combined']['wall']['samples']
            groups.setdefault((row['architecture'], row['variant']), []).append(row)
    results = []
    for (mode, variant), rows in sorted(groups.items()):
        maximum = max(rows, key=lambda row: row['timing']['combined']['wall']['max_us'])
        outliers = [dict(stage=row['stage'], case_id=row['case_id'], **item)
                    for row in rows for item in row['timing']['outliers']]
        results.append(dict(architecture=mode, variant=variant, rows=len(rows),
            cycles=sum(row['timing']['combined']['wall']['samples'] for row in rows),
            largest_row_p99_us=max(row['timing']['combined']['wall']['p99_us'] for row in rows),
            maximum_wall_us=maximum['timing']['combined']['wall']['max_us'],
            maximum_thread_cpu_us=max(row['timing']['combined']['thread_cpu']['max_us'] for row in rows),
            maximum_case=dict(stage=maximum['stage'], case_id=maximum['case_id']),
            overrun_cycles=len(outliers), all_outliers=outliers,
            maximum_wall_outlier=max(outliers, key=lambda value: value['combined']['wall_ns']) if outliers else None,
            global_GC_changes=False, serial_owned_simulations=1))
    return dict(schema='R3_TIMING_REVIEW.v1', source_sha256=hashes, results=results,
        p99_selection_limit_us=800, maximum_goal_us=1000,
        samples_are_per_case_not_pooled_p99=True, all_outliers_retained=True,
        aggregation='publication-only exact episode inventory; nominal parity has no paired timing',
        frozen_helper_limitation='r3_review timing glob includes parity records; frozen source unchanged')


def build_timing():
    protocol = read('PROTOCOL_R3.json')
    selected = read('R3_DIAGNOSTIC_SELECTION.json')['chosen_specs']
    inventory = [('R3_DIAGNOSTIC_R1_torque_4.4.json', 1), ('R3_HOLDOUT.json', 76)]
    for mode in ('servo', 'torque'):
        inventory.append((f'R3_VALIDATION_R1_{mode}.json', 4))
    inventory += [(f"R3_VALIDATION_{spec['id']}_{spec['architecture']}.json", 4)
                  for spec in protocol['candidate_grid']]
    for stage, count in [('DEVELOPMENT', 26), ('D4', 385), ('PERTURB', 34)]:
        for spec in selected:
            inventory += [(f"R3_{stage}_{spec['id']}_{spec['architecture']}.json", count),
                          (f"R3_{stage}_R1_{spec['architecture']}.json", count)]
    result = aggregate_episode_timings([(D / name, count) for name, count in inventory])
    if sum(row['rows'] for row in result['results']) != 1885:
        raise ValueError('all 1885 instrumented R3 episodes must be retained')
    if sum(row['cycles'] for row in result['results']) != 11310000:
        raise ValueError('all 11310000 measured cycles must be retained')
    result['unwrapped_nominal_parity_episodes'] = 8
    result['equivalent_action_replay'] = read('R3_REPLAY_TIMING.json')
    result['source_sha256']['results/v2_protocol/R3_REPLAY_TIMING.json'] = sha(D / 'R3_REPLAY_TIMING.json')
    return result


def documentation_details():
    """Read-only presentation summaries; never change a measurement or gate."""
    from tools import r3_select as S
    forced = read('R3_FORCED_STABILITY.json')['analyses']
    actual = read('R3_ACTUAL_STABILITY.json')['analyses']
    baseline = read('R3_R1_STABILITY.json')['analyses']
    key = lambda r: (r['architecture'], r['mass_kg'], tuple(r['com_m']), r['time_s'])
    r1 = {key(row): row for row in baseline}
    stability = []
    for name in P.CANDIDATES:
        rows = [r for r in forced if r['candidate'] == name]
        own = [r for r in actual if r['candidate'] == name]
        matched = [r1[key(r)] for r in rows]
        count = lambda values: f"{sum(r['combined_radius_fd_pass'] for r in values)}/{len(values)}"
        stability.append(dict(candidate=name, forced=count(rows),
                              actual=count(own) if own else '미실행: 비선정', r1=count(matched)))
    a_failed = [r for r in forced if r['architecture'] == 'servo' and not r['combined_radius_fd_pass']]
    delta = max(abs(r['closed_loop']['spectral_radius'] - r1[key(r)]['closed_loop']['spectral_radius'])
                for r in a_failed)
    assert all(not r1[key(r)]['combined_radius_fd_pass'] and 3.1 <= r['time_s'] <= 3.6 for r in a_failed)
    b = [r for r in forced if r['candidate'] == 'B_mass']
    radius_key = 'protocol_radius_pass_lt_1_plus_1e_minus_6'
    radius_failed = [r for r in b if not r['closed_loop'][radius_key]]
    fd_only = [r for r in b if r['closed_loop'][radius_key] and not r['fd_convergence']['pass']]
    assert len(radius_failed) == 3 and len(fd_only) == 4
    assert all(not r1[key(r)]['closed_loop'][radius_key] for r in radius_failed)
    cost_rows, averages = [], []
    metric_names = dict(rms_mm='TCP RMS', max_mm='TCP 최대', torque_step_pick_nm='집기 계단',
                        torque_step_place_nm='놓기 계단', table_force_place_max_n='놓기 힘')
    for stage, total in [('D4', 189), ('PERTURB', 28)]:
        for mode, name in [('servo', 'A_handoff_long'), ('torque', 'B_mass')]:
            pairs = S.pair_results(read(f'R3_{stage}_R1_{mode}.json'), read(f'R3_{stage}_{name}_{mode}.json'))
            review = S.review_costs(pairs, references_only=True)
            assert len(review['rows']) == total
            failures = [r for r in review['rows'] if not r['pass']]
            metric_counts, masses, cross = Counter(), Counter(), Counter()
            for row in failures:
                masses[row['mass']] += 1
                for metric, value in row['metrics'].items():
                    if not value['pass_le_1_05']:
                        metric_counts[metric] += 1
                        cross[row['mass'], metric] += 1
            cost_rows.append(dict(stage='D4' if stage == 'D4' else '분석 섭동', architecture=mode,
                exceeded=len(failures), total=total, worst_ratio=review['worst_ratio'],
                metrics='; '.join(f'{metric_names[m]} {n}행' for m, n in metric_counts.items()),
                masses='; '.join(f'{m:g} kg {n}행' for m, n in sorted(masses.items())),
                mass_metrics='; '.join(f'{m:g} kg {metric_names[k]} {n}행' for (m, k), n in sorted(cross.items()))))
            if stage == 'PERTURB':
                positive = [(l, r) for l, r in pairs if r['kind'] == 'reference' and r['mass'] > 0
                            and r['collision'] is None and r['pick_offset'] is None]
                assert len(positive) == 22
                averages.append(dict(architecture=mode, rows=len(positive),
                    rms_ratio=sum(S.ratio(r['rms_mm'], l['rms_mm']) for l, r in positive) / len(positive),
                    max_ratio=sum(S.ratio(r['max_mm'], l['max_mm']) for l, r in positive) / len(positive)))
    return dict(stability=stability, costs=cost_rows, positive_mass_perturbation=averages,
        stability_notes=[
            f'임계값은 R2와 같지만 검사점은 R3에서 늘었다. A의 강제 실패 6점과 실제 실패 10점은 모두 추가한 3.1–3.6 s 구간이고 기존 2.5, 3.75, 3.89 s에는 없다.',
            f'A의 강제 실패 6점은 R1도 같은 점에서 실패했다. 강제 실패 6점에서, 두 설정을 통틀어 R1 대비 radius 차이 최대는 {delta:.2e}로 약 3.8e-5다.',
            'B-mass의 강제 실패 7점은 radius 실패 3점과 radius 통과·FD 미수렴 4점이다. radius 실패 3점은 R1도 실패했고, 5.65/4.4 kg의 2.5 s 두 점은 B가 1에 더 가깝다. R1이 실패한 7.7 kg, 2.5 s는 B가 통과했다.',
            '같은 강제점 R1은 R1 관측·물리점 격자의 비교이며 후보 실제 궤적의 R1 재생이 아니다. 이 국소 관문만으로 후보와 R1을 가를 수 없다. 규칙과 탈락 판정은 그대로 두고 D4·섭동 비용·오감지 근거를 함께 판단한다.',
            'D4 전체 385조건에서 R1 A도 포화 4회/최대 0.166667%로 상한 2회/0.15%를 넘었다. A-long도 4회이고 최대율은 0.150000%다. 무충돌 189조건의 2회와 혼동하지 않는다.'],
        redesign_context=[
            '재설계 결정은 R1 비교 파일보다 먼저 저장됐다. 결정은 연구 작업 커밋 5a86c02, R1 비교 파일은 그 뒤 커밋 8b57c79에서 추가됐다. 두 커밋은 비공개 작업 이력이라 공개 이력에서는 찾을 수 없다. A의 당시 국소 관문 실패는 그 뒤 R1과 같은 위치로 확인돼 단독 식별 근거가 약해졌다.',
            '결정 기록 자체는 바꾸지 않는다. 질량별 비선형 위상·원인 미식별과 추가 시각/gain의 인과적 해결책 부재 사유는 그대로 유효하다. 이후 holdout·공식 결과로 재설계를 하지 않았다.'],
        cost_notes=[
            '비용은 같은 무충돌 조건마다 후보/R1 비를 계산했다. 한 행에서 여러 지표가 초과할 수 있어 지표 합계와 초과 행 수는 다르다. 관문 1.05는 완화하지 않았다.',
            'B-mass의 8 kg, 강성 1.2배 조건(case 16)은 오감지 2→32회, 강성 1.2배·마찰 0.8배 복합 모서리 조건(case 17)은 2→28회였다. 반면 강성 0.8배(case 12)는 3→2회로 감소했다.',
            '양의 질량 무충돌 섭동 22행은 각 같은 조건의 비를 먼저 계산한 산술평균이다. 아래 평균은 효능 관문이 아닌 참고값이며, 평균 개선이 조건별 비용·오감지 통과를 뜻하지 않는다.'],
        causes=[
            'A: 놓기 계단 순간 직접 DOB 보상은 이미 0이었다. 그 순간 보상을 더 감쇠한다는 가설은 맞지 않았다. 차이는 이전 궤적의 feedforward·속도 피드백 이력에 연결되며 5.65 kg만 불리한 비선형 위상은 완전히 식별하지 못했다.',
            'B: 경량에서 남은 P/V 피드백 기여 3.148/3.859 N·m가 하중 보간 0.336/0.838 N·m보다 컸다. 분해는 고정 순서의 명령 차분이며 마찰·클리핑이 있는 폐루프의 유일한 인과 분해가 아니다.'])


def build(verification):
    protocol = read('PROTOCOL_R3.json')
    validation = read('R3_BOUND_STABILITY_REVIEW.json')
    reviews = {row['candidate']: row for row in validation['candidates']}
    timing = read('R3_TIMING_REVIEW.json')
    times = {(row['architecture'], row['variant']): row for row in timing['results']}
    decision = read('R3_REDESIGN_DECISION.json')
    pre = read('R3_PREFREEZE_DECISION.json')
    holdout = read('R3_HOLDOUT.json')
    official = read('R3_OFFICIAL_REGRESSION.json')
    integrity = read('R3_INTEGRITY.json')
    math = read('R3_PRNE_MATH.json')
    perturb_review = read('R3_PERTURB_REVIEW.json')
    replay = read('R3_REPLAY_TIMING.json')
    replay_p99 = [row['timing']['combined']['wall']['p99_us'] for row in replay['results']]
    if not integrity['passed'] or not all(integrity['final_R1_alias'].values()):
        raise ValueError('this negative-round report requires verified final R1 aliases')
    candidates, per_mass = [], []
    for spec in protocol['candidate_grid']:
        name, mode = spec['id'], spec['architecture']
        review, elapsed = reviews[name], times[mode, name]
        forced = review['forced_stability']['rows']
        actual = review['actual_stability']['rows']
        gain = review['tracking_efficacy']
        base = read(f'R3_VALIDATION_R1_{mode}.json')['results']
        candidate = read(f'R3_VALIDATION_{name}_{mode}.json')['results']
        mean_rms = sum(r['rms_mm'] / l['rms_mm'] for l, r in zip(base, candidate, strict=True)) / len(base)
        mean_max = sum(r['max_mm'] / l['max_mm'] for l, r in zip(base, candidate, strict=True)) / len(base)
        gain_text = f"평균 RMS 비 {mean_rms:.3f} / 최대 비 {mean_max:.3f}"
        if mode == 'servo':
            gain_text += ' (참고; 효능은 섭동에서 판정)'
        failed = []
        if review['costs']['status'] != 'PASS':
            failed.append('질량별 비용')
        if review['forced_stability']['status'] != 'PASS':
            failed.append('국소 안정성/FD')
        if mode == 'torque' and gain['status'] != 'PASS':
            failed.append('10% 효능')
        candidates.append(dict(
            id=name, method='DOB 4.5 + handoff' if mode == 'servo' else 'P-IDC 4.5/1 + ' + spec['schedule'],
            architecture=mode,
            validation_cost=f"{review['costs']['status']}; 최악 {review['costs']['worst_ratio']:.3f}×",
            validation_gain=gain_text,
            forced_stability=f"{sum(r['pass'] for r in forced)}/{len(forced)} radius+FD",
            actual_stability=(f"{sum(r['pass'] for r in actual)}/{len(actual)} radius+FD"
                              if actual else '미실행: 비선정'),
            timing=f"{elapsed['largest_row_p99_us']:.1f}/{elapsed['maximum_wall_us']:.1f} us",
            decision='reject', reason='·'.join(failed)))
        for left, right in zip(base, candidate, strict=True):
            if (left['case_id'], left['mass'], left['com']) != (right['case_id'], right['mass'], right['com']):
                raise ValueError('unmatched per-mass validation conditions')
            per_mass.append(dict(candidate=name, architecture=mode, mass_kg=right['mass'],
                r1_rms_mm=left['rms_mm'], diagnostic_rms_mm=right['rms_mm'],
                r1_max_mm=left['max_mm'], diagnostic_max_mm=right['max_mm'],
                r1_place_step_nm=left['torque_step_place_nm'],
                diagnostic_place_step_nm=right['torque_step_place_nm'],
                validation_status='reject', final_r3_alias='R1'))
    old_metrics = read('R2_REPORT_MANIFEST.json')['metrics']
    metrics = [{**{key: row[key] for key in ('id', 'name', 'condition', 'unit', 'baseline', 'v1', 'r1')},
                'r3': row['r1'], 'status': row['r2_status']} for row in old_metrics]
    final = [dict(architecture=mode, decision='reject',
                  selected=pre['architectures'][mode]['challenger'], final_alias='R1',
                  my_sha256=sha(ROOT / f'my_{mode}.py'), r1_sha256=sha(ROOT / f'r1_{mode}.py'),
                  reason='사전 관문 탈락 기록 유지; D4·섭동 비용·오감지 실패도 확인')
             for mode in ('servo', 'torque')]
    redesign = [dict(architecture=mode, used=decision['redesigns_used'][mode],
                    reason=('비용 외 수렴한 국소 관문도 실패했다. 미식별 위상에 추가 시각/gain을 정할 근거가 없어 비용 중심 재설계 조건이 아니다.'
                            if mode == 'servo' else
                            'B_mass는 비용·효능을 이미 통과했다. 막힌 것은 국소 관문이므로 비용 중심 재설계 조건이 아니다.'),
                    recorded_before_development=True, arbitrary_parameter_search_performed=False)
                for mode in ('servo', 'torque')]
    lie = [dict(method='A motor DOB', section='자체 유도', equation='dhat=z+LBv; quintic handoff',
                elastic_1ms_zoh_delay_adaptation='elastic motor + 자체 Drive queue; 1 ms ZOH, command/JTS delay',
                comparison='유클리드 R1 대비 A-long 최대 TCP 약4.5–5.6% 감소; 독립 Lie 효과 아님', decision='reject'),
           dict(method='B P-IDC', section='Lie Group Robot Control §10.1.6',
                equation='tau=M*a_r+C*v_r+g+M*r*s',
                elastic_1ms_zoh_delay_adaptation='elastic lift + clip + mass/phase schedule; 1 ms ZOH, one-cycle delay',
                comparison='B-mass 질량별 RMS비 평균 .833553; 제어·스케줄 동시 변경이라 기하 단독 효과 미분리', decision='reject'),
           dict(method='B pRNE', section='Lie Group Robotics 목차 §8.3, 본문 미확인; 자체 유도',
                equation='ad(V), X^T, C(q,v)u; Mdot=C+C^T',
                elastic_1ms_zoh_delay_adaptation='rigid nominal C만 계산; elastic 플랜트는 별도 1 ms ZOH/delay 국소 검사',
                comparison=f"유한차분 항등식 오차 {math['worst']['mdot_minus_C_plus_CT_max_abs_error']:.3g}; 같은 관측 p99 {replay_p99[0]:.1f}→{replay_p99[1]:.1f}us (고정 순서 참고, 기하 단독 효과 미분리)", decision='reject')]
    normal = []
    for mode in ('servo', 'torque'):
        rows = [row for spec in protocol['candidate_grid'] if spec['architecture'] == mode
                for row in read(f"R3_VALIDATION_{spec['id']}_{mode}.json")['results']]
        count = sum(row['false_alarms'] for row in rows)
        normal.append(dict(stage='validation 후보 전체', architecture=mode,
                           count=count, rise_vs_r1=count, status='PASS' if count == 0 else 'FAIL'))
    stage_outcomes = []
    for stage, file in [('development', 'R3_DEVELOPMENT_REVIEW.json'),
                        ('D4', 'R3_D4_REVIEW.json'), ('분석 섭동', 'R3_PERTURB_REVIEW.json')]:
        data = read(file)['architectures']
        parts = []
        for mode in ('servo', 'torque'):
            value = data[mode]
            base, candidate = value['R1_summary'], value['candidate_summary']
            parts.append(f"{mode}: 검출 {base['counts']['detected']}→{candidate['counts']['detected']}회, {value['review']['status']}")
            for label, summary in [('R1', base), ('후보', candidate)]:
                normal.append(dict(stage=stage + ' ' + label, architecture=mode,
                    count=summary['reference_false_alarms'],
                    rise_vs_r1=summary['reference_false_alarms'] - base['reference_false_alarms'],
                    status='PASS' if summary['reference_false_alarms'] == 0 else 'FAIL'))
        stage_outcomes.append(stage + ': ' + '; '.join(parts) + ' (재승격 없음)')
    stage_outcomes += [
        'D4 동일 조건 비용 최악 비 A 1.136/B 1.255. A 편향 조건 오감지 8→9회, B 6→4회지만 동일 조건 추가 오감지 1회가 있어 두 방식 모두 실패했다.',
        '분석 섭동 A 비용 비 1.058/기존 효능 점수 0.951, B 비용 비 1.473/정상 오감지 9→64회로 실패했다. 두 방식 모두 40 N, 6방향을 6/6 검출했다.',
    ]
    from tools.r3_review import summarize
    h = {mode: summarize([row for row in holdout['results'] if row['architecture'] == mode])
         for mode in ('servo', 'torque')}
    stage_outcomes += [
        'D4 포화 A: 기준 무충돌 189조건은 2회, 전체 385조건은 R1/후보 모두 4회다. 사전 전체 한도 2회/0.15%는 완화하지 않았다.',
        '후보 실제점 radius/FD 동시 통과는 A-long 54/64, B-mass 45/48이다. R1의 68/78은 별도의 같은 강제격자 범위다. 이 관문 단독으로 후보와 R1을 가를 수 없으며 전역 발산 증거도 아니다.',
        '새 holdout 한 배치에서 최종 R1만 76회 검사했다: ' + '; '.join(
            f"{mode} 검출 {value['counts']['detected']}/30·편향 조건 오감지 {value['offset_false_alarms']}회"
            for mode, value in h.items()),
        f"마지막 공식 58회: {official['status']}. 8 kg 추적·감지는 최종 R1 소스의 회귀 확인이며 후보 효능 자료가 아니다.",
    ]
    time_rows = [dict(stage=row['variant'], architecture=row['architecture'],
                     p99_us=row['largest_row_p99_us'], max_us=row['maximum_wall_us'],
                     outlier_count=row['overrun_cycles'], outliers_retained_raw=True,
                     p99_gate_pass=row['largest_row_p99_us'] < 800,
                     max_1000_goal_met=row['maximum_wall_us'] <= 1000,
                     status='PASS' if row['largest_row_p99_us'] < 800 else 'FAIL')
                 for row in timing['results']]
    curation = dict(schema='R3_REPORT_MANIFEST.v1', status='frozen',
        candidates=candidates, per_mass_validation=per_mass, metrics=metrics,
        final_architectures=final, redesign=redesign, lie_group_usage=lie,
        normal_false_alarms=normal, timing=time_rows,
        exposure=dict(holdout=dict(opened_after_freeze=True, batches=holdout['metadata']['batch_attempt'],
                                  no_tuning_after=True),
                      official=dict(executed_last=True, no_tuning_after=official['tuning_after_exposure'] is False)),
        stage_outcomes=stage_outcomes, verification=verification,
        limitations=[
            '모든 개선 수치는 유한 비공식 조건의 진단 결과다. R2에서 고른 방법의 선택 편향과 새 validation/고정 위상 국소 분석의 한계가 있다.',
            'G4 지속 지지, 엄격 최대1ms, 격자 포화·섭동 오감지·약한 힘 감지의 기존 미달은 해소하지 못했다.',
            '진단 후보의 새 holdout/공식 효능은 미실행이다. 사전 탈락 뒤 최종 R1만 검사했고 재승격·튜닝·공식 재실행하지 않았다.',
            '기준/v1은 새 R3 split에서 재실행하지 않았다. 네 열 T1–T6은 보존 동일조건 수치와 exact-source alias다.',
            '분석 전용 runner/원시 trace를 포함한 새 수집부터 완전 재현, 실제 로봇·전역/LTV/hybrid/수동성 보장은 검증하지 않았다.',
        ],
        document_details=documentation_details(),
        conclusion='새 채택 0개. 서보와 토크 모두 독립 판정해 R1을 유지한다. 국소 관문 단독으로 R1과 후보를 가를 수는 없지만 규칙을 완화하지 않았다. D4·섭동의 비용·오감지 악화도 확인해 탈락 판정을 유지한다. 음성 결과를 방법 계열 자체의 불가능성으로 일반화하지 않는다.')
    paths = {
        'protocol': 'PROTOCOL_R3.json', 'r2_report_manifest': 'R2_REPORT_MANIFEST.json',
        'validation_review': 'R3_BOUND_STABILITY_REVIEW.json', 'challengers': 'R3_CHALLENGERS.json',
        'forced_stability': 'R3_FORCED_STABILITY.json', 'actual_stability': 'R3_ACTUAL_STABILITY.json',
        'redesign_decision': 'R3_REDESIGN_DECISION.json', 'development_review': 'R3_DEVELOPMENT_REVIEW.json',
        'd4_review': 'R3_D4_REVIEW.json', 'perturbation_review': 'R3_PERTURB_REVIEW.json',
        'freeze': 'R3_FREEZE.json', 'holdout_review': 'R3_HOLDOUT_REVIEW.json',
        'official_review': 'R3_OFFICIAL_REGRESSION.json', 'timing_review': 'R3_TIMING_REVIEW.json',
        'final_selection': 'R3_FINAL_SELECTION.json', 'integrity': 'R3_INTEGRITY.json'}
    paths['r1_stability'] = 'R3_R1_STABILITY.json'
    paths['cause_analysis'] = 'R3_CAUSE_ANALYSIS.json'
    for stage in ('D4', 'PERTURB'):
        for mode, candidate in [('servo', 'A_handoff_long'), ('torque', 'B_mass')]:
            paths[f'{stage.lower()}_{mode}_r1'] = f'R3_{stage}_R1_{mode}.json'
            paths[f'{stage.lower()}_{mode}_candidate'] = f'R3_{stage}_{candidate}_{mode}.json'
    artifact_specs = {key: (D / value, 'literal_copy' if key == 'r2_report_manifest' else 'fresh')
                      for key, value in paths.items()}
    artifact_specs['r2_presentation'] = (ROOT / 'results/presentation_R2.pdf', 'reused_exact_source')
    return curation, P.build_manifest(curation, artifact_specs, ROOT)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--verification', type=Path)
    group.add_argument('--timing-only', action='store_true')
    args = parser.parse_args()
    if args.timing_only:
        path = D / 'R3_TIMING_REVIEW.json'
        if path.exists():
            raise RuntimeError('do not overwrite timing evidence')
        path.write_text(json.dumps(build_timing(), indent=2) + '\n')
        print('all 1885 measured episodes retained; frozen scientific source unchanged')
        return
    curation, manifest = build(json.loads(args.verification.read_text()))
    for name, value in [('R3_REPORT_CURATION.json', curation), ('R3_REPORT_MANIFEST.json', manifest)]:
        (D / name).write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    print('R3 report inputs validated')


if __name__ == '__main__':
    main()
