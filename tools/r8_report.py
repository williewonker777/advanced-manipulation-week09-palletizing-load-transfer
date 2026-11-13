"""Render R8 evidence without executing or changing any scientific algorithm."""
from pathlib import Path
import argparse
import hashlib
import json
import sys
import tempfile
import textwrap
from datetime import datetime, timezone

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from tools.r7_json import read_json,write_json,replace_json

D='results/v2_protocol/'
INPUTS=['PROTOCOL_R8.json','R8_FREEZE.json','R8_ADOPTION.json',
        'R8_HOLDOUT_REVIEW.json','R8_OFFICIAL_REGRESSION.json',
        'R8_SAVED_REVIEW.json','R8_AUDIT_TIMING.json','R8_PARITY_REVIEW.json',
        'R8_DEVELOPMENT_SUMMARY.json','R7_OFFICIAL_REGRESSION.json','R8_ARCHIVE.json','R8_PUBLIC_SCOPE.json']
OUTPUTS=['README.md','docs/R8_RESULTS.md','docs/R8_METHODS.md',
         'report/PRESENTATION_R8.md','results/presentation.pdf']


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def context(root):
    c={name[:-5]:read_json(root/(D+name)) for name in INPUTS}
    c['old_readme']=read_json(root/(D+'R8_ARCHIVE.json'))['original_readme']
    c['root']=Path(root)
    validate_restoration(c)
    return c


def validate_restoration(c):
    a=c['R8_ADOPTION']; r=a.get('restoration')
    if r is None:return
    historical={k:v for k,v in a.items() if k!='restoration'}
    assert hashlib.sha256(json.dumps(historical,sort_keys=True,ensure_ascii=False).encode()).hexdigest()==r['historical_stage_B_fields_sha256']
    assert not historical['adopted'] and historical['final_selection']=='fallback'
    assert r['effective_adopted'] and r['effective_final_selection']=='C1'
    assert r['effective_detector_sha256']==a['candidate_sha256']==sha(c['root']/'my_detector.py')==sha(c['root']/'experiments/r8_final/C1.py')
    assert r['post_result_exception'] and r['candidate_bytes_unchanged'] and r['models_thresholds_numeric_gates_unchanged']
    for name,digest in r['source_sha256'].items():assert sha(c['root']/(D+name))==digest
    for name in ('new_physical_evaluations','new_holdout_batches','new_official_batches','remote_operations'):
        assert type(r[name]) is int and r[name]==0


def restoration_note():
    return ('C1 채택(사용자 결정으로 복원). B5에서 fallback된 이유는 우리가 만든 import 허용 목록(numpy, mujoco, sim)이었다. '
            '가이드에는 import 제한이 없으며, 숨은 내부값 접근과 sim 변경 금지·Part 3 허용 입력 규칙은 그대로다. '
            '허용 목록에 표준 모듈 io, base64, collections만 추가했고 금지 이름 검사는 유지했다. '
            'R7 기록 시험은 보관된 R7 파일에 묶어 SHA 결속·변경 거부 검사를 유지했다. '
            '후보 바이트·모델·임계·관문 숫자와 B4 holdout·B5 공식 결과는 그대로이며 다시 실행하지 않았다. '
            '이는 결과를 본 뒤의 사용자 결정에 따른 예외이며 R7에도 채택 복원 선례가 있다. '
            'B5 fallback 기록은 R8_ADOPTION.json의 기존 최상위 필드에 보존하고, 현재 판정은 추가한 restoration 항목에 기록했다.')


def readme_body(c):
    # Keep the archived R7 evidence unchanged; update only the current summary rows.
    rows={
      '| 약한 집기·놓기 충돌 놓침 |':'| 약한 집기·놓기 충돌 놓침 | 서보는 기존 최종 대비 holdout 66→74/90(+8), 공식 18→23/24로 개선됐다. 사건·툴 손실과 조건별 오감지 증가 0이다. 토크 공식 18/24는 그대로다. | 기존 R1 신호 분석은 게이트·연속 표본 조건·정상 점수 겹침을 확인했다. L2d의 부분 개선이지 모든 수직 약충돌 분리 보장은 아니다. | [R8 결과](docs/R8_RESULTS.md), [과거 신호 측정](docs/R4_README.md#평가기-점검과-미검출-원인) |',
      '| 최대 1 ms 초과 |':'| 최대 1 ms 초과 | R8 서보 holdout 합산 p99 406.149→675.362 µs, 최대 701.350→1587.216 µs, 1 ms 초과 0→14주기로 더 나빠졌다. 감사 최대는 7991.764 µs다. | p99 800 µs 관문은 통과하나 최대 1 ms는 미달이다. 최대치의 단일 원인은 미확정이며 서로 다른 감사/holdout 사건을 합치지 않는다. | [R8 시간](docs/R8_RESULTS.md#3-계산-시간--이상치를-모두-포함) |',
      '| 계산 비용 증가 |':'| 계산 비용 증가 | R8 서보 holdout 합산 p99는 기존 최종 R7 406.149→C1 675.362 µs다. | 새 L2d 계산이 추가됐다. OS·측정 영향도 섞여 증가분 전체를 순수 알고리즘 비용으로 단정할 수 없다. | [R8 시간](docs/R8_RESULTS.md#3-계산-시간--이상치를-모두-포함) |',
      '| 충돌 감지(전체 D4·운반) |':'| 충돌 감지(전체 D4·운반) | 부분 | D4 서보 R7 119→C1 138/180, 토크 111/180 그대로. 새 holdout 서보 66→74/90, 공식 18→23/24. 손실·오감지 증가 0이다. | D4 서보 놓침 42, 토크 69. 토크 감지 이득은 없고 모든 약충돌을 해결한 것은 아니다. [R8 원자료](docs/R8_RESULTS.md) |',
      '| 약한 집기·놓기 충돌 |':'| 약한 집기·놓기 충돌 | 부분 | 서보 holdout 추가 8/90(집기 1·놓기 7), 공식 18→23/24. 사건·툴 손실·조건별 오감지 증가 0. | 토크는 개선 없음. 약한 수직 충돌의 모호성과 숨은 조건은 여전히 미달이다. [R8](docs/R8_RESULTS.md) |',
      '| 계산 시간(1 ms) |':'| 계산 시간(1 ms) | 부분 | 서보 holdout p99 675.362 µs로 800 µs 관문 통과. | 기존 최종 대비 1 ms 초과 0→14주기, 최대 1.59 ms로 악화. 감사 최대 7.99 ms. 엄격한 1 ms 보장은 없다. [시간](docs/R8_RESULTS.md#3-계산-시간--이상치를-모두-포함) |',
      '| 숨은 조건 일반화 |':'| 숨은 조건 일반화 | 미검증 | R8 봉인 holdout 1배치 456회와 공식 회귀를 통과했다. 새 독립 실행을 추가한 것이 아니라 기존 B4·B5 결과로 복원했다. | 비공개 조건 통과의 보장은 없다. 과거 R7의 152회 결과와 별개이며 R6 holdout은 미실행이다. |',
    }
    lines=[]
    for line in c['old_readme'].splitlines()[1:]:
        if line.startswith('**최종 제출:'):
            line='**최종 제출: 제어는 서보·토크 모두 R1, 감지는 서보 C1(R7 + L2d), 토크 R1이다. C1은 사용자 결정으로 복원했다.**'
        else:
            for prefix,replacement in rows.items():
                if line.startswith(prefix):line=replacement;break
        lines.append(line)
    return '\n'.join(lines)+'\n'


def table(head,rows):
    return '| '+' | '.join(head)+' |\n| '+' | '.join(['---']*len(head))+' |\n'+''.join('| '+' | '.join(map(str,r))+' |\n' for r in rows)


def efficacy(c):
    saved=c['R8_SAVED_REVIEW']['summary'];hold=c['R8_HOLDOUT_REVIEW']['entries']
    rows=[]
    for key,label in [('servo_D4','서보 D4 개발'),('servo_A6_weak','서보 독립 약충돌'),('servo_perturb','서보 섭동'),('torque_D4','토크 D4 개발')]:
        x=saved[key];rows.append([label,x['conditions'],x['base_detected'],x['C1_detected'],x['gain'],x['loss'],x['FA_increase']])
    for e in hold:
        rr=[r for r in e['rows'] if r['case']['kind']=='collision']
        rows.append([('서보' if e['mode']=='servo' else '토크')+' 새 holdout',len(rr),sum(r['fallback']['outcome']=='detected' for r in rr),sum(r['C1']['outcome']=='detected' for r in rr),e['gain'],sum(r['loss'] for r in rr),sum(r['FA_increase'] for r in e['rows'])])
    return table(['세트','조건','기존 최종','C1','추가','손실','오감지 증가 조건'],rows)


def timing(c):
    s=c['R8_AUDIT_TIMING']['summary'];rows=[]
    for v,label in [('fallback','기존 최종'),('C1','C1 후보')]:
        x=s[v];rows.append([label,'감사 96,000',f"{x['p99_us']:.3f}",f"{x['maximum_wall_us']:.3f}",f"{x['maximum_CPU_us']:.3f}",x['wall_ge1ms_cycles']])
    for e in c['R8_HOLDOUT_REVIEW']['entries']:
        for v in ['fallback','C1']:
            rr=e['timing'][v];rows.append([('서보' if e['mode']=='servo' else '토크')+' '+('기존' if v=='fallback' else 'C1'),'holdout 684,000',f"{max(t['p99_us'] for t in rr):.3f}",f"{max(t['maximum_us'] for t in rr):.3f}",f"{max(t['maximum_CPU_us'] for t in rr):.3f}",sum(t['ge1ms_cycles'] for t in rr)])
    return table(['구성','주기 수','최악 p99 (µs)','최대 wall (µs)','최대 CPU (µs)','1 ms 초과'],rows)


def official(c):
    old=c['R7_OFFICIAL_REGRESSION']['records'];new=c['R8_OFFICIAL_REGRESSION']['records'];rows=[]
    for mode,label in [('servo','서보 A'),('torque','토크 B')]:
        a=old[mode]['control_and_collisions'];b=new[mode]['control_and_collisions'];x=b['control']['8.0']
        count=lambda p:sum(v!='놓침' for r in p['collisions'] for v in r['cells'][:4])
        rows.append([label,str(count(a))+'/24',str(count(b))+'/24',x[0]+'/'+x[1],x[4],'/'.join(x[6:9]),x[2]])
    return table(['방식','기존 검출','C1 검출','8 kg RMS/최대 (mm)','놓기 계단 (Nm)','정상 오감지 0/4/8 kg','포화 (%)'],rows)


def results(c,prefix='../'):
    intro='''# R8 결과 — C1 채택(사용자 결정으로 복원)

A=서보, B=토크, 오감지는 충돌 밖의 새 경보를 뜻한다. 비교 기준은 기존 최종(서보 R7, 토크 R1)이다.
''' + restoration_note() + '''

최종 제어는 두 방식 모두 R1, 최종 감지는 서보 C1(R7 + L2d)·토크 R1이다. 실제 C1에서 작업 전체 778시험·제출 계약 8건과 보고 check를 통과했다. 아래는 B4·B5의 보존 측정이며 BR에서 새 평가나 시간 측정을 하지 않았다.

## 1. 검출과 일반화
'''
    text=intro+efficacy(c)+'''
D4와 독립 약충돌은 개발 근거다. 새 holdout은 동결 뒤 새 시드로 한 번만 열었고, 참조 12·충돌 90·오프셋 12조건을 방식별 두 구성에서 실행했다(456 실제 루프). 조건별 사건/툴 손실·오감지 증가 0, 참조 오감지 0, 물리 8배열 바이트 동일, 토크 출력 바이트 동일, p99 관문을 모두 통과했다. 서보 추가 8건은 최소 6건 관문 이상이다. 숨은 모든 조건의 일반화 보장은 아니다.

### 서보 holdout의 위상·힘·방향 분포
'''
    strata=c['R8_HOLDOUT_REVIEW']['entries'][0]['strata'];rows=[]
    labels={'phase:pick':'집기','phase:transport':'운반','phase:place':'놓기','force:<10':'10 N 미만','force:10..15':'10–15 N','force:>15':'15 N 초과'}
    for k,x in strata.items():rows.append([labels.get(k,k.replace('direction:','방향 ')),x['cases'],x['fallback_detected'],x['C1_detected'],x['gain'],x['loss']])
    text+=table(['구간','조건','기존','C1','추가','손실'],rows)+'''
## 2. 공식 회귀 — 노출된 조건의 회귀 확인만
'''+official(c)+'''
원래 evaluate.py를 방식별 1회(각 29 episode, 합계 58) 실행했다. 기존 검출 유지·지연 추가 ≤5 ms·정상 오감지 0·포화 0·발산 없음·충돌 밖 추가 오감지 없음 관문을 통과했다. 공식 문자열은 원래 출력 자릿수를 그대로 썼다. 공식 결과로 학습·선택·수정을 하지 않았다. 이 표는 최종 파일과 바이트가 같은 C1의 B5 보존 측정이다. BR에서 다시 재지 않았다.

## 3. 계산 시간 — 이상치를 모두 포함
'''+timing(c)+'''
합산은 제어기+감지기의 실제 알고리즘 시간이다. 기록용 진단은 밖에 두고 새 채널 계산은 전부 포함했다. 감사는 직렬 저장 재생, holdout은 직렬 실제 루프다. p99 ≤800 µs는 통과했지만 C1의 최대 1 ms는 실패했다. 서로 다른 측정의 최대치를 한 사건으로 합치지 않는다. 토크 출력은 동일해도 실행 시간은 OS·측정 부하 때문에 같지 않을 수 있다.

## 4. B5 fallback 이력과 사후 복원 예외

- C1은 자체 완결이지만 추가 io·base64·collections import가 기존 test_detector의 생산 경로 허용 목록(numpy·mujoco·sim)에 없었다. 마지막 C1 전체 tools 703시험에서 실패 3·오류 1이 났다. 그중 **생산 import 실패 1건이 당시 되돌림 근거**였다. 가이드가 아니라 우리 자체 허용 목록이었다. 나머지 3건은 과거 R7 문서 fixture가 현재 최종 파일을 복사해 생긴 불일치이며 문서 실패 자체는 채택 veto가 아니다.
- B1에서는 C1 제출 overlay 계약·바이트 동등성·새 실제 대조 8회를 확인했지만, 전체 시험은 fallback 작업 경로에서 돌렸다. 그래서 이 조립 문제를 동결 전에 발견하지 못했다. C1 조립/검증 절차의 미흡이며 L2d의 검출 이득이 없었다는 결론은 아니다.
- 후보를 고치거나 검사를 약하게 하지 않았다. 기존 최종으로 되돌린 **775시험(tools 703+R8 72)**과 정확한 제출 이름 overlay 계약은 다시 통과했다. 이것은 Stage B 당시의 처리다. 이후 사용자 결정으로 import 허용 목록에 세 표준 모듈만 추가하고 R7 과거 fixture를 바로잡았다. 실제 C1 최종에서 작업 전체 778시험과 제출 계약 8건·보고 check를 통과했다. holdout·공식은 재실행하지 않고 B4·B5의 동일 바이트 결과를 쓴다. 결과 이후 예외라는 사실과 B1 전체 시험 경로 누락을 숨기지 않는다.
- 메모리 예약도 처음에는 부족했다. A6 재생 reader 3개가 각각 약 0.75 GiB를 잡은 채 자기 용량 guard를 기다렸다. 자기 프로세스만 정리해 1 worker·시작 전 1 GiB 여유로 바꿨다. 원래 미완료 시도 3건과 로그를 보존했으며 다른 세션은 조작하지 않았다. 후보·자료·관문은 그대로다.

## 5. 착지 연구와 기각된 가설

| 단계 | 시도와 관측 | 처리 |
|---|---|---|
| A–A2 | 접촉 모델·침하·미분과 잡음 분기를 분리 검토 | 비용·오감지 관문을 그대로 적용 |
| A3–A4 | 질량 구간 착지와 정상 보정 확대 | 일부 구간의 G4 개선, 경계 비용·release 오감지 보존 |
| A5 | release_memory_v5, 끝 3.88 s 착지 | 개발 G4·비용 통과, release 충돌 13→11로 비채택 |
| A5–A6 | L2d 집합 소속, 강성/수평 위치 불확실성 | 서보 개발 +19, 독립 약충돌 +14; 토크는 섭동 오감지로 제외 |
| A6 | 수직 충돌 지도 | R1 48→착지 33/56, release 뒤 +z 최소 15→30 N |
| A7 | release_reinit 관측기 상태 재초기화 | H15 기준 0/9, D4 손실 3·오감지 증가 49조건으로 탈락 |

H13(더 이른 종료로 계단 감소)은 3.85–3.86 s의 peak 1.47–2.26 mg 때문에 기각했다. H15는 지정한 재초기화 후보의 사전 기준에 미달했다. 작업대 기억 자체가 무관하다는 증명은 아니다. 이전 지지 투영 55.854 N을 현재 46.990 N에서 통째로 빼 정상 해제에서 −8.864 N이 됐고, 이후 fast innovation이 +7.369 N으로 바뀌었다. 착지는 이번 채택 범위에서 제외했으며 원래 G4 기준을 완화하지 않았다.

## 6. 남은 문제와 범위

G4는 최종 두 방식에서 여전히 미달이다. 최대 1 ms, 서보 포화가 있는 개발 조건, 기존 섭동 오감지, 토크 감지 이득 없음, 약한 수직 충돌의 모호성, 숨은 조건 일반화도 해결되지 않았다. 착지의 부분 성공은 최종에 적용하지 않는다. C1의 검출 개선과 달리 G4·최대 1 ms 등의 미달은 남는다. 자동 충돌 후 정지·회피는 구현하지 않았다.

새 학습·패키지 설치·GPU·원격 작업 0, 원본/작업 sim·model 변경 0이다. Stage B의 실제 루프는 parity 8+holdout 456+공식 58=522회와 별도 합성 Plant 1스텝이다. BR에서는 새 실제 평가 루프 0회다. 성공 저장 대조는 B1/B3 각 980회, 감사 시간 재생은 32회다. 실시간 우선순위의 하드 상한, 연속 강성/비선형 COM 보장, 추가 실제 로봇 시험은 미실행이다. 기존 영상은 R7 최종을 설명하며 C1의 새 영상은 만들지 않았다. 현재 C1 트리에서 validate_media는 R7 후보 source binding 실패를 냈다. 동결된 검사 코드는 고치지 않고, 공개 미디어·manifest·기록은 같은 바이트인 별도 과거 영상 검사 사본에서 my_detector.py만 보관 R7 A_STATE.py로 재기준화해 출처를 확인한다. 이 사본에서는 제출 전체 시험을 돌리지 않는다. 실제 C1의 전체 시험·계약·공개 파일 감사와 과거 영상 출처 검사는 별개다.

'''
    text+='원자료: '+', '.join(f'[{label}]({prefix}{D}{name})' for label,name in [('판정','R8_ADOPTION.json'),('holdout','R8_HOLDOUT_REVIEW.json'),('공식','R8_OFFICIAL_REGRESSION.json'),('저장 대조','R8_SAVED_REVIEW.json'),('감사 시간','R8_AUDIT_TIMING.json'),('사전 protocol','PROTOCOL_R8.json')])+'.\n'
    return text


def methods():
    return '''# R8 방법과 재현

## 고정된 C1
서보 제어 R1·감지 R7에 L2d를 OR로 더한다. 토크는 기존 R1 감지 그대로다. 제어 지령·모델·기존 분기의 임계와 게이트를 바꾸지 않았다. 알려진 질량·COM, 엔코더·그 차분 속도, 모터/JTS 토크, grip, 교시 기준과 공개 명목 모델만 입력으로 쓴다. 실제 링크각·실제 접촉력·센서 숨은 offset/seed·충돌 라벨은 runtime 입력이나 학습에 쓰지 않는다.

## 수직 하중에 둔감한 채널
수직력은 수직 축 둘레 토크를 만들지 않는다. 이를 이용해 J1의 정상 역동역학 차이를 별도 관측한다. 임계는 정상-only 자료로 정했고 새 충돌 결과로 낮추지 않았다. 임계·모델은 A5B 이후 불변이다. 0 kg에는 별도 정상 모델을 쓴다.

강성 가설 K/K0∈{0.8,1,1.2}마다 엔코더−K⁻¹JTS로 링크 자세를 근사하고, 극 40 s⁻¹의 임계 감쇠 필터와 25 s⁻¹ 외란 필터를 쓴다. 수평 COM 위치 ±1 mm 중앙 차분을 같은 필터에 통과시켜 민감도 S_j를 얻는다. 공칭 교시 파지 회전을 쓰며 실제 파지 회전을 읽지 않는다. 창 앞 100 ms 상수 기준선과 정상 회귀를 뺀 오차를 e_j라 하면 새 통계는 직접 유도한

`s(t) = min_j max(0, |e_j(t)| − 0.03 ||S_j(t)||₂)`

이다. 30 mm 국소 위치 집합과 이산 강성 가설 가운데 하나라도 정상이면 새 경보를 억제한다. 연속 강성 전체나 비선형 위치 오차의 안정성·검출 보장이 아니다. 접촉/진입 조건·정상 임계·4표본 지속·기존 경보 뒤 latch는 기존 L2d 그대로다. 기존 경보를 지우지 않는다. 새로운 제어기가 아니므로 강체 안정성 정리를 탄성 관절·1 ms 플랜트의 보장으로 쓰지 않는다.

## 검증 순서와 재현 경계
[PROTOCOL_R8](../results/v2_protocol/PROTOCOL_R8.json) → 조립·저장 동등성·감사 시간·새 실제 대조 → [동결](../results/v2_protocol/R8_FREEZE.json) → 동결본 사전 재생 → 새 봉인 holdout 한 번 → 원래 공식 평가 방식별 한 번이다. R7 생성 규칙을 3회 독립 추출해 참조/충돌/offset을 12/90/12로 늘렸다. 개발 질량·COM·시드를 제외했다. 서보 추가 검출 최소 6/90, 손실/오감지 비증가/물리 바이트/p99 관문을 유지했다. 비교는 기존 최종 R7/R1이다.

키트와 별도 NPZ 파일은 공개본에 포함하지 않는다. 후보 소스에는 고정 모델 바이트가 내장돼 있다. 별도 실행 사본에 허가된 키트의 sim·model을 설치하고 `tools/r8_materialize_models.py --root 실행사본경로`로 정확한 모델 archive를 복원한 뒤 `env -u PYTHONPATH .venv/bin/python tools/r8_build.py --check`와 `-m unittest discover -s tools`를 쓴다. 연구 작업 사본의 tools/r8_study.py는 등록 plan만 실행하고 조건 RNG가 없다. tools/r8_sealed.py는 연구 작업 이력의 동결을 검증한다. 이 수집 도구와 해당 시험은 공개하지 않는 키트 유래 fixture에 의존하므로 공개 배포에서는 제외했다. 공개본에서 제공하는 도구인 것처럼 실행하면 안 된다. Stage B의 fallback 전체 시험 기록은 보존한다. BR에서는 같은 동결 C1을 실제 최종 my_detector.py에 놓고 tools+R8 전체 778시험과 8건 계약을 통과했다. 동결 목록 331개 중 과학 파일은 불변이며 BR에서 명시한 시험 파일 2개만 바뀌었다(329개 원래 SHA 유지). 동결 manifest 자체는 고치지 않았다. 동결의 source_head와 전체 시험 목록은 연구 작업 이력이며 공개 이력/배포 파일 전체와 같지 않다. 원시 센서/물리 스트림과 실행 로그는 별도 연구 증거이며 공개 패키지만으로 그 저장 재생을 새 측정인 것처럼 실행할 수 없다. 공개 문서 검사는 `env -u PYTHONPATH .venv/bin/python tools/r8_report.py --check`다. 공개 staging은 키트 없는 배포 트리여서 실행 검사는 키트를 넣은 C1 overlay에서 한다. validate_media의 R7 검사만은 현재 감지기 경로를 과거 R7에 고정해 두어 C1 공개 트리에서는 실패한다. 검사 코드는 변경하지 않는다. 별도 영상 보관 검사 사본에서 다른 공개 파일·모든 영상·manifest는 그대로 두고 my_detector.py만 보관 R7로 놓아 과거 출처를 확인한다. 실제 최종 C1 전체 시험과 계약 검사를 이 사본에서 실행하거나 그 결과를 대체하지 않는다.

최종 감지는 서보 C1(R7 + L2d)·토크 R1이다. Stage B의 자체 import 허용 목록 실패 뒤 사용자 결정으로 복원했다. 이는 결과를 본 뒤의 예외이며 후보·모델·임계·수치 관문은 바꾸지 않았다. B4·B5를 재실행하지 않고 같은 바이트의 보존 결과를 쓴다. R7 기록 시험은 보관된 R7에, 현재 제출 검사는 실제 C1에 결속한다. 기존 R7 README·PDF는 별도 보존했다. 추가 문헌·교재 인용 없이 기존 개발의 기하·정상-only 관측기를 검증했다.
'''


def render(c):
    short='## R8 추가 검증 — C1 채택(사용자 결정으로 복원)\n\n'+restoration_note()+'\n\n'+efficacy(c)+"""
- 최종 제어는 서보·토크 모두 R1, 감지는 서보 C1(R7 + L2d)·토크 R1이다. 바뀐 제출 파일은 my_detector.py 하나다.
- B4 한 배치 456회에서 서보 66→74/90(+8), 손실·조건별 오감지 증가 0. B5 공식은 서보 18→23/24, 토크 18→18/24다. 아래 수치는 보존 결과이며 재실행하지 않았다.
- 1 ms 최대는 더 나빠졌다. 서보 holdout 1 ms 초과 0→14주기·최대 1.59 ms, 감사 최대 7.99 ms다. p99 675.362 µs는 관문 800 µs 안이다.
- 실제 C1 최종에서 작업 전체 778시험·기본 계약 8건·보고 check 통과. B1의 잘못된 전체 시험 경로와 B5 fallback 기록은 보존했다.
- G4·기존 섭동 오감지·약한 수직 충돌·숨은 조건·토크 감지 이득 없음은 미달 그대로다. 착지는 채택하지 않았다.

[상세 결과](docs/R8_RESULTS.md) · [방법·재현](docs/R8_METHODS.md) · [원래 R7 README](docs/R7_README.md)

아래 R7 전용 표·실행 횟수·판정 복원 경위는 당시 기록이다. 현재 문제 표만 R8 결과에 맞게 갱신했다. 기존 영상은 R7 설명이며 C1의 새 효능 영상이 아니다.

"""
    readme='# Week09 R8 — C1 감지 채널 채택과 남은 과제\n\n'+short+readme_body(c)
    detail=results(c)
    notes=detail.replace('# R8 결과 —','# R8 발표 —',1)
    return [readme,detail,methods(),notes]


def make_pdf(path,c):
    from tools.r7_report import _configure_plot
    _configure_plot()
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    pages=[('R8: C1 채택 — 사용자 결정으로 복원',[
        '최종 제어 R1 / 감지 서보 C1(R7 + L2d)·토크 R1',
        '서보 holdout 66→74(+8/90), 손실·오감지 증가 0',
        '공식 18→23/24. 자체 import 검사 실패 뒤 사용자 결정 복원',
        '후보 바이트·모델·임계·수치 관문 불변, 평가 재실행 0']),
        ('검출 근거',efficacy(c)),('공식 회귀: 원본 출력 자릿수',official(c)),
        ('계산 시간: 모든 이상치 보존',timing(c)),
        ('B5 fallback과 사후 복원 예외',[
        '가이드에는 import 제한 없음. 자체 목록에 표준 모듈 3개 추가',
        'B1 전체 시험을 fallback 경로에서 돌린 조립 검증 누락',
        'R7 과거 시험은 보관된 R7에 결속. SHA·변경 거부 유지',
        '결과 이후 사용자 결정 예외. 실제 C1 전체 778시험·8건 계약 통과']),
        ('착지 연구와 남은 문제',[
        'A5 release 충돌 13→11, A6 수직 지도 48→33/56',
        'H13 더 이른 종료: peak 1.47–2.26 mg, 기각',
        'A7 재초기화 H15 유효 검출 0/9, 후보 탈락',
        'G4, 최대 1 ms, 토크 감지, 섭동 오감지, 수직 모호성 미달']),
        ('검증 범위',[
        '물리 parity 8 + holdout 456 + 공식 58 = 522회',
        '저장 B1/B3 각 980회, 감사 시간 32회, 미완료 재생 3시도 보존',
        '학습·GPU·패키지 추가·원격 작업 0',
        'R7 영상 출처는 보관 R7 사본 검사, 실제 제출 전체 시험은 C1'])]
    meta={'Title':'R8 detector validation','Creator':'r8_report.py','CreationDate':datetime(2000,1,1,tzinfo=timezone.utc),'ModDate':datetime(2000,1,1,tzinfo=timezone.utc)}
    with PdfPages(path,metadata=meta) as pdf:
        for i,(title,body) in enumerate(pages,1):
            f=plt.figure(figsize=(13.333,7.5),facecolor='#f6f8fb');f.text(.04,.92,title,fontsize=23,weight='bold',color='#183153')
            if isinstance(body,list):
                y=.78
                for bullet in body:
                    for line in textwrap.wrap('• '+bullet,66,break_long_words=False):f.text(.055,y,line,fontsize=16);y-=.07
                    y-=.02
            else:
                lines=body.strip().splitlines();head=[x.strip() for x in lines[0].strip('|').split('|')];data=[[x.strip() for x in line.strip('|').split('|')] for line in lines[2:]]
                ax=f.add_axes([.02,.12,.96,.7]);ax.axis('off');t=ax.table(cellText=data,colLabels=head,loc='center',cellLoc='left');t.auto_set_font_size(False);t.set_fontsize(9)
                for (row,col),cell in t.get_celld().items():cell.set_height(.09);cell.set_edgecolor('#b0bec5');cell.set_facecolor('#dbe7f3' if row==0 else 'white')
            f.text(.04,.025,f'R8 검증 | {i}/{len(pages)}',fontsize=9);pdf.savefig(f);plt.close(f)


def generate(root,check=False):
    c=context(root);texts=render(c);pins={D+n:sha(root/(D+n)) for n in INPUTS};pins['docs/R7_README.md']=sha(root/'docs/R7_README.md')
    record={'schema':'R8_REPORT_CURATION.v1','sources':pins,'outputs':OUTPUTS,'measurements_changed':False}
    if check:
        assert read_json(root/(D+'R8_REPORT_CURATION.json'))==record
        assert all((root/p).read_text()==s for p,s in zip(OUTPUTS[:4],texts))
        with tempfile.TemporaryDirectory() as temp:
            p=Path(temp)/'p.pdf';make_pdf(p,c);assert p.read_bytes()==(root/OUTPUTS[4]).read_bytes()
    else:
        for p,s in zip(OUTPUTS[:4],texts):(root/p).parent.mkdir(parents=True,exist_ok=True);(root/p).write_text(s)
        make_pdf(root/OUTPUTS[4],c)
        target=root/(D+'R8_REPORT_CURATION.json')
        (replace_json if target.exists() else write_json)(target,record)
    return True


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,default=ROOT);p.add_argument('--check',action='store_true');a=p.parse_args();generate(a.root,a.check);print('R8 REPORT PASS')
