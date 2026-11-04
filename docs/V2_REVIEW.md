# V2 자체 검토

## 결론

최종 v2는 **제어기 개선안이 아니라 제한된 연구용 검출기 확장**으로 채택한다. 제어기 A/B는 v1과 같고, 정상 동작만으로 학습한 수평 wrench 잔차 채널을 기존 v1 경보와 논리 OR로 결합했다. 봉인 holdout에서는 검출 수가 늘고 무충돌 reference 오감지가 증가하지 않았지만, D4의 B 방식과 분석 전용 섭동에서 오감지가 늘었다. 따라서 모든 조건에서 v1보다 낫거나 현장 배치에 적합하다고 결론내리지 않는다.

공식 평가 8조합은 L1의 오류를 찾는 데 이미 사용했다. 공식 표는 제출 호환성·명목 회귀 증거일 뿐 독립 효능 검정이 아니다. 최종 선택 근거는 공식 결과 뒤 새로 실행한 L3 development와, 소스·가중치·임계값을 동결한 뒤 처음 연 holdout이다. 상세 집계는 [`V2_RESULTS.md`](V2_RESULTS.md), 방법과 수식은 [`V2_METHODS.md`](V2_METHODS.md), 재현 경계는 [`REPRODUCE.md`](../report/REPRODUCE.md)에 있다.

## 방법과 변경 경계

### 정상 잔차 learner

- 예측기 특징은 현재·과거 공개 reference, 알려진 질량과 COM뿐이다. score는 허용 encoder/JTS 신호로 만든 현재 관측기 wrench와 비교한다. plant/log/oracle, case index와 미래값은 쓰지 않는다.
- 충돌 없는 정상 조건에서 77개 basis를 구성하고, v1 관측기가 허용 입력으로 계산한 수평 wrench 두 축을 ridge target으로 사용했다.
- train 조건으로 표준화와 가중치를 정하고, 분리된 무충돌 validation으로 임계값을 한 번 고정했다.
- runtime은 잔차 2축 norm의 `alpha=0.1` EMA, 4-cycle persistence, 8-cycle release를 사용한다. 기존 경보 boolean은 OR로 보존한다. 다만 조기 추가 경보로 평가상 masking될 가능성도 있으므로 검출 성공 수의 보장은 아니며, 대응 사례를 직접 비교했다.

### L3는 재학습이 아닌 gate 계약 복구

L3는 모델을 다시 학습하거나 수치 임계값을 바꾼 방법이 아니다. 최초 protocol은 grip 중이면서 기하학적 작업대 높이 band 밖일 때만 learner를 활성화하도록 등록했다. L1은 payload 질량 조건까지 포함한 기존 `table_allowed` 값을 잘못 재사용해 0 kg에서 learner를 켰다. L3는 공개 nominal Jacobian과 직전 추정 관절각으로 상자 바닥 중심 높이를 계산하여, 모든 질량에 동일한 기하 gate를 적용했다. ridge 가중치, 표준화, 임계값, EMA, persistence, release, 높이 band는 그대로다.

Lie Group 자료는 frame·작용점·power duality 감사를 설계하는 참고로만 사용했다. 공개 문헌 목록과 확인 절은 [`LITERATURE.md`](LITERATURE.md)의 서지와 절 번호를 따른다. 유도는 자체 작성했으며 강체·연속시간 안정성 결과를 탄성 관절, JTS 지연, 1 ms 이산 제어 플랜트의 보장으로 옮기지 않았다.

## 후보 판정

| 후보 | 결과 | 판정 |
|---|---|---|
| C: 법선 bounded admittance | A/B 모두 G4 0/4. 마지막 20 ms 지지 증가 기준을 통과하지 못함 | 탈락. 실제 force sensor가 아닌 지연된 추정 wrench를 쓰며, 고정 0.10 s 복귀와 등록된 전체 독립 bound box의 양립 한계도 남음 |
| L1: 정상 잔차 learner | development 이득 뒤 공식 0 kg 무충돌 오감지 발견 | 탈락. 공식 결과 뒤 임계값을 다시 맞추지 않음 |
| L2: train bounding-box fallback | 탐색 이득은 있었으나 공식 노출 뒤 적용 범위를 정한 후보 | 탐색 기록만 보존, 최종 비선정 |
| L3: 기하 높이 gate 복구 | 새 development와 봉인 holdout에서 검출 증가. 가중치·임계값 불변 | 제한적 연구 채택. D4/G6 오감지 회귀 때문에 일반 우월성·배치 적합성은 부정 |

## 검출 결과

### D4: 동일 180충돌/구조

| 구조 | v1 검출 | v2 검출 | v1→v2 충돌 중 추가 FA | 순수 reference FA | 판정 |
|---|---:|---:|---:|---:|---|
| A | 94 | 111 | 2→2 | 0→0 | 검출 증가, 추가 FA 비증가 |
| B | 91 | 111 | 1→3 | 0→0 | 검출 증가지만 추가 FA 증가 |

G3는 두 구조가 모두 검출 증가와 추가 FA 비증가를 만족해야 하므로 **미달**이다. B의 추가 회귀 두 건을 버리지 않았다.

1. case 293: 8.0 kg, COM `[0.0, -0.02, 0.015]` m, 운반 3.4 s의 `+z` 40 N. v1/v2 모두 46 ms에 검출했지만 추가 FA가 0→1로 늘었다.
2. case 303: 2.0 kg, COM `[0.0, 0.0, -0.02]` m, 운반 3.4 s의 `+z` 100 N. v1/v2 모두 24 ms에 검출했지만 추가 FA가 0→1로 늘었다.

원시 행과 판정 근거는 [`S5_D4.json`](../results/v2_protocol/S5_D4.json)과 [`S5_D4_REVIEW.json`](../results/v2_protocol/S5_D4_REVIEW.json)에 있다.

### 봉인 holdout: 동일 30충돌/구조

| 구조 | v1 검출 | v2 검출 | 충돌 중 추가 FA | 순수 reference FA | offset FA |
|---|---:|---:|---:|---:|---:|
| A | 15 | 19 | 0→0 | 0→0 | 0→0 |
| B | 14 | 18 | 0→0 | 0→0 | 1→1 |

holdout은 동결 후 처음 생성했고 결과를 본 뒤 모델, 임계값 또는 적용 범위를 바꾸지 않았다. 이 60개 충돌에서 A/B 모두 4건씩 더 검출했고 추가 오감지는 늘지 않았다. 다만 유한한 사전 고정 조건의 결과이며 미공개 평가나 임의 분포의 성공 확률을 뜻하지 않는다. 근거는 [`S5_HOLDOUT.json`](../results/v2_protocol/S5_HOLDOUT.json)과 [`S5_HOLDOUT_REVIEW.json`](../results/v2_protocol/S5_HOLDOUT_REVIEW.json)에 있다.

## 제어·하중 이양과 물리 parity

검출기만 바뀌었으므로 물리 제어 개선은 0이다. D4 전체 770쌍, holdout 76쌍, 분석 섭동 68쌍, 합계 **914개의 v1/v2 대응 episode**에서 TCP 오차, 토크 계단, 작업대 힘, 포화, 발산, tool contact 지표가 모두 동일했다.

이 parity는 회귀가 없다는 뜻이지 기존 제어 한계를 해결했다는 뜻이 아니다.

- A는 D4 reference의 두 COM 조건에서 포화 `0.15 %`, `0.0333 %`가 v1과 v2에 그대로 남았다.
- A의 최악 놓기 작업대 peak는 `1.6098 mg`로 그대로다.
- 8 kg 마지막 20 ms 평균 지지는 A `0.164664 N` (`0.002098 mg`), B `8.751053 N` (`0.111507 mg`)이며 v1과 v2가 같다.
- 마지막 20 ms minimum은 A `0 N`, B `0.298418 N`; 놓기 peak는 각각 `1.004467 mg`, `1.013993 mg`이다. peak 상한은 지켰지만 지속 지지와 v1 대비 증가 조건을 만족하지 못해 G4는 미달이다.

## 분석 섭동과 G6

분석 전용 섭동은 noise, 새 noise seed, friction, stiffness, corner reference와 별도 collision 조건을 포함한다. 공식 점수나 학습 데이터가 아니다. 무충돌 reference 28조건/구조에서 false-alarm rise는 다음처럼 악화했다.

| 구조 | v1→v2 전체 FA | 실패 family의 v1→v2 기여 |
|---|---:|---|
| A | 4→6 | stiffness 2→3, corner 2→3 |
| B | 7→9 | stiffness 4→5, corner 3→4 |

collision 6조건/구조는 모두 검출했지만, 무충돌 섭동 FA가 구조별 2회씩 늘었다. 상속된 제어 악화 조건도 줄지 않았으므로 G6는 **미달**이다. 전체 기록은 [`S5_PERTURBATIONS.json`](../results/v2_protocol/S5_PERTURBATIONS.json)에 있다.

## 계산시간과 outlier 조사

G5는 D4, holdout, 별도 timing, 분석 섭동의 최종 v2 A/B 관측 wall 최대가 모두 1 ms 미만이어야 통과한다. 각 행의 median/p99 중 가장 큰 값과 전 데이터 최대는 다음과 같다.

| 구조 | 측정 행 | 최대 row median | 최대 row p99 | 관측 wall 최대 | 같은 cycle thread CPU | context switch (자발/비자발) |
|---|---:|---:|---:|---:|---:|---:|
| A | 460 | 134.314 us | 346.762 us | 12,151.685 us | 12,148.944 us | 0 / 0 |
| B | 460 | 173.455 us | 350.531 us | 160,578.126 us | 39,373.253 us | 2,919 / 0 |

A 최악은 D4 case 91의 3.7 kg 무충돌 reference였다. 직전/직후 가용 RAM은 약 17.05/17.04 GiB, 1분 load average는 2.47이었다. B 최악은 D4 case 328의 8 kg 충돌 조건이었다. 직전/직후 가용 RAM은 약 22.30/22.23 GiB, 1분 load average는 1.86이었다. A는 wall과 CPU가 거의 같고, B는 큰 CPU 시간과 많은 자발 context switch가 함께 나타났다. 따라서 모든 초과를 단순히 외부 부하나 스케줄링 탓으로 돌릴 수 없다. 최종 v2에서 보존한 1 ms 이상 outlier는 A 6개, B 18개이며 G5는 **미달**이다.

후향적 원인 조사는 원래 최악 조건을 v1/v2 각 3회 재실행해 자연 GC, page fault와 context switch를 수동 관측했다. 12회 중 두 outlier가 자연 GC와 겹쳤다. 두 건은 v1 A 재현에서 관측했으며, v2 6회는 각각 최대 1 ms 미만이었다. 작은 선택 조건의 재실행으로 원래 v2 최대값을 지우거나 버전별 시간 우위를 주장하지 않는다.

- 7.294 ms outlier: generation 1 GC span 약 7.140 ms와 중첩, page fault와 context switch 증분 0.
- 14.193 ms outlier: generation 2 GC span 약 13.937 ms와 중첩, page fault와 context switch 증분 0.

이는 일부 CPU형 지연과 GC가 동시에 일어난 관측 증거일 뿐 인과 증명은 아니다. 특히 원래 B의 160.578 ms outlier는 재현되지 않아 고유 원인은 **미해결**이다. 후향 진단은 원래 최대값을 대체하지 않는다. 원 측정은 [`FINAL_ASSESSMENT.json`](../results/v2_protocol/FINAL_ASSESSMENT.json), 수동 진단은 [`S5_DIAGNOSTICS.json`](../results/v2_protocol/S5_DIAGNOSTICS.json)에 있다.

## G7과 기하 감사의 한계

frame·작용점 변환과 virtual-power 일관성은 수치 감사로 확인했다. 그러나 같은 순간의 가능한 작업대 법선 wrench는 지지와 수직 충돌에서 동일할 수 있다. 좌표변환은 새 정보를 만들지 않으므로 현재 허용 입력만으로 그 두 원천을 일반적으로 식별하지 못한다. 이 반례는 **한 순간의 비식별성**을 보인 것이며 모든 동적 history가 구분 불가능하다고 주장하지 않는다. G7은 방법 평가와 한계 명시는 완료했지만 문제 자체는 미해결이다.

## 효용성 판정

### 원리

정상 동작의 질량·COM·reference 의존 wrench baseline을 작은 ridge로 예측하고 기존 관측기와의 수평 잔차를 감시한다. 기존 경보를 OR로 보존하며, 기하 높이 gate로 정상 작업대 지지 구간의 learner 활성화를 제한한다.

### 범위

새 development와 봉인 holdout에서는 약한 수평 충돌 검출 증가를 확인했다. D4의 B 방식과 분석 섭동에서는 오감지가 악화했다. 결과는 실행한 유한 시뮬레이션 조건에만 적용하며 확률적 일반화, 미공개 평가 통과, 실제 로봇 성능을 보장하지 않는다.

### 대가

77-basis NumPy 추론, EMA와 gate 계산이 추가된다. 일반 주기의 p99는 1 ms보다 작았지만 보존한 최대 outlier가 기준을 크게 넘었다. 모델·임계값 관리, 적용 범위 감사와 outlier 조사가 추가 비용이다.

### 현장 적용성

현재 결과만으로 hard real-time, collision stop, 기능 안전 또는 사람-로봇 협업 안전을 주장할 수 없다. 실제 배치 전에는 별도 sensing/process constraint, 안전 반응, 실기 하드웨어 검증, 실행시간 상한 제어가 필요하다. 수직 동일-wrench 모호성도 추가 관측 없이 남는다.

## 최종 목표 판정

| 목표 | 판정 | 근거 |
|---|---|---|
| G1 | 충족* | 공식 호환성 표에서 8 kg 15 N 운반 A/B 45/43 ms. `*` 공식 세트는 독립 효능 증거가 아님 |
| G2 | 충족* | 공식 호환성 표에서 8 kg 10 N 운반 A/B 62/56 ms. 같은 독립성 제한 적용 |
| G3 | 미달 | D4 B의 충돌 중 추가 FA 1→3 |
| G4 | 미달 | C 후보 A/B 0/4, 최종 v2 지속 이양은 v1과 동일 |
| G5 | 미달 | 최종 v2 관측 최대 A 12.152 ms, B 160.578 ms |
| G6 | 미달 | 제어 악화 조건 불변, 분석 섭동 무충돌 FA A 4→6/B 7→9 |
| G7 | 미해결 | 순간 동일 법선 wrench 반례와 식별 한계를 확인했지만 원천 분리는 못함 |

## 미실행·검증 한계

- 실제 로봇, 하드웨어 safety reaction, hard real-time scheduler는 검증하지 않았다.
- 15주차 미공개 조건과 Windows `.bat` 실행은 검증하지 않았다.
- 공식 환경에 pytest를 추가하지 않았고 dependency-free `unittest`를 사용했다.
- 전용 LSP, mypy, pyright 정적진단은 사용할 수 없어 실행하지 않았다. AST 입력 감사와 독립 runtime review로 보완했지만 이를 형식 검증으로 보지 않는다.
- 이 문서는 결과 데이터의 자체 검토다. 최종 공개 패키지 감사, media 검증, 기본 제출 파일명 overlay의 완료를 선행 사실로 주장하지 않는다.
