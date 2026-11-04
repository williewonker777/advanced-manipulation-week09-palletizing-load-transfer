# V2 결과

## 증거 분리

공식 8조합은 호환성 회귀표이며 디버깅에 소비되어 독립 효능 증거가 아니다. 최종 선택 주장은 fresh non-official L3 연구와 sealed holdout에서만 가져온다.

## 목표 판정

| 목표 | 판정 |
|---|---|
| G1 | True |
| G2 | True |
| G3 | False |
| G4 | False |
| G5 | False |
| G6 | False |
| G7 | not solved; identical instantaneous vertical wrench remains ambiguous |

## 누락 없는 supplemental 집계

| 데이터/그룹 | pure ref FA | offset episodes/FA/contact miss | collision extra FA | detected | miss | masked |
|---|---:|---:|---:|---:|---:|---:|
| fresh_validation/L3/servo | 0 | 0/0/0 | 0 | 0 | 0 | 0 |
| fresh_validation/L3/torque | 0 | 0/0/0 | 0 | 0 | 0 | 0 |
| fresh_validation/v1/servo | 0 | 0/0/0 | 0 | 0 | 0 | 0 |
| fresh_validation/v1/torque | 0 | 0/0/0 | 0 | 0 | 0 | 0 |
| fresh_development/L3/servo | 0 | 0/0/0 | 0 | 28 | 36 | 0 |
| fresh_development/L3/torque | 0 | 0/0/0 | 0 | 38 | 26 | 0 |
| fresh_development/v1/servo | 0 | 0/0/0 | 0 | 28 | 36 | 0 |
| fresh_development/v1/torque | 0 | 0/0/0 | 0 | 27 | 37 | 0 |
| d4/v1/servo | 0 | 16/8/0 | 2 | 94 | 86 | 0 |
| d4/v1/torque | 0 | 16/6/0 | 1 | 91 | 89 | 0 |
| d4/v2/servo | 0 | 16/8/0 | 2 | 111 | 69 | 0 |
| d4/v2/torque | 0 | 16/6/0 | 3 | 111 | 69 | 0 |
| sealed_holdout/v1/servo | 0 | 4/0/0 | 0 | 15 | 15 | 0 |
| sealed_holdout/v1/torque | 0 | 4/1/0 | 0 | 14 | 16 | 0 |
| sealed_holdout/v2/servo | 0 | 4/0/0 | 0 | 19 | 11 | 0 |
| sealed_holdout/v2/torque | 0 | 4/1/0 | 0 | 18 | 12 | 0 |
| perturbations/v1/servo | 4 | 0/0/0 | 0 | 6 | 0 | 0 |
| perturbations/v1/torque | 7 | 0/0/0 | 1 | 6 | 0 | 0 |
| perturbations/v2/servo | 6 | 0/0/0 | 0 | 6 | 0 | 0 |
| perturbations/v2/torque | 9 | 0/0/0 | 1 | 6 | 0 | 0 |

## 해석

검출기만 바뀌었으므로 v2 제어 개선은 0이다. 제어 추적 회귀가 없다는 범위와 검출 개선을 분리한다. D4와 holdout의 질량/COM/방향별 failure map은 유한 조건 결과이며 확률이나 실제 현장 성능이 아니다.

Timing은 D4/holdout/timing/perturbation의 모든 outlier를 포함한다. G5는 final v2 A/B만 판정하고 v1 및 전체 최악은 비교로 분리한다. 정확한 worst-wall cycle의 thread CPU/context switch를 함께 보여 주며 wall-only 초과를 context switch로 단정하지 않는다.
