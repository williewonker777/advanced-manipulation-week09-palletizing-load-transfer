# R3 결과

A=서보, B=토크, FA=오감지다. 강제점은 R1 관측·물리점에 후보를 재생한 검사, 실제 후보점은 후보 자체 궤적의 검사다. radius는 이산 폐루프 고유값의 최대 절댓값, FD는 유한차분 수렴 검사다.

## 최종 방식별 판정

| 방식 | 판정 | 탈락 진단 후보 | 최종 제출 | 근거 |
|---|---|---|---|---|
| 서보 A | 탈락 | A_handoff_long | R1 | 사전 관문 탈락 기록 유지; D4·섭동 비용·오감지 실패도 확인 |
| 토크 B | 탈락 | B_mass | R1 | 사전 관문 탈락 기록 유지; D4·섭동 비용·오감지 실패도 확인 |

탈락 진단 후보는 최종 제출이 아니다. 두 방식 모두 독립적으로 판정해 최종 R1을 유지한다.

## 등록 후보 5개

| 후보 | 방법 | 방식 | 사전 검증 비용 | 사전 검증 추적비 | 강제점 국소 검사 | 실제 후보점 국소 검사 | 계산 p99/최대 [us] | 판정 | 근거 |
|---|---|---|---|---|---|---|---|---|---|
| A_handoff_short | DOB 4.5 + handoff | 서보 A | 실패; 최악 1.095× | 평균 RMS 비 0.955 / 최대 비 0.850 (참고; 효능은 섭동에서 판정) | 22/28 radius·FD 동시 통과 | 미실행: 비선정 | 320.0/615.0 us | 탈락 | 질량별 비용·국소 안정성/FD |
| A_handoff_long | DOB 4.5 + handoff | 서보 A | 실패; 최악 1.061× | 평균 RMS 비 0.976 / 최대 비 0.948 (참고; 효능은 섭동에서 판정) | 22/28 radius·FD 동시 통과 | 54/64 radius·FD 동시 통과 | 364.4/7156.1 us | 탈락 | 질량별 비용·국소 안정성/FD |
| B_phase | P-IDC 4.5/1 + phase | 토크 B | 실패; 최악 3.841× | 평균 RMS 비 0.906 / 최대 비 1.054 | 45/50 radius·FD 동시 통과 | 미실행: 비선정 | 554.4/843.3 us | 탈락 | 질량별 비용·국소 안정성/FD·10% 효능 |
| B_mass | P-IDC 4.5/1 + mass | 토크 B | 통과; 최악 1.034× | 평균 RMS 비 0.834 / 최대 비 0.838 | 43/50 radius·FD 동시 통과 | 45/48 radius·FD 동시 통과 | 672.0/9649.0 us | 탈락 | 국소 안정성/FD |
| B_mass_phase | P-IDC 4.5/1 + mass_phase | 토크 B | 실패; 최악 1.620× | 평균 RMS 비 0.911 / 최대 비 0.928 | 45/50 radius·FD 동시 통과 | 미실행: 비선정 | 598.9/798.2 us | 탈락 | 질량별 비용·국소 안정성/FD·10% 효능 |

후보 표의 근거는 사전 검증 시점 기록이다. R1과의 국소 비교 및 최종 탈락을 뒷받침하는 D4·섭동 근거는 아래에 함께 제시한다.

## 서보 A 사전 검증: R1 → 미채택 진단 후보

| 후보 | 방식 | 질량 [kg] | R1 RMS [mm] | 후보 RMS [mm] | R1 최대 [mm] | 후보 최대 [mm] | R1 놓기 계단 [N·m] | 후보 놓기 계단 [N·m] | 판정 | 최종 R3 |
|---|---|---|---|---|---|---|---|---|---|---|
| A_handoff_short | 서보 A | 1.27 | 1.158 | 1.117 | 3.838 | 3.257 | 4.753 | 5.203 | 탈락 | R1 |
| A_handoff_short | 서보 A | 3.67 | 1.148 | 1.101 | 3.788 | 3.207 | 6.948 | 7.304 | 탈락 | R1 |
| A_handoff_short | 서보 A | 5.47 | 1.154 | 1.097 | 3.826 | 3.249 | 10.636 | 11.108 | 탈락 | R1 |
| A_handoff_short | 서보 A | 7.17 | 1.177 | 1.111 | 3.868 | 3.301 | 14.311 | 14.089 | 탈락 | R1 |
| A_handoff_long | 서보 A | 1.27 | 1.158 | 1.140 | 3.838 | 3.666 | 4.753 | 4.690 | 탈락 | R1 |
| A_handoff_long | 서보 A | 3.67 | 1.148 | 1.125 | 3.788 | 3.589 | 6.948 | 7.372 | 탈락 | R1 |
| A_handoff_long | 서보 A | 5.47 | 1.154 | 1.123 | 3.826 | 3.614 | 10.636 | 10.846 | 탈락 | R1 |
| A_handoff_long | 서보 A | 7.17 | 1.177 | 1.139 | 3.868 | 3.653 | 14.311 | 13.789 | 탈락 | R1 |

## 토크 B 사전 검증: R1 → 미채택 진단 후보

| 후보 | 방식 | 질량 [kg] | R1 RMS [mm] | 후보 RMS [mm] | R1 최대 [mm] | 후보 최대 [mm] | R1 놓기 계단 [N·m] | 후보 놓기 계단 [N·m] | 판정 | 최종 R3 |
|---|---|---|---|---|---|---|---|---|---|---|
| B_phase | 토크 B | 1.27 | 2.048 | 2.012 | 4.006 | 4.981 | 4.923 | 5.304 | 탈락 | R1 |
| B_phase | 토크 B | 3.67 | 2.263 | 2.101 | 4.075 | 5.081 | 6.191 | 6.863 | 탈락 | R1 |
| B_phase | 토크 B | 5.47 | 2.587 | 2.292 | 5.727 | 5.190 | 9.102 | 9.650 | 탈락 | R1 |
| B_phase | 토크 B | 7.17 | 3.028 | 2.500 | 7.608 | 6.238 | 12.190 | 12.572 | 탈락 | R1 |
| B_mass | 토크 B | 1.27 | 2.048 | 2.048 | 4.006 | 4.006 | 4.923 | 4.923 | 탈락 | R1 |
| B_mass | 토크 B | 3.67 | 2.263 | 2.123 | 4.075 | 4.140 | 6.191 | 6.301 | 탈락 | R1 |
| B_mass | 토크 B | 5.47 | 2.587 | 1.932 | 5.727 | 4.356 | 9.102 | 9.249 | 탈락 | R1 |
| B_mass | 토크 B | 7.17 | 3.028 | 1.967 | 7.608 | 4.376 | 12.190 | 12.599 | 탈락 | R1 |
| B_mass_phase | 토크 B | 1.27 | 2.048 | 2.048 | 4.006 | 4.006 | 4.923 | 4.923 | 탈락 | R1 |
| B_mass_phase | 토크 B | 3.67 | 2.263 | 2.115 | 4.075 | 4.014 | 6.191 | 6.191 | 탈락 | R1 |
| B_mass_phase | 토크 B | 5.47 | 2.587 | 2.290 | 5.727 | 5.184 | 9.102 | 9.644 | 탈락 | R1 |
| B_mass_phase | 토크 B | 7.17 | 3.028 | 2.500 | 7.608 | 6.238 | 12.190 | 12.572 | 탈락 | R1 |

## T1–T6

기준/v1/R1은 보존된 동일 조건 수치이고 최종 R3=R1은 같은 소스다. 특히 T3는 R2와 같은 34조건의 보존값이며 새 R3 조건에서 네 버전을 재실행한 효능 표가 아니다. 새 검증·D4·섭동은 아래에서 분리한다.

| 항목 | 조건·단위 | 기준 | v1 | R1 | R3 |
|---|---|---|---|---|---|
| T1 놓기 전 마지막 20 ms 평균 지지율 | 8 kg, 동일 COM, 놓기 직전 20 ms 평균 [%mg, A/B] | A 0.0000 / B 0.0557 | A 0.2098 / B 11.1507 | A 0.2098 / B 11.1507 | A 0.2098 / B 11.1507 |
| T2 놓기 토크 계단 | 8 kg, 동일 COM, 놓기 토크 계단 [N·m, A/B] | A 73.842 / B 74.797 | A 14.626 / B 13.126 | A 14.626 / B 13.126 | A 14.626 / B 13.126 |
| T3 분석 섭동 최악 TCP/놓기힘/오감지 | R2와 같은 34조건의 보존값: 방식별 무충돌 28+외력 6 [mm/N/오감지] | A 22.731 mm / 208.997 N / 오감지 221; B 17.171 mm / 352.140 N / 오감지 309 | A 14.777 mm / 212.734 N / 오감지 4; B 45.652 mm / 362.600 N / 오감지 8 | A 14.777 mm / 212.734 N / 오감지 6; B 45.652 mm / 362.600 N / 오감지 10 | A 14.777 mm / 212.734 N / 오감지 6; B 45.652 mm / 362.600 N / 오감지 10 |
| T4 질량·COM 격자 포화/놓기 최대값 | 방식별 무충돌 189조건 [포화 조건 수/놓기 힘 mg] | A 포화 0/189, 최대값 2.0359 mg; B 포화 0/189, 최대값 9.8822 mg | A 포화 2/189, 최대값 1.6098 mg; B 포화 0/189, 최대값 1.0358 mg | A 포화 2/189, 최대값 1.6098 mg; B 포화 0/189, 최대값 1.0358 mg | A 포화 2/189, 최대값 1.6098 mg; B 포화 0/189, 최대값 1.0358 mg |
| T5 TCP 추적 RMS/최대 | 8 kg, 동일 COM, 무충돌 [RMS/최대 mm, A/B] | A 6.144/14.000; B 7.042/13.183 | A 1.174/3.914; B 3.232/8.444 | A 1.174/3.914; B 3.232/8.444 | A 1.174/3.914; B 3.232/8.444 |
| T6 D4 감지/놓침/기존 경보·구분 불가 | 방식별 충돌 180조건, 기존 경보·구분 불가 별도 [검출/놓침/구분 불가] | A 88/43/49; B 78/42/60 | A 94/86/0; B 91/89/0 | A 111/69/0; B 111/69/0 | A 111/69/0; B 111/69/0 |

## 국소 안정성: 같은 점의 R1 비교

| 후보 | 강제점 통과 | 실제 후보점 통과 | 같은 강제점 R1 |
|---|---|---|---|
| A_handoff_short | 22/28 | 미실행: 비선정 | 22/28 |
| A_handoff_long | 22/28 | 54/64 | 22/28 |
| B_phase | 45/50 | 미실행: 비선정 | 46/50 |
| B_mass | 43/50 | 45/48 | 46/50 |
| B_mass_phase | 45/50 | 미실행: 비선정 | 46/50 |

표의 분자는 radius와 FD를 동시에 통과한 점 수다. R1 열은 같은 강제점 격자만 비교하며 후보 실제점에 대응하는 R1 재실행은 아니다.

- 임계값은 R2와 같지만 검사점은 R3에서 늘었다. A의 강제 실패 6점과 실제 실패 10점은 모두 추가한 3.1–3.6 s 구간이고 기존 2.5, 3.75, 3.89 s에는 없다.
- A의 강제 실패 6점은 R1도 같은 점에서 실패했다. 강제 실패 6점에서, 두 설정을 통틀어 R1 대비 radius 차이 최대는 3.78e-05로 약 3.8e-5다.
- B-mass의 강제 실패 7점은 radius 실패 3점과 radius 통과·FD 미수렴 4점이다. radius 실패 3점은 R1도 실패했고, 5.65/4.4 kg의 2.5 s 두 점은 B가 1에 더 가깝다. R1이 실패한 7.7 kg, 2.5 s는 B가 통과했다.
- 같은 강제점 R1은 R1 관측·물리점 격자의 비교이며 후보 실제 궤적의 R1 재생이 아니다. 이 국소 관문만으로 후보와 R1을 가를 수 없다. 규칙과 탈락 판정은 그대로 두고 D4·섭동 비용·오감지 근거를 함께 판단한다.
- D4 전체 385조건에서 R1 A도 포화 4회/최대 0.166667%로 상한 2회/0.15%를 넘었다. A-long도 4회이고 최대율은 0.150000%다. 무충돌 189조건의 2회와 혼동하지 않는다.

## D4·분석 섭동의 비용 초과 분포

| 구분 | 방식 | 1.05 초과 행 | 무충돌 전체 행 | 초과 지표별 행 | 초과 질량별 행 | 최악 비 |
|---|---|---|---|---|---|---|
| D4 | 서보 A | 20 | 189 | 놓기 계단 20행 | 1.3 kg 8행; 2 kg 4행; 3.7 kg 3행; 4 kg 1행; 6.25 kg 2행; 8 kg 2행 | 1.136 |
| D4 | 토크 B | 42 | 189 | 놓기 계단 38행; 집기 계단 4행 | 3.7 kg 7행; 4 kg 27행; 6.25 kg 6행; 8 kg 2행 | 1.255 |
| 분석 섭동 | 서보 A | 2 | 28 | 놓기 계단 2행 | 8 kg 2행 | 1.058 |
| 분석 섭동 | 토크 B | 16 | 28 | 놓기 계단 10행; 집기 계단 9행; 놓기 힘 3행; TCP 최대 1행 | 4.3 kg 9행; 8 kg 7행 | 1.473 |

### 질량·지표별 분포

| 구분 | 방식 | 질량·지표별 행 |
|---|---|---|
| D4 | 서보 A | 1.3 kg 놓기 계단 8행; 2 kg 놓기 계단 4행; 3.7 kg 놓기 계단 3행; 4 kg 놓기 계단 1행; 6.25 kg 놓기 계단 2행; 8 kg 놓기 계단 2행 |
| D4 | 토크 B | 3.7 kg 놓기 계단 7행; 4 kg 놓기 계단 27행; 6.25 kg 집기 계단 2행; 6.25 kg 놓기 계단 4행; 8 kg 집기 계단 2행 |
| 분석 섭동 | 서보 A | 8 kg 놓기 계단 2행 |
| 분석 섭동 | 토크 B | 4.3 kg TCP 최대 1행; 4.3 kg 놓기 힘 2행; 4.3 kg 집기 계단 3행; 4.3 kg 놓기 계단 6행; 8 kg 놓기 힘 1행; 8 kg 집기 계단 6행; 8 kg 놓기 계단 4행 |

- 비용은 같은 무충돌 조건마다 후보/R1 비를 계산했다. 한 행에서 여러 지표가 초과할 수 있어 지표 합계와 초과 행 수는 다르다. 관문 1.05는 완화하지 않았다.
- B-mass의 8 kg, 강성 1.2배 조건(조건 16)은 오감지 2→32회, 강성 1.2배·마찰 0.8배 복합 모서리 조건(조건 17)은 2→28회였다. 반면 강성 0.8배(조건 12)는 3→2회로 감소했다.
- 양의 질량 무충돌 섭동 22행은 각 같은 조건의 비를 먼저 계산한 산술평균이다. 아래 평균은 효능 관문이 아닌 참고값이며, 평균 개선이 조건별 비용·오감지 통과를 뜻하지 않는다.

### 양의 질량 무충돌 섭동의 참고 평균

| 방식 | 평균한 행 수 | 평균 RMS 비 | 평균 최대 TCP 비 |
|---|---|---|---|
| 서보 A | 22 | 0.979 | 0.951 |
| 토크 B | 22 | 0.654 | 0.688 |

**효능 관문이 아닌 참고값이다.** 평균 개선으로 같은 조건의 비용·오감지 관문을 대체하지 않는다.

## 재설계 사용 여부

| 방식 | 사용 횟수 | 근거 | 개발 검증 전 기록 | 임의 파라미터 탐색 |
|---|---|---|---|---|
| 서보 A | 0 | 비용 외 수렴한 국소 관문도 실패했다. 미식별 위상에 추가 시각/gain을 정할 근거가 없어 비용 중심 재설계 조건이 아니다. | 예 | 아니오 |
| 토크 B | 0 | B_mass는 비용·효능을 이미 통과했다. 막힌 것은 국소 관문이므로 비용 중심 재설계 조건이 아니다. | 예 | 아니오 |

- 재설계 결정은 R1 비교 파일보다 먼저 저장됐다. 결정은 연구 작업 커밋 5a86c02, R1 비교 파일은 그 뒤 커밋 8b57c79에서 추가됐다. 두 커밋은 비공개 작업 이력이라 공개 이력에서는 찾을 수 없다. A의 당시 국소 관문 실패는 그 뒤 R1과 같은 위치로 확인돼 단독 식별 근거가 약해졌다.
- 결정 기록 자체는 바꾸지 않는다. 질량별 비선형 위상·원인 미식별과 추가 시각/gain의 인과적 해결책 부재 사유는 그대로 유효하다. 이후 holdout·공식 결과로 재설계를 하지 않았다.

## Lie group 적용

| 방법 | 서지·절 | 사용 식 | 탄성·1 ms ZOH·지연 변경 | 측정 효과·한계 | 판정 |
|---|---|---|---|---|---|
| A motor DOB | 자체 유도 | dhat=z+LBv; quintic handoff | elastic motor + 자체 Drive queue; 1 ms ZOH, command/JTS delay | 유클리드 R1 대비 A-long 최대 TCP 약 4.5–5.6% 감소; 독립 Lie 효과 아님 | 탈락 |
| B P-IDC | Lie Group Robot Control §10.1.6 | tau=M*a_r+C*v_r+g+M*r*s | elastic lift + clip + mass/phase schedule; 1 ms ZOH, 한 주기 지연 | B-mass 질량별 RMS 비 평균 0.834; 제어·스케줄 동시 변경이라 기하 단독 효과 미분리 | 탈락 |
| B pRNE | Lie Group Robotics 목차 §8.3, 본문 미확인; 자체 유도 | ad(V), X^T, C(q,v)u; Mdot=C+C^T | rigid nominal C만 계산; elastic 플랜트는 별도 1 ms ZOH/delay 국소 검사 | 유한차분 항등식 오차 2.8e-09; 같은 관측 p99 772.2→475.5 us (고정 순서 참고, 기하 단독 효과 미분리) | 탈락 |

## 정상 오감지

| 구분 | 방식 | 오감지 수 | R1 대비 증가 | 상태 |
|---|---|---|---|---|
| 사전 검증 후보 전체 | 서보 A | 0 | 0 | 통과 |
| 사전 검증 후보 전체 | 토크 B | 0 | 0 | 통과 |
| 개발 검증 R1 | 서보 A | 0 | 0 | 기준 |
| 개발 검증 후보 | 서보 A | 0 | 0 | 통과 |
| 개발 검증 R1 | 토크 B | 0 | 0 | 기준 |
| 개발 검증 후보 | 토크 B | 0 | 0 | 통과 |
| D4 R1 | 서보 A | 0 | 0 | 기준 |
| D4 후보 | 서보 A | 0 | 0 | 통과 |
| D4 R1 | 토크 B | 0 | 0 | 기준 |
| D4 후보 | 토크 B | 0 | 0 | 통과 |
| 분석 섭동 R1 | 서보 A | 6 | 0 | 기준 |
| 분석 섭동 후보 | 서보 A | 5 | -1 | 실패 |
| 분석 섭동 R1 | 토크 B | 9 | 0 | 기준 |
| 분석 섭동 후보 | 토크 B | 64 | 55 | 실패 |

R1 행은 비교 기준이다. 기준의 오감지를 통과로 바꾼 것이 아니며 수치는 그대로 보존했다.

## 계산 시간과 모든 이상치

| 구분 | 방식 | 최대 행 p99 [us] | 관측 최대 [us] | 1 ms 초과 주기 | 이상치 전수 보존 | p99 관문 통과 | 최대 1 ms 목표 | 상태 |
|---|---|---|---|---|---|---|---|---|
| A_handoff_long | 서보 A | 364.355 | 7156.108 | 7 | 예 | 예 | 아니오 | 통과 |
| A_handoff_short | 서보 A | 319.959 | 614.987 | 0 | 예 | 예 | 예 | 통과 |
| R1 | 서보 A | 327.671 | 11702.850 | 8 | 예 | 예 | 아니오 | 기준 |
| B_mass | 토크 B | 671.990 | 9648.991 | 12 | 예 | 예 | 아니오 | 통과 |
| B_mass_phase | 토크 B | 598.943 | 798.176 | 0 | 예 | 예 | 예 | 통과 |
| B_phase | 토크 B | 554.443 | 843.326 | 0 | 예 | 예 | 예 | 통과 |
| R1 | 토크 B | 350.440 | 14162.769 | 4 | 예 | 예 | 아니오 | 기준 |

모든 이상치를 보존했다. p99<800 us는 관문, 최대 1 ms는 미달을 포함해 기록하는 목표다. R1에는 추가 진단·holdout이 있어 전체 행 수가 다르므로 서로 다른 전수 집계를 같은 조건의 속도 비로 해석하지 않는다.

## 단계별 결과

- 개발 검증: 서보 A: 검출 15→15회, 통과; 토크 B: 검출 15→15회, 통과 (재승격 없음)
- D4: 서보 A: 검출 111→111회, 실패; 토크 B: 검출 111→111회, 실패 (재승격 없음)
- 분석 섭동: 서보 A: 검출 6→6회, 실패; 토크 B: 검출 6→6회, 실패 (재승격 없음)
- D4 동일 조건 비용 최악 비 A 1.136/B 1.255. A 편향 조건 오감지 8→9회, B 6→4회지만 동일 조건 추가 오감지 1회가 있어 두 방식 모두 실패했다.
- 분석 섭동 A 비용 비 1.058/기존 효능 점수 0.951, B 비용 비 1.473/정상 오감지 9→64회로 실패했다. 두 방식 모두 40 N, 6방향을 6/6 검출했다.
- D4 포화 A: 기준 무충돌 189조건은 2회, 전체 385조건은 R1/후보 모두 4회다. 사전 전체 한도 2회/0.15%는 완화하지 않았다.
- 후보 실제점 radius/FD 동시 통과는 A-long 54/64, B-mass 45/48이다. R1의 68/78은 별도의 같은 강제격자 범위다. 이 관문 단독으로 후보와 R1을 가를 수 없으며 전역 발산 증거도 아니다.
- 새 holdout 한 배치에서 최종 R1만 76회 검사했다: 서보 A 검출 24/30·편향 조건 오감지 0회; 토크 B 검출 22/30·편향 조건 오감지 1회
- 마지막 공식 58회: 통과. 8 kg 추적·감지는 최종 R1 소스의 회귀 확인이며 후보 효능 자료가 아니다.

## 마지막 공식 8 kg 회귀

최종 R1 바이트를 원본 evaluator로 재실행한 값이다. 진단 후보의 공식 효능이 아니며 0/4/8 kg 전체 회귀와 기존 미검출은 원자료에 보존했다.

| 방식 | RMS / 최대 mm | 놓기 계단 Nm | 운반 15 N / 10 N ms | 0/4/8 kg 무충돌 FA | 8 kg 포화 % | 판정 |
|---|---|---|---|---|---|---|
| 서보 A | 1.17 / 3.91 | 14.6 | 45 / 62 | 0/0/0 | 0.0 | 통과 |
| 토크 B | 3.23 / 8.44 | 13.1 | 43 / 56 | 0/0/0 | 0.0 | 통과 |

## 검증 범위

### 실행

- 보존 실험: 사전 검증 28회·개발 104회·D4 1540회·섭동 136회·일치 검사 8회·holdout 76회·공식 58회
- 강제점 206점·실제 후보점 112점·같은 강제격자 R1 78점, pRNE 16표본·전체 시간·동일 관측 재생 검사
- 문서 검증: 전체 unittest 작업 352개/공개 overlay 333개, 보고서 17개, 별도 함수 18회씩(일부 중복)
- 동결 101/기존 R2 53/보호 20/원본 29/환경 21 불변, 최종 3파일은 R1과 바이트 동일
- 작업·공개 보고서 검사, AST, 공개 감사(원본 코드·가이드·ffprobe), 매체·전수 SHA·PDF 시각 검사

### 실행하지 않음

- 이번 문서 정리에서 시뮬레이션·holdout·공식 재실행 없음; 탈락 후보의 holdout·공식 효능 없음
- 실물·hidden·Windows·전역/LTV/hybrid/수동성 보장
- 분석 전용 runner/원시 trace 포함 새 수집부터의 독립 완전 재현
- Python LSP: 실행파일 없음; AST/단위검사로 별도 확인

## 한계

- 모든 개선 수치는 유한 비공식 조건의 진단 결과다. R2에서 고른 방법의 선택 편향과 새 사전 검증/고정 위상 국소 분석의 한계가 있다.
- G4 지속 지지, 엄격 최대 1 ms, 격자 포화·섭동 오감지·약한 힘 감지의 기존 미달은 해소하지 못했다.
- 진단 후보의 새 holdout/공식 효능은 미실행이다. 사전 탈락 뒤 최종 R1만 검사했고 재승격·튜닝·공식 재실행하지 않았다.
- 기준/v1은 새 R3 split에서 재실행하지 않았다. 네 열 T1–T6은 보존 동일조건 수치와 exact-source alias다.
- 분석 전용 runner/원시 trace를 포함한 새 수집부터 완전 재현, 실제 로봇·전역/LTV/hybrid/수동성 보장은 검증하지 않았다.

## 결론

새 채택 0개. 서보와 토크 모두 독립 판정해 R1을 유지한다. 국소 관문 단독으로 R1과 후보를 가를 수는 없지만 규칙을 완화하지 않았다. D4·섭동의 비용·오감지 악화도 확인해 탈락 판정을 유지한다. 음성 결과를 방법 계열 자체의 불가능성으로 일반화하지 않는다.

## 원자료와 자체 유도

- [R3 방법·가정](R3_METHODS.md), [구현 전 원인 기록](R3_CAUSE_ANALYSIS.md)
- [충돌 신호 분석](../results/v2_protocol/R3_SIGNAL_ANALYSIS.json), [pRNE 수치 검사](../results/v2_protocol/R3_PRNE_MATH.json)
- [동일 관측 계산비용](../results/v2_protocol/R3_REPLAY_TIMING.json)

| 증거 | 원자료 경로 | 출처 |
|---|---|---|
| protocol | [results/v2_protocol/PROTOCOL_R3.json](../results/v2_protocol/PROTOCOL_R3.json) | R3 기록 |
| r2_report_manifest | [results/v2_protocol/R2_REPORT_MANIFEST.json](../results/v2_protocol/R2_REPORT_MANIFEST.json) | 보존값 인용 |
| validation_review | [results/v2_protocol/R3_BOUND_STABILITY_REVIEW.json](../results/v2_protocol/R3_BOUND_STABILITY_REVIEW.json) | R3 기록 |
| challengers | [results/v2_protocol/R3_CHALLENGERS.json](../results/v2_protocol/R3_CHALLENGERS.json) | R3 기록 |
| forced_stability | [results/v2_protocol/R3_FORCED_STABILITY.json](../results/v2_protocol/R3_FORCED_STABILITY.json) | R3 기록 |
| actual_stability | [results/v2_protocol/R3_ACTUAL_STABILITY.json](../results/v2_protocol/R3_ACTUAL_STABILITY.json) | R3 기록 |
| redesign_decision | [results/v2_protocol/R3_REDESIGN_DECISION.json](../results/v2_protocol/R3_REDESIGN_DECISION.json) | R3 기록 |
| development_review | [results/v2_protocol/R3_DEVELOPMENT_REVIEW.json](../results/v2_protocol/R3_DEVELOPMENT_REVIEW.json) | R3 기록 |
| d4_review | [results/v2_protocol/R3_D4_REVIEW.json](../results/v2_protocol/R3_D4_REVIEW.json) | R3 기록 |
| perturbation_review | [results/v2_protocol/R3_PERTURB_REVIEW.json](../results/v2_protocol/R3_PERTURB_REVIEW.json) | R3 기록 |
| freeze | [results/v2_protocol/R3_FREEZE.json](../results/v2_protocol/R3_FREEZE.json) | R3 기록 |
| holdout_review | [results/v2_protocol/R3_HOLDOUT_REVIEW.json](../results/v2_protocol/R3_HOLDOUT_REVIEW.json) | R3 기록 |
| official_review | [results/v2_protocol/R3_OFFICIAL_REGRESSION.json](../results/v2_protocol/R3_OFFICIAL_REGRESSION.json) | R3 기록 |
| timing_review | [results/v2_protocol/R3_TIMING_REVIEW.json](../results/v2_protocol/R3_TIMING_REVIEW.json) | R3 기록 |
| final_selection | [results/v2_protocol/R3_FINAL_SELECTION.json](../results/v2_protocol/R3_FINAL_SELECTION.json) | R3 기록 |
| integrity | [results/v2_protocol/R3_INTEGRITY.json](../results/v2_protocol/R3_INTEGRITY.json) | R3 기록 |
| r1_stability | [results/v2_protocol/R3_R1_STABILITY.json](../results/v2_protocol/R3_R1_STABILITY.json) | R3 기록 |
| cause_analysis | [results/v2_protocol/R3_CAUSE_ANALYSIS.json](../results/v2_protocol/R3_CAUSE_ANALYSIS.json) | R3 기록 |
| d4_servo_r1 | [results/v2_protocol/R3_D4_R1_servo.json](../results/v2_protocol/R3_D4_R1_servo.json) | R3 기록 |
| d4_servo_candidate | [results/v2_protocol/R3_D4_A_handoff_long_servo.json](../results/v2_protocol/R3_D4_A_handoff_long_servo.json) | R3 기록 |
| d4_torque_r1 | [results/v2_protocol/R3_D4_R1_torque.json](../results/v2_protocol/R3_D4_R1_torque.json) | R3 기록 |
| d4_torque_candidate | [results/v2_protocol/R3_D4_B_mass_torque.json](../results/v2_protocol/R3_D4_B_mass_torque.json) | R3 기록 |
| perturb_servo_r1 | [results/v2_protocol/R3_PERTURB_R1_servo.json](../results/v2_protocol/R3_PERTURB_R1_servo.json) | R3 기록 |
| perturb_servo_candidate | [results/v2_protocol/R3_PERTURB_A_handoff_long_servo.json](../results/v2_protocol/R3_PERTURB_A_handoff_long_servo.json) | R3 기록 |
| perturb_torque_r1 | [results/v2_protocol/R3_PERTURB_R1_torque.json](../results/v2_protocol/R3_PERTURB_R1_torque.json) | R3 기록 |
| perturb_torque_candidate | [results/v2_protocol/R3_PERTURB_B_mass_torque.json](../results/v2_protocol/R3_PERTURB_B_mass_torque.json) | R3 기록 |
| r2_presentation | [results/presentation_R2.pdf](../results/presentation_R2.pdf) | 동일 원본 보존 |
