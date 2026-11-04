# R2 발표 자료 원고

PDF는 아래 16개 슬라이드를 같은 순서로 생성한다.

## 1. R2 질문과 고정 규칙
- R1을 비교 기준으로 고정
- G4 기준 불변
- holdout 동결 후 1회
- 공식 평가는 마지막 회귀 확인만

## 2. 증거 흐름과 독립성
- P0 → protocol → C1–C6 → 동결 → holdout → 공식
- fresh와 exact-source 재사용을 분리
- 불리한 결과와 outlier를 모두 유지

## 3. P0 물리량 측정
- 후보 실행 전 baseline 8회와 probe 32회를 순차 측정했다.
- 40회 모두 지속 G4를 통과하지 못해 순간 force crossing을 하중 이양으로 세지 않았다.
- 관측 경계는 normal 0.689439/2.678555 mm, 속도 20.3744 mm/s, 가속도 0.254511 m/s^2이며 측정 최적값으로 주장하지 않는다.
- 최대 첫 접촉-힘 손실 34 ms와 보수적 시작 3.809 s를 protocol에 고정했다.

## 4. 후보군 C1–C6
- 제어 구성 25개와 C5 observer ablation 3개
- architecture별 판정 행: 채택 0, 탈락 59
- 후보별 gate·효과·비용을 함께 기록

## 5. 탄성 관절 안정성 모델
- 실제 탄성 관절 국소 FD
- 1 ms ZOH
- 명시적 한 주기 지연
- 유한 고정모드 결과만 주장

## 6. 안정성 결과
- 전체 300건
- protocol-radius 222건
- strict 190건

## 7. pRNE 수학·계산 검증
- 표본 9건
- Mdot = C + C^T 유한차분 확인
- reverse VJP는 같은 출력에서 helper 비용만 감소
- R1 observer 대비 고유 속도 향상 주장은 하지 않음

## 8. 정상 검증과 final timing
- 제어 변경마다 무충돌 정상 데이터 확인
- A p99/max 334.8/5576.3 us; overrun 1
- B p99/max 363.1/6016.5 us; overrun 5
- strict 1 ms 미달; 원인 단정 없음

## 9. T1–T3 동일 조건
| 항목 | 조건·단위 | 기준 | v1 | R1 | R2 |
|---|---|---|---|---|---|
| T1 놓기 전 마지막 20 ms 평균 지지율 | 8 kg, 동일 COM, release 직전 20 ms<br>[% of mg (A/B)] | A 0.0000 / B 0.0557 | A 0.2098 / B 11.1507 | A 0.2098 / B 11.1507 | A 0.2098 / B 11.1507 |
| T2 놓기 토크 계단 | 8 kg, 동일 COM, place step<br>[N m (A/B)] | A 73.842 / B 74.797 | A 14.626 / B 13.126 | A 14.626 / B 13.126 | A 14.626 / B 13.126 |
| T3 분석 섭동 최악 TCP/놓기힘/오감지 | fresh 34 perturbation rows per architecture; 34=28 normal+6 external<br>[mm / N / rises] | A 22.731 mm / 208.997 N / FA 221; B 17.171 mm / 352.140 N / FA 309 | A 14.777 mm / 212.734 N / FA 4; B 45.652 mm / 362.600 N / FA 8 | A 14.777 mm / 212.734 N / FA 6; B 45.652 mm / 362.600 N / FA 10 | A 14.777 mm / 212.734 N / FA 6; B 45.652 mm / 362.600 N / FA 10 |

## 10. T4–T6 동일 조건
| 항목 | 조건·단위 | 기준 | v1 | R1 | R2 |
|---|---|---|---|---|---|
| T4 질량·COM 격자 포화/놓기 peak | 189 no-collision reference rows per architecture<br>[rows / mg] | A sat 0/189, peak 2.0359 mg; B sat 0/189, peak 9.8822 mg | A sat 2/189, peak 1.6098 mg; B sat 0/189, peak 1.0358 mg | A sat 2/189, peak 1.6098 mg; B sat 0/189, peak 1.0358 mg | A sat 2/189, peak 1.6098 mg; B sat 0/189, peak 1.0358 mg |
| T5 TCP 추적 RMS/최대 | 8 kg, 동일 COM, no collision<br>[mm (A/B)] | A 6.144/14.000; B 7.042/13.183 | A 1.174/3.914; B 3.232/8.444 | A 1.174/3.914; B 3.232/8.444 | A 1.174/3.914; B 3.232/8.444 |
| T6 D4 감지/놓침/masked | 180 collision rows per architecture; masked kept separate<br>[episodes] | A 88/43/49; B 78/42/60 | A 94/86/0; B 91/89/0 | A 111/69/0; B 111/69/0 | A 111/69/0; B 111/69/0 |

## 11. 후보 판정
- C1: 4개 변형, 모두 reject
- C2: 4개 변형, 모두 reject
- C3: 6개 변형, 모두 reject
- C4: 8개 변형, 모두 reject
- C5: 3개 변형, 모두 reject
- C6: 3개 변형, 모두 reject

## 12. Lie group 적용
- C1 wrench planning / power-dual mapping: reject
- C2 SE(3) normal coordinate: reject
- C3 NRIC / motor DOB: reject
- C4 P-APD / P-IDC with pRNE action: reject
- C5 metric-compatible pRNE: reject
- C6 SO(3)+R3 separated interpolation: reject

## 13. 강건성·holdout
- 동결 뒤 marker batch_attempt=1로 152행을 한 번만 열었고 이후 튜닝하지 않았다.
- 최종 R1 alias는 A/B 각각 20 detected, 10 missed였다. false rise는 A 0, B offset 1이었다.
- 이미 탈락한 C5 pRNE+ridge도 A/B 각각 20/10으로 감지 이득이 없었고 false rise는 A 4(충돌 2, offset 2), B 2(offset 2)였다.
- holdout의 0.75 kg은 C5 학습 질량 하나와 겹친다. COM·시각·힘·방향과 seed는 새 조건이지만 네 질량 모두 unseen이라고 주장하지 않는다.

## 14. 공식 회귀 확인
- holdout 뒤 unchanged evaluate.py를 default 이름 overlay에서 A/B 각각 한 번 실행했다.
- 0/4/8 kg 무충돌 control은 추가 오감지 0, 포화 0 %, 발산 없음으로 R1과 같은 셀을 유지했다. B의 8 kg 운반 collision 행 false max 1회도 R1과 같다.
- 8 kg 운반은 A/B에서 15 N 45/43 ms, 10 N 62/56 ms를 유지했다.
- 집기 10/15 N과 놓기 10 N의 기존 미감지는 남았고, 공식 노출 뒤 수정하거나 재승격하지 않았다.

## 15. 미달·한계·outlier
- T1~T6은 R1보다 개선되지 않아 모두 미달로 기록했다.
- D4 baseline의 masked를 버리지 않고 A 88/43/49, B 78/42/60 detected/missed/masked로 분리했다.
- R1과 최종 R2 alias의 D4는 A/B 모두 111 detected, 69 missed이며 독립 R2 재실행으로 세지 않는다.
- C5 pRNE+ridge holdout은 감지 이득 0에 추가 false rise가 있어 음성 결과로 유지했다.
- P0는 PARTIAL이다. 40회에서 지속 G4를 만드는 feasible displacement/speed를 식별하지 못했고 경계는 관측·보수 선택이지 측정 최적값이 아니다.
- holdout 0.75 kg은 C5 학습 질량 하나와 겹친다. 새 joint condition과 seed만으로 모든 질량 unseen 일반화를 주장하지 않는다.
- pRNE-only의 4회/architecture 감지 손실은 ridge 제거가 함께 있어 pRNE에만 귀속할 수 없다. matched ridge와 pRNE+ridge의 15/24 대 15/24가 isolated zero effect다.

## 16. 최종 결론
- 이 유한 시뮬레이션에서 사전 등록한 새 제어 25개 구성과 C5 관측기 변형은 모두 엄격 게이트를 통과하지 못했다. 방법 계열 자체가 불가능하다는 뜻은 아니다. 최종 R2는 R1 제어기와 감지기의 exact-source alias이며, 새 후보를 억지로 채택하지 않았다.
- 새 후보 채택 0개; 최종 R2=R1 exact-source alias
- 재설계 가족별 1회 허용·사용 0회; 미사용 사유 기록 없음
- P0/G4와 T1–T6 추가 개선 목표는 미달
- 실기·hidden·전역 안정성은 검증하지 않음
