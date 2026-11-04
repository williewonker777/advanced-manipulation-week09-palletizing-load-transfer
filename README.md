# 팔레타이징 하중 이양 · 제어/리그룹 2차 실험

기준 = 배포 키트 제어·감지, **v1** = 탄성·하중전이 제어와 기하 감지, **R1** = v1 제어 + 정상-only ridge/EMA 감지, **R2** = 2차 후보 검증 후 R1을 유지한 최종 구현.

**최종 구현은 R1을 유지한다.** 새 제어25설정과 pRNE 관측기를 실제 구현·측정했지만 사전 채택 기준을 통과하지 못했다. 수학 검증 성공을 성능 개선이나 탄성 플랜트의 안정성 보장으로 바꾸어 쓰지 않는다.

## 영상으로 먼저 보기

GIF를 누르면 MP4가 열린다. 07·08은 **기준/v1/R1/R2 네 열**, 05·06은 이전 기준/v1/R1 세 열 비교다. R2=R1 표시는 같은 소스라는 뜻이지 독립적인 새 개선이 아니다.

[![07 서보 A · 기준/v1/R1/R2 네 열 비교](artifacts/media/07_r2_servo_comparison.gif)](artifacts/media/07_r2_servo_comparison.mp4)
[![08 토크 B · 기준/v1/R1/R2 네 열 비교](artifacts/media/08_r2_torque_comparison.gif)](artifacts/media/08_r2_torque_comparison.mp4)

### 이전 R1 세 열 비교: 05 서보 A / 06 토크 B

[![05 서보 A · 기준/v1/R1 비교](artifacts/media/05_v2_servo_transport_collision.gif)](artifacts/media/05_v2_servo_transport_collision.mp4)
[![06 토크 B · 기준/v1/R1 비교](artifacts/media/06_v2_torque_transport_collision.gif)](artifacts/media/06_v2_torque_transport_collision.mp4)


## 같은 조건의 비교

A/B는 위치 서보/모터 토크 방식이다. R2는 R1과 같은 소스의 alias이며, 같은 수치를 독립 신규 재현이라고 세지 않는다.

| 항목 | 기준 | v1 | R1 | 최종 R2 |
|---|---:|---:|---:|---:|
| T1: 8kg 마지막20ms 평균 지지율 [%mg, A/B] | 0.000 / 0.056 | 0.210 / 11.151 | 0.210 / 11.151 | 0.210 / 11.151 |
| T2: 8kg 놓기 최대1ms 토크 변화 [Nm, A/B] | 73.8 / 74.8 | 14.6 / 13.1 | 14.6 / 13.1 | 14.6 / 13.1 |
| T3: 분석섭동28무충돌 조건 최대TCP [mm, A/B] | 17.43 / 15.13 | 5.28 / 17.26 | 5.28 / 17.26 | 5.28 / 17.26 |
| T3: 40N 외력6개 포함 전체34조건 최대TCP [mm, A/B] | 22.73 / 17.17 | 14.78 / 45.65 | 14.78 / 45.65 | 14.78 / 45.65 |
| T3: 같은28무충돌 조건 오감지 [A/B] | 177 / 248 | 4 / 7 | 6 / 9 | 6 / 9 |
| T4: 질량·COM 격자 포화 조건 [189조건/방식, A/B] | 0 / 0 | 2 / 0 | 2 / 0 | 2 / 0 |
| T4: 격자 최대 놓기 힘 [mg, A/B] | 2.036 / 9.882 | 1.610 / 1.036 | 1.610 / 1.036 | 1.610 / 1.036 |
| T5: 8kg TCP RMS/최대 [mm, A] | 6.14 / 14.00 | 1.17 / 3.91 | 1.17 / 3.91 | 1.17 / 3.91 |
| T5: 8kg TCP RMS/최대 [mm, B] | 7.04 / 13.18 | 3.23 / 8.44 | 3.23 / 8.44 | 3.23 / 8.44 |
| T6: 180충돌 중 검출 [A/B] | 88 / 78 | 94 / 91 | 111 / 111 | 111 / 111 |

- T1/T2/T5는 동일8kg·COM·무충돌 조건의 보존된 분석 trace와 공식 회귀를 구분해 보고한다. 평균 지지율 자체는 공식 evaluator가 출력하는 항목이 아니다.
- T3는 잡음/마찰/강성 **분석섭동**이며 공식 수치가 아니다. T4/T6는 동일385조건/방식이다. 기준770회는 새 실행, v1/R1 격자는 exact-source 과거 기록이다.
- 기준의 T6에는 기존경보/구분불가가 A49/B60건 더 있다. 이를 빼고 기준의 놓침43/42건만 비교하면 안 된다. R1/R2는 각각69/180회를 놓친다.

## 무엇을 시도했고 왜 유지했나

- **P0:** 접촉은 놓기50–57ms 전이지만 반발·접촉 손실 때문에 지지가 이어지지 않았다. 측정 기반 ±0.689/±2.679mm probe를 포함한40회 모두 G4 미달이다. 지속 지지에 필요한 실현 가능한 변위·속도는 식별하지 못했다. 경계값은 관측/보수적 설계값이지 실측 최적값이 아니다.
- **C1 렌치 계획·C2 SE(3) 방향별 순응:** force schedule, wrench dual, 탄성 모터 lift를 구현했다. G4와 TCP/오감지 비용 관문을 넘지 못했다. A의 순수 법선 병진에서는 Euclidean/SE(3)가 수치적으로 같고 계산만 늘었다.
- **C3 NRIC·DOB:** 독립 명목 입력 queue와 탄성 nominal 적분, 미분 없는 motor DOB를 구현했다. NRIC는 국소 불안정성/오감지로 탈락했다. **A DOB(w4.5)는 추적이 개선**됐다(7.55kg 최대 TCP 3.890→3.226mm 등). 다만 5.65kg 놓기 토크 계단이 10.752→11.335Nm(+5.42%)로 사전5% 비용 상한을 넘었으며, 이 설정은 등록 관문 중 `cost`만 실패했다. 반면 **B DOB(w4.5)는 추적이 악화**됐다(1.05kg RMS 2.045→5.273mm 등). [R1 원자료](results/v2_protocol/R2_VALIDATION_R1.json) / [DOB 원자료](results/v2_protocol/R2_VALIDATION_C3_DOB_w4.5.json).
- **C4 P-APD·P-IDC:** 실제속도/설계속도를 분리한 pRNE로 링크 법칙을 만들고 탄성 모터 기준에 올렸다. 아래는 **B P-IDC(rate 4.5, damping 1)의 validation 4조건**이다. 각 수치는 같은 질량·COM의 **R1 → 후보**이며 공식 평가가 아니다. RMS는 개선됐지만 경량 최대오차와 놓기 계단 비용 때문에 **탈락**했다. 이 설정도 등록 관문 중 `cost`만 실패했다.

| 질량 [kg] | TCP RMS [mm], R1 → P-IDC | TCP 최대 [mm], R1 → P-IDC | 놓기 계단 [Nm], R1 → P-IDC |
|---:|---:|---:|---:|
| 1.05 | 2.045 → 1.938 | 4.010 → 4.397 | 4.874 → 7.286 |
| 3.15 | 2.205 → 1.926 | 4.058 → 4.361 | 5.528 → 8.782 |
| 5.65 | 2.702 → 1.936 | 6.430 → 4.367 | 9.282 → 9.475 |
| 7.55 | 3.140 → 1.975 | 8.060 → 4.382 | 12.686 → 12.958 |

[R1 validation 원자료](results/v2_protocol/R2_VALIDATION_R1.json) / [B P-IDC 원자료](results/v2_protocol/R2_VALIDATION_C4_P-IDC_w4.5_d1.json). 네 조건 중 최악 최대오차 비는1.09654, 놓기 계단 비는1.58874다.

- **C5 pRNE:** 직접 유도한 metric-compatible C의 Mdot=C+Cᵀ 오차는 최대3.89e-10이다. 그러나 정상 잔차는 거의 그대로다. 균형 development에서 R1·pRNE+새 ridge·matched ridge는 A/B 각각15/24 검출, pRNE 단독은11/24다. 단독은 ridge도 끄므로 손실 원인을 pRNE만으로 돌리지 않는다.
- **C6 보간:** 7차/S-curve/SO(3)+R³의 조기 도착·dwell을 구현했지만 원래 궤적 대비 TCP 비용과 다른 관문을 넘지 못했다.

### 재설계 사용 여부

[사전 protocol](results/v2_protocol/PROTOCOL_R2.json)은 물리적 실패 설명을 먼저 기록하고 새 validation seed를 쓰는 조건으로 **family별 최대1회** 재설계를 허용했다. [판정 기록](results/v2_protocol/R2_ADOPTION.json)의 `redesigns_used`는 **0**이다. cost 관문 중심으로 탈락한 A DOB와 B P-IDC를 포함해, 재설계를 사용하지 않은 구체적 **사유 기록 없음**. 비용 관문 실패 사실과 재설계 미사용 사유를 혼동하지 않는다. 공식 회귀 이후의 재설계·튜닝은 해당 protocol에서 금지된다. 평가 전 구현 오류 수정은 결과를 본 재설계로 세지 않았다.

300개의 탄성 관절·1ms ZOH·명령/JTS 지연 국소 분석을 보존했다. protocol-radius 통과222/300과 strict 안정190/300을 구분한다. 이는 공통 R1 물리점의 고정모드 분석이며 전역/하이브리드 안정성·수동성 증명이 아니다.

## 결과·설명

- [최종 R2 보고서](docs/R2_RESULTS.md) · [발표 PDF](results/presentation.pdf)
- [요구사항 판정](docs/REQUIREMENTS_AUDIT.md)
- [방법 자체 유도·적용 경계](docs/R2_METHODS.md) · [NRIC/DOB/APD/IDC 탄성 변형](docs/R2_ROBUST_DERIVATION.md)
- [직접 확인한 문헌과 사실/추론](docs/LITERATURE.md)
- [사전 protocol](results/v2_protocol/PROTOCOL_R2.json) · [동결](results/v2_protocol/R2_FREEZE.json) · [채택 판정](results/v2_protocol/R2_ADOPTION.json)
- [새 holdout](results/v2_protocol/R2_HOLDOUT.json) · [마지막 공식 회귀](results/v2_protocol/R2_OFFICIAL_REGRESSION.json)
- [서보 4열 비교 영상](artifacts/media/07_r2_servo_comparison.mp4) · [토크 비교 영상](artifacts/media/08_r2_torque_comparison.mp4)
- [재실행/환경](report/REPRODUCE.md) · [전체 파일 SHA256](artifacts/SHA256SUMS)

영상은 기준/v1/R1/R2 네 열의 두 비공식 시나리오이며 방향은 A +y, B -x다. R2=R1일 때 같은 소스라는 표시를 한다. 선정이나 새로운 비밀 평가로 사용하지 않는다. 원자료/교재 문장·그림·가이드·키트 코드는 배포하지 않는다.

## 제출과 한계

제출 진입점은 self-contained `my_servo.py`, `my_torque.py`, `my_detector.py`다. 런타임은 NumPy/MuJoCo와 제공된 공개 nominal/trajectory/config만 사용하며, 새 실험 모델 파일을 읽지 않는다. 학습은 별도 CPU 환경의 정상 관측 타깃으로 했고 공식 환경에 패키지를 추가하지 않았다. 원본 키트는 외부에 별도로 필요하다.

G4는 마지막20ms 평균≥0.5mg·최소≥0.25mg·R1 대비 평균증가≥0.25mg·놓기 peak≤1.1mg이며 완화하지 않았다. 현재 미달이다. 무충돌 격자 포화, 분석섭동 오감지, 약한 힘 놓침, 1ms 최대 계산 목표도 해결하지 못했다. outlier와 실패 사례를 버리지 않았다. 실기 안전성이나 hard real-time을 보장하지 않는다.

공식 무충돌 오감지0은 모든 충돌행의 추가 오감지가0이라는 뜻이 아니다. B의8kg 운반 충돌행은 R1과 같은 추가 오감지 최대1회를 유지한다.

이전 v2 표기는 R1 기록이다. [R1 README 원문 스냅샷](docs/R1_README.md)과 [R1 PDF](results/presentation_R1.pdf)를 보존한다. README 스냅샷의 상대 링크는 원래 저장소 루트 기준이다. 새 holdout은 새 조건 배치이지만 최종 구현은 과거 R1과 같다. 0.75kg 한 질량은 C5 학습과 겹치므로 전부 미학습 질량이라고 주장하지 않는다.
