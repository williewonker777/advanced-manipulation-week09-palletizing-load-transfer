# R2 결과: 제어·Lie group 2차 검증

## 결론

이 유한 시뮬레이션에서 사전 등록한 새 제어 25개 구성과 C5 관측기 변형은 모두 엄격 게이트를 통과하지 못했다. 방법 계열 자체가 불가능하다는 뜻은 아니다. 최종 R2는 R1 제어기와 감지기의 exact-source alias이며, 새 후보를 억지로 채택하지 않았다.

공식 평가는 마지막 회귀 확인에만 사용했다. 후보 선정·임계값 조정에는 사용하지 않았다.

## T1–T6 동일 조건 비교

| 항목 | 동일 조건 | 단위 | 기준 | v1 | R1 | R2 | 판정 | 출처 |
|---|---|---:|---:|---:|---:|---:|---|---|
| T1 놓기 전 마지막 20 ms 평균 지지율 | 8 kg, 동일 COM, release 직전 20 ms | % of mg (A/B) | A 0.0000 / B 0.0557 | A 0.2098 / B 11.1507 | A 0.2098 / B 11.1507 | A 0.2098 / B 11.1507 | not_met | common_reference, adoption |
| T2 놓기 토크 계단 | 8 kg, 동일 COM, place step | N m (A/B) | A 73.842 / B 74.797 | A 14.626 / B 13.126 | A 14.626 / B 13.126 | A 14.626 / B 13.126 | not_met | common_reference, adoption |
| T3 분석 섭동 최악 TCP/놓기힘/오감지 | fresh 34 perturbation rows per architecture | mm / N / rises | A 22.731 mm / 208.997 N / FA 221; B 17.171 mm / 352.140 N / FA 309 | A 14.777 mm / 212.734 N / FA 4; B 45.652 mm / 362.600 N / FA 8 | A 14.777 mm / 212.734 N / FA 6; B 45.652 mm / 362.600 N / FA 10 | A 14.777 mm / 212.734 N / FA 6; B 45.652 mm / 362.600 N / FA 10 | not_met | perturbations, adoption |
| T4 질량·COM 격자 포화/놓기 peak | 189 no-collision reference rows per architecture | rows / mg | A sat 0/189, peak 2.0359 mg; B sat 0/189, peak 9.8822 mg | A sat 2/189, peak 1.6098 mg; B sat 0/189, peak 1.0358 mg | A sat 2/189, peak 1.6098 mg; B sat 0/189, peak 1.0358 mg | A sat 2/189, peak 1.6098 mg; B sat 0/189, peak 1.0358 mg | not_met | d4, d4_reused, adoption |
| T5 TCP 추적 RMS/최대 | 8 kg, 동일 COM, no collision | mm (A/B) | A 6.144/14.000; B 7.042/13.183 | A 1.174/3.914; B 3.232/8.444 | A 1.174/3.914; B 3.232/8.444 | A 1.174/3.914; B 3.232/8.444 | not_met | common_reference, adoption |
| T6 D4 감지/놓침/masked | 180 collision rows per architecture; masked kept separate | episodes | A 88/43/49; B 78/42/60 | A 94/86/0; B 91/89/0 | A 111/69/0; B 111/69/0 | A 111/69/0; B 111/69/0 | not_met | d4, d4_reused, development, official, adoption |

## P0 측정

- 후보 실행 전 baseline 8회와 probe 32회를 순차 측정했다.
- 40회 모두 지속 G4를 통과하지 못해 순간 force crossing을 하중 이양으로 세지 않았다.
- 관측 경계는 normal 0.689439/2.678555 mm, 속도 20.3744 mm/s, 가속도 0.254511 m/s^2이며 측정 최적값으로 주장하지 않는다.
- 최대 첫 접촉-힘 손실 34 ms와 보수적 시작 3.809 s를 protocol에 고정했다.

## 후보별 판정

| 후보 | 방식 | 실패 관문 | 오감지 | G4 통과 수 | TCP 최대/RMS 최악 비 | 놓기 계단 최악 비 | p99/최대 계산시간 [us] | 국소 radius 통과 수 |
|---|---|---|---:|---:|---:|---:|---:|---:|
| `C1_F0.5_fb0` | servo | no_false_alarm, cost, G4 | 1 | 0/4 | 2.978/1.565 | 3.528 | 312.221/561.578 | 6/6 |
| `C1_F0.5_fb0` | torque | no_false_alarm, cost, local_radius, G4 | 2 | 0/4 | 2.282/1.408 | 4.053 | 315.740/498.249 | 4/6 |
| `C1_F0.5_fb0.25` | servo | no_false_alarm, cost, G4 | 2 | 0/4 | 3.422/1.812 | 4.218 | 326.920/596.618 | 6/6 |
| `C1_F0.5_fb0.25` | torque | no_false_alarm, cost, local_radius, G4 | 3 | 0/4 | 2.691/1.759 | 4.801 | 303.492/501.168 | 5/6 |
| `C1_F0.85_fb0` | servo | no_false_alarm, cost, G4 | 4 | 0/4 | 4.002/1.896 | 7.340 | 330.103/579.418 | 6/6 |
| `C1_F0.85_fb0` | torque | no_false_alarm, cost, local_radius, G4 | 4 | 0/4 | 3.074/1.510 | 7.775 | 299.451/461.239 | 4/6 |
| `C1_F0.85_fb0.25` | servo | no_false_alarm, cost, G4 | 5 | 0/4 | 4.544/2.112 | 8.175 | 302.821/4291.646 | 6/6 |
| `C1_F0.85_fb0.25` | torque | no_false_alarm, cost, local_radius, G4 | 4 | 0/4 | 3.558/1.903 | 8.821 | 296.879/553.788 | 4/6 |
| `C2_euclidean_small` | servo | no_saturation, cost, G4 | 0 | 0/4 | 1.300/1.069 | 3.522 | 268.781/433.688 | 6/6 |
| `C2_euclidean_small` | torque | no_false_alarm, no_saturation, cost, local_radius, G4 | 10 | 0/4 | 1.245/1.199 | 17.536 | 304.565/421.159 | 1/6 |
| `C2_euclidean_stroke` | servo | no_saturation, cost, local_radius, G4 | 0 | 0/4 | 1.600/1.210 | 35.110 | 293.112/521.088 | 5/6 |
| `C2_euclidean_stroke` | torque | no_false_alarm, no_saturation, cost, local_radius, G4 | 2 | 0/4 | 1.115/1.196 | 6.853 | 313.710/450.968 | 3/6 |
| `C2_se3_small` | servo | no_saturation, cost, G4 | 0 | 0/4 | 1.300/1.069 | 3.522 | 471.603/686.368 | 6/6 |
| `C2_se3_small` | torque | no_false_alarm, no_saturation, cost, local_radius, G4 | 11 | 0/4 | 1.188/1.180 | 17.530 | 427.056/707.018 | 1/6 |
| `C2_se3_stroke` | servo | no_saturation, cost, local_radius, G4 | 0 | 0/4 | 1.600/1.210 | 35.110 | 457.632/695.077 | 5/6 |
| `C2_se3_stroke` | torque | no_false_alarm, no_saturation, cost, local_radius, G4 | 3 | 0/4 | 1.220/1.265 | 6.670 | 409.089/622.727 | 3/6 |
| `C3_NRIC_g1` | servo | no_false_alarm, cost, local_radius | 13 | 0/4 | 1.063/1.222 | 1.110 | 744.079/1048.376 | 4/6 |
| `C3_NRIC_g1` | torque | no_false_alarm, cost, local_radius | 10 | 0/4 | 2.026/1.987 | 1.212 | 721.401/1128.636 | 0/6 |
| `C3_NRIC_g2` | servo | no_false_alarm, cost, local_radius | 10 | 0/4 | 1.033/1.193 | 1.004 | 755.328/1046.916 | 4/6 |
| `C3_NRIC_g2` | torque | no_false_alarm, cost, timing_p99, local_radius | 10 | 0/4 | 2.596/1.977 | 1.116 | 802.157/5149.171 | 0/6 |
| `C3_NRIC_g4` | servo | no_false_alarm, cost, local_radius | 10 | 0/4 | 1.161/1.269 | 1.014 | 691.008/1845.293 | 4/6 |
| `C3_NRIC_g4` | torque | no_false_alarm, cost, local_radius | 9 | 0/4 | 3.401/2.479 | 1.374 | 771.875/1201.896 | 0/6 |
| `C3_DOB_w4.5` | servo | cost | 0 | 0/4 | 0.829/0.957 | 1.054 | 350.816/475.058 | 6/6 |
| `C3_DOB_w4.5` | torque | cost, local_radius | 0 | 0/4 | 2.562/2.578 | 1.133 | 254.350/400.699 | 5/6 |
| `C3_DOB_w9` | servo | cost | 0 | 0/4 | 0.825/0.964 | 1.085 | 379.788/517.818 | 6/6 |
| `C3_DOB_w9` | torque | cost, local_radius | 0 | 0/4 | 2.582/2.634 | 1.131 | 374.411/547.507 | 5/6 |
| `C3_DOB_w18` | servo | cost | 0 | 0/4 | 0.823/0.964 | 1.068 | 303.949/463.799 | 6/6 |
| `C3_DOB_w18` | torque | cost, local_radius | 0 | 0/4 | 2.596/2.654 | 1.129 | 306.975/486.899 | 5/6 |
| `C4_P-APD_w4.5_d0.5` | servo | cost, timing_p99, tracking_gain | 0 | 0/4 | 1.034/1.061 | 1.074 | 813.287/1102.865 | 6/6 |
| `C4_P-APD_w4.5_d0.5` | torque | cost, timing_p99, local_radius, tracking_gain | 0 | 0/4 | 1.244/1.112 | 1.526 | 829.478/1037.046 | 5/6 |
| `C4_P-APD_w4.5_d1` | servo | cost, tracking_gain | 0 | 0/4 | 1.028/1.090 | 1.455 | 723.616/1113.075 | 6/6 |
| `C4_P-APD_w4.5_d1` | torque | cost, timing_p99, local_radius, tracking_gain | 0 | 0/4 | 1.200/1.088 | 1.716 | 810.120/5197.160 | 5/6 |
| `C4_P-APD_w9_d0.5` | servo | cost, tracking_gain | 0 | 0/4 | 1.021/1.115 | 1.681 | 687.178/1093.685 | 6/6 |
| `C4_P-APD_w9_d0.5` | torque | cost, local_radius | 0 | 0/4 | 1.089/0.994 | 2.048 | 751.050/1027.556 | 4/6 |
| `C4_P-APD_w9_d1` | servo | cost, tracking_gain | 0 | 0/4 | 1.001/1.165 | 2.370 | 592.267/985.146 | 6/6 |
| `C4_P-APD_w9_d1` | torque | no_false_alarm, cost, local_radius | 10 | 0/4 | 0.991/0.923 | 3.236 | 716.750/4859.009 | 0/6 |
| `C4_P-IDC_w4.5_d0.5` | servo | cost, tracking_gain | 0 | 0/4 | 1.050/1.055 | 1.073 | 762.426/1037.335 | 6/6 |
| `C4_P-IDC_w4.5_d0.5` | torque | cost, local_radius | 0 | 0/4 | 1.109/0.944 | 1.564 | 787.285/1047.046 | 5/6 |
| `C4_P-IDC_w4.5_d1` | servo | cost, timing_p99, tracking_gain | 0 | 0/4 | 1.051/1.099 | 1.408 | 800.219/1157.895 | 6/6 |
| `C4_P-IDC_w4.5_d1` | torque | cost | 0 | 0/4 | 1.097/0.948 | 1.589 | 718.389/1070.206 | 6/6 |
| `C4_P-IDC_w9_d0.5` | servo | cost, tracking_gain | 0 | 0/4 | 1.054/1.140 | 1.304 | 711.589/4746.880 | 6/6 |
| `C4_P-IDC_w9_d0.5` | torque | cost, timing_p99, local_radius | 0 | 0/4 | 0.945/0.787 | 1.749 | 838.599/1071.745 | 5/6 |
| `C4_P-IDC_w9_d1` | servo | cost, timing_p99, tracking_gain | 0 | 0/4 | 1.001/1.231 | 2.431 | 815.822/1094.835 | 6/6 |
| `C4_P-IDC_w9_d1` | torque | no_false_alarm, cost, local_radius | 7 | 0/4 | 0.896/0.776 | 2.927 | 757.532/1076.945 | 1/6 |
| `C5_prne_only` | servo | 없음(validation; observer 판정 별도) | 0 | 0/4 | 1.000/1.000 | 1.000 | 575.829/878.836 | N/A |
| `C5_prne_only` | torque | 없음(validation; observer 판정 별도) | 0 | 0/4 | 1.000/1.000 | 1.000 | 646.098/1032.757 | N/A |
| `C5_prne_plus_ridge` | servo | 없음(validation; observer 판정 별도) | 0 | 0/4 | 1.000/1.000 | 1.000 | 694.179/1144.245 | N/A |
| `C5_prne_plus_ridge` | torque | 없음(validation; observer 판정 별도) | 0 | 0/4 | 1.000/1.000 | 1.000 | 605.652/1060.185 | N/A |
| `C5_ridge_only_matched` | servo | not_an_adoptable_method | 0 | 0/4 | 1.000/1.000 | 1.000 | 247.600/815.748 | N/A |
| `C5_ridge_only_matched` | torque | not_an_adoptable_method | 0 | 0/4 | 1.000/1.000 | 1.000 | 288.467/418.228 | N/A |
| `C6_seventh_joint` | servo | no_false_alarm, no_saturation, cost, local_radius, G4 | 10 | 0/4 | 12.879/5.352 | 1.885 | 293.767/462.457 | 5/6 |
| `C6_seventh_joint` | torque | no_false_alarm, no_saturation, cost, local_radius, G4 | 11 | 0/4 | 11.703/2.941 | 1.906 | 317.292/591.637 | 4/6 |
| `C6_S_curve_joint` | servo | no_false_alarm, no_saturation, cost, local_radius, G4 | 6 | 0/4 | 12.289/5.102 | 1.567 | 347.530/539.798 | 5/6 |
| `C6_S_curve_joint` | torque | no_false_alarm, no_saturation, cost, local_radius, G4 | 9 | 0/4 | 11.240/2.875 | 1.578 | 328.834/450.267 | 4/6 |
| `C6_SO3_R3_seventh` | servo | no_false_alarm, no_saturation, cost, local_radius, G4 | 10 | 0/4 | 12.905/5.342 | 1.881 | 368.411/759.707 | 5/6 |
| `C6_SO3_R3_seventh` | torque | no_false_alarm, no_saturation, cost, local_radius, G4 | 11 | 0/4 | 11.738/2.946 | 1.903 | 374.689/547.847 | 4/6 |
| `C5_prne_only` | observer (A/B) | no_lost_or_extra_false, benefit | 0/0 | N/A | N/A/N/A | N/A | A 692.634/N/A; B 704.021/N/A | N/A |
| `C5_prne_plus_ridge` | observer (A/B) | benefit | 0/0 | N/A | N/A/N/A | N/A | A 687.486/N/A; B 701.250/N/A | N/A |
| `C5_ridge_only_matched` | observer (A/B) | normal, benefit, not_diagnostic | 0/0 | N/A | N/A/N/A | N/A | A 279.145/N/A; B 298.750/N/A | N/A |

표의 59개 판정 행은 모두 최종 reject다. C5 정상 validation 6행은
제어 gate를 통과했지만 새 채택이 아니며, observer 효능 판정은 별도 행에서
reject로 기록했다. 모든 등록 후보와 불리한 결과를 유지했다. 검증 단계에서 버린 후보를 holdout이나
공식 평가 결과를 본 뒤 되살리지 않았다.
[원자료: `R2_ADOPTION.json`](../results/v2_protocol/R2_ADOPTION.json)

### 재설계 사용 여부

- protocol 허용량: 가족별 최대 1회; 실제 사용 0회.
- 공식 회귀 이후 재설계 금지. holdout·공식 결과를 본 뒤 후보를 고치거나 재승격하지 않았다.
- `C3_DOB_w4.5/servo`, `C4_P-IDC_w4.5_d1/torque`: cost 관문만 실패한 상태에서 재설계 미사용; 사유 기록 없음.

## Lie group 사용 내역

| 방법 | 참조 절 | 직접 쓴 식 | 탄성 플랜트 적용 | 유클리드 대비 효과 | 판정 |
|---|---|---|---|---|---|
| C1 wrench planning / power-dual mapping | Lie Group Robotics §2.2; 접촉점·탄성 lift는 자체 유도 | J_b=Q J_w, W_b=Q W_w, J_b^T W_b=J_w^T W_w; theta_d=q_d+K_e^-1 tau_d | 접촉점 Jacobian에 원점 이동을 한 번만 포함하고 알려진 COM 작용선·COP 제한을 썼다. A/B 탄성 모터 lift와 기존 tracker, 1 ms ZOH·한 주기 지연을 분석했다. elastic joint + 1 ms ZOH + one-cycle delay의 유한 국소 검사이지 전역 보장은 아니다. | 비리그룹 R1 대비 4설정 모두 A/B G4 0/4, TCP·오감지·토크 비용 관문 실패. 좌표변환만 바꾼 별도 ablation은 없어 효과를 Lie 기하에만 귀속하지 않는다. | reject |
| C2 SE(3) normal coordinate | Lie Group Robotics §§2.2,6.1; Lie Group Robot Control §§11.4,11.5 | E=T_des^-1 T, xi=log(E), e=-n^T xi, normal covector=J_r(xi)^-T n | elastic joint의 motor reference로 올리고 1 ms exact ZOH, one-cycle delay, measured displacement/speed/acceleration projection을 포함했다. | A: Euclidean/SE(3) max RMS 차이 4.1e-12/2.3e-13 mm, FA 0/0이지만 p99 269/293→472/458 us. B: RMS 차이 최대 0.222/0.219 mm, FA 10→11 및 2→3, 비용·국소 안정성 gate 실패. | reject |
| C3 NRIC / motor DOB | Lie Group Robot Control §10.4(NRIC); DOB 모터식·정규화 보상은 자체 유도 | Delta tau=r tanh((1+gamma^-2)s); d_hat=z+L B v_m | 명목 탄성 플랜트에 독립 무보상 queue와 implicit midpoint를 쓰고 motor DOB를 샘플 정확 적분했다. 실제 Drive 내부값 대신 허용 신호·자체 queue로 재생하며 1 ms ZOH·명령/JTS 지연을 포함했다. elastic joint + 1 ms ZOH + one-cycle delay의 유한 국소 검사이지 전역 보장은 아니다. | R1 대비 NRIC는 오감지·국소 불안정성, A DOB는 최대 TCP 약17–18% 감소에도 놓기 torque-step 최악+5.42–8.54%로 탈락. SE(3) 오차를 쓰지 않으므로 독립 Lie-group 이득으로 해석하지 않는다. | reject |
| C4 P-APD / P-IDC with pRNE action | Lie Group Robot Control §§10.1.3,10.1.6; Lie Group Robotics 목차 §8.3, 본문 미확인 | tau_d=M qdd_r+C(q,v_actual) qd_r+g+K_r(qd_r-v); theta_d=q_d+K_e^-1 tau_d | 설계속도와 실제속도를 분리한 자체 pRNE action, mass-scaled gain·load blend·clip·anti-windup·탄성 motor lift를 썼다. P-APD는 자체 변형 이름이며 강체 수동성 정리를 전용하지 않고 1 ms·지연 국소 분석을 수행했다. elastic joint + 1 ms ZOH + one-cycle delay의 유한 국소 검사이지 전역 보장은 아니다. | B P-IDC(rate4.5,damping1)는 7.55kg RMS3.140→1.975mm, 최대8.060→4.382mm지만 경량 최대+9.65%, 놓기 step 최악+58.9%로 탈락. 제어 법칙 전체 비교이며 pRNE/기하 단독 효과는 분리하지 않았다. | reject |
| C5 metric-compatible pRNE | Lie Group Robotics 목차 §8.3, 본문 미확인; 자체 유도는 R2_METHODS의 C5 절 | Mdot(q,v)=C(q,v)+C(q,v)^T; C(q,v)^T u는 reverse VJP recursion으로 계산 | elastic plant-controller는 R1 그대로이고 1 ms ZOH와 one-cycle delay 물리 폐루프 변경이 없어 C5-specific pole analysis는 N/A다. math·normal·collision·timing·156개 physical parity를 각각 검사했으며 강체 수동성 보장을 옮기지 않았다. | matched new ridge와 pRNE+ridge는 개발 A/B 모두 15/24로 isolated gain 0. pRNE-only 11/24는 ridge 제거가 함께 있어 pRNE 단독 효과로 귀속하지 않는다. reverse VJP helper는 같은 수학 출력에서 median 825→178 us였지만 R1 observer 대비 속도 향상은 아니다. | reject |
| C6 SO(3)+R3 separated interpolation | Lie Group Robot Control §§9.1,9.3; Lie Group Robotics §6.1 및 SO(3)+R3 자체 유도 | R(s)=R0 exp(s log(R0^T R1)); p(s)=p0+s(p1-p0), s는 7차 또는 S-curve | elastic joint reference와 기존 clipping을 유지하고 1 ms ZOH 및 one-cycle delay가 든 폐루프 국소 고유값을 다시 계산했다. | 고차 보간과 SO(3)+R3 후보는 무충돌 false rise, 포화, TCP 비용, 국소 radius 또는 G4 gate를 통과하지 못했다. | reject |

교재의 강체 안정성 결과를 그대로 이 탄성 관절 플랜트의 보장으로 사용하지 않았다.
각 제어 후보는 탄성 관절, 1 ms ZOH, 한 주기 지연을 포함한 유한 국소 분석으로만
판정했다. 구성 의존 PD gain `M(q)`, load blending, clipping도 강체 정리의 전제를
바꾸므로 수동 플랜트 보장을 주장하지 않는다.

## 명시적 한계

- P0는 PARTIAL이다. 40회에서 지속 G4를 만드는 feasible displacement/speed를 식별하지 못했고 경계는 관측·보수 선택이지 측정 최적값이 아니다.
- holdout 0.75 kg은 C5 학습 질량 하나와 겹친다. 새 joint condition과 seed만으로 모든 질량 unseen 일반화를 주장하지 않는다.
- pRNE-only의 4회/architecture 감지 손실은 ridge 제거가 함께 있어 pRNE에만 귀속할 수 없다. matched ridge와 pRNE+ridge의 15/24 대 15/24가 isolated zero effect다.
- 안정성은 강제 R1 물리점·고정 접촉 모드의 유한 국소 선형화이며 수동성·전역·hybrid 보장이 아니다.
- all 25 configurations rejected는 이 유한 실험의 결론이며 NRIC, DOB, P-APD, P-IDC 또는 Lie-group 방법 계열이 일반적으로 불가능하다는 증명이 아니다.
- pRNE timing 비용은 shared core를 중복 계산한 이 독립 observer 구현의 비용이다. 알고리즘에 본질적으로 피할 수 없는 비용이라고 주장하지 않는다.
- 최악 timing cycle의 CPU와 wall이 거의 같아 interpreter/allocation/GC 작업과 양립하지만 원인을 유일하게 확정하지 않는다. 과거 R1 GC 상관은 fresh R2 원인 증명이 아니다.
- R2 final은 R1 exact-source alias다. common reference와 v1/R1 D4 행을 fresh R2 독립 재실행으로 주장하지 않는다.

## 수학·안정성 검증

- pRNE 유한 표본: `{"all_protocol_gates_pass": true, "cases": 9, "maximum_bias_max_abs": 8.348877145181177e-14, "maximum_direct_transpose_max_abs": 6.67395472220278e-10, "maximum_selected_mdot_normalized_inf": 3.893921841004257e-10, "maximum_vjp_inner_product_abs": 4.440892098500626e-15}`
- R2 후보 안정성: 전체 300건, protocol-radius 통과
  222건, 엄격 안정 190건.
- R1 비교 안정성: `{"non_strict_cases": [{"architecture": "torque", "mass_kg": 7.7, "spectral_radius": 1.0011726546962387, "time_s": 2.5}], "protocol_radius_fail_cases": [{"architecture": "torque", "mass_kg": 7.7, "spectral_radius": 1.0011726546962387, "time_s": 2.5}], "protocol_radius_pass_cases": 11, "spectral_radius_max": 1.0011726546962387, "spectral_radius_min": 0.9979360768714778, "strict_stable_cases": 11, "total_cases": 12}`
- compact ledger는 all_cases 300건을 모두 유지한다.
- 위 결과는 유한 국소·고정 접촉 모드 검증이며 비선형/하이브리드 플랜트의 전역
  안정성 또는 안전 보장이 아니다.

### pRNE 계산·할당 진단

- full_C_basis_transpose: median 825.0 us, p99 1522.3 us, peak 11000 B
- reverse_VJP: median 178.1 us, p99 206.4 us, peak 10736 B

이 비교는 같은 pRNE 수학 출력을 만드는 두 helper의 순수 microbenchmark다. R1
observer 대비 속도 향상이나 제어기 채택 근거로 사용하지 않았고, plant episode는
0회이며 전역 GC 설정도 바꾸지 않았다.

### 최종 R1/R2 alias timing

- servo: rows 102, cycles 612000, largest row p99 334.763 us, maximum wall/CPU 5576.324/5573.779 us, overruns 1
- torque: rows 102, cycles 612000, largest row p99 363.109 us, maximum wall/CPU 6016.548/6014.207 us, overruns 5

1 ms strict goal은 미달이다. 같은 cycle CPU가 wall과 거의 같고 context switch는
최악 cycle에서 0/0이지만, 이는 interpreter/allocation/GC 작업과 양립한다는 뜻일
뿐 원인을 유일하게 확정하지 않는다. 과거 R1 GC 상관은 역사적 보조 증거로만 둔다.

## 미달·outlier

```json
{
  "outliers_retained": [
    "후보 안정성 300건 중 protocol-radius 통과 222건, strict 190건만 통과했다.",
    "R1 비교 12건 중 B 7.7 kg, 2.5 s의 spectral radius 1.00117265 실패를 유지했다.",
    "최종 alias timing 1,224,000 paired cycles에서 A/B 1/5회의 1 ms 초과를 모두 유지했다. 최대 wall/same-cycle CPU는 A 5576.324/5573.779 us, B 6016.548/6014.207 us다.",
    "baseline D4 collision의 masked 109건과 모든 false rise를 missed와 분리해 보존했다.",
    "공개본 full stability JSON은 공백만 제거해 11,964,321→6,540,838 bytes로 줄였으며 300건의 모든 case·pole·outlier를 유지했다. 작업 사본 원본은 바꾸지 않았다."
  ],
  "summary": [
    "T1~T6은 R1보다 개선되지 않아 모두 미달로 기록했다.",
    "D4 baseline의 masked를 버리지 않고 A 88/43/49, B 78/42/60 detected/missed/masked로 분리했다.",
    "R1과 최종 R2 alias의 D4는 A/B 모두 111 detected, 69 missed이며 독립 R2 재실행으로 세지 않는다.",
    "C5 pRNE+ridge holdout은 감지 이득 0에 추가 false rise가 있어 음성 결과로 유지했다."
  ],
  "unmet": [
    "T1: 마지막 20 ms 평균 지지율 50 % 목표 미달(A 0.2098 %, B 11.1507 %).",
    "T2: 놓기 토크 계단은 R1과 동일해 추가 개선 없음.",
    "T3: 분석 섭동 악화와 false rise가 줄지 않음.",
    "T4: A 포화 2/189 및 peak 1.6098 mg가 남아 있음.",
    "T5: TCP RMS/최대는 R1과 동일하며 B 3.232/8.444 mm가 남아 있음.",
    "T6: D4 A/B 각각 69회 놓침과 공식 집기·놓기 미감지가 남아 있음."
  ]
}
```

## 동결·holdout·공식 회귀

### Holdout
- 동결 뒤 marker batch_attempt=1로 152행을 한 번만 열었고 이후 튜닝하지 않았다.
- 최종 R1 alias는 A/B 각각 20 detected, 10 missed였다. false rise는 A 0, B offset 1이었다.
- 이미 탈락한 C5 pRNE+ridge도 A/B 각각 20/10으로 감지 이득이 없었고 false rise는 A 4(충돌 2, offset 2), B 2(offset 2)였다.
- holdout의 0.75 kg은 C5 학습 질량 하나와 겹친다. COM·시각·힘·방향과 seed는 새 조건이지만 네 질량 모두 unseen이라고 주장하지 않는다.

### 공식 회귀
- holdout 뒤 unchanged evaluate.py를 default 이름 overlay에서 A/B 각각 한 번 실행했다.
- 0/4/8 kg 무충돌 control은 추가 오감지 0, 포화 0 %, 발산 없음으로 R1과 같은 셀을 유지했다. B의 8 kg 운반 collision 행 false max 1회도 R1과 같다.
- 8 kg 운반은 A/B에서 15 N 45/43 ms, 10 N 62/56 ms를 유지했다.
- 집기 10/15 N과 놓기 10 N의 기존 미감지는 남았고, 공식 노출 뒤 수정하거나 재승격하지 않았다.

## 실행 및 미실행 검증

### 실행
- 작업 사본 unittest 238개와 pure function 18개(P0 stability 7, matched-ridge 3, detector 8) 통과. 이 함수 검사에서 episode는 실행하지 않았다.
- Python 138파일 syntax 검사와 production AST 3파일 통과.
- freeze 53파일 불변 검증 통과.
- pristine-kit+공개 overlay unittest 219개, pure function 18개, 보호 파일 20개 검증 통과. overlay에서 공식 평가는 재실행하지 않았다.
- pRNE 9개 유한 표본의 Mdot=C+C^T, VJP, bias 항등식 protocol gate 통과.
- 탄성 관절+1 ms ZOH+one-cycle delay 국소 분석 300건과 R1 비교 12건을 전부 보존.
- 동결 뒤 holdout 152행을 한 batch로 실행하고 마지막 공식 A/B 58 episode를 완료.
- 최종 R1/R2 alias 204행의 1,224,000 paired cycles를 검토하고 1 ms 초과 6회를 모두 보존.
- 원본 tree 29파일, 보호 파일 20개, 공식 환경 package 21개가 불변임을 확인.
- 공개 staging 399파일의 audit_public(kit·guide·ffprobe·require-media·SHA), published-byte report hash 재결합/검사를 통과했다. 최종 파일별 byte SHA는 artifacts/SHA256SUMS에 기록한다.
- validate_media legacy/v2/R2 통과, media 16개 7,641,724 bytes 통과, 복사된 public report --check 통과.

### 미실행
- 실제 로봇, 안전 반응, hidden 조건, Windows 환경은 검증하지 않았다.
- 유한 국소 분석은 비선형·하이브리드 전역 안정성 또는 확률 보장이 아니다.

### 자원·재현 조건
- Python은 PYTHONPATH를 제거한 환경에서 실행했다.
- 작은 모델과 분석은 CPU로 수행했고 GPU 작업은 0개였다.
- 소유 simulation worker는 1개였고 다른 세션 프로세스에 신호를 보내지 않았다.
- 공식 실행 중 최소 가용 RAM은 21.509 GiB였다.
- R2 four-way 영상은 2쌍·16 illustrative episode이며 Mesa llvmpipe를 사용했다.

## 변경 경로

- `experiments/r2_*.py`
- `tools/r2_*.py`
- `models/r2_*.json`
- `results/v2_protocol/R2_*.json`
- `docs/R2_METHODS.md`
- `docs/R2_ROBUST_DERIVATION.md`
- `docs/R2_RESULTS.md`
- `report/PRESENTATION_R2.md`
- `results/presentation.pdf`

## 증거 출처

`fresh`는 R2에서 새로 실행한 증거이고, `reused_exact_source`는 명시된 원본 결과를
그대로 재사용한 비교 증거다. 재사용 행을 독립 재실행으로 주장하지 않는다.

| 증거 묶음 | 파일 | 출처 |
|---|---|---|
| adoption | `results/v2_protocol/R2_ADOPTION.json` | fresh |
| allocation_comparison | `results/v2_protocol/R2_ALLOCATION_COMPARISON.json` | fresh |
| common_reference | `results/v2_protocol/R2_COMMON_REFERENCE.json` | reused_exact_source |
| control_detector_parity | `results/v2_protocol/R2_CONTROL_DETECTOR_PARITY.json` | fresh |
| d4 | `results/v2_protocol/R2_D4_base.json` | fresh |
| d4_reused | `results/v2_protocol/S5_D4.json` | reused_exact_source |
| development | `results/v2_protocol/R2_DEVELOPMENT_R1.json` | fresh |
| development | `results/v2_protocol/R2_DEVELOPMENT_C5_prne_only.json` | fresh |
| development | `results/v2_protocol/R2_DEVELOPMENT_C5_prne_plus_ridge.json` | fresh |
| development | `results/v2_protocol/R2_DEVELOPMENT_C5_ridge_only_matched.json` | fresh |
| development_review | `results/v2_protocol/R2_DEVELOPMENT_REVIEW.json` | fresh |
| environment_integrity | `results/v2_protocol/R2_ENVIRONMENT_INTEGRITY.json` | fresh |
| freeze | `results/v2_protocol/R2_FREEZE.json` | fresh |
| holdout | `results/v2_protocol/R2_HOLDOUT.json` | fresh |
| holdout_opened | `results/v2_protocol/R2_HOLDOUT_OPENED.json` | fresh |
| holdout_review | `results/v2_protocol/R2_HOLDOUT_REVIEW.json` | fresh |
| lie_comparison | `results/v2_protocol/R2_LIE_COMPARISON.json` | fresh |
| official | `results/v2_protocol/R2_OFFICIAL_REGRESSION.json` | fresh |
| official_started | `results/v2_protocol/R2_OFFICIAL_STARTED.json` | fresh |
| p0 | `results/v2_protocol/R2_P0_BASELINE.json` | fresh |
| p0 | `results/v2_protocol/R2_P0_PROBE_DESIGN.json` | fresh |
| p0 | `results/v2_protocol/R2_P0_REVIEW.json` | fresh |
| p0 | `results/v2_protocol/R2_P0_down_small.json` | fresh |
| p0 | `results/v2_protocol/R2_P0_down_stroke.json` | fresh |
| p0 | `results/v2_protocol/R2_P0_up_small.json` | fresh |
| p0 | `results/v2_protocol/R2_P0_up_stroke.json` | fresh |
| perturbations | `results/v2_protocol/R2_PERTURBATIONS.json` | fresh |
| preholdout_review | `results/v2_protocol/R2_PREHOLDOUT_REVIEW.json` | fresh |
| prne_math | `results/v2_protocol/R2_PRNE_MATH.json` | fresh |
| prne_normal | `results/v2_protocol/R2_PRNE_NORMAL.json` | fresh |
| protocol | `results/v2_protocol/PROTOCOL_R2.json` | fresh |
| r1_stability | `results/v2_protocol/R2_R1_STABILITY.json` | fresh |
| stability_compact | `results/v2_protocol/R2_CANDIDATE_STABILITY_COMPACT.json` | fresh |
| stability_full | `results/v2_protocol/R2_CANDIDATE_STABILITY.json` | fresh |
| timing_review | `results/v2_protocol/R2_TIMING_REVIEW.json` | fresh |
| validation_review | `results/v2_protocol/R2_VALIDATION_REVIEW.json` | fresh |

### 공개 유도·문헌

- `docs/R2_METHODS.md`
- `docs/R2_ROBUST_DERIVATION.md`
- `docs/LITERATURE.md`

- [Nonlinear robust internal loop compensator for robust control of robotic manipulators](https://pure.kaist.ac.kr/en/publications/nonlinear-robust-internal-loop-compensator-for-robust-control-of-/)
- [Bringing nonlinear H-infinity optimality to robot controllers](https://pure.kaist.ac.kr/en/publications/bringing-nonlinear-h-optimality-to-robot-controllers/)
- [Contact-responsive high-gain robot motion controllers using a constrained disturbance observer](https://link.springer.com/article/10.1007/s11370-026-00744-z)
- [Linear Model Predictive Control of 6-DOF ROV Using NRIC](https://jkros.org/xml/39598/39598.pdf)
- [Sliding Mode Momentum Observers for Estimation of External Torques and Joint Acceleration](https://elib.dlr.de/127719/1/root.pdf)
