# R3 원인 분석: 새 코드보다 먼저 확인한 것

R3은 R2 공식 회귀 뒤의 튜닝이 아니라 새 protocol의 별도 연구다. 두 방법은 이미 본 R2 validation4조건에서 골랐으므로 선택 편향이 있다. 아래 수치는 **비공식 R2 관측 trace의 분석/반사실 controller replay**이며 새 물리 실험의 효능 근거가 아니다. 공식 trace는 쓰지 않았다. 원인 기록은 새 후보 코드와 protocol보다 먼저 남겼다.

## A: place 구간 직접 DOB 감쇠 가설은 맞지 않았다

5.65kg의 최대 놓기 계단은3.902s J2에서10.752119→11.335217Nm(+5.423%)다. 한 주기 지연을 포함한 8개 관측 replay는 원래 모터 토크와 최대오차0으로 일치했다. 해당 계단 양쪽 command의 **직접 DOB 보상은 모두0**이다. 기존 방법은3.6s부터 이미 `z=-p`로 추종하고 보상을 끈다.

증가분0.583098Nm는 feedforward 이력+0.254429, velocity feedback+0.331350, integral−0.002681Nm로 분해된다. 직전 J2 양자화 속도가 R1 0.023968/DOB0.047937rad/s에서 둘 다0으로 바뀌어 속도 feedback 계단이 커졌다. 다른 질량에서는 FF와 속도항이 상쇄되거나 최대 사건이1ms 이동했다. 정확히5.65kg만 불리한 연속 비선형 위상이 되는 이유까지 식별했다는 뜻은 아니다.

기존 전역 TCP 최대 이득17.06–19.06%도3.824–3.827s, DOB가 이미 꺼진 놓기 하강에 나타났다. 초기 transport(3.10s 전) RMS는6.87–7.48% 개선됐지만3.10–3.35s는 소폭 악화했다. 따라서 직접 place 보상을 더 줄이는 대신 **이전 상태 이력을 완만하게 R1에 넘기는 handoff**를 시험한다. 새 handoff가 전역 최대 이득을 보존할지는 미검증이다. 조기 hard-off는 나쁜 계단을 평가 창 밖으로 옮길 수 있어 진단 전용이고 채택하지 않는다.

## B: 경량 실패는 하중 보간보다 reference feedback과 terminal phase에 연결된다

경량1.05/3.15kg의 P-IDC TCP 최대는 release 뒤4.450/4.451s J1이다. 그 시점 effective load·integral·clipping은0이지만 P/V feedback은 계속 켜져 있다. 기존 freeze는 적분만 감쇠하며 P/V를 끄지 않는다.

| 질량 kg | 놓기 최대 사건 | reference feedback 기여 Nm | load blend 기여 Nm | 해당 사건 clip |
|---:|---|---:|---:|---|
|1.05|3.875s J2|3.148|0.336|거의0|
|3.15|3.878s J2|3.859|0.838|거의0|

이 분해는 고정된 빈모델→하중→feedback→clip 순서의 signed command 차분이다. 비선형 friction/clip이 있어 유일한 인과 분해라고 주장하지 않는다. 전체 place 창에서 clip을 없애면 최대가7.286→10.294/8.782→9.948Nm로 더 나빠지므로 clip 제거가 해법은 아니다.

무거운5.65/7.55kg의 lift RMS는4.825→1.137/6.159→1.272mm다. R1의1.737/1.741s J2 최대를 줄인 이득이3.7s 이전에 집중된다. 무거운 하중의 놓기 계단은 R1과 같은 release J2 사건이며 load fraction 변화가 지배한다. P-IDC 관측에 R1 link law를 replay하면9.440/12.959Nm, P-IDC9.475/12.958Nm로 가까워 원래 R1 대비 약+2% 비용과 양립한다. 새 폐루프 궤적의 인과 효과는 별도 검증이 필요하다.

따라서 **3.60s부터 기존 UNLOAD_START3.84s까지 link law를 quintic으로 R1에 넘기는 phase handoff**를 우선한다. 연속 질량 스케줄 가설은 비교 후보로 남기며 mass×phase도 비교한다. raw load fraction을 activation으로 곱하면 무거운 release의 급격한 변화가 새 계단을 만들 수 있어 쓰지 않는다.

## 계산과 수학

empty/full의 kinematics는 같고 pRNE action은 body inertia에 선형이다. `I(lambda)=I0+lambda*(I1-I0)`를 쓰는 한 recursion이 두 endpoint action 보간과 최대7.11e−15Nm로 일치했다. 공통 kinematics·workspace를 재사용하고 baseline FF를 중복 계산하지 않는 것이 R3-B의 계산 축이다. fractional-payload nominal M/g는 bit-exact가 아니므로 endpoint M/g 보간을 유지한다. 시간 개선은 아직 측정하지 않았다.

## 검증 범위와 근거

- 관측 replay와 원래 모터 토크 일치: A/B 각각8개 trace, 최대오차0.
- 독립 elastic lift replay도 B 원시 command와 일치.
- 허용된 encoder/속도/JTS/grip/알려진 mass·COM으로 replay; 실제 물리 내부값은 사후 사건 분석에만 사용.
- 기존4조건은 설계 자료이며 새로운 validation/development/holdout과 분리한다.
- 탄성 관절·1ms ZOH·1주기 지연과 전환 구간 안정성은 새 구현 뒤 별도 검사한다. 강체 정리·유한 국소 고유값을 전역 보장으로 쓰지 않는다.

[분석 수치·증거 hash](../results/v2_protocol/R3_CAUSE_ANALYSIS.json), [기존 A validation](../results/v2_protocol/R2_VALIDATION_C3_DOB_w4.5.json), [기존 B validation](../results/v2_protocol/R2_VALIDATION_C4_P-IDC_w4.5_d1.json).

공개 정리에서 내부 작업 표기 1곳만 중립 표현으로 바꿨고 수치와 판단은 구현 전 기록 그대로다.
