# R3 설계와 검증 범위

R3은 R2의 공식 회귀 이후 튜닝이 아닌 별도 사전 등록 실험이다. 이미 노출된 R2 사전 검증은 설계 자료, 기존 공식 수치는 회귀 기준으로만 쓴다. 새 사전 검증/개발 검증/holdout을 분리하고 서보와 토크를 독립 판정한다. [원인 기록](R3_CAUSE_ANALYSIS.md) → [사전 protocol](../results/v2_protocol/PROTOCOL_R3.json) → [재설계 결정](../results/v2_protocol/R3_REDESIGN_DECISION.json)의 순서를 따른다.

## A: derivative-free 모터 DOB와 출력 handoff

모터식을 `B vdot = u - tau_s - f(v) + d`로 두고, `p=L B v`, `dhat=z+p`, `L=4.5 rad/s`로 정한다. 관측 입력을 한 주기 동안 고정하면

```
a = exp(-L * 0.001)
z[k+1] = a*z[k] - (1-a)*(p[k]+u_hat[k-1]-tau_s[k]-f(v[k]))
dhat[k] = z[k+1]+p[k]
```

이다. 차분 가속도 대신 허용된 encoder 속도/JTS와 **자체 명령 queue로 재구성한** Drive PI 출력을 쓴다. 실제 Drive 내부 상태나 플랜트의 참 토크는 런타임에서 읽지 않는다. 식의 `u_hat`에는 한 주기 지연이 있고, 자체 Drive의 속도 필터·적분 상태도 분석에 포함한다. 이는 강체 관절의 이상 DOB 보장이 아니다.

`h(x)=10x^3-15x^4+6x^5`를 `[0,1]` 밖에서 상수로 연장한다. `gamma(t)=1-h((t-t0)/(3.60-t0))`를 써서 R1의 서보 A feedforward에만 `gamma*clip(-dhat)`을 더한다. `t0=3.35/3.10 s` 두 설정이다. 감쇠 중 observer는 계속 진화하며, 원래 집기/놓기/비흡착/검출 freeze는 유지한다. 직접 놓기 보상은 이미 0이었으므로 **이전 궤적 이력**의 비용을 줄일 수 있는지 시험하는 가설이지 순간 보상 제거 실험이 아니다. 조기 hard-off로 계단을 평가 창 밖에 옮기지 않고 전체 주기 최대 계단도 따로 기록한다.

## B: P-IDC action 최적화와 연속 스케줄

`e=q_ref-q_hat`, `edot=v_ref-v_hat`, `r=4.5`, `zeta=1`에서

```
v_r = v_ref + 2*zeta*r*e + r^2*integral(e)
a_r = a_ref + 2*zeta*r*edot + r^2*e
s   = v_r - v_hat
tau_PIDC = M(q_hat)*(a_r+zeta*r*s) + C(q_hat,v_hat)*v_r + g(q_hat)
tau_link = tau_R1 + alpha*clip(tau_PIDC-tau_R1, reserve)
```

로 쓴다. **clip한 차이를 나중에 alpha로 곱한다.** 질량 스케줄은 알려진 `box_mass`만 써서 `h((mass-3.15)/2.50)`, phase 스케줄은 `1-h((t-3.60)/0.24)`다. phase/mass/두 값의 곱 세 설정을 비교한다. 알 수 없는 실제 하중이나 raw load fraction을 activation으로 쓰지 않는다. R1과 같은 탄성 모터 lift `theta_ref=q_ref+K_e^-1*tau_link`를 적용하고, load estimator와 lift는 주기당 한 번만 갱신한다. `alpha=0`은 수학·관측 replay 검사에서 R1과 같은 출력이다.

명목 endpoint의 `inverse_dynamics(q,0,a)`를 각각 한 번 사용해 `M*a+g`를 정확히 보간한다. fractional-payload 모델의 근사 동등성으로 대체하지 않는다. `C*v_r`는 공통 kinematics와 미리 할당한 배열을 쓰는 한 번의 polarized RNE recursion으로 계산한다. 전역 GC 설정은 바꾸지 않았다.

### 자체 유도한 pRNE action

body twist 순서를 `(omega,linear)`로 두면 `ad(V)=[[omega_x,0],[linear_x,omega_x]]`이고 motion transform `X`의 힘 쌍대 변환은 `X^T`다. 관절 screw `S_i`, 실제 속도 `v`, 설계 벡터 `u`에 대해 다음을 순방향 계산한다.

```
V_i = X_i*V_parent + S_i*v_i
U_i = X_i*U_parent + S_i*u_i
P_i = X_i*P_parent + ad(X_i*V_parent)*S_i*u_i
F_i = I_i*P_i - ad(V_i)^T*I_i*U_i
```

역방향으로 `F_parent += X_i^T*F_i`, `c_i=S_i^T*F_i`를 모으면 `c=C(q,v)u`다. 빈/적재 모델의 같은 kinematics에서 `I(lambda)=(1-lambda)I0+lambda*I1`이므로 이 recursion도 endpoint 보간과 같다. 운동에너지의 미분과 양쪽 power pairing을 정리하면 고정 질량에서 `Mdot(q;v)=C(q,v)+C(q,v)^T`가 된다. 시간 변화하는 load fraction에는 별도 `lambda_dot*dM/dlambda`가 있어 이 항등식을 가변 하중 폐루프의 수동성 증명으로 옮길 수 없다.

[별도 유한표본 수치 검증](../results/v2_protocol/R3_PRNE_MATH.json)은 새 표본에서 endpoint/action 보간과 `h,h/2` 유한차분 `Mdot`을 확인한다. endpoint 비교에는 기존 pRNE 구현을 재사용하므로 독립 재구현 검증은 아니다. 명목 강체 행렬의 항등식일 뿐 탄성 관절·clipping·delay의 안정성을 증명하지 않는다. 공간 벡터 계산과 feedback/schedule을 함께 바꿨으므로 R1 대비 효과를 Lie 기하 하나에 귀속하지 않는다.

## 리그룹 자료의 사용과 변경

| 대상 | 서지·절 | 사용 식 | 탄성 관절에 맞춘 변경 | 분리 측정의 한계 |
|---|---|---|---|---|
| A 모터 DOB | 모터 DOB 식·handoff 자체 유도 | `dhat=z+LBv` | 모터/Drive queue, JTS 지연, 1 ms ZOH | SE(3) 오차를 쓰지 않아 독립 Lie 효과 아님 |
| B P-IDC | Lie Group Robot Control §10.1.6; Lie Group Robotics 목차 §8.3, 본문 미확인 | `M*a_r+C*v_r+g+M*zeta*r*s` | 추정 링크 상태→탄성 모터 lift·한 주기 지연·clip·부드러운 스케줄 | R1과 제어법·스케줄이 달라 독립 기하 ablation 아님 |
| B pRNE | Lie Group Robotics 목차 §8.3, 본문 미확인; recursion은 자체 유도 | `ad`, `X^T`, `C(q,v)u`, `Mdot=C+C^T` | rigid nominal 항만 사용하고 elastic 지연 플랜트는 별도 분석 | endpoint 수치 동등성은 확인하되 성능 개선을 리그룹 단독 기여로 주장하지 않음 |

자료의 문장·그림을 재현하지 않았다. 정확한 확인 범위와 기존 원문 확인 이력은 [R2 자체 유도](R2_METHODS.md)와 [문헌](LITERATURE.md)에 남아 있다.

## 안정성·선정에서 허용하지 않는 추론

- 모든 설정의 강제점 검사 R1 물리점, 고정된 challenger의 **실제 candidate** 정상 trace에서 탄성 관절·Drive PI·JTS 필터/지연·encoder 차분·1 ms exact ZOH·한 주기 명령 queue를 포함한다. 실제 TCP/놓기 계단 최대값도 넣는다.
- 중앙 유한차분 `h/h2`의 기계 입력과 폐루프 Jacobian 수렴을 구분한다. FD 존재는 수렴 통과가 아니다. inactive 적분의 인위적 단위 pole은 분석 전용 adapter로 제외하고 원래 compute를 상속해 런타임은 바꾸지 않는다. 초기 분석도 보존했다.
- moving R1 자체에서도 고정 위상 국소 pole 관문 실패가 있다. 새 방법만의 비선형 발산 원인이나 전역 불안정성을 입증한 것이 아니며, 그 사실로 새 방법의 관문을 면제하지도 않는다. 수치 경계의 불확실성은 실패/미확인으로 남긴다.
- 비용은 **모든 동일 조건** RMS·최대 TCP·집기/놓기 토크 계단·놓기 힘 각각 R1의 1.05배 이하다. 질량별 최댓값의 비를 전체 최대의 비로 대체하지 않는다. G4는 원래 mean/min/increase/최대값 기준을 유지하고 C3/C4에 새 통과 수 요건을 추가하지 않는다. 미달은 그대로 미달이다.
- p99 wall<800us는 선정 관문이고 최대 1ms는 목표다. 모든 outlier를 보존한다. 충돌 자료로 임계값을 맞추지 않고 holdout·공식 노출 뒤 source/파라미터를 고치지 않는다. 이 유한 시뮬레이션은 실제 로봇 안전성·전역/LTV/hybrid/passivity 증명이 아니다.

## 같은 관측의 계산비용 비교 (선정과 분리)

새 정상 사전 검증의 R1 7.17 kg 관측 6000주기를 R2 P-IDC와 R3 B-mass에 같은 순서로 재생했다. 이 질량에서 alpha=1이므로 최적화 전후 전체 토크의 최대 차이는 1.71e-13Nm다. 플랜트는 진행하지 않았고 R2→R3 직렬 한 번씩, warm-up/이상치 제거/전역 GC 변경 없이 측정했다.

| 전체 controller+detector wall [us] | R2 원 구현 | R3 action 최적화 |
|---|---:|---:|
| median |450.194|319.489|
| p99 |772.189|475.509|
| maximum |1001.346|760.157|

[모든 시간·CPU·이상치 원자료](../results/v2_protocol/R3_REPLAY_TIMING.json). 고정 순서·단일 호스트 realization이라 무작위 반복 실험이나 전역 실시간 보장은 아니다. 줄인 중복 계산/할당의 참고 이득이며 Lie 기하만의 기여를 분리하지 않았다. 실제 다조건 실행의 모든 outlier와 최대 1 ms 목표 판정은 별도로 유지한다. 이 속도 이득으로 B의 격자 비용·분석 섭동 오감지·국소 관문 실패를 면제하지 않는다.
