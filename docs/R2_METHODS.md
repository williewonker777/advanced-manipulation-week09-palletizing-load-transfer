# 2차 실험: 자체 유도와 적용 경계

사전 protocol은 `results/v2_protocol/PROTOCOL_R2.json`이다. 모든 런타임은
엔코더·샘플 차분 속도·JTS·grip·교시 궤적·제공된 명목 모델/하중만 쓴다.
실제 작업대 힘·링크각·weld 상태는 사후 물리 분석과 선형화에만 썼다.
아래 식은 구현의 자체 유도다. 강체 연속시간의 수동성/안정성 정리를
탄성 관절, 센서 지연, 명령 지연, 1 ms 구현의 보장으로 옮기지 않는다.

## P0와 C1: 지지 렌치에서 탄성 모터 기준까지

첫 접촉부터 release까지 50–57 ms가 있어도, 순간 지지력 crossing 뒤
반발과 접촉 손실이 있어 지속 지지가 성립하지 않았다. 관측된 동적
힘-침투 데이터를 정적 강성으로 회귀하면 음의 강성이 나오는 경우가 있어
그 회귀는 설계에 쓰지 않았다. 단순 위치 하강 probe의 최대 충격 증가도
탈락 근거로 보존한다. 경계는 P0에서 측정한 0.689439/2.678555 mm와
접촉 시각에서 정했으며, 이것이 필요한 G4 변위를 식별했다는 뜻은 아니다.

접촉점을 박스 바닥 중심으로 두면 world Jacobian `J_w`는 병진·회전
순서다. 같은 원점에서 body 좌표로 바꾸는 회전 블록 `Q`에 대해
`J_b=Q J_w`, `W_b=Q W_w`이므로 `J_b.T W_b=J_w.T W_w`다.
원점 이동은 이미 접촉점 Jacobian에 포함되어 중복 이동하지 않는다.
정상력의 작용선은 알려진 COM의 중력선과 바닥 면의 교점으로 만들고,
COP를 알려진 직사각형 바닥 범위로 제한한다.

계획한 지지력 `F_des=fraction*m*g*h(t)`의 관절 효과를 한 번만 뺀다.

```
tau_link = (1-h)*tau_R1 + h*tau_full - J_b.T W_b(F_des)
theta_des = q_ref + K_elastic^-1 tau_link
```

강체 축의 `K_elastic^-1`은 0이다. R1의 필터된 토크 미분, 모터 기준
속도/가속도 제한, 모터 마찰 보상과 A Drive/B motor feedback은 유지한다.
추정 피드백 비교는 실제 힘이나 자체 투영한 table wrench가 아니라
허용 신호로 계산한 raw observer wrench를 사용한다. 경보 중에는 새
힘 피드백을 동결한다. 지지력 평균만이 아니라 마지막20ms의 최소값,
R1 대비 증가량, 놓기 peak, 원래 TCP 비용을 함께 판정한다.

## C2: normal coordinate와 power-dual을 함께 유도

A는 법선 외부 admittance의 임계감쇠 2차 상태를 1 ms exact ZOH로
진행시킨 뒤 측정 기반 변위/속도/가속도 경계로 투영한다. 경계 투영이
걸린 구간은 원래 선형 ZOH 시스템과 같다는 주장을 하지 않는다.
순수 world 법선 병진이고 목표 회전이 같으면 log와 Euclidean 변형은
수학적으로 동일하므로, 그 경우 리그룹의 독립 이득을 주장할 수 없다.

B에서는 `E=T_des^-1 T_actual`, `xi=log(E)`,
`n=[R_des.T e_z,0]`, `e=-n.T xi`를 쓴다. 여기의 spatial 순서는
병진·회전이다. `J_r(xi)=dexp(-xi)`라 하면

```
xi_dot = J_r^-1 (V_actual - Ad(E)^-1 V_des)
n_dot  = [-omega_des x n_linear, 0]
e_dot  = -n.T xi_dot - n_dot.T xi
```

실제 body perturbation `delta`에 대해 `d xi=J_r^-1 delta`이므로
`-d e/d delta=J_r^-T n`이다. 따라서 normal coordinate의 spring/damper와
목표 힘은 **같은 covector**로 관절에 사상해야 한다. 원래 반대 군 오차
정의를 그대로 쓴 채 Adjoint의 dual을 생략하지 않는다. 비항등 자세에서
virtual work와 양쪽 프레임이 움직이는 오차 미분을 유한차분으로 검증했다.
곡률 때문에 생기는 작은 회전 dual 항은 Euclidean과의 차이이자 대가다.
임피던스는 링크 목표 토크를 만든 뒤 위 탄성 모터 기준으로 올린다.

## C3/C4

[탄성 NRIC·DOB·P-APD·P-IDC 유도](R2_ROBUST_DERIVATION.md)를 참조한다.
NRIC의 명목 플랜트에는 **보상하지 않은 독립 명령 queue**를 넣고,
실제 출력만 명목 외부 명령+오차 보상으로 만든다. 명목 플랜트를 이미
보상한 실제 명령으로 진행시키는 구현은 후보 평가 전에 제외했다.
A의 DOB는 실제 Drive의 내부 상태를 읽지 않고 최초 queue부터 공개
Drive 식을 자체 재생한다. C4의 `C(q,v_actual)u_ref`는 아래 pRNE action을
쓰며 `bias(q,u_ref)`로 대체하지 않는다.
질량 행렬로 스케일한 PD 게인, 시간에 따른 load-fraction 보간, clip,
anti-windup, 링크에서 모터로의 lift는 원래 강체 법칙의 가정을 바꾼다.
이름에 P가 있어도 실제 탄성·샘플 플랜트의 수동성이 증명된 것은 아니다.

## C5: metric-compatible pRNE와 transpose recursion

여기는 angular-first `[omega,linear]` 순서다. body motion transform을
`X_i`, joint subspace를 `S_i`, body inertia를 `I_i`로 둔다. 실제 속도 `v`와
설계 속도 `u`를 분리한다.

```
V_i = X_i V_parent + S_i v_i
W_i = X_i W_parent + S_i u_i
```

설계 `u`를 고정하고 실제 `qdot=v`로 미분하면
`A_i=X_i A_parent-ad(S_i v_i) X_i W_parent`다. transport 항에 전체
`V_i`를 잘못 넣지 않는다. `ad*_V=-ad_V.T`에 대해
`B(V)=ad*_V I+I ad_V`는 skew다. `P_i=A_i+ad(V_i)W_i`로 묶으면

```
P_i = X_i P_parent + ad(X_i V_parent) S_i u_i
F_i = I_i P_i + ad*_Vi I_i W_i
```

가 된다. 일반 wrench backward recursion으로 `C(q,v)u`를 얻는다.
각 링크에서 `W.T B(V)W=0`이므로, 고정된 임의의 `u`에 대해
`u.T C u = (1/2) u.T Mdot u`다. 모든 `u`에 성립하므로
`Mdot=C+C.T`가 따른다. `u=v`이면 중력 없는 일반 bias와 일치한다.
이는 metric-compatible C의 한 convention이며 유일한 Christoffel
행렬을 계산했다고 주장하지 않는다. 고정 tool/payload inertia는 부모
링크에 spatial 합성하고 모터 관성/탄성은 이 링크 RNE에 섞지 않는다.

출력 dual `y`를 힘 재귀의 adjoint로 전달하고, 부분 가속도/설계 twist의
adjoint를 역방향 누적하면 `C.T y`도 O(n)에 구한다. 런타임에서 여섯
기저나 유한차분으로 C 전체를 만들지 않는다. basis와 Mdot 유한차분은
독립 검사 oracle에만 썼다. `Mdot` 항의 대체는

```
beta_old = bias(q,v) - Mdot_backward v
beta_pRNE = gravity(q) - C(q,v).T v
```

다. 관측기 gain50과 grip 모델 전환 때 integral 보존 정책은 유지했다.
C5 detector는 별도 추정기를 가지므로 제어기의 R1 load core는 바꾸지
않는다. pRNE 단독·R1 ridge 단독·pRNE+새 normal-only ridge를 비교한다.
학습 입력 파일은 `t,x,y,active`만 담고, 실제 힘/qpos가 든 사후 로그와
물리적으로 분리한다. 공식 조건과 충돌 결과는 fit/threshold에 쓰지 않는다.
추가로 같은 정상 관측을 R1 관측기에 재생해 **matched ridge**를 학습했다.
따라서 pRNE+새 ridge와 matched ridge의 비교가 pRNE의 기여 분리다.
pRNE 단독 대 R1 비교는 기존 ridge도 끄므로, 검출 손실을 pRNE 계산만의
원인으로 돌리지 않는다. 동결된 새 holdout의0.75kg은 C5 학습 질량 하나와
겹친다. COM·시각·힘·방향 조건은 새 시드로 분리하지만, 모든 질량을
처음 본 질량이라고 표현하지 않는다. 최종 유지한 R1의 과거 학습 질량과는
이0.75kg이 겹치지 않는다.

## C6와 안정성 해석

교시 접근3.6s부터 도착3.809s까지 7차/S-curve joint 보간과 SO(3)+R³
분리 보간을 비교한다. 후자는 위치와 회전을 별도로 보간하고 full 6D
IK/Jacobian/Jdot로 q/qd/qdd를 일관되게 만든다. 그립은3.9s까지 유지한다.
S-curve는 bounded piecewise-constant jerk의 C2 궤적이고, 7차처럼 jerk까지
연속이라고 쓰지 않는다. 조기 도착의 원래 궤적 대비 TCP 비용을 숨기지 않는다.

선형 분석은 실제 탄성 기구의 고정 접촉 모드 FD와 exact ZOH, 명령/JTS
한 주기 지연, Drive PI, encoder 샘플 차분, 출력에 영향을 주는 제어 상태를
포함한다. Coulomb는 실제 runner처럼 샘플 후 ZOH, 모델 native viscosity는
연속 항으로 취급한다. 구조화 command-sensitivity 배율은 고전 SISO gain
margin이나 일반 MIMO 강건 margin이 아니다. 강체 교재 정리·유한 국소
고유값·실제 에피소드 성공은 서로 다른 종류의 증거다.

## 서지·절만의 교재 참고

- J. Park, Y. Choi, W. K. Chung, *Lie Group Robotics*, 원고(2024-03-02),
  §§2.2,6.1 및 계산 동역학 목차§8.3. 해당 원고의 pRNE 본문은 확인 범위에 없다.
- Y. Choi, *Lie Group Robot Control*, 2024-2학기, §§9.1,9.3,
  10.1.3,10.1.6,10.4,11.4,11.5. APD의 탄성/수동성 지향 자체 변형을
  P-APD라 부르며, 자료에 동일 이름의 정리가 있다고 주장하지 않는다.

원자료·쪽 이미지·교재 문장/그림은 포함하지 않는다. 직접 확인한 논문과
원방법 대비 변경점은 [문헌 조사](LITERATURE.md)에 별도로 구분한다.
