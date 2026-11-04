# R2 C3/C4 탄성 관절 변형

이 문서는 실험 후보의 자체 유도와 구현 경계를 기록한다. 연속시간 강체
결과를 이 탄성·이산 플랜트의 안정성 보장으로 사용하지 않는다. 실제 채택은
탄성 관절, 1 ms ZOH, 한 주기 명령/JTS 지연을 포함한 별도 선형화와 측정으로
판정한다.

## C3: NRIC

내부 명목 상태는

\[
z=[\theta_{1:3},q_{1:6}],\qquad
M_z(z)\ddot z + h_z(z,\dot z)=G\tau_m
\]

로 둔다. J1--J3 모터 관성은 링크와 분리하고, J4--J6 모터 관성은 강체
링크 관성에 더한다. 스프링은
`K(theta-q)+D(theta_dot-q_dot)`이고, 마찰은 공개된 평활 Coulomb/점성
모델이다. 1 ms 갱신은 forward Euler가 아니라 implicit midpoint의 최대
8회 고정점 해를 사용한다. 순수 탄성 시험의 이산식 잔차는 `1e-9`보다
작아야 하며, 후보 선형화에서도 step/convergence를 별도로 기록한다.

내부 명목 플랜트에는 실제 보상 토크를 넣지 않는다. 명목 outer controller는
명목 모터 위치·속도와 명목 JTS로 닫고, 한 주기 전의 **무보상 명목 명령**만
명목 플랜트에 적용한다. 실제 출력은 이 명목 outer 출력에 아래 정규화 보상을
더한다. 서보 방식의 명목 Drive도 명목 encoder와 독립 `v/I`를 사용한다.
따라서 실제 feedback controller를 그대로 복제해 보상만 더한 구조가 아니다.
하중 feedforward의 fraction은 허용 JTS/encoder로 만든 R1 shared packet을
명목·실제 outer가 같이 사용한다. 이는 내부 상태 feedback 공유가 아니라
알려진 payload에 대한 동일한 외생 schedule이며 결과에서 명시한다.

실제 허용 신호로 복원한 링크 추정 `q,v`와 명목 상태의 차이로

\[
s_i={\dot e_i\over \omega\sigma_i}+{e_i\over\sigma_i}
    +{\int e_i\,dt\over\sigma_i/\omega},\qquad
\Delta\tau_i=r_i\tanh((1+\gamma^{-2})s_i)
\]

를 만든다. `sigma`는 P0 무충돌 encoder-reference RMS, `r`은 P0에서 남은
토크 여유의 25%다. 따라서 `gamma`는 무차원이고, 단위가 없는
`1/gamma^2`를 토크에 직접 더하지 않는다. 접촉 전환·경보에서는 적분을
동결/감쇠하고 명목 상태를 grip 모델 전환 때 실제 추정으로 재초기화한다.

## C3: motor DOB

모터측 식

\[
B\dot v_m=u-\tau_s-f(v_m)+d
\]

에 대해 `p=L B v_m`, `d_hat=z+p`를 두고

\[
\dot z=-L\{z+p+u-\tau_s-f(v_m)\}
\]

를 한 샘플 동안 정확히 적분한다. 사용되는 `u`는 현재 출력이 아니라 한
주기 앞서 실제로 적용된 자체 명령의 추정이다. 서보 방식은 공개 Drive 식과
자체 보관한 이전 command/filter/integral만 재생하며 Drive의 비공개 상태나
센서 전 실제 토크를 읽지 않는다. 보상은 `-d_hat`이고 P0 토크 여유로
성분별 제한한다. 접촉 구간에서는 `z=-p`로 추종해 해제 순간 stale 보상이
튀어나오지 않게 한다.

접촉 gate의 픽업 전후 80 ms와 공개 궤적의 놓기 접근 구간은 충돌 결과에서
맞춘 수치가 아니다. 공개 grip 시각과 P0에서 접촉이 반복적으로 사라진 구간을
보수적으로 감싼 설계 구간이며, 실제 접촉 보장이나 측정 최적값으로 해석하지
않는다.

## C4: P-APD와 P-IDC

두 후보 모두 먼저 링크측 목표 토크를 만들고

\[
\theta_d=q_d+K_e^{-1}\tau_{link,d}
\]

로 탄성 모터 기준에 올린 뒤 R1의 motor-side tracker/JTS feedback 구조를
유지한다. rigid axis에서는 `K_e^{-1}=0`이다.

P-APD는 `Phi=omega I`, `D=2 zeta omega M`, `K=D Phi`로

\[
\tau_d=M(\ddot q_d+\Phi\dot e)
 +C(q,v)(\dot q_d+\Phi e)+g+D\dot e+Ke
\]

를 계산한다. P-IDC는

\[
\dot q_r=\dot q_d+K_Ve+K_P\int e,\quad
\ddot q_r=\ddot q_d+K_V\dot e+K_Pe
\]

와 `tau_d=M qdd_r+C(q,v) qd_r+g+K_r(qd_r-v)`를 쓴다. 핵심은
`C(q,v_actual)u_ref`를 pRNE action으로 계산한다는 점이며,
`bias(q,u_ref)`로 바꾸지 않는다. P-IDC 적분은 P0 오차 규모로 제한하고
grip 전환·접촉·경보에서 동결/감쇠한다.

빈 도구/하중 모델의 토크를 R1 load fraction으로 보간한다. 후보 링크 토크는
R1 inverse-dynamics 토크 주변의 P0 reserve 안으로 제한한다. 이 제한과
anti-windup은 안정성 증명이 아니며, 충돌 신호를 보고 조정하지 않는다.
