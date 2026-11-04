# V2 방법

## 등록된 최소 변경

### 원래 learner 가설

Learner는 gate 수정과 별개의 가설이다. 충돌 없는 정상 동작에서만 77개 basis `Phi(q_ref, qd_ref, qdd_ref, jerk, m, c)`를 만들고, v1 관측기의 허용된 수평 wrench `w_v1,xy`를 target으로 표준화·clip[-8,8] 후 ridge `beta=(Phi^T Phi+Lambda)^-1 Phi^T w_v1,xy`를 맞춘다. `Lambda=diag(0,0.02,...,0.02)`로 상수 intercept는 벌점을 주지 않는다. 입력은 현재/과거 공개 reference, 질량과 COM뿐이며 plant/log/oracle, case index와 미래값은 없다. train 조건으로 mean/scale/weight를 구하고 분리된 no-collision validation으로 threshold를 한 번 동결했다. runtime 점수는 `e_k=w_v1,xy-Phi_k beta`, `s_k=(1-0.1)s_(k-1)+0.1||e_k||_2`이며 4-cycle persistence와 8-cycle release를 쓴다. 동기는 정상 운반의 질량/COM/운동 의존 baseline을 제거해 약한 수평 외력을 드러내는 것이다.

### L3 gate 구현 복구

L3는 위 learner를 다시 학습한 것이 아니다. packet의 `q_prev`와 공개 nominal 점 Jacobian으로 상자 바닥 중심 높이 `z_b(q_prev)`를 계산한다. 게이트는 모든 질량에서

`active = grip AND min_i |z_b(q_prev)-h_table,i| > h_band`

이며, 기존 v1 경보와 논리 OR 한다. 원래 모델의 ridge 가중치, 표준화, `EMA alpha=0.1`, 4-cycle persistence, 8-cycle release, 높이 band와 임계값은 그대로다. 이 수정은 새 수치 검출 규칙이 아니라 ORIGINAL 기하 높이 계약의 구현 복구다.

공식 데이터는 L1 오류 디버깅에 이미 노출되었으므로 효능의 독립 검정으로 사용하지 않는다. L3 뒤 재학습·재보정이나 holdout 이후 tuning은 없다.

### Frame, power와 유한 기하 감사

점 `p`에서 `q`로 wrench 기준점을 옮기면 `f_q=f_p`, `tau_q=tau_p+(p-q)×f_p`이고, twist는 `v_p=v_q+omega×(p-q)`이므로 `f_p·v_p+tau_p·omega=f_q·v_q+tau_q·omega`다. 또한 `tau_joint·qdot = wrench·(J qdot)`인 virtual-power 항등을 coadjoint shift와 Jacobian 수치 검사로 확인한다. 이는 frame/sign 오류를 찾는 감사이지 관측성을 만들지 않는다. nonlinear bottom-centre 위치와 1차 Jacobian shift의 유한 오차도 등록 범위에서만 검사했으며, 전역 SE(3) 정확성·접촉모델 타당성·실기 안전성을 증명하지 않는다. 기하 원리는 `docs/LITERATURE.md`에 열거한 Park·Choi·Chung (2024) §§2.2, 6.1, 7.2와 Choi (2024) §§11.1, 11.4, 11.5의 frame/power 범위만 참고하며 원자료·그림을 재사용하지 않는다.

## 제어 후보의 자체 변형과 탄성·이산화 한계

C는 full SE(3) impedance나 논문의 ILC/QP를 재현하지 않는다. 공개 nominal endpoint의 6D 점 Jacobian으로 `J a=[0,0,1,0,0,0]^T`를 풀고, 한 개의 scalar 보정에 대해 `q_ref+=a*x`, `qd_ref+=a*v`, `qdd_ref+=a*accel`을 일관되게 더했다. 놓기 전 0.12 s의 5차 blend로 목표 지지를 0.85mg까지 올리고 `v_raw=(Fz_hat-Fz_target)/12000`을 10 ms filter 뒤 제한했다(변위1.5 mm,속도0.01 m/s,가속도0.5 m/s²). 기존 탄성 보상 inner loop,1 ms sampling과고정grip시각을 유지한다. `Fz_hat`은 허용된 JTS 관측기 추정값이며 실제 작업대 힘은 사후 G4 계산에만 썼다.

이 설계는 지연/잡음이 있는 추정 렌치를 쓰는 국소 외부 admittance다. 강체·연속시간 교재 안정성 결과를 탄성 관절·1 ms 구현의 보장으로 전용하지 않는다. 경보 시 추가 하강 억제가 가속도 smoothness보다 우선하고, release 뒤 고정0.10 s C2 복귀는 전체 독립 bound box를 모두 덮지 못한다. 이 예외를 숨겨 안전하게 bounded라고 단정하지 않으며, 실제 검증의 G4 실패와 함께 후보를 탈락시켰다.

## 후보 처분

- C bounded-normal admittance: 각 구조 G4 0/4. 고정 0.10 s 복귀와 1.5 mm/0.01 m/s 경계는 등록된 **전체 state box**를 모두 덮을 수 없어 무조정 탈락했다. 실제 시험 trajectory가 모두 경계를 위반했다는 뜻은 아니다.
- L1: development 검출 이득 뒤 공식 무충돌 경보 회귀로 탈락.
- L2: train bounding-box 탐색 결과로 보존하되 critic이 선호한 최종 규칙이 아님.
- L3: 정확한 기하 높이 gate bugfix. weight/threshold 변경 없음. 최종 선택은 `L3`이다.

## 평가와 한계

공식 표는 변경 없는 `evaluate.py` JSON만 사용한다. D4, 봉인 holdout, timing, perturbation은 명시적으로 별도 supplemental 데이터다. 모든 무충돌 false alarm rise, 충돌 중 추가 false alarm, miss, prior-alarm/indistinguishable masked 사례를 버리지 않고 센다. G5는 모든 source의 **최종 v2 A/B** 관측 wall 최대가 각각 1 ms 미만일 때만 통과한다. baseline/v1 및 전체 버전 최악은 비교값으로 별도 보존한다. 초과 시 outlier와 같은 cycle의 wall--thread CPU 차이를 보존하고 context switch 가능성은 추론으로만 한정한다.

같은 순간의 가능한 table-normal wrench는 현재 입력만으로 외력과 지지를 식별할 수 없다. 실제 하드웨어 안전성, 확률적 일반화, hard real-time, collision stop은 입증하지 않는다.

## 문헌 사용 범위

문헌 표기와 채택 범위는 `docs/LITERATURE.md`의 bibliography/sections만 따른다. Albu-Schaeffer et al. (2007)은 유연관절 제어 원리, van Steen et al. (2024)은 bounded reference 동기, Chi et al. (2022)은 residual 분해의 원리 참고일 뿐 원방법 재현이나 안전성 증명이 아니다.
