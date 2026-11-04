# 문헌 조사와 후보 근거

열람일 2026-11-04. 아래 20편은 조사 중 원문·저자 공개본·출판사 초록·arXiv 페이지를 **직접 열어 확인**했다. 초록만 확인한 경우를 원문 전체 재현이라고 쓰지 않는다. 저자 코드는 가져오지 않았으며, 논문 실험 성능과 이 과제 효과는 별개다. 고전과 2022–2026 문헌을 함께 조사했다. 제어/관측 12편, 학습 8편이다. 원문 PDF 직접 열람 12편은 C1/C2/C4/C5/C6/C10/C11/C12 및 L2/L3/L4/L8이다(제어 8·학습 4). 나머지는 직접 열린 출판사/기관/arXiv 페이지에서 확인한 범위만 사용한다. C8/C9 PDF는 용량 제한으로 열리지 않아 PDF 열람 수에 넣지 않았다.

| ID | 제목·저자·발표처·연도·직접 확인 링크 | 확인 사실 | 과제 적용 추론·예상 효과·위험 |
|---|---|---|---|
| C1 | N. Hogan, *Impedance Control: An Approach to Manipulation: Part I—Theory*, ASME JDSMC107,1985,DOI10.1115/1.3140702. [MIT 원문](https://bpb-us-e1.wpmucdn.com/sites.mit.edu/dist/5/1384/files/2025/02/1985-impedance-control-an-approach-to-manipulation-part-I-theory.pdf) | 환경과 결합한 동적 상호작용 및 impedance/admittance를 구분 | 법선만 순응하는 작은 외부루프 후보. 센서지연·탄성·이산화 보장은 별도 |
| C2 | C.Ott,A.Albu-Schäffer,A.Kugi,S.Stramigioli,G.Hirzinger, *A Passivity Based Cartesian Impedance Controller for Flexible Joint Robots—Part I: Torque Feedback and Gravity Compensation*, ICRA2004,DOI10.1109/ROBOT.2004.1307462. [저자 원문](https://www.robotic.de/fileadmin/robotic/ott/papers/icra2004.pdf) | 탄성관절 torque feedback과 gravity compensation | v1에 이미 관련 구조가 있어 새 기여로 중복 주장하지 않음 |
| C3 | A.Albu-Schäffer,C.Ott,G.Hirzinger, *A Unified Passivity-based Control Framework for Position, Torque and Impedance Control of Flexible Joint Robots*, IJRR26,2007,DOI10.1177/0278364907073776. [출판사](https://journals.sagepub.com/doi/10.1177/0278364907073776) | inner torque feedback을 inertia shaping으로 분석 | outer compliance의 참고. 이 플랜트의 수동성/안정성 보장으로 전용 불가 |
| C4 | A.De Luca,R.Mattone, *Sensorless Robot Collision Detection and Hybrid Force/Motion Control*, ICRA2005,DOI10.1109/ROBOT.2005.1570247. [원문](https://www.diag.uniroma1.it/deluca/pHRI_elective/ICRA05_ADL_Mattone_SensorlessCollisionDetection.pdf) | generalized momentum으로 미지 외력의 filtered torque 추정 | 허용 JTS 잔차 관측의 기반. 현재 가속도 오차/탄성·센서잡음은 남음 |
| C5 | S.Haddadin,A.De Luca,A.Albu-Schäffer, *Robot Collisions: A Survey on Detection, Isolation, and Identification*, IEEE T-RO33,2017,DOI10.1109/TRO.2017.2723903. [저자 원문](https://iris.uniroma1.it/retrieve/e383532a-2aa9-15e8-e053-a505fe0a3de9/Haddadin_Postprint_Robot-Collisions_2017.pdf) | 검출·위치추정·식별·분류·반응을 분리,탄성도 검토 | 접촉 원천 식별을 단순 검출 성공과 혼동하지 않음 |
| C6 | J.J.van Steen,N.van de Wouw,A.Saccon, *Robot Control for Simultaneous Impact Tasks via Quadratic Programming-based Reference Spreading*, ACC2022,DOI10.23919/ACC53348.2022.9867812(최초 preprint2021),[원문 PDF](https://arxiv.org/pdf/2111.05211) | 충돌 전후 겹치는 reference와 interim mode | 짧은 전환 reference 발상만 참고; QP/impact map은 미채택 |
| C7 | Y.Wang,N.Dehio,A.Tanguy,A.Kheddar, *Impact-aware task-space quadratic-programming control*, IJRR42,2023,DOI10.1177/02783649231198558. [출판사](https://journals.sagepub.com/doi/abs/10.1177/02783649231198558) | impact bounds와 post-impact feasible set,Q P실험 | 물리적 제한 필요성 참고. full QP는 비용·접촉모델 요구로 제외 |
| C8 | J.van Steen,G.van den Brandt,N.van de Wouw,J.Kober,A.Saccon, *Quadratic Programming-based Reference Spreading Control for Dual-Arm Robotic Manipulation with Planned Simultaneous Impacts*, preprint2023,T-RO acceptance표시2024,[arXiv2305.08643](https://arxiv.org/abs/2305.08643) | 순차접촉 interim에서 velocity error peak 억제 | 단일팔·고정 grip와 다름,그대로 재현했다고 하지 않음 |
| C9 | J.van Steen,N.van de Wouw,A.Saccon, *Impact-Aware Control using Time-Invariant Reference Spreading*, preprint2024,[arXiv2411.09870](https://arxiv.org/abs/2411.09870) | 전후 vector field와 partial-contact interim을 사용 | bounded reference의 동기. contact mode/impact model 없는 본 변형은 훨씬 제한적 |
| C10 | L.Yan,T.Stouraitis,J.Moura,W.Xu,M.Gienger,S.Vijayakumar, *Impact-Aware Bimanual Catching of Large-Momentum Objects*, IEEE T-RO40,2024,DOI10.1109/TRO.2024.3381551. [저자 원문](https://www.research.ed.ac.uk/files/419545467/Impact-Aware_YAN_DOA19022024_AFV_CC_BY.pdf) | 접촉위치·운동·강성·힘을 최적화한 양팔 catching | 외부인지/양팔/최적화가 필요해 이 과제 직접이식 제외 |
| C11 | B.Zhou,Y.Sun,W.Liu,R.Jiao,F.Fang,S.Li, *Active Admittance Control with Iterative Learning for General-Purpose Contact-Rich Manipulation*, preprint2024,[원문 PDF/arXiv2403.16560](https://arxiv.org/pdf/2403.16560) | low-level admittance와 high-level ILC | 고정 bounded admittance만 실험; holdout 중 학습은 금지 |
| C12 | K.Okabe,H.Honda, *Sensorless Collision-Detection Method Exploiting Static Redundancy for Robustness to Manipulation Forces*, J.Robotics&Mechatronics38,2026,DOI10.20965/jrm.2026.p0589. [원문](https://www.jstage.jst.go.jp/article/jrobomech/38/2/38_589/_pdf/-char/en) | Jacobian null-space의 static redundancy 활용 | square 6×6 full-rank wrench 모델에는 null-space 이득이 보장되지 않아 비선정 |
| L1 | K.M.Park,Y.Park,S.Yoon,F.C.Park, *Collision Detection for Robot Manipulators Using Unsupervised Anomaly Detection Algorithms*, IEEE/ASME T-Mech27,2022,DOI10.1109/TMECH.2021.3119057. [저자기관 초록](https://snu.elsevierpure.com/en/publications/collision-detection-for-robot-manipulators-using-unsupervised-ano/) | 충돌 없는 주행만 쓰는 one-class SVM/AE | 정상 잔차 ridge+score로 자체 단순화. SVM/AE의 논문 재현은 아님 |
| L2 | D.Kim,D.Lim,J.Park, *Transferable Collision Detection Learning for Cooperative Manipulator Using Versatile Modularized Neural Network*, IEEE T-RO38,2022,DOI10.1109/TRO.2021.3129630. [저자 원문](https://daegyulim.github.io/assets/TRO2021.pdf) | modular NN과 free-motion target adaptation | 정상-only 학습 동기. 사전로봇데이터/NN 이식은 재현비용 때문에 제외 |
| L3 | C.Chi,B.Burchfiel,E.Cousineau,S.Feng,S.Song, *Iterative Residual Policy for Goal-Conditioned Dynamic Manipulation of Deformable Objects*, RSS2022,[원문](https://www.roboticsproceedings.org/rss18/p016.pdf),arXiv2203.00663 | delta-action의 trajectory 효과를 반복 학습 | 잔차분해 발상 참고; 시각·sampling policy는 현재 입력/시간 범위 밖 |
| L4 | G.Tiboni,A.Protopapa,T.Tommasi,G.Averta, *Domain Randomization for Robust, Affordable and Effective Closed-loop Control of Soft Robots*, IROS2023,DOI10.1109/IROS55552.2023.10342537,[원문 PDF/arXiv2303.04136](https://arxiv.org/pdf/2303.04136) | 물성 분포를 바꾸는 soft-robot RL | 다양한 조건 필요성 참고. 본 protocol은 물성섭동을 학습에 안 쓰고 사후 분석으로만 검사 |
| L5 | Y.De Santis,M.Pierallini,F.Angelini,M.Garabini, *Reinforced Iterative Learning Control for elastic joint arms: Guaranteed cross-trajectory generalization under model uncertainties*, preprint2026,[arXiv2610.28329](https://arxiv.org/abs/2610.28329) | elastic arm에서 ILC/RL 결합을 제안 | preprint의 가정·보장은 본 플랜트에 전용 불가; 동결 평가 중 update 제외 |
| L6 | J.Kübel,H.Krauss,J.Li,M.Zhao, *Energy-based Regularization for Learning Residual Dynamics in Neural MPC for Omnidirectional Aerial Robots*, preprint2026,[arXiv2604.14678](https://arxiv.org/abs/2604.14678) | analytical model+NN residual에 energy loss | 잔차제한 동기, aerial NN-MPC/내부상태 target은 미채택 |
| L7 | M.Al-Ghaili,Y.Qasim,H.Almubarak, *Differentiable Iterative Learning Control for Black-Box Position-Commanded Manipulators*, preprint2026,[arXiv2611.01220](https://arxiv.org/abs/2611.01220) | bilevel optimization 기반 constrained ILC | reference-only 학습 사례이나 현재 실행 중 최적화/holdout 적응은 제외 |
| L8 | T.Johannink,S.Bahl,A.Nair,J.Luo,A.Kumar,M.Loskyll,J.A.Ojea,E.Solowjow,S.Levine, *Residual Reinforcement Learning for Robot Control*, ICRA2019,[원문 PDF/arXiv1812.03201](https://arxiv.org/pdf/1812.03201) | 수작업 controller에 RL output을 가산 | 기존 채널을 남기는 작은 확장 발상. RL exploration/policy는 미채택 |

## 미달과 후보 연결

| 항목 | 우선 후보 | 기대와 위험 | 선정 이유 |
|---|---|---|---|
| U1/U2 지연·작은힘 | L1/L2 정상 잔차 ridge+수평 대칭 EMA | 정상 모델오차 제거로 감도 가능;필터 지연·미지조건 오감지 위험 | numpy·허용입력·자기지도 타깃·CPU학습 가능 |
| U3/U6 검출·오감지 | 기존 채널 OR learned 채널,무충돌 calibration | 기존검출 보존 가능,추가오감지는 늘 수 있어 검사필수 | 원인별 ablation·회귀 방어 가능 |
| U4 이양 | C1/C3/C9/C11 기반 bounded normal admittance | 추정support만 사용,힘센서지연·windup·TCP대가 | 1.5mm bound와정해진시간;실패하면제어v1 유지 |
| U5 계산 | 작은 matrix-vector/스칼라 상태 | Python/OS outlier 가능 | full QP·MPC·대형NN보다 검증단순 |
| U7/U8 강건성 | 제한된변위·안티윈드업,독립격자/섭동 | 마찰/강성모델오류 미해결 가능 | 보장대신실측범위만보고 |
| U9 동일법선렌치 | power-duality/frame audit,비식별 반례 | 좌표변환은정보를새로만들지 않음 | 추가센서없이원천식별을주장하지않음 |

NN이 NumPy로 불가능해서가 아니라, 구현·감사·학습 재현 비용 대비 최소가설 검증에 ridge를 선택했다. 논문 기반은 **원리 참고와 자체 변형**을 뜻하며 원방법 재현·실기 안전성 증명이 아니다. 출판사 원문을 열지 못한 고전과 DBLP 서지만 확인한 문헌은 위20편 및 원문 수용 개수에 포함하지 않았다.

## 기하학 참고 (위 논문 수에 포함하지 않음)

- J.Park,Y.Choi,W.K.Chung, *Lie Group Robotics: Mathematical and Computational Approach to Robot Kinematics, Dynamics, and Control*, 원고(2024-03-02), §§2.2,6.1,7.2.
- Y.Choi, *Lie Group Robot Control*, 2024-2학기 강의자료, §§11.1,11.4,11.5.

프레임·작용점이 다른 렌치를 직접 빼지 않으며 힘-속도의 power duality를 단위검사한다. 법선 병진만의 작은 보정은 일반 SE(3)제어 전체와 다르다. 공개 모델의 taught endpoint Jacobian으로 접선/회전을 유지하는 방향을 만들고, 그 고정 방향에 스칼라 변위·속도·가속도를 일관되게 더하는 **국소 근사**를 비교한다. 강체 연속시간 안정성 결과를 탄성·JTS지연·1ms이산제어의 보장으로 사용하지 않는다. 원자료·노트·그림·문장은 배포하지 않는다.
