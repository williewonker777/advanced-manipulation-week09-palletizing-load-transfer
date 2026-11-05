# 새 문헌과 적용 가설

열람일 2026-11-05. 이전 조사와 겹치지 않는 직접 열린 출처 10편이다. 원문 7편, 기관 초록·서지 2편, 출판사 서지 1편을 구분한다. 원문을 열지 않은 문헌의 식·실험을 재현했다고 주장하지 않는다. 저자 코드는 가져오지 않았다.

| 문헌 | 직접 확인한 원문·출처와 범위 | 확인 사실 | 이 과제의 적용 추론·제한 |
|---|---|---|---|
| P1 P. Sundvall, P. Jensfelt, *Fault Detection for Mobile Robots Using Redundant Positioning Systems*, ICRA 2006 | [저자 PDF](https://www.csc.kth.se/~patric/publications/icra06-paul.pdf), 원문 | 혁신 에너지의 reset CUSUM을 설명한다 | 팔의 잔차에 고정 정상 보정 누적기를 적용할 근거. 이동로봇 센서·오류모델은 가져오지 않음 |
| P2 F. Franzel, T. Eiband, D. Lee, *Detection of Collaboration and Collision Events during Contact Task Execution*, Humanoids 2020(출판2021), DOI10.1109/HUMANOIDS47582.2021.9555677 | [기관 초록·서지](https://portal.fis.tum.de/en/publications/detection-of-collaboration-and-collision-events-during-contact-ta-2/) | 작업 지식과 접촉 유형 분류를 나눈다 | 라벨/SVM·추가 힘센서 의존 때문에 원방법은 제외 |
| P3 R. J. Kirschner, J. Jantalia, N. Mansfeld, S. Abdolshah, S. Haddadin, *CSM: Contact Sensitivity Maps for Benchmarking Robot Collision Handling Systems*, ICRA2021, DOI10.1109/ICRA48506.2021.9561528 | [저자 PDF](https://iliad-project.eu/wp-content/uploads/2021/06/CSM-Contact-sensitivity.pdf), 원문 | 물체·속도·자세 조건별 감도 평가를 다룬다 | 질량·구간·힘·방향을 나눠 보고한다. 런타임 검출 알고리즘의 재현은 아님 |
| P4 D. Zurlo, T. Heitmann, M. Morlock, A. De Luca, *Collision Detection and Contact Point Estimation Using Virtual Joint Torque Sensing Applied to a Cobot*, ICRA2023, DOI10.1109/ICRA48891.2023.10160661 | [저자 PDF](https://www.diag.uniroma1.it/~deluca/pHRI_elective/ICRA23_Zurlo_etal_CollisionDetection_and_ContactPointEstimation.pdf), 원문 | 에너지·운동량 잔차를 함께 쓰고 저속 민감도 한계를 설명한다 | 교차 채널·구간 평가의 참고. 전류 관측기와 안정성 결과를 이식하지 않음 |
| P5 C. Zhang, C. Mu, Y. Wang, J. Li, Z. Liu, *Collision Detection for Six-DOF Serial Robots Force/Position Hybrid Control Based on Continuous Friction Model*, Measurement and Control56(3–4),2023,571–582, DOI10.1177/00202940221091575 | [출판사 PDF](https://journals.sagepub.com/doi/pdf/10.1177/00202940221091575), 원문 | 마찰·힘/위치 방향에 따른 검출의 차이를 다룬다 | 문헌의 장치 임계값은 쓰지 않는다. 접촉 전환의 정상 변동을 별도 확인 |
| P6 I. Gibbs, E. J. Candès, *Adaptive Conformal Inference Under Distribution Shift*, NeurIPS34,2021 | [원문 PDF](https://papers.nips.cc/paper/2021/file/0d441de75945e5acbc865406fc9a2559-Paper.pdf) | 온라인 분포 변화에 따라 보정량을 갱신하는 ACI를 제안한다 | 온라인 적응은 제외. 정상 적합/보정 분할 원리만 참고하고 coverage 보장을 전용하지 않음 |
| P7 M. Zaffran, O. Féron, Y. Goude, J. Josse, A. Dieuleveut, *Adaptive Conformal Predictions for Time Series*, ICML/PMLR162,2022,25834–25866 | [원문 PDF](https://proceedings.mlr.press/v162/zaffran22a/zaffran22a.pdf) | 시간 의존성과 교환가능성, 온라인 보정의 한계를 다룬다 | 시간 인접 표본을 섞지 않는 정상 episode 분할. AgACI 자체와 확률 보장은 제외 |
| P8 E. S. Page, *Continuous Inspection Schemes*, Biometrika41(1–2),1954,100–115, DOI10.1093/biomet/41.1-2.100 | [출판사 서지·미리보기](https://academic.oup.com/biomet/article-abstract/41/1-2/100/456627), 전문 미열람 | 저자·연도·발표처·DOI를 확인했다 | 고전 순차 검정의 서지 근거. 구체식은 실제 원문을 확인한 P1을 사용 |
| P9 M. Pierallini 외, *A Provably Stable Iterative Learning Controller for Continuum Soft Robots*, RA-L8(10),2023,6427–6434, DOI10.1109/LRA.2023.3307007 | [기관 서지](https://elib.dlr.de/197435/), [최종 원문](https://repository.tudelft.nl/file/File_d3f9b7f4-fd73-4bc5-9f0c-04912b04ba8f) | 연속체 soft robot의 모델·데이터 기반 feedforward ILC를 분석한다 | 후순위의 오프라인 동결 feedforward 가설. 탄성 관절·1 ms·지연에 수렴 결과를 전용하지 않음 |
| P10 Y. Dou, E. Prempain, L. Su, *Adaptive Iterative Learning Control for Robotic Manipulators*, CONTROL2024,96–97, DOI10.1109/CONTROL60310.2024.10531843 | [기관 초록·서지](https://research.manchester.ac.uk/en/publications/adaptive-iterative-learning-control-for-robotic-manipulators/) | 피드백과 반복 오차 feedforward의 adaptive gain 결합을 설명한다 | 적응·반복 갱신은 평가 독립성·온라인 학습 금지로 제외. 전문·보장 재현 아님 |

## 후보를 정하기 전의 경계

논문은 이 플랜트에서의 이득 증거가 아니다. 먼저 허용 신호만의 경험적 분리 상한을 분석하고, 개선 여지가 있는 방법만 새 정상 데이터와 사전 protocol로 검증한다. CUSUM의 누적 drift, 조건부 보정의 누설, 예상 접촉과 충돌의 비식별성, 시간 의존성으로 인한 보장 부재를 함께 다룬다. 명목 false-alarm 수준이나 유한 조건 검사를 실물 안전성·전역 안정성·확률 보장으로 쓰지 않는다.
