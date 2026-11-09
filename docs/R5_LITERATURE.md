# 문헌 근거와 적용 경계

## 범위와 용어

이 문서는 정상 주행 자료만으로 잔차를 학습·보정하는 충돌 감지 후보의 문헌
근거를 정리한다. **정상-only**는 학습·보정에 충돌 표지나 충돌 시뮬레이션
신호를 넣지 않는다는 뜻이다. **FA**는 무충돌 또는 충돌 사건 밖의 오감지이며, **A**와
**B**는 각각 서보 방식과 토크 방식이다. 논문의 실험·정리와 아래의 자체
적용 추론은 구분한다.

문헌은 일반 웹에서 직접 연 원문, 저자 공개본 또는 학회 원문 페이지로 확인했다.
이 목록의 방법을 그대로 재현했거나, 이 플랜트의 통계적 coverage·안전·실시간
성능을 보장한다는 뜻은 아니다.

## 직접 확인한 문헌

| 구분 | 서지·직접 확인 링크 | 확인 범위와 확인 사실 | 자체 적용 추론·한계·이식 차이 |
|---|---|---|---|
| N1 | D. Lim, D. Kim, J. Park, *Momentum Observer-Based Collision Detection Using LSTM for Model Uncertainty Learning*, ICRA 2021, 4516–4522, DOI `10.1109/ICRA48506.2021.9561667`. [원문](https://daegyulim.github.io/assets/ICRA2021.pdf) | 원문. free-motion proprioception 시계열로 모델 불확실성을 LSTM으로 추정하고, 저속·정지마찰 근처의 오차 한계도 보고한다. | 짧은 과거창 정상 예측의 근거다. LSTM을 재현하지 않고, 고정된 NumPy 특징과 ridge 예측으로 축소한다. 탄성 관절, 1 ms 주기, 사용 가능한 신호가 다르므로 논문 결과를 이식하지 않는다. |
| N2 | G. Evangelisti, S. Hirche, *Data-Driven Momentum Observers With Physically Consistent Gaussian Processes*, IEEE T-RO 40, 2024, 1938–1951, DOI `10.1109/TRO.2024.3366818`. [저자 공개본](https://mediatum.ub.tum.de/doc/1739795/document.pdf) | 저자 공개본. 물리 제약 GP와 covariance-adaptive momentum observer를 결합하고, 특정 관측기 계열의 오차 경계와 안정성 조건을 다룬다. | 불확실성 크기를 상태에 따라 나타내려는 동기다. GP 관측기·물리 가정·안정성 결과는 사용하지 않으며, 작은 상태별 scale만 별도로 시험한다. |
| N3 | Y. S. Shao, B. Singh, N. Morozovsky, P. Yu, *Collision Detection for Low-Cost Robot Manipulators Using Probabilistic Residual Torque Modeling*, IROS 2025, 14766–14773, DOI `10.1109/IROS60139.2025.11246650`. [원문](https://events.infovaya.com/uploads/documents/pdfviewer/b2/f3/204893-2932.pdf) | 원문. excitation trajectory의 momentum-observer residual과 상태에서 GMM을 만들어 미모델 동역학을 표현한다. | 상태조건 정상 잔차 모델의 비교 근거다. 4 자유도 장치와 GMM을 재현하지 않으며, 작은 상태별 정규화가 먼저다. |
| N4 | D. A. Nix, A. S. Weigend, *Learning Local Error Bars for Nonlinear Regression*, NIPS 7, 1994, 489–496. [원문](https://papers.nips.cc/paper/1994/file/061412e4a03c02f9902576ec55ebbe77-Paper.pdf) | 원문. 입력에 따라 달라지는 오차분산을 학습하는 방법과, 분산 가중 학습의 이점·국소해 위험을 설명한다. | 위상·상태별 residual scale의 근거다. 신경망 이분산 우도를 쓰지 않는다. 평균·scale은 정상 적합 자료에서만, threshold는 분리된 정상 보정 자료에서만 정한다. |
| N5 | A. Stirn, H.-H. Wessels, M. Schertzer, L. Pereira, N. Sanjana, D. Knowles, *Faithful Heteroscedastic Regression with Neural Networks*, AISTATS 2023, PMLR 206, 5593–5613. [원문](https://proceedings.mlr.press/v206/stirn23a/stirn23a.pdf) | 원문. 통상 이분산 NLL 학습이 평균 정확도 또는 분산 보정을 해칠 수 있음을 분석한다. | 복잡한 이분산 신경망을 바로 채택하지 않는 근거다. 이 논문의 학습 수정이나 성능을 사용하지 않는다. |
| N6 | Y. Romano, E. Patterson, E. J. Candès, *Conformalized Quantile Regression*, NeurIPS 2019, 3538–3548. [원문](https://papers.nips.cc/paper/8613-conformalized-quantile-regression.pdf) | 원문. 분리된 validation residual 보정과 입력별 prediction interval 폭을 결합한다. | 학습과 calibration을 분리하는 설계 근거다. 시간상관 제어 trace는 교환가능 표본이 아니므로 coverage 보장을 주장하지 않는다. |
| N7 | V. Vovk, *Conditional Validity of Inductive Conformal Predictors*, ACML 2012, PMLR 25, 475–490. [원문](https://proceedings.mlr.press/v25/vovk12/vovk12.pdf) | 원문. inductive conformal의 무조건 validity와 조건부 validity를 구분하고, 교환가능성 가정을 명시한다. | 전역 보정값을 상태별 보장으로 오해하지 않는 경계다. 고정된 정상자료 분위수는 경험적 임계값일 뿐이다. |
| N8 | A. Bellotti, *Approximation to Object Conditional Validity with Inductive Conformal Predictors*, COPA 2021, PMLR 152, 4–23. [원문](https://proceedings.mlr.press/v152/bellotti21a/bellotti21a.pdf) | 원문. 유한 표본의 강한 object-conditional validity가 일반적으로 보장되지 않음을 설명하고 경험적 근사를 제안한다. | 상태별 threshold는 각 조건의 완전한 보장이 아니다. 조건 수를 작게 고정하고 독립 무충돌 감사로만 판정한다. |

## 실제 후보와 적용 범위

| 후보 | 문헌과의 관계 | 사전 분석 상태 | 적용 경계 |
|---|---|---|---|
| 위상 정규화 `phaseMAD` | N4의 입력 의존 오차 크기와 N6의 분리 보정 원리를 단순화한다. | A·B 모두 추가 검출 0으로 진입 미달이며, 영구 진단 대조다. | 위상은 미리 정한 작은 범주만 쓰며, 충돌 결과로 scale·threshold를 바꾸지 않는다. |
| 상태 정규화 `state-logscale` | N2–N5의 상태 의존 불확실성 논의를 작은 정상 residual scale로 축소한다. | A·B 모두 사전 상한 진입 기준을 통과한 후보이며, 아직 채택 판정은 아니다. | GP, GMM, LSTM, 이분산 NN을 재현하지 않는다. 정상 train·calibration·감사 자료를 분리하고, 경험적 FA 관문으로만 판정한다. |
| 20/50 ms AR 정상 ridge | N1의 과거창 정상 불확실성 예측을 선형·고정 가중치로 축소한 진단 후보다. | A는 추가 검출이 있으나 충돌 외 오감지가 늘었고, B도 같은 이유로 채택 불가다. | 진단 결과를 보고 threshold를 재조정하거나 다시 설계하지 않는다. 이 후보는 최종 제출 방식이 아니다. |
| RFF 정상 모델 | 비선형 정상 모델의 비교 후보다. | 사전 분석에서 진입 기준에 닿지 않아 구현하지 않는다. | 성능 불가능의 주장이 아니라, 이번의 사전 고정한 구현 범위 밖이라는 뜻이다. |

## 해석 원칙

- `state-logscale`의 사전 상한 통과는 이후 validation, development, 강건성,
  동결 후 holdout, 마지막 회귀 확인을 통과했다는 뜻이 아니다.
- 과거창 AR의 추가 검출은 충돌 외 오감지 증가를 상쇄하지 못한다. 따라서 이 문서에서는
  진단 결과로만 남기며 최종 방식으로 주장하지 않는다.
- 정상자료 보정의 임계값은 충돌 표지·공식 조건·holdout으로 선택하지 않는다.
  문헌의 conformal 용어는 train/calibration 분리의 근거일 뿐, 이 조건에서의
  확률 coverage 또는 안전 보장이 아니다.
