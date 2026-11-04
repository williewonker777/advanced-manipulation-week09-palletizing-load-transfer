# 사전 protocol

기계 판독 사전 설계는 [`PROTOCOL.json`](../results/v2_protocol/PROTOCOL.json)이다. 모델 학습·후보 구현·development 충돌·holdout 실행 전에 로컬 커밋했다. 구현 버그 수정과 결과를 보고 하는 재튜닝을 구분하며 후자는 금지한다.

- C: 허용 JTS 관측 support만으로 법선 bounded admittance. 최대1.5 mm·10 mm/s·0.5 m/s². 기존 elastic inner controller를 보존한다. 전체 Lie Group controller 재현이 아니라 좌표 감사·국소 근사다.
- L: 정상-only 자기지도 ridge. 질량/COM/알려진 reference 기반 수평 wrench 예측,방향대칭 norm+EMA. 비공식 무충돌 validation 최대에서 threshold 1회고정,균형 ±방향 development에서 방법 채택만 결정. 기존감지 OR 유지.
- G4: 마지막20ms 평균support≥0.50mg,최소≥0.25mg,평균개선≥0.25mg,놓기peak≤1.10mg. validation 이득이 없으면 C탈락.
- 공식조건은학습/threshold선정에미사용. holdout 평가 전 source/weights동결. 미달/outlier 보존.

최종 결과·구현 차이는 `V2_RESULTS.md`. 효과나 안정성의 보장이 아니다.
