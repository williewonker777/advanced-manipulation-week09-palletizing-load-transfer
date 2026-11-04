# 원본 요구사항 재판정

가이드 전체(Part 1–4·부록 A–E), 배포 텍스트 18개, binary STL 10개와 HTML 삽입 그림 12개를 새로 확인했다. 표는 가이드 문장을 재배포하지 않는 요약이다. 원본과 `sim/`, `model/`은 읽기 전용이다. 총 원본29파일(키트28+가이드HTML1)을 확인했다. 현재 판정은 동결된 최종v2의 코드·공식8조합·D4·사전분리holdout·분석섭동·기본파일명overlay 재실행을 반영한다. v1 기준 재현4조합도 보존했다. 공식세트는 디버깅에 소비된 호환성 증거이고, 독립 관측 개선은 새 비공식 연구·holdout으로만 판단한다. 상세 수치와 대가는 `V2_RESULTS.md`, `V2_REVIEW.md`와 함께 읽는다.

판정의 **충족**은 해당 유한 조건의 증거이지 실기 안전·비공개 평가 통과를 뜻하지 않는다. 수치 목표와 규칙 준수를 분리한다.

| ID | 요약 | 출처 | 종류 | 현재 증거 | 판정 | 조치 |
|---|---|---|---|---|---|---|
| R01 | 하중 전환의 제어 문제에 아이디어와 효과를 함께 제시 | Part 2 목표1 | 제출물 | v1 제어 원리 유지·정상-only ridge/EMA 채널 추가; C/학습/교차조합 실실행 | 충족 | v2 제어 신규이득 없음 명시 |
| R02 | 토크 변화·착지 충격·원래 경로 오차를 줄임 | Part 2 목표1, Part 4 기준 | 평가 항목 | 8kg A step9.7/14.6N·m·table79N·TCPmax3.91mm; B9.0/13.1·80N·8.44mm, v1동일 | 충족 | 명목 기준 개선은 v1 상속; stress 한계 별도 |
| R03 | 포화·진동·발산을 확인 | Part 2 목표1 | 평가 항목 | 공식0/4/8kg 포화0·정상; D4 A COM2조건0.15/0.0333%포화·발산0; 진동부재 보장안함 | 부분 | 격자 포화와 진동 검증 한계 유지 |
| R04 | 감지 아이디어와 시뮬레이션 근거 제시 | Part 2 목표2 | 제출물 | 정상-only77basis ridge+대칭EMA+기존OR; S4 ablation·D4·sealed holdout | 충족 | 정상 학습과 L3 gate수정 기여 구분 |
| R05 | 무충돌 집기·놓기 때 경보 없음 | Part 2 목표2 | 평가 항목 | 공식0/4/8kg와D4각189ref·holdout각4ref는FA0; 분석섭동FA A4→6/B7→9 | 부분 | 명목 충족과 섭동 실패를 구분 |
| R06 | 더 약한 충돌을 더 빨리 판정 | Part 2 목표2, Part 4 기준 | 평가 항목 | 8kg운반15N45/43ms·10N62/56ms; D4검출111/111회이나B추가FA1→3,miss69/69회 | 부분 | G1/G2 호환성 통과와 G3 미달 함께 보고 |
| R07 | 하나의 추정 공유 시 각 목표 기여를 분리 | Part 2 두 목표, Part 4 예시 | 제출물(공유 선택 시) | 공식ctrl/detector8교차조합, v1/v2 D4·holdout·섭동914physical쌍 동일 | 충족 | 제어변경0·감지만기여;공유cache검사 |
| R08 | 정지·후퇴 반응은 선택 사항 | Part 2 두 목표 | 권고 | 반응 미구현, 시뮬레이터 기본 유지 | 판정 불가 | 선택하지 않음; 필수요구 아님 |
| R09 | 접촉모델 사용 시 파라미터 변화·수렴 한계 보고 | Part 2 방법A | 권고 | 일방향/COP 집합은 강성 접촉모델A와 다름;A 미선택 | 판정 불가 | 강성 섭동 별도 보고 |
| R10 | 강인 기법의 불확실성–감도 대가 제시 | Part 2 방법B | 권고 | 15N지연·10N개선과 D4 B추가FA+2,분석섭동FA각+2를함께기록 | 충족 | 전범위 우월·실기배치 권고 안함 |
| R11 | 감지 무효구간을 쓰면 위험·시간 대가 필요 | Part 2 방법C | 권고(방법C 선택 시 조건부) | 전체감지off구간없음; learned만기하height gate, v1채널상시OR; 방법C전체mask 미선택 | 판정 불가 | 추가 gate 근처 감도 한계 공개 |
| R12 | 논문·학습·추가 센서 등 선택 가능, 미검증은 계획으로 구분 | Part 2 방법D | 권고 | 직접연20문헌(PDF12편), C·학습 구현평가, geometry/power검사·수직반례 | 충족 | 원리참고와자체변형/미해결을명시 |
| R13 | 궤적 변경 시 원래 기준 TCP·주기시간 대가 보고 | Part 2 방법D 말미 | 제출물 | v1의5ms endpoint advance 상속; C국소보정은G4실패로미채택;원래경로오차평가 | 충족 | 최종 reference/grip 변경0 |
| R14 | 공식 evaluator 결과를 기준과 나란히 제시 | Part 2 시뮬레이션, 부록A/B | 제출물 | 변경없는evaluate.py로기준/v1재현4+최종공식8조합,기본이름A/B도동일66셀 | 충족 | 공식 표는독립효능아닌호환성 |
| R15 | 하중 전환 토크·TCP·작업대 힘·잔차 그래프 | Part 2 시뮬레이션 | 제출물 | 기준/v1/v2 torque·TCP·table·residual자체그림7개,16쪽PDF,새3way05/06영상 | 충족 | 01–04는역사적v1로구분 |
| R16 | 질량·COM·충돌 시각·크기·방향 다양화 | Part 2 시뮬레이션 | 제출물 | D4 1540행,사전봉인holdout152행,분석섭동136행;질량/COM/시각/힘/±축/사선/offset | 충족 | 모든실패/오감지분리집계;확률보장아님 |
| R17 | 실패하는 조건을 숨기지 않음 | Part 2 시뮬레이션 | 제출물 | D4 miss69/69·B추가FA·offsetFA8/6,섭동FA6/9·포화·모든timing outlier보존 | 충족 | 실패원인과미확정원인도기록 |
| R18 | 1 ms 제어주기 내 계산 가능한지 측정 | Part 2 시뮬레이션, Part 3 | 평가 항목 | 최종worst-row p99 A346.762/B350.531µs; 관측max12.152/160.578ms;passiveGC/CPU/context진단 | 부분 | strict1ms 미달; 원래160.578ms 단일원인미확정 |
| R19 | 원리·범위·대가·현장 적용성을 설명 | Part 2 효용성 | 제출물 | V2_REVIEW의원리/범위/대가/현장적용,미달G3/G4/G5/G6·U9미해결 | 충족 | 실기/안전/hardRT 보장 미주장 |
| R20 | 문제→아이디어→검증→효용성 발표 | Part 2 발표 구성 | 권고 | 최신16쪽PDF 문제→방법→공식/독립검증→효용·한계;전체쪽시각확인 | 충족 | 논문·교재그림 재사용0 |
| R21 | 실제 내부 상태·작업대 힘·진짜 토크를 런타임에 안 씀 | Part 2 규칙, Part 3, 부록B/C | 필수 규칙 | production+learnedcore AST통과; 허용observer자기지도target; plant/log는사후분석전용 | 충족 | AST는임의Python정보흐름의증명아님 |
| R22 | 원래 sim/model, 구동부 구조 보존 | Part 2 규칙, 부록A | 필수 규칙 | 보호20파일SHA일치·원본29inventory/SHA일치;모듈monkeypatch0 | 충족 | final publication전후 재검사 |
| R23 | 미공개 질량·COM·외력에 일반화 필요 | Part 2 규칙, evaluate.py 머리말 | 평가 항목 | 비정수·미학습COM/시각/방향의seed369503봉인holdout완료,비공개15주차조건접근불가 | 판정 불가 | holdout은유한대리검사이며비공개통과아님 |
| R24 | controller 입력은 t/encoder/JTS/grip/reference 및 공개모델 | Part 3 측정·모델, 부록C | 필수 규칙 | controller시그니처·공개nominal,learner현재/과거reference+known mass/COM,추정JTSwrench만 | 충족 | 가중치내장·numpy추론·외부파일I/O0 |
| R25 | motor torque는 detector 전용 입력 | Part 3 측정 | 필수 규칙 | controller 입력에 tau_m 없음, detector는 허용 | 충족 | 계약 테스트 |
| R26 | trajectory/config는 읽기만 | Part 3 모델 | 필수 규칙 | AST public-module쓰기금지·hash보존;실제offset/drift/seed 런타임미사용 | 충족 | 분석전용섭동과runtime엄격분리 |
| R27 | 선택한 방식의 출력 준수: A는 지령4개·gain만, B는 토크6개 | Part 3 변경 가능, 부록C | 필수 규칙 | A4×6지령·gain/B6토크 shape·finite·reset검사;최종기본이름A/B전체공식평가동일66셀 | 충족 | self-contained3파일·공식venv추가패키지0 |
| R28 | detector→controller 호출 순서 불변, self.ctrl은 공유 선택 시 | Part 3 변경 가능 | 필수 규칙 | 원runner update→compute 유지,공유cache timestamp/리셋/외부controllerfallback검사통과 | 충족 | S5계측은별도인스턴스subclass만 |
| R29 | 충돌평가 판정과 무충돌 경보를 혼동하지 않음 | Part 4, 부록B, evaluate.py | 평가 항목 | 공식우선순위·±20ms참조 그대로; ref/offset/extraFA/miss/masked를별도집계 | 충족 | 실패·outlier삭제0 |
| R30 | 모델·센서·흡착의 가정과 실기 차이 고지 | 부록D | 제출물 | 허용신호·rigid교재vselastic/1ms변형·동일법선wrench반례·실기미검증명시 | 충족 | 동역학전체history비식별의증명아님 |
| R31 | 실행환경·모델 사용권 고지 | 부록A/C/E, requirements/model license | 권고 | Python3.12.3/numpy2.5.4/MuJoCo3.17.0·package목록S0동일,외부모델별도·저자코드0 | 충족 | 학습별도venv·키트/가이드/수업스캔미배포 |
| R32 | GPU 시뮬레이터는 선택일 뿐 | 부록E | 권고 | CPU학습·CPU시뮬레이션·Mesa llvmpipe영상; GPU/Isaac미선택 | 판정 불가 | 선택사항이며GPU학습0 |
| R33 | detector API는 update(t,th,thd,tau_m,tau_s,grip)→잔차6개·bool | Part 3, 부록C | 필수 규칙 | 최종Detector.update 6잔차·bool,reset/단독/공유/외부이름import 및defaultname전체평가 | 충족 | 가중치·threshold·EMA·persistence 동결유지 |

## 제출 형식 판정

가이드에는 파일 확장명·API·바꿀 수 있는 위치는 있으나 **제출 ZIP/단일 파일 강제/학습 가중치 형식/추가 모듈 금지의 명시 규정은 없다**. 부록 B는 별도 controller 파일·상속 예시도 허용한다. 따라서 규정이 없는 부분을 필수라고 단정하지 않는다.

보수적 제출 계약은 A `servo_controller.py`, B `torque_controller.py`, 공통 `collision_detector.py` 각각 자기완결이다. 원본 키트에 해당 방식 controller와 detector만 복사한다. Python3.12, numpy, mujoco, public sim.config/trajectory만 런타임 사용한다. 그림용 matplotlib 외에 새 패키지를 요구하지 않는다. 가중치는 파일 안 숫자 배열로 내장하며 torch/pickle/ONNX/가중치 파일 I/O는 하지 않는다. `tools/validate_submission.py`는 별도 사본에서 세 기본 모듈명을 사용해 A/B 공식 평가를 실행한다. v1 overlay와 최종 v2 overlay를 따로 실행했고, 최종 A/B의 결정적66셀씩이 공식 최종 조합과 모두 같았다. Windows .bat 실동작은 Linux PC에서 미검증이다.

가이드의 config 공개 범위에는 센서 오프셋도 포함되지만 이 연구에서는 더 엄격한 계약으로 실제 offset/drift/seed를 사용하지 않는다. 내부값 학습 타깃은 원본의 명시적 학습 규정이 아니라 이번 연구의 보수 정책에 따라 사용하지 않는다. 충돌 라벨은 시나리오 정의에서, 자기지도 타깃은 허용 입력의 관측기 잔차에서만 만든다.

## 가이드와 별개인 연구 자체 목표

| ID | 자체 수치 목표 | 가이드와의 구분 | v1 재확인/처리 |
|---|---|---|---|
| U1/G1 | 8 kg15 N 운반68 ms이하 | 더 빠르게는 가이드,68 ms cutoff는 자체 | A/B83 ms →45/43 ms,호환성목표통과(공식독립성제약) |
| U2/G2 | 8 kg10 N 운반검출 | 더 약하게는 가이드,10 N 통과는 자체 | v1놓침→v2 A62/B56ms |
| U3/G3 | 180충돌 sweep 검출수 증가·추가오감지 비증가 | 수/조건 자체설계 | 직접재측정94/91→111/111이나B extraFA1→3로G3미달 |
| U4/G4 | 놓기 직전 지속이양·힘≤1.1mg | 부드러운착지는 가이드,정확 수치는 자체 | S3수치고정후C각0/4탈락;최종mean0.164664/8.751053N미달 |
| U5/G5 | 모든 관측 주기<1 ms | 계산시간검사는 가이드,관측max cutoff는 자체 엄격목표 | 최종max12.152/160.578ms미달,원outlier보존·GC수동진단 |
| U6 | offset·충돌 후 추가오감지 제거 | 자체stress 목표 | 일부 실패 |
| U7/U8/G6 | COM·마찰·강성 악화 감소 | 변화조건검사는 가이드,격자/개수는 자체 | 제어악화불변,분석섭동FA4→6/7→9로G6미달 |
| U9/G7 | 수직동일렌치 모호성 다루기 | 추가센서대안 자유,완전식별 강제없음 | 입력만으로 구분불가,해결 주장안함 |
| X1 | A+B 비교 | 가이드는 A 또는 B,둘 다는 연구범위 | 양쪽 평가 |
| X2 | 직접확인 논문12편,제어/학습 각4편 이상 | 추가 연구요구,가이드 수량규정없음 | PDF12편(제어8·학습4),총20편 |
| X3 | 제어기법/학습 각각 구현·ablation | 추가 연구요구 | C·L1·L2·L3실행/탈락/채택근거보존 |
| X4 | numpy추론·분리학습·무충돌임계값·holdout선고정 | 추가 보수 연구계약 | 사전protocol→freeze→holdout실행·source/target감사완료 |

**누락/미달을 모두 해결했다고 판정하지 않는다.** 강인성·최대시간·감지 전체 우위·비공개 평가는 부분/미검증이다. G1/G2의 명목 개선을 G3/G6의 오감지 대가를 상쇄하는 보편적 성공으로 쓰지 않는다. 현재 필수 입력/무결성 계약 위반은 발견하지 못했다.

독립 요구사항 검토 후 조건부/선택항목을 보완했다. 판정 불가 중 선택 미사용은 필수 미충족으로 세지 않는다. R03은 saturation/발산만으로 진동 부재를 증명하지 않아 부분으로 둔다.

## R2 재판정: 새로운 개선과 규칙 준수를 분리

| 구분 | R2 증거 | 판정 |
|---|---|---|
| 제출 형식·허용 입력 | 최종3파일 R1 동일, 새 default-name overlay 원본 공식 회귀·production AST, 학습 allowed-only 파일 | 형식·입력 규칙 충족; 성능 전체 충족 아님 |
| 원본/보호 소스·환경 | 원본29/보호20/공식21패키지 불변 | 충족 |
| P0 물리 측정 | 8기준+32probe;40/40 G4 미달, 정적 강성 fit 제외 | 측정 수행, **필요 변위·속도 식별은 부분/미달** |
| 실제 후보 구현 | C1/C2/C3/C4/C6 25설정×A/B×4조건=200회, pRNE9수학 조건·정상 보정·균형 ablation | 시도 충족; 새 방법 채택0 |
| 탄성/1ms/지연 분석 | 300공통점 국소 분석,222 protocol-radius·190 strict 통과, R1 B의 국소 미달도 보존 | 분석 수행; 전역·수동성 보장 미검증 |
| 사전 동결·새 holdout·공식 순서 | 사전 protocol/source 커밋,53파일 동결, 새152배치1회 후 공식58회, 사후 수정0 | 순서 충족;0.75kg 질량 중복 한계 명시 |
| G4/T1 | 마지막20ms 평균·최소·증가·peak 기준 미완화 | **미충족** |
| T2–T6 신규 이득 | 원래 R1 유지; 일부 후보의 이득보다 회귀/비용이 큼 | 새 개선 미달, 기존 R1 수치 유지 |
| 강건성·미검출·시간 상한 | 격자/섭동/holdout의 실패·오감지·포화·시간 outlier 전부 보존 | 부분/미충족; 실기·hard real-time 주장 없음 |

상세 동일조건 표와 공개 관문 결과는 R2_RESULTS.md 및 R2 공식/holdout JSON을 따른다. R1의 과거 공식 노출 이력과 이번 R2의 사전 동결 절차를 섞지 않는다. 이번 R2에서는 공식 조건을 학습·임계값 선정·수정에 쓰지 않았다.
