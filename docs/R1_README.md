# ICPBL v2 결과

## 결론

v2는 **v1 제어기 + 정상-only ridge 잔차/대칭 EMA 감지 채널**이며, 기존 v1 경보와 OR한다. 제한된 연구용 채택이며 실기 배치를 권고하는 결과가 아니다. 제어기는 바꾸지 않았다. 새 L3 검증이 등록 규칙을 통과해 정확한 기하 높이 gate 복구를 채택했다. L3의 가중치, 임계값, EMA와 지속시간은 바꾸지 않았다. 공식 세트는 디버깅에 사용되어 독립 효능 증거가 아니며 G1/G2는 공식 호환성 결과로만 공개한다. 최종 선택 근거는 새 비공식 L3 연구와 봉인 holdout에 한정한다.

| 구조 | 버전 | TCP RMS/max mm | torque step pick/place N m | table N | FA pick/place/other | 8 kg transport 15 N delay ms | 10 N delay ms | minimum |
|---|---|---:|---:|---:|---:|---|---|---|
| A | baseline | 6.14/14.00 | 57.5/73.8 | 101 | 5/2/0 | 68 | 놓침 | 15 N |
| A | v1 | 1.17/3.91 | 9.7/14.6 | 79 | 0/0/0 | 83 | 놓침 | 15 N |
| A | v2 | 1.17/3.91 | 9.7/14.6 | 79 | 0/0/0 | 45 | 62 | 10 N |
| B | baseline | 7.04/13.18 | 57.2/74.8 | 295 | 5/4/0 | 68 | 97 | 10 N |
| B | v1 | 3.23/8.44 | 9.0/13.1 | 80 | 0/0/0 | 83 | 놓침 | 15 N |
| B | v2 | 3.23/8.44 | 9.0/13.1 | 80 | 0/0/0 | 43 | 56 | 10 N |

- G1/G2/G3/G5: `True` / `True` / `False` / `False`. D4 B의 추가 오감지 1→3 때문에 G3는 미달이며 검출 수 증가만으로 성공이라 하지 않는다.
- G4: C 후보는 각 구조 0/4로 탈락했다. v2 제어 개선 주장은 없으며 검출기 변경으로 인한 추적 회귀도 없다.
- G6: `False`. 제어 악화조건은 그대로이며 분석 섭동 무충돌 FA는 A 4→6, B 7→9로 별도 보고한다. detector-only 변경을 제어 개선으로 주장하지 않는다.
- G7: 같은 순간 수직 wrench의 지지/충돌 모호성은 해결하지 못했다.

## 영상과 재현

기존 `01`--`04` 영상은 v1이다. 새 `05`/`06`은 3.7 kg 운반 중 15 N의 baseline/v1/v2 3-way 비교이며 `v2_manifest`로 대응 관계를 공개한다. 내부 trace는 그래프 전용 분석값이며 학습에 사용하지 않았다.

## 실행과 제출 파일명 overlay

아래 명령은 키트 작업 사본의 `.venv`를 사용하고 외부 `PYTHONPATH`를 제거한다. 제출 파일은 `my_servo.py`, `my_torque.py`, `my_detector.py`이며 검증기는 이를 기본 이름 `servo_controller.py`, `torque_controller.py`, `collision_detector.py`로 복사한 격리 overlay에서 실행한다.

```bash
env -u PYTHONPATH PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python -c "import numpy, mujoco; print(numpy.__version__, mujoco.__version__)"
env -u PYTHONPATH PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python tools/validate_submission.py --prefix my --out results/v2_submission_check
env -u PYTHONPATH PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python tools/evaluate_checked.py --ctrl my_servo --detector my_detector --out results/evaluation_v2_A.txt
env -u PYTHONPATH PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python tools/evaluate_checked.py --ctrl my_torque --detector my_detector --out results/evaluation_v2_B.txt
```

깨끗한 data collection은 MuJoCo가 있는 공식 `.venv`에서 수행하고, fitting만 NumPy 전용 `.venv-train`에서 offline으로 수행한다. 공식/holdout을 다시 학습에 넣지 않는다. 같은 saved data/manifest, script와 seed로 byte-identical model JSON 재학습을 검증했다. 새 collection부터의 end-to-end byte 비교는 미실행이며 수치 동일성을 보장하지 않는다. 새 manifest의 host telemetry와 `data_sha256`도 전체 model JSON hash를 바꿀 수 있다. 아래는 별도 재현 환경의 준비 예시이며 이번 작업에서 공식 `.venv`의 패키지를 추가하지 않았다.

```bash
env -u PYTHONPATH python3.12 -m venv --without-pip .venv-train
env -u PYTHONPATH pip3 --python .venv-train/bin/python install numpy==2.5.4
: "${V2_EVIDENCE:?set an external private evidence directory}"
env -u PYTHONPATH PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python tools/v2_collect.py --out "$V2_EVIDENCE/learning-data"
env -u PYTHONPATH PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv-train/bin/python tools/v2_train.py --data-dir "$V2_EVIDENCE/learning-data" --out "$V2_EVIDENCE/model-reproduced.json" --seed 369401
```

최종 보고서 생성은 최종 JSON과 여섯 private NPZ trace가 모두 있을 때만 수행한다.

```bash
env -u PYTHONPATH PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python tools/v2_report.py --trace-dir "${V2_EVIDENCE:?set private evidence root}/traces"
```

## 연구 범위와 산출물

- [방법과 수식](docs/V2_METHODS.md), [결과와 누락 없는 집계](docs/V2_RESULTS.md), [음성 결과·시간 진단 자체 검토](docs/V2_REVIEW.md)
- [요구사항 감사](docs/REQUIREMENTS_AUDIT.md), [문헌과 채택 범위](docs/LITERATURE.md)
- [발표 구성](report/PRESENTATION_V2.md), [16쪽 PDF](results/presentation.pdf)
- [최종 기계판독 판정](results/v2_protocol/FINAL_ASSESSMENT.json), [자체 작성 그림](results/figures/)
- 실제 하드웨어, hard real-time, 확률적 일반화, safety stop을 주장하지 않는다. 내부 trace는 학습 입력이 아니다.
