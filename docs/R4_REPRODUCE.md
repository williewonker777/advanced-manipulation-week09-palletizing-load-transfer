# R4 재현·검증 범위

## 최종 구현과 자료

최종 `my_servo.py`, `my_torque.py`, `my_detector.py`는 모두 R1과 바이트가 같다.
W40과 CUSUM은 미채택 실험이다. 원래 네 생성 소스, W40 수리본, 보정값, 모든 조건별
집계와 outlier를 보존했다. [결과](R4_RESULTS.md), [방법](R4_METHODS.md),
[문헌 확인 범위](R4_LITERATURE.md)를 함께 읽는다.

원본 키트와 가이드는 포함하지 않는다. 동일 키트를 별도로 준비하고 `sim/`, `model/`,
기본 evaluator를 겹친 환경에서 검사한다. `tools/kit_manifest.sha256`의 보호 20파일이
일치해야 한다. 환경에는 패키지를 추가하지 않았으며, 기록된 기존 NumPy·MuJoCo 환경을
그대로 사용했다. 렌더링을 재현한다면 소프트웨어 Mesa llvmpipe만 사용한다.

## 결과를 다시 실행하지 않는 검사

Python 실행은 모두 `env -u PYTHONPATH`로 시작한다. bytecode를 끄고 BLAS/OMP를 1로,
GPU를 숨긴다. 다음 검사는 새 holdout·공식 평가를 열지 않는다.

```bash
env -u PYTHONPATH PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES='' .venv/bin/python -m unittest discover -s tools -p 'test_*.py'
env -u PYTHONPATH PYTHONDONTWRITEBYTECODE=1 .venv/bin/python tools/r4_report.py --check
env -u PYTHONPATH PYTHONDONTWRITEBYTECODE=1 .venv/bin/python tools/validate_media.py --root . --ffprobe "$FFPROBE"
sha256sum -c artifacts/SHA256SUMS --quiet
```

공개 본문 검사는 `tools/audit_public.py`에 동일 키트·가이드·ffprobe와
`--require-media`를 지정한다. `FFPROBE`는 준비된 실행 파일이다. 검사 도구 설치를 위해
공식 Python 환경을 변경하지 않는다. 보고서 manifest는 포함된 집계의 바이트와
curation을 묶고, 문서·PDF `--check`는 같은 입력으로 재생성해 대조한다.

보고서 `--check`는 공개 트리의 모든 Markdown·HTML(README, `docs/`, `report/`,
`results/`, `tools/` 포함)의 상대 파일 링크도 확인한다. 중첩 GIF/MP4 링크를 포함하며 외부 URL은 접속하지 않는다.
같은 문서의 제목 앵커 존재 여부는 이 파일 경로 검사 범위 밖이다.

과거 R1/R2/R3 README는 루트에서 `docs/`로 옮기며 깨진 URL만 각각 9/31/20개 고쳤다.
따라서 그 세 파일의 현재 전체 해시는 과거 보존 기록과 다르다. 과거 JSON·동결·검증 기록을
다시 쓰지 않고 [별도 대조 증거](../results/v2_protocol/R4_REPORT_LINK_REPAIRS.json)에서
현재 URL을 원래 URL로 역복원해 고정된 원문 SHA와 같은지 검사한다.
옮기기 전 루트 README와 현재 문서가 가리키는 저장소 대상도 같아야 하며,
쿼리 문자열과 제목 앵커를 바꾸는 것은 허용하지 않는다.
이 세 문서의 URL 외 차이 0만 허용하며 다른 문서·과학 파일 변경의 예외로 쓰지 않는다.

## 새 실험과 기존 기록의 구분

원시 관측 캐시는 포함하지 않았다. 공개된 집계·소스·manifest 대조와, 원시 관측을
사용한 ROC·보정·감지 재계산은 서로 다르다. 후자는 해당 원시 입력과 원래 연구 작업
이력이 있어야 정확한 출처 검사를 통과한다. 연구 커밋 해시는 공개 이력의 조상이 아니다.

D4는 이미 노출된 캐시 재생이며 독립 효능이 아니다. 섭동은 기존 물리 25조건과 새
잡음 9조건을 구분한다. 최대 시간·outlier는 OS 부하에 영향받아 수치의 완전 재현이나
hard real-time 보장이 아니다. 재생 행의 기존 R1 timing을 새 감지기 비용으로 쓰지 않는다.

등록된 holdout/공식 실행 도구는 배타적 marker로 재실행을 금지한다. 원래 holdout 시드의
사전 메모리 생성 사고, 재봉인, W40 1회 구조 수리, 새 배제 질량을 결과 문서에 기록했다.
이 기록의 시드를 새 실험에 재사용하거나, 동결 뒤 후보·임계값·가중치를 수정해서는 안 된다.
새 연구는 별도 사전 계획·미사용 시드·봉인 데이터로 시작해야 한다. 이번 실제 holdout은
모든 후보 탈락 뒤 R1 보존만 확인했으며 W40의 독립 효능 시험이 아니다.
