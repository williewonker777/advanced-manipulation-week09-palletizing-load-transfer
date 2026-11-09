# R5 재현·검증 범위

R5는 제어기를 바꾸지 않고 정상 데이터로 학습한 감지 분기만 비교한다. 채택 여부와 수치는
[결과](R5_RESULTS.md), 고정 수식과 데이터 분할은 [방법](R5_METHODS.md), 구현 전 가설은
[원인 분석](R5_CAUSE_ANALYSIS.md), 외부 근거는 [문헌 조사](R5_LITERATURE.md)에서 확인한다.
R4까지의 기준은 [R4 보존 문서](R4_README.md)에 남아 있다.

## 공개 파일로 확인할 수 있는 것

공개 안전 투영에는 R5 core, 생성 감지기 4개, 고정 모델 JSON 8개, 최종·R1 비교 파일 4개와
`r5_build.py`, `r5_normal_learning.py`, `r5_metrics.py`, R5 보고서 도구 및 무시뮬레이션 검사가
포함된다. 고정 추론은 NumPy만 사용하고 실행 중 파일을 읽거나 가중치·임계값을 갱신하지 않는다.
`my_servo.py`와 `my_torque.py`는 R1에서 바꾸지 않았다. 패키지의 Python 환경에 새 패키지를
설치하지 않는다.

다음 연구용 파일은 공개 재현 경로에 포함하지 않는다.

- 키트의 plant·episode loop를 사용하는 `tools/r5_fixture.py`
- `tools/r5_study.py`, `r5_review.py`, `r5_robustness.py`, `r5_sealed.py`와 이들을 필수
  import하는 검사
- 원시 NPZ 신호, 비공개 물리 비교 배열, 실제 획득·개봉 driver

결과·protocol·holdout 조건과 출처 JSON은 동결과 평가가 끝난 뒤 사후 증거로 포함한다. 이 JSON은
원시 신호나 비공개 실행 driver를 제공하지 않으며 학습·임계값 선택에도 쓰이지 않았다.

따라서 공개 파일만으로 정상 캐시를 처음부터 수집하거나, 전체 실험·별도 평가·공식 평가를
끝까지 다시 실행할 수는 없다. 누락된 driver를 비슷하게 다시 만들거나 키트 코드를 복사해
이를 전체 재현이라고 부르지 않는다.

## 고정 산출물과 문서 검사

저장소 루트에서 기존 Python을 사용한다. 모든 명령은 외부 `PYTHONPATH`를 제거하고 bytecode,
GPU 사용과 BLAS/OMP 병렬화를 끈다.

```bash
export PYTHON=python3
env -u PYTHONPATH PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES='' \
  "$PYTHON" tools/r5_build.py --check
env -u PYTHONPATH PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES='' \
  "$PYTHON" tools/r5_report.py --check
```

`r5_build.py --check`는 등록 모델과 생성 감지기를 메모리에서 다시 조립해 포함된 바이트와
비교한다. 이 모드는 현재 모델·생성 소스·빌드 기록을 쓰지 않는다. 보고서 `--check`는 포함된
결과와 문서 산출물이 같은 입력에서 만들어졌는지 검사하며 새 시뮬레이션을 실행하지 않는다.

## 외부 키트를 겹친 공개 검사

원본 키트는 저장소에 포함하지 않는다. 합법적으로 제공받은 읽기 전용 키트를 새 빈 디렉터리에
복사한 뒤 공개 트리를 겹친다. `$KIT`는 키트 루트, `$PUBLIC`은 이 저장소 루트, `$OVERLAY`는
새 빈 작업 디렉터리다. 키트 원본을 직접 수정하지 않는다.

```bash
test -d "$KIT" && test -d "$PUBLIC" && test -d "$OVERLAY"
test -z "$(find "$OVERLAY" -mindepth 1 -maxdepth 1 -print -quit)"
cp -a "$KIT"/. "$OVERLAY"/
cp -a "$PUBLIC"/. "$OVERLAY"/
cd "$OVERLAY"
env -u PYTHONPATH PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES='' \
  "$PYTHON" -m unittest discover -s tools -p 'test_*.py'
```

외부 키트는 import와 API 검사용일 뿐 공개 파일로 바뀌지 않는다. 공개 감사의 `--kit`도
키트를 허용 목록에 넣는 옵션이 아니다. 공개 코드와 키트 코드 사이의 연속 6줄 복사를 검사하는
읽기 전용 입력이다.

## 정상-only 학습을 다시 실행하려면

재학습에는 위 키트 overlay와 별도로 제공된 정상 캐시가 모두 필요하다. 캐시는 `fit`과
`calibration`을 서보·토크 방식별로 나누고, 각 index가 protocol의 공개 설계, 역할, 신호 파일
SHA256과 일치해야 한다. 신호는 1 ms 간격 6000표본의 허용 배열만 담는다. 충돌 라벨, 외력,
실제 링크 상태, 접촉력, 강성·마찰·센서 시드 같은 물리·평가 정보가 있으면 로더가 거부한다.

```bash
export NORMAL_DATA=/path/to/r5-normal-cache
export OUT=/path/to/new-empty-output
env -u PYTHONPATH PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES='' \
  "$PYTHON" tools/r5_normal_learning.py --dataset "$NORMAL_DATA" --output "$OUT"
```

`$OUT`은 입력 캐시 밖의 새 디렉터리이거나 빈 디렉터리여야 한다. `models/r5`, `logs`, 입력 캐시
아래 경로를 지정할 수 없으므로 포함된 현재 모델을 덮어쓰지 않는다. 출력은 별도 모델 JSON과
manifest뿐이며, 채택된 모델로 자동 승격하지 않는다.

공개 결과의 신호 SHA와 출처 ID는 별도로 받은 캐시가 같은 바이트인지 확인하는 근거다. 해시나
집계 JSON만으로 원시 배열을 복원할 수 없으며 원시 NPZ를 재배포한다는 뜻도 아니다. 기존 영상은
R4까지의 기준을 설명하는 자료다. R5 채택이나 새 영상의 존재는 [결과](R5_RESULTS.md)에서만 판정한다.

## 연구 기록과 공개 파일의 해시

동결 JSON의 `head`는 연구 작업 이력의 커밋이다. 공개 저장소의 커밋이 아니므로 공개 이력에서
찾을 수 없다. 공개 파일의 실제 바이트는 [공개 파일 목록](../results/v2_protocol/R5_PUBLIC_PROJECTION.json)과
`artifacts/SHA256SUMS`로 확인한다.

과거 집계의 연구 원본과 공개본은 비공개 경로 제거 때문에 해시가 다를 수 있다. 공개 기준 지표
목록과 보고서 입력 목록은 실제 공개 파일의 해시·크기에 연결한다. 이 연결에서 바꾸는 것은 출처
메타데이터뿐이며 측정값·판정은 바꾸지 않는다. 원본/공개 해시와 변환 대조도 공개 파일 목록에 남긴다.

최종 R1의 전체 검사는 작업 사본 529개, 외부 키트를 겹친 공개본 466개가 통과했다.
[최종 검증 기록](../results/v2_protocol/R5_FINAL_VERIFICATION.json)은 앞선 후보의 526개 검사·8오류도
함께 연결한다. 동결 뒤 후보를 고친 것이 아니라 미리 동결한 R1로만 되돌렸다.
