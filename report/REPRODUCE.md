# Reproduction

## Final v2 / historical v1 boundary

`my_servo.py` and `my_torque.py` equal the frozen `v1_*` controllers, but
`my_detector.py` is the final normal-residual learner with the mass-independent
geometry gate. Historical `FINAL_*`, `D3_*`, `D4_*`, `report/RESULTS.md`, and media
01–04 describe **v1**, not the new detector. Current results live in
`results/v2_protocol/`, `docs/V2_RESULTS.md`, and media 05–06. Historical scripts
that hard-code `my_*` now run v2; reproduce v1 only in a disposable copy with
the frozen v1 files overlaid under those names. Never overwrite a frozen
experiment record to make it agree with a later source revision.

### Final protocol and supplementary studies

The original protocol, two disclosed debugging amendments, final source freeze,
and failed variants are all retained. Canonical official conditions were
consumed for debugging, **not independent efficacy evidence**. Thresholds and
weights were not retuned from collisions or holdout. The saved protocol's
holdout seed is 369503; rerunning it is reproduction, not a new sealed test.
Use a disposable kit work copy and run these commands sequentially:

```bash
env -u PYTHONPATH PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python tools/v2_study.py --stage d4 --variants v1,v2 --out results/reproduced_v2/D4.json
env -u PYTHONPATH PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python tools/v2_study.py --stage holdout --variants v1,v2 --out results/reproduced_v2/HOLDOUT.json
env -u PYTHONPATH PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python tools/v2_study.py --stage timing --variants v1,v2 --out results/reproduced_v2/TIMING.json
```

These are analysis studies, not official evaluator scores. The private
perturbation adapter is not distributed because it contains course-runner
structure; its conditions, all numeric results, nominal parity, noise seeds,
and timing events remain in `S5_PERTURBATIONS.json`. This is an explicit
independent-reproduction limitation, not a claim that all published runs can
be regenerated without the external materials.

### Collection, fitting, and media

Only collection uses the existing simulator `.venv`; it records allowed-input
self-supervised observer targets, not plant/log truth. Fit on CPU in a separate
environment containing the same NumPy version; do not add packages to the
official environment. Keep this private evidence directory outside the work
copy. Neither fitting nor collection reads holdout results.

The following creates a new reproduction-only training environment; it does not
change the official `.venv`. Use the existing system pip driver if the Python
installation lacks ensurepip. These installation commands are documentation,
not operations performed on the official environment during this study.

```bash
env -u PYTHONPATH python3.12 -m venv --without-pip .venv-train
env -u PYTHONPATH pip3 --python .venv-train/bin/python install numpy==2.5.4
: "${V2_EVIDENCE:?set an external private evidence directory}"
env -u PYTHONPATH PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python tools/v2_collect.py --out "$V2_EVIDENCE/learning-data"
env -u PYTHONPATH PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv-train/bin/python tools/v2_train.py --data-dir "$V2_EVIDENCE/learning-data" --out "$V2_EVIDENCE/model-reproduced.json" --seed 369401
env -u PYTHONPATH PYTHONDONTWRITEBYTECODE=1 .venv/bin/python tools/v2_build_candidate.py --model models/v2_residual.json --geometry-gate --out "$V2_EVIDENCE/detector-reproduced.py"
cmp my_detector.py "$V2_EVIDENCE/detector-reproduced.py"
env -u PYTHONPATH PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -c 'from tools.v2_study import require_holdout_freeze, verify_freeze_sources; verify_freeze_sources(require_holdout_freeze()); print("frozen source/weights verified")'
env -u PYTHONPATH PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python tools/v2_trace.py --out "$V2_EVIDENCE/traces"
env -u PYTHONPATH PYTHONDONTWRITEBYTECODE=1 .venv/bin/python tools/v2_report.py --trace-dir "$V2_EVIDENCE/traces"
env -u PYTHONPATH PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MUJOCO_GL=glfw LIBGL_ALWAYS_SOFTWARE=1 GALLIUM_DRIVER=llvmpipe __GLX_VENDOR_LIBRARY_NAME=mesa .venv/bin/python tools/render_v2_comparison.py --raw-dir "$V2_EVIDENCE/media" --ffmpeg ffmpeg
```

Byte-identical retraining was verified with the **same saved data and manifest**,
script, NumPy version, seed, and single-thread BLAS. A fresh collection-to-fit byte comparison was not run; equality of fresh
samples, coefficients or thresholds is not guaranteed. Recollection also records
new host metadata, so the complete JSON provenance hash can differ even if
numeric samples and coefficients match under the recorded deterministic setup. Do not install OCR or copy
private teaching scans/notes into this package. Renderer output must identify
Mesa llvmpipe and an existing `ffmpeg`/`ffprobe` is required; no GPU is used.
The `--geometry-gate` flag is required for final L3; omitting it reconstructs
the rejected L1 candidate instead. The builder command above deliberately uses
the published frozen model, so it checks exact embedded-source reproduction
without replacing any accepted source or weight file.

## Environment and boundary

Same-PC recorded environment: Ubuntu 24.04, Python 3.12.3, MuJoCo 3.17.0,
NumPy 2.5.4, Matplotlib 3.11.2. Full package versions: results/ENV.md.
Obtain the course kit independently; this project contains no simulator/models.
Use a disposable work copy. Overlay our three final modules and tools without changing protected trees; create `results/` before evaluator output. For published artifact/README tests and media validation, also copy our `results/` and `artifacts/media/` into the work copy.

On a Python installation without ensurepip, the recorded installation method is:

```bash
env -u PYTHONPATH python3.12 -m venv --without-pip .venv
env -u PYTHONPATH pip3 --python .venv/bin/python install -r requirements.txt
```

Run from the kit work-copy root. Suppress inherited ROS paths and bytecode.
Check available RAM >=6 GiB before sweeps; keep total simulation workers <=4.
Do not interfere with unrelated processes. Integrity manifest covers the exact
external kit version used here; do not regenerate it to bless a different kit.

```bash
env -u PYTHONPATH PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m unittest discover -s tools -p 'test_*.py'
env -u PYTHONPATH PYTHONDONTWRITEBYTECODE=1 .venv/bin/python tools/production_audit.py
env -u PYTHONPATH PYTHONDONTWRITEBYTECODE=1 .venv/bin/python tools/sync_shared_core.py
env -u PYTHONPATH PYTHONDONTWRITEBYTECODE=1 .venv/bin/python tools/evaluate_checked.py --ctrl my_servo --detector my_detector --out results/evaluation_A.txt
env -u PYTHONPATH PYTHONDONTWRITEBYTECODE=1 .venv/bin/python tools/evaluate_checked.py --ctrl my_torque --detector my_detector --out results/evaluation_B.txt
env -u PYTHONPATH PYTHONDONTWRITEBYTECODE=1 .venv/bin/python tools/sweep.py --name reproduce-A --ctrl my_servo --detector my_detector --baseline --workers 4
env -u PYTHONPATH PYTHONDONTWRITEBYTECODE=1 .venv/bin/python tools/compare_baseline.py results/evaluation_A.txt results/sweeps/reproduce-A/results.json
```

Only deterministic displayed score cells are compared; host timing is not a
deterministic score. Two recorded final sweeps each matched all66 displayed cells.

## Historical v1 supplemental studies

With the v1 overlay described above, the fixed robustness design is results/D4_ROBUSTNESS_PROTOCOL.json, registered
before final collision tests. Each architecture runs385 episodes. Run A and B
sequentially (not concurrently with4workers each):

```bash
env -u PYTHONPATH PYTHONDONTWRITEBYTECODE=1 .venv/bin/python tools/d4_robustness.py --controller my_servo --workers 4
env -u PYTHONPATH PYTHONDONTWRITEBYTECODE=1 .venv/bin/python tools/d4_robustness.py --controller my_torque --workers 4
```

Presentation tools use only the external runner and independently written plots.
Use a separate temporary raw-output directory for the required --raw-dir and
--evidence-dir options. Do not add raw traces to the public repository.
The private parameter-perturbation runner is intentionally not distributed:
it duplicates the course runner for analysis. Its methods and aggregate results
are documented in D4_PERTURBATIONS; those runs are NOT official scores.

## Historical ablations

Owned experiments/d2_servo.py,d2_torque.py,d3_detector.py preserve pre-sharing
algorithms; control/detector wrappers import these historical files rather than
the final shared modules. Evaluate them with the same integrity-gated wrapper.
FAST calibration protocol and recorded training/holdout are included; the
corresponding historical source is experiments/d3_detector_fast.py.
Do not use collision labels to change thresholds or fit holdout cases.

## Public package check (no kit needed)

Media inspection requires a local `ffprobe`; add it to PATH or pass `--ffprobe`.
An unreadable media file or unavailable probe fails the check rather than skipping it.
On a public-only tree, four intentionally excluded course sources (the three
baseline modules and `sim/runner.py`) are reported as recorded provenance, not
locally authenticated bytes. All published owned-source hashes are mandatory.
The same validator in the disposable kit overlay additionally checks those four
external files against their recorded hashes. Do not copy them into publication.

```bash
FFPROBE="$(command -v ffprobe)"
test -n "$FFPROBE"
env -u PYTHONPATH PYTHONDONTWRITEBYTECODE=1 python3 tools/audit_public.py . --require-media --ffprobe "$FFPROBE"
env -u PYTHONPATH PYTHONDONTWRITEBYTECODE=1 python3 tools/validate_media.py --root . --ffprobe "$FFPROBE"
sha256sum -c artifacts/SHA256SUMS
```

Supplying --kit to the audit additionally checks for large verbatim Python
blocks against a read-only original. The package excludes the kit, guide,
contacts, machine paths, private workflow material, environments and raw logs.

## R2: 제어·리그룹 2차 실험 재현 범위

최종 R2 진입점은 보존한 R1과 byte-identical이다. 새25제어설정·pRNE의 음성 결과는 실험 파일로 남기고, 실패한 설정을 최종 진입점에 넣지 않았다. 기준/v1/R1/R2의 같은 수치는 독립 신규 구현 반복으로 세지 않는다. 최종 공개 검증 결과는 R2 보고서와 해당 JSON을 우선한다.

### 환경과 입력

- 공식 환경의 기존21패키지와 원본29파일, 작업 사본의 보호20파일을 유지했다. 공식 환경에는 패키지를 추가하지 않았다.
- 모든 Python은 PYTHONPATH를 제거하고 bytecode 생성을 막는다. 작은 ridge 학습은 별도 CPU 환경에서만 한다.
- 런타임은 문서화된 encoder/차분 속도/JTS/grip/reference/알려진 mass·COM/public nominal만 쓴다. 실제 table/tool 힘·링크각·센서 전 토크·잡음 seed/offset/drift를 제어 입력이나 학습 타깃으로 넣지 않는다.
- 학습 파일은 t/x/y/active만 담는다. 사후 물리 로그는 별도 외부 디렉터리에 보관한다.
- 실제 episode worker는1개, 전체 시뮬레이션≤4, MemAvailable≥6GiB, GPU 작업0이다. 다른 세션 프로세스를 조작하지 않는다.

### 비공식 방법 재현

원본 키트는 배포하지 않으므로, 별도의 로컬 작업 사본에 제공된 키트가 있어야 한다. 아래 PRIVATE_DIR는 공개 저장소 밖의 Git-ignored 위치다. 원본 자료나 sim/model 소스는 수정하지 않는다.

```bash
export R2_PRIVATE_EVIDENCE_ROOT="$PRIVATE_DIR"
env -u PYTHONPATH PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  .venv/bin/python tools/r2_p0.py --raw-dir "$PRIVATE_DIR/p0" --public-out "$PRIVATE_DIR/p0_summary.json"
env -u PYTHONPATH PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 \
  .venv/bin/python tools/r2_p0.py --raw-dir "$PRIVATE_DIR/p0" --snapshots-only
env -u PYTHONPATH PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 \
  .venv/bin/python tools/r2_stability.py --private-dir "$PRIVATE_DIR/p0" --output "$PRIVATE_DIR/stability.json"
env -u PYTHONPATH PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 \
  .venv/bin/python -m unittest tools.test_r2_prne tools.test_r2_contact tools.test_r2_robust
```

후보별 사전 설정은 PROTOCOL_R2.json, 실행 factory는 tools/r2_study.py에 있다. pRNE+ridge는 새 정상 모델·protocol·source SHA가 없으면 실행을 거부한다. matched ridge는 동일 정상 관측을 R1 관측기에 재생한 기여 분리용이며 새 채택 fallback이 아니다. 분석섭동은 private instance harness에 의존한다. 공개 코드만으로 그 private harness를 재구성했다고 주장하지 않으며, 원본 runner 대비8쌍의 nominal parity와 전체272개 결과를 함께 보존한다.

### 동결·holdout·공식 순서

사전 protocol 커밋 → 무충돌 검증/정상 보정 → 균형 development → source/weights 동결 → 새 holdout1회 → 원본 공식 평가의 순서다. source 동결은53파일이며, opening marker는 source와 로컬 Git object를 확인한 다음 RNG보다 먼저 배타적으로 만든다.

원래 실험의 holdout은 이미 열었다. 같은 seed를 다시 실행해도 독립 holdout이 되지 않는다. 새로운 선정 연구라면 별도 protocol·새 seed·새 동결을 먼저 만들어야 한다. 보존된 FREEZE의 source commit은 작업 저장소 이력의 기록이므로 공개 staging의 Git object와 같다고 가정하지 않는다. 실파일 SHA와 출처를 확인하고 자기 실험의 로컬 동결을 따로 만든다. 공식 결과를 본 뒤 같은 후보를 고쳐 재승격하지 않는다.

이번 마지막 공식 회귀는 기본 제출 파일명으로 바꾼 임시 키트에서 A/B 한 번씩 총58 episode를 실행했다. 원본 evaluate.py의 모든 비시간 셀이 R1과 일치했다. 공식 최대 계산시간이나 실기 안전성까지 검증했다는 뜻은 아니다. 이번 배치 뒤에는 추가 공식 평가를 실행하지 않았다.

### 영상·공개 검사

영상은 학습/선정/holdout과 별개인 비공식 예시다. 이미 있는 ffmpeg·ffprobe 실행파일을 지정한다. Mesa software GL이 실제 vendor/renderer 문자열로 확인되지 않으면 렌더링을 거부한다.

```bash
env -u PYTHONPATH PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 \
  MUJOCO_GL=glfw LIBGL_ALWAYS_SOFTWARE=1 GALLIUM_DRIVER=llvmpipe __GLX_VENDOR_LIBRARY_NAME=mesa \
  .venv/bin/python tools/render_r2_comparison.py --raw-dir "$PRIVATE_DIR/media" --ffmpeg "$FFMPEG"
env -u PYTHONPATH PYTHONDONTWRITEBYTECODE=1 \
  .venv/bin/python tools/validate_media.py --root publish/repo --ffprobe "$FFPROBE"
env -u PYTHONPATH PYTHONDONTWRITEBYTECODE=1 \
  .venv/bin/python tools/audit_public.py publish/repo --require-media --ffprobe "$FFPROBE"
```

공개 검사는 source/자료/연락처/경로·매체 메타데이터·SHA·8개 MP4/GIF 쌍을 확인한다. Python 행렬곱 @는 토큰으로 구분하지만 문자열·주석의 실제 연락처는 계속 거부한다.

## R3: DOB/P-IDC 원인 분석과 새 protocol

R3은 R2의 사후 재설계가 아니라 별도 사전 protocol의 새 라운드다. R2 validation에서 방법을 골랐다는 선택 편향과 이미 노출된 공식 조건을 숨기지 않는다. 최종 세 진입점은 R1과 바이트가 같고, A-long/B-mass는 탈락한 진단 후보일 뿐 최종 제출이 아니다. R1/R2 문서·PDF와 기존 영상은 보존한다.

### 완료한 비공식 검사와 기록 범위

- 원인 분석을 먼저 커밋한 뒤 PROTOCOL_R3.json을 고정했다. R2의 1.05배 비용, p99<800us, 국소 radius/FD·감지 관문을 완화하지 않았다. 새 validation28회, development104회, D4 전체1540회, 분석섭동136회와 별도 nominal parity8회를 직렬 실행했다. 같은 관측6000주기씩의 R2/R3 계산 replay는 플랜트 episode가 아니다.
- A2/B3설정과 별도 탄성 관절·1ms ZOH·한 주기 지연 분석이 실험 소스에 있다. 재설계 A0/B0의 그 시점 사유는 R3_REDESIGN_DECISION.json에 고정했다. 정상 감지기·가중치·임계값은 R1 그대로이고, 충돌/holdout에서 재보정하지 않았다.
- 최종101파일과 선정 근거를 커밋 동결한 뒤 새 holdout369903을 한 배치76회 열었다. 현재 기록은 이미 개봉된 조건이므로 재실행해도 새 독립 holdout이 아니다. marker를 지우거나 실패를 숨긴 재개는 허용하지 않는다.
- 1885개 계측 episode/11,310,000cycle의 시간은 R3_TIMING_REVIEW.json에 전수 집계했다. 비계측 parity8회와 별도 관측 replay12000cycle은 구분한다. 최대 row p99와 관측 최대를 혼동하지 않고 outlier·thread CPU·context-switch를 모두 남겼다. R1과 challenger의 전체 행 수가 달라, 속도 비는 동일6000관측 replay에서만 별도로 비교한다.
- 동결된 연구 집계기의 timing glob이 비계측 parity도 잡는 알려진 오류는 고치지 않았다. 명시적 입력 inventory를 쓰는 publication-only r3_report_manifest.py로 같은 측정치를 표현했다. 기존 R3_TIMING_REVIEW.json의 덮어쓰기는 거부한다.

### 공개된 값의 재검증

원본 키트는 별도로 제공되어야 한다. 제공된 키트를 복사한 임시 작업 폴더에 이 코드/결과를 overlay하고 다음 검사를 실행한다. 원본 sim/model이나 다른 실행 세션은 수정하지 않는다. 아래 명령은 시뮬레이션이나 이미 열린 공식/holdout을 재실행하지 않는다.

```bash
env -u PYTHONPATH PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES='' \
  .venv/bin/python -m unittest discover -s tools -p 'test*.py'
env -u PYTHONPATH PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 CUDA_VISIBLE_DEVICES='' \
  .venv/bin/python tools/r3_report.py --check
sha256sum -c artifacts/SHA256SUMS
```

보고서 manifest는 표현한 증거 파일의 SHA를 검증한다. 큰 JSON은 값 손실 없는 공백 축약, 프로세스 PID는 개수로 바꿨고 전후 SHA를 남긴다. 모든 조건·수치·pole·outlier는 보존했다. 원 연구 Git 객체를 요구하는101파일 동결 증명은 연구 작업 저장소의 기록이며, 다른 저장소의 Git 이력이 같다고 가정하지 않는다. 배포 파일 바이트는 별도 SHA256SUMS로 확인한다.

새 수집부터 전 과정을 독립 재현한 것은 아니다. 분석 전용 비공개 runner와 원시 trace는 외부 자료 구조를 포함해 배포하지 않으며, source SHA·AST parity·모든 조건/결과·원본 runner와의8회 nominal parity를 근거로 범위를 한정한다. 최초 비공식 smoke에서 Python 환경 prefix 누락으로 생긴 bytecode3개는 별도 증거를 남겨 제거하고 공식 venv로 재검증했다. 최종 원본29/보호20/기존R2동결53/환경21패키지 불변을 확인하며, 전 과정에 사건이 없었다고 주장하지 않는다. 실제 로봇·hidden 조건·Windows·전역/LTV/hybrid/수동성 보장은 미검증이다.

마지막 공식 회귀는 원본 기본 파일명 overlay로 A/B 각 한 번, 총58episode였고 두 방식 모두 PASS했다. 0/4/8kg 정상FA0/0/0·포화0·발산없음, 8kg 운반15N45/43ms와10N62/56ms를 유지했다. 집기 약한힘·놓기10N 미검출과 B운반FA1회는 남겼다. 공식 노출 뒤 과학 source/가중치/임계값/파라미터 수정이나 공식 재실행은 없다.

공개 검사에서 안정성 test fixture의 일반적인 zero-buffer 할당과 공개 compute 서명이 이어져 가이드10토큰 일치가 발생했다. 원 연구101파일과 audit 기준은 바꾸지 않고, 배포용 그 test 할당 한 줄에 괄호만 추가했다. 전체 Python AST가 동일함을 강제하고 공개 overlay의 전체 unittest도 다시 실행했다. 이 단일 test의 원 연구 SHA와 배포 SHA는 서로 다르며, 나머지 측정·제어·임계값 변경을 뜻하지 않는다.
