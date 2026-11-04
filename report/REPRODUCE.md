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
