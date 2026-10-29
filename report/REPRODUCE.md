# Reproduction

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

## Supplemental studies

The fixed robustness design is results/D4_ROBUSTNESS_PROTOCOL.json, registered
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
