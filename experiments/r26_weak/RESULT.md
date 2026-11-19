# R26 weak-pulse lifecycle result

## Verdict

Frozen v1 is a **partial mechanism success and deployment rejection**.  It
recovers the missing R25 close-transfer short second without extending any
incumbent-owned event, but it mistakes two disclosed actuator-skew support
transients for unowned physical pulses.  The preregistered stop was honored;
no v2 thresholds or source model were fit.

Frozen entrypoint: `experiments.r26_weak.candidate:Detector`  
Reusable mixin: `experiments.r26_weak.candidate:WeakLifecycleMixin`

Candidate SHA256:
`25267785e3bed87f64ad2a81da75e261cdf6640fa8db0dc27faecee5e0635faf`

Lifecycle SHA256:
`b29f78c765760f702a64a95921be018b8b2da0a567f11bd0a649ddb3d1475ca4`

Freeze record SHA256:
`b44ef76babf2f77aa763ceb9a7db1dc0f36ce2bf3c5d0b58ad432466f533a602`

## Targeted results

- R25 stock weak signed carry: incumbent and candidate are identical—one rise
  at tick 2448, +35 ms, 73 active samples.
- R25 stock transfer cross short second:
  - isolated: gained one rise at tick 1753, +60 ms, 8 samples;
  - together: gained one rise at tick 1750, +57 ms, 8 samples;
  - long and first events retain exact onset and occupancy.
- S24 transfer off-axis mid: incumbent has zero rises; candidate gains one rise
  at tick 1613, +125 ms, 8 samples.
- Old late +/- panel: 16/16 rows pass and are false-free.  Every incumbent
  event has exactly zero delay delta and zero active-sample delta; occupancy is
  40–72 samples.
- Prior 3.855 s controls: only 4/6 rows pass.  False rises remain in
  `p23_post_detach_diagonal_pair` and `s23_post_detach_radial` under
  `actuator_skew`.

## Diagnostic distinction

The rejected nuisance pulses last 10–15 samples and have table support for
every sample: table normal 34.35–47.97 N and valid support throughout.  The
real transfer-cross short-second pulses last 23–26 samples with zero table
normal and no admissible/valid support.  S24 mid has support during 16/55
samples, but table normal is zero at onset, peak, and end and never exceeds
4.32 N.  `hybrid_alpha` is zero in every compared pulse and is not useful.

This is evidence for a future, separately preregistered physical-source model;
it was not used to change v1.

## Reproduction

From the repository root:

```sh
source "$E/env.sh"
export PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1

.venv/bin/python -B -m unittest experiments.r26_weak.test_lifecycle -v

.venv/bin/python -B -m experiments.r26_weak.targeted \
  --plan experiments/r25_holdout/plan.json \
  --out "$E/r26/weak/reproduced-targeted.json"

.venv/bin/python -B -m experiments.r26_weak.late_panel \
  --r24-plan experiments/r24_holdout/plan.json \
  --weak-plan experiments/r25_weak/PLAN.json \
  --out "$E/r26/weak/reproduced-late.json" --workers 2

.venv/bin/python -B -m experiments.r26_weak.diagnose_sources \
  --r23-plan experiments/r23_holdout/plan.json \
  --r24-plan experiments/r24_holdout/plan.json \
  --r25-plan experiments/r25_holdout/plan.json \
  --out "$E/r26/weak/reproduced-diagnostic.json"
```

Use fresh output paths.  Raw results are under `E/r26/weak/`, notably
`targeted-v1.json`, `s24-target-v1.json`, `late-v1.json`,
`controls-3855-v1.json`, and `source-diagnostic-v1.json`.
