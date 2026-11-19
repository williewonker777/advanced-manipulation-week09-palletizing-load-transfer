# R26 robust support-origin certificate result

## Verdict

The preregistered certificate is **frozen and rejected**.  It preserves the
useful weak-pulse gains, but the full worst-case Coulomb disk is not large
enough to explain the two 3.855 s actuator-skew nuisances.  No threshold or
hard contact mask was added after evaluation.

Standalone entrypoint:
`experiments.r26_support_certificate.candidate:Detector`

Composable mixin:
`experiments.r26_support_certificate.candidate:CertifiedWeakLifecycleMixin`

Candidate SHA256:
`6c34b05f2d481472b9fc6c244e02ce09dad921ea9e91c557d3a5818c5f632555`

Certificate SHA256:
`20e50940627d5afc5db6d5e878b61a35b6b20d4944cb97e3636cf2be14208ad8`

Freeze record SHA256:
`a32f90e040be03f3c086d3b9c5dcb84f60bdbbb4d9ec3523e6fd0cc58a536921`

## Targeted results

- S24 mid: gained one strict +125 ms, 8-sample event.
- R25 weak carry: preserved the incumbent +35 ms, 73-sample event exactly.
- R25 transfer cross: gained the isolated and joint short second at +60 ms
  and +57 ms, both 8 samples; long/first events are unchanged.
- Prior 3.855 controls: only 4/6 rows false-free.  Pair and radial
  actuator-skew nuisances remain false.
- Old late +/-: 16/16 pass and are false-free; every event record is identical.
- Matched +/- support-contact panel: measured table normal reaches 25.1 N and
  the lag-window certificate uses up to 31.6 N.  The negative sign preserves a
  strict event exactly; the positive sign preserves an already fragmented
  three-rise incumbent record without adding a gain or regression.
- `s25_long_transfer_diag`: baseline has zero attributable rises; candidate has
  three rises and two duplicate edges.  Three same-direction lifecycle episodes
  are separated by only 10–12 ms after their prior end.  This is a pulse-identity
  fragmentation problem, not uncertain support origin.

## Reproduction

```sh
source "$E/env.sh"
export PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1

.venv/bin/python -B -m unittest \
  experiments.r26_support_certificate.test_certificate \
  experiments.r26_weak.test_lifecycle -v

.venv/bin/python -B -m experiments.r26_support_certificate.targeted \
  --r23-plan experiments/r23_holdout/plan.json \
  --r24-plan experiments/r24_holdout/plan.json \
  --r25-plan experiments/r25_holdout/plan.json \
  --disclosed results/r24/disclosed.json \
  --out "$E/r26/support/reproduced-targeted.json" --workers 2

.venv/bin/python -B -m experiments.r26_support_certificate.guards \
  --r24-plan experiments/r24_holdout/plan.json \
  --r25-plan experiments/r25_holdout/plan.json \
  --weak-plan experiments/r25_weak/PLAN.json \
  --contact-panel experiments/r26_support_certificate/contact_panel.json \
  --out "$E/r26/support/reproduced-guards.json" --workers 2

.venv/bin/python -B -m experiments.r26_support_certificate.diagnose_long \
  --plan experiments/r25_holdout/plan.json \
  --out "$E/r26/support/reproduced-long-diagnostic.json"
```

Use fresh output paths.  Frozen raw evidence is under `E/r26/support/`.
