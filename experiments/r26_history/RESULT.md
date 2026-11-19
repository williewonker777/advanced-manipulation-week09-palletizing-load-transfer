# Frozen R26 long-event historical guard

## Verdict

The frozen candidate
`experiments.r26_long.candidate:Detector`
(`a0570079ad5ce8425e9fbdc291b20401a9056b3bca35824237d157ec37a73057`)
**fails historical protection**.

The original torque evaluator passed exactly, but the complete D4 matrix found
one duration regression.  Per the requested fail-fast rule, legacy66, weak, and
pairs were not started after this concrete loss.

## Coverage

- Runtime: Python 3.12.3, MuJoCo 3.17.0.
- Original torque evaluator: fresh R25 and R26, 29 episodes per arm, 58 total.
  All 24 collision rows have zero delay delta, zero active-duration delta, and
  zero false-alarm delta.  No strict gains or regressions.
- D4: complete 385-case matrix, every R26 episode fresh.
- Historical R25 archive authentication: fresh R25 case 0 full record exactly
  matches the archived record; fresh R26 physical hashes also match.  Matrix
  SHA256 is `bc4b3dbb925cfb8609f535e1420f8b3f20785769b7f339708ca914b6992db791`.

## Verified regression

D4 case 299:

- controller/detector: `my_torque` / `my_detector`
- mass: 6.25 kg
- COM: `[0.03, 0.0, 0.015]`
- collision: transport at 1.9 s, +x 100 N
- R25 and R26: strict, +19 ms, zero duplicate/false/mask/tool/tail
- physics, signed force, and prefix: exact
- R25 interval: `[1919, 2085)`; true-window occupancy 166 samples
- R26 interval: `[1919, 2250)`; true-window occupancy 301 samples
- regression: +135 samples versus the allowed +5

This is the sole D4 gate failure.  There are no D4 strict gains and all scored
event classifications are otherwise unchanged.

## Reproduction

```sh
source "$E/env.sh"
export PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1

.venv/bin/python -B -m experiments.r26_history.official \
  --out "$E/r26/history/reproduced-official.json" --workers 2

.venv/bin/python -B -m experiments.r26_history.d4 \
  --archive "$E/r25/composed/d4" --preflight \
  --out "$E/r26/history/reproduced-d4-auth.json"
```

Run `experiments.r26_history.d4` in candidate shards, then call it with
`--auth`, `--chunks`, and a fresh `--out` for full comparison.  Existing raw
shards and the compact stop receipt are under `E/r26/history/`; the receipt
SHA256 is `2e43a0848a5263bf2a6a64c5de7ff540d2a256197f20c0c1325c7f5f363d633d`.
