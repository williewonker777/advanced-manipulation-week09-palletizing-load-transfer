# R31 single counterfactual — FAIL

Status: **scientific FAIL; no enrollment, no retune, no third panel**.

The earlier 15-case registration remains closed as
`INVALID_INFRASTRUCTURE` with zero released case outputs. This independent
study first validated the generic arbitrary-onset prefix method on a synthetic
fixture and on disclosed D4 case 330/reference 39. Generic and native
historical scores, prefixes, and physical signatures matched exactly for C3
and R30 headroom.

Seed `310031` then fixed 12 disjoint loaded-transport collisions and their 12
exact calm twins before any novel output. The three source-pinned arms produced
72 retained episodes.

## One-shot result

- all 12 collisions were strict for C3, R30 headroom, and R31;
- all inputs, signed forces, prefixes, and cross-arm physics passed;
- every calm twin was false-rise free;
- R30 produced one duplicate on `s06` and `s07`; R31 removed both and retained
  strict detection;
- R31 introduced no false, delayed-false, duplicate, mask, tool-prealarm,
  stuck-tail, detection, or latency regression;
- the predeclared primary-duration gate failed against C3 on five cases:
  `s01`, `s05`, `s06`, `s07`, and `s10`.

On those rows R31/R30 emitted a four-sample early optional interval while C3's
corresponding primary interval lasted 67, 78, 112, 106, and 116 samples. The
shortfalls were 63, 74, 108, 102, and 112 samples, all beyond the fixed
five-sample allowance. Thus the candidate removed the exercised duplicates
but did so with a protected occupancy loss, so the frozen verdict is FAIL.

## Pins

- candidate: `1e33b4698ef3678afe72cc4e286fcab5d27b2232e7bf087fdef7500ccc674764`
- plan: `23b99746576c8b35f9d6254655194c020f503624b0ed46389549c1c3cd5b630d`
- scorer: `d2a6a380d3bf8144a491ba220c0a068869ca6f5c0707b635290cefcc812f46eb`
- runner: `ba7d7c4c6e024fa17c8d2ab6e74fe4a4aadf417a3bdb5cd60b4fcce4d9198723`
- generator: `59dce36a0b74ad3e27dc84854fff13566a6f9f2f4f3a7c76850e546e49fa61fc`
- preflight receipt: `2ce24c4ae008524fc81d314e016a908b105ce11a9125220a8fe3101d585460b2`
- result: `e2236c93017546969a82454f0dd16eaf12740aea111af6827034231c76664013`

## Reproduction

The plan is single-use research evidence. A fresh-path technical replay is:

```bash
. "$E/env.sh"
cd "$R"
export PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
.venv/bin/python -B -m experiments.r31_single_counterfactual.preflight --out /fresh/preflight.json
.venv/bin/python -B -m experiments.r31_single_counterfactual.run --workers 2 --out /fresh/result.json
```

This command is documentation, not authorization to create another selection
panel or revise the frozen verdict.
