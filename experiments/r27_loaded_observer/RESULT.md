# R27 loaded observer — bounded failed research result

No R27 detector qualified.  The final frozen research module is
`experiments.r27_loaded_observer.candidate:Detector`, SHA-256
`6c24a8d1944e3c5c23fd2c6a24f1b2afce1054c4817ec443cf5513bb21f87d4b`.
It is retained to reproduce the failure and must not be promoted.

## Normal-only development

The data comprise eight fit and eight disjoint audit payload/COM/offset cases,
each under two separate mechanics.  Every episode is force-free.  The split was
locked before outcomes and the audit never changed parameters.

1. A global 8/32 ms change bound failed before forced testing: normal acquisition
   and placement produced 40.74/62.82 N fitted thresholds.
2. Observer-effective payload-settle and table-geometry gates reduced those to
   20.74/17.40 N, still above the intended weak-force class.
3. The selected research proposal fitted signed normal horizontal change from
   `[1, vertical change]` and detected only the unexplained lateral residual.
   Its thresholds were 6.420 N at 8 ms and 5.529 N at 32 ms.  It produced no
   candidate rise in 16 fit or 16 disjoint audit normal episodes.

The candidate embeds its fitted parameters, performs no runtime file I/O, uses
only the public encoder/JTS/grip models and public table-geometry state, and
subclasses frozen R26 C3.  Continuous telemetry owns rearming only; output is an
eight-sample pulse and is suppressed while native R26 is live.

## Frozen disclosed test

After the independent R27 challenge was confirmed presealed, but without opening
it, the frozen candidate ran on the two disclosed R26 misses and the disclosed
detected comparator, across stock, actuator-deadzone, and surface-shift mechanics.
R26/R27, authentic none/forced yielded 36 episodes.

- Both weak targets remained undetected in all six mechanics rows: strict 0/6.
- No target duplicate or no-force false rise was added.
- All three comparator events were exactly R26, including first rise and active
  duration.
- All nine none detection traces and all force, causal-prefix, and physical
  comparisons were exact.

The target gate therefore failed, and the candidate was not changed.

## Root cause and finite next step

The 8 ms onset source never crossed its 6.420 N normal-only bound.  In the stock
target windows it peaked at 3.585 N (carry) and 4.740 N (place); no sample was
above threshold, so pending/sustain logic never ran.

A fixed post-failure diagnostic found real separation at the already fitted
32 ms residual: carry was 7.141 N forced versus 3.134 N none, with ten
consecutive above-threshold samples; place was 8.050 versus 4.595 N, with 23.
This establishes an onset-timescale mismatch, not a missing force or runtime
gate failure.  It is not a retroactive candidate.

After three failed mechanisms, this lane stops.  A future independent design
should preregister the 32 ms residual as onset and obtain ownership/sustain rules
from new normal-only data.  It cannot simply reuse 5.529 N globally: the disjoint
normal audit had isolated 32 ms residuals up to 5.758 N.

The independent challenge was never inspected.  Pregrasp-tool and wider legacy
panels were not run after the primary falsifier and circuit breaker.  Full hashes,
counts, limitations, and machine-readable results are in local evidence
`r27/observer/summary.json`.
