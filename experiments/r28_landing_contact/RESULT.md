# R28 landing-contact result

Verdict: **rejected on a disclosed known protection loss**. Neither the frozen
`f764e6e...` detector nor the finite-input adapter `9402c6d...` is eligible for
a fresh challenge or promotion.

The physical contact veto did remove the four known V3 loaded-normal false
edges. On the complete disclosed 36-episode R27 holdback it changed false rises
from 6 to 2, added no flags, did not change active duration, and retained 6/8
strict event slots. Two absolute failures in that panel are inherited behavior,
not new R28 regressions.

The decisive counterexample is the disclosed 8 kg, centered-COM, +Y 15 N
pickup at 1.42 s. With byte-matched mechanics, force, and trajectory, V3 rises
at sample 1464 (+43 ms); C1, f764, and the finite adapter do not rise. At sample
1456 the observed table normal is 1.619685 N, above the fixed 0.5 N air veto.
That loss of ownership resets and disarms the R28 optional episode. Ownership
returns only after horizontal force has passed its release/rearm level, so the
later two-sample threshold crossing cannot reacquire. This is a causal loss
caused by applying the air veto to a resolved 8 kg load, not a numerical-NaN
artifact.

The finite adapter correctly fails closed on non-finite observations but is
otherwise identical to f764 on finite waveforms, so it retains the loss.
Official-58 and any fresh R28 challenge were intentionally not run after this
known falsifier failed. Detailed receipts are under `E/r28/landing/finite/`,
especially `verdict.json`, `pickup.json`, `pickup-diagnostic.json`, and
`holdback-v3/result.json`.
