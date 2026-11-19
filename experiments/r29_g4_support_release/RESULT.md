# R29 G4 support-release result

Verdict: **the single preregistered attempt is rejected**. No R29 detector from
this lane is eligible for promotion or a prospective run.

## Causal diagnosis

The inherited rises at 3903 and 3923 are both C1 counterfactual-route onsets in
C1 and frozen observation-memory. The first is the slow route (`cf_alarm=4`,
`cf_fast=0`); the second is the fast route (`cf_alarm=0`, `cf_fast=8`). Neither
is produced by the R27/R28 optional branch.

Offline, box/table reaction coincides with both rises, but the gripper is open;
that does not establish a force on the arm and was not used by the detector.
Online, table-wrench eligibility and controller support validity are absent.
The fresh released-vector observer is also unavailable: its empty-model window
contains only 1/31 residual samples at 3903 and 21/31 at 3923 after the measured
grip/model switch.

A separately continuous empty-arm encoder/JTS inverse observer is finite. It
measures `[-3.243, 6.179, -31.360] N` at 3903 and
`[-2.338, -2.256, 0.028] N` at 3923. This supported one frozen attempt to own a
negative-z unloading episode and at most one unsettled CF rebound using the
pre-existing R26 signed bounds, without time, case, mass, contact, or actual
force inputs.

## Outcome

The mechanism removed exactly 3903 and 3923, retained the known 0 kg/8 kg +Y
delays (+38/+43 ms), and added no normal-control edge. It nevertheless failed
two prespecified protections:

- A genuine matched quiet-state `-Z 15 N` impulse at 3.88 s was completely
  missed: observation-memory reported one strict attributable rise at +130 ms
  for 79 samples; R29 reported none.
- A genuine late post-release `-Z 15 N` impulse retained its +127 ms rise but
  active duration grew from 8 to 68 samples, violating the +5-sample guard.

The same-phase negative-z control therefore supplies a physical
counterexample: with the allowed public signals and this signed inverse
certificate, genuine negative-z force is not safely distinguishable from the
proposed unloading source. The incumbent alarms remain active; no threshold
retune or second cutoff was attempted.

Full 54-episode records, exact force, physical hashes, prefixes, absolute
misses, strict/false/duplicate/delay/duration metrics, and online state are in
the private `result.json` and `verdict.json` receipts.
