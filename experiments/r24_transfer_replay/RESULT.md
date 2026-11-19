# R24 authenticated external-kit replay

## Verdict

The frozen R24 controller was replayed twice from newly materialized, kit-free
public source snapshots against a separately authenticated stock kit. The three
known rows reproduced their pinned physical hashes, support metrics, false-event
times, and G4 results exactly. The preregistered new challenge **did not pass in
full**, so this is not promotion evidence and no controller, detector, case,
force, mechanics setting, or gate was retuned.

## Distribution and execution boundary

- Runtime authored closure: 42 byte-pinned relative files.
- Selected public snapshot: those 42 files plus `closure.json` (43 total).
- The public snapshot contains no `sim/`, `model/`, or `evaluate.py`.
- The CLI requires an external authenticated K and builds a temporary overlay.
- No network, public-tree write, stock-kit write, hardware action, or remote
  operation occurs.

## Known reproduction

`sh5`, `heavy8_offset`, and independent `ec3:up` all passed exact replay pins.
Their R24 G4/support/false-event outcomes match the frozen R24 report:

- `sh5`: R19 `[3.903]`, R24 `[]`; G4 pass.
- `heavy8_offset`: R19/R24 `[3.917]`; G4 pass.
- `ec3:up`: R19/R24 `[3.919]`; G4 pass.

All inherited R10 events remain disclosed in `RESULT.json`; they were not used
as a reason to relabel the R19/R24 result.

## Predeclared new normal cases

The interpolated-mass, shifted-COM, and 10%-softer-table-contact rows all passed
G4, cost <=1.05 against R10/R19, saturation, and R24 false-not-above-R19
reporting clauses. The softer-mechanics row inherited the R19/R24 `[3.919]`
event; it is reported, not suppressed.

## Predeclared signed late impulses and twins

Every force trace exactly matched the programmed signed trapezoid and every
forced/twin physical prefix was exact.

- `+X 12 N` and `-X 12 N`: R10, R19, and R24 all failed to produce a genuine
  event. Nonloss is therefore vacuous; these are explicit inherited and
  candidate absolute true-detection failures.
- `+Z 16 N`: all three arms detected at `38 ms`.
- `-Z 16 N`: R10 `141 ms`, R19 `129 ms`, R24 `145 ms`. R24 retained the event
  and remained within R10+5 ms, but failed R19+5 ms by `11 ms` (`16 ms` slower
  than R19). This single strict comparator failure makes the new challenge
  result false.
- No forced or twin false event was added versus either comparator. The shared
  no-force twin still contains inherited R10 rises `[1.283, 3.905]` and
  R19/R24 rise `[1.283]`; all are preserved in the compact result.

## Reproducibility

Two fresh external-kit executions reproduced all scientific fields exactly,
SHA-256 `08de6fb59184fead67065f4f29483d8c87d649ac0d1b4af85e0cd5bb18e363a6`.
Only the declared host wall-clock fields `ctrl_p99_us` and `stream_timing_us`
varied.
