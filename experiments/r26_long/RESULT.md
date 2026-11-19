# R26 long signed-force event identity

## Result

**PASS on all 99 already-disclosed R25 candidate episodes; no independent R26 claim.** The frozen candidate is `experiments.r26_long.candidate:Detector`, SHA-256 `a0570079ad5ce8425e9fbdc291b20401a9056b3bca35824237d157ec37a73057`.

The target is `s25_long_transfer_signed`: 3.77 kg, COM `[-0.024,0.013,-0.006]`, one 180-sample force `[-6,0,+2.3] N` physically active from ticks 1624 through 1803. Frozen R25 reported it three times on stock (`+60/+84/+154 ms`) and twice on the two shifted mechanics.

## Root cause

Per-source reproduction proves every target edge comes from **yaw only**. R10, learned normal, horizontal, light, detached, pickup, and continuation routes remain inactive. The yaw signal never reverses; it fluctuates around the frozen bound, so the inherited three-sample onset / eight-sample clear latch repeatedly drops and reacquires one continuous force:

- stock: `1684–1696`, `1708–1747`, `1778–1813`;
- actuator shift: `1684–1743`, `1762–1808`;
- contact shift: `1675–1747`, `1767–1803`.

This is an event-identity failure, not multiple impacts or a cross-route handoff.

## Minimal causal fix

`SignedYawOwnership` can start only after the inherited yaw detector reports. It learns two public sensor estimates during the event: signed yaw momentum and the independent learned horizontal residual. At a yaw-route clear, both current measurements must be nearer their learned event means than zero. Only then does signed yaw retain ownership until its sample becomes nearer zero. Sign reversal, grip/model switch, nonfinite data, or R25 pickup completion releases ownership immediately.

The mixin calls frozen R25 exactly once and ORs only the admitted ownership bridge. It contains no case/time/impact/label branch, force truth, schedule gate, cooldown, global threshold change, or scorer relaxation.

## Target and protection

Final target rises are:

| mechanics | frozen R25 delays | R26 delays | duplicates |
|---|---:|---:|---:|
| stock | 60, 84, 154 ms | 60 ms | 0 |
| actuator shift | 60, 138 ms | 60 ms | 0 |
| contact shift | 51, 143 ms | 51 ms | 0 |

Exact frozen-byte replay covers 99 disclosed R25 episodes: six target twins, 60 paired episodes, and 33 remaining negative/single episodes. All physical and applied-force hashes match. All 54 previously strict slots remain strict; maximum latency change is 0 ms and maximum protected duration increase is 3 samples against the +5 floor. There are zero new false rises, duplicate rises, or stuck tails. The R25 pickup repair remains exactly 60 ms / 6 samples / one edge. Corrected short-second rows and every five-slot opposed-close row pass across all three mechanics.

## Retained failed trial

A yaw-only nearest-event owner repaired the target but extended `stock:p25_close_transfer_cross` from 45 to 53 active samples, violating the +5 floor. At its clear tick, actual force was already zero: yaw remained event-like, while the independent residual was closer to zero. Requiring redundant agreement only at the ownership handoff removes that failure without changing target results.

## Scope and evidence

- `plan.json`: fixed before candidate trials.
- `diagnosis.json`: exact frozen reproduction and per-source evolution.
- `protection-v1.json`, `v1-regression-probe2.json`: retained rejected hypothesis and counterexample.
- `target-final.json`, `protection-final.json`, `remaining-final.json`: exact frozen-byte final replay.
- `candidate.freeze.json`: immutable source and input pins.

Future R26 independent conditions, R23/R24/nearby/D4 panels, promotion, packaging, and default changes are outside this lane.
