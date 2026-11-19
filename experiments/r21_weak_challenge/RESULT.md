# R21 independent weak-horizontal challenge

Status: **both frozen candidates pass all retention gates; the yaw-balance witness
has the stronger independent gain. Neither is a default submission here.**

The case plan was frozen before either candidate saw these outcomes. It contains
12 early weak impacts (6–15 N, ±x/±y/diagonal, 0–8 kg), one stronger early
control, two impacts outside the candidate phase, and three no-impact controls
including lateral offset, a real tool contact, and zero payload. Every forced
case has a same-arm no-force twin.

## Final v3 result

| arm | strict early weak | all forced strict | candidate-only weak gain vs R20 | lost R20 strict |
|---|---:|---:|---:|---:|
| R10 | 8/12 | 11/15 | — | — |
| R20 horizontal | 8/12 | 11/15 | — | — |
| R21 pick observer | 9/12 | 12/15 | 1 | 0 |
| R21 yaw witness | **10/12** | **13/15** | **2** | 0 |

The pick observer newly detects the 6.7513 kg, −y 8 N impact at 72 ms. The yaw
witness detects it at 37 ms and also detects a 4.5785 kg, −x 7.5 N impact at
46 ms. The no-force twins are quiet at the credited onset; these are not
relabels of existing normal rises. Both candidates still miss the weakest 6 N
+x and 6.5 N +y cases.

All 15 forced cases passed exact pre-impact prefixes, exact applied signed/scalar
force, cross-arm physical equality, no added unmatched false rise or prealarm,
no lost tool event, no stuck tail, and no lost R10 strict positive. The
205.96 N offset-induced real tool contact remained detected. All three
no-impact controls added zero false rises.

The two independent candidate runs repeated every R10 and R20 case record
exactly. Candidate/model bytes matched their pre-outcome pins before and after
198 final episodes. The original course overlay retained all 20 protected
sim/ and model/ files.

## Scorer correction retained in the audit trail

An initial 132-episode run failed every preimpact_prefix_exact check because
the first scorer accidentally compared the no-force and force logs over all
6,000 samples, including the future force. It did not alter simulation or
policy output. That result is retained as invalid evidence. The v3 scorer
changes only this comparison to [:collision_index], adds a regression test,
is re-pinned, and reruns every final episode. No case, policy, force, threshold,
or other decision rule changed.

## Reproduction interface

Inside an authenticated external course overlay:

    python -B -m experiments.r21_weak_challenge.evaluator       --plan experiments/r21_weak_challenge/plan.json       --out /path/to/new-pick-result.json       --arms r10,r20,r21_pick       --candidate-pin 30e56f7ac752c6072ef88869378147be3534484b097ee23044c328f787f50364

    python -B -m experiments.r21_weak_challenge.yaw_driver       --plan experiments/r21_weak_challenge/plan.json       --out /path/to/new-yaw-result.json

verify.py crosschecks both outputs, source pins, cohort completeness, comparator
repeat identity, every case gate, and the aggregate counts. This remains finite
CPU simulation evidence, not hardware safety or hidden-test proof.
