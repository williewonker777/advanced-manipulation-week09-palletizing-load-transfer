# R21 support-aware yaw result: rejected

## Frozen mechanism

The candidate retains the plain R10 detector and adds the already frozen cyclic
joint-1 yaw observer.  It subtracts only a symmetric interval that the current
R10 table-support projection can physically explain:

`|tau_z| <= Fz * (mu * (||p_xy|| + box_half_diagonal) + torsion_radius)`.

The interval is enabled only while gripping and `last_table_allowed` is true.
It uses R10's projected `Fz`, `friction_mu`, box COP dimensions, torsion radius,
and nominal projected point.  It has no force-trained threshold, payload rule,
prior-hit timer, refractory period, or change to R10 alarms.  The unchanged
normal-only servo bins and 3-set/8-clear rule remain the decision envelope.

The design, source pins, four protected twin cases, positive-retention guards,
and two unopened reserved cases were frozen in `plan.json` before any force
outcome.  See raw `r21/support_yaw/frozen.sha256`.

## Protected result

`protected.json` contains 32 episodes: R10 and candidate, each over
none/first/second/both twins for four cases.  All committed signed force
profiles and plant traces matched exactly between detector arms.  Both actual
near-release second impacts remained detectable in every candidate case, and
all R10 genuine events were retained within 5 ms.

However, the central repair failed:

- after the exposed 8 kg, 100 N +y prior impact, the first-only candidate still
  raised at 3.872 s;
- after the exposed 7.15 kg, 40 N -y prior impact, it still raised at 3.866 s.

At those rises, the physically derived support intervals were only 1.970 and
1.638 Nm, leaving unexplained yaw magnitudes of 4.720 and 5.624 Nm.  The maximum
intervals from 3.84--3.90 s were 2.592 and 2.395 Nm while measured yaw reached
8.619 and 8.647 Nm.  These values are in `exposed_diagnostic.json`.

Two other protected cases passed, including the zero-mass 10 N early and
near-release positives.  The zero-mass near-release positive remained an
actual candidate-only rise, demonstrating that the implementation is not a
blanket release kill.  That does not offset the two required exposed failures.

## Decision

**Reject.** The known physical support interval is too small to explain the
delayed yaw transients.  Increasing friction, lever arm, or torsion solely to
clear them would be force-outcome retuning rather than the preregistered
mechanistic test.  The protected stage failed, so the stage-gated non-axis
reserved cases and full official evaluator were intentionally not opened.
No prior candidate or default module was changed.
