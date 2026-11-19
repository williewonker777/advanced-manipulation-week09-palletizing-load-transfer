# R32 servo source ownership — rejected

Status: **rejected / no enrollment**.

The frozen candidate keeps the R31 dual-observer weak notice but treats it as
one owned physical source. After that source completes, it cannot rearm until
both public horizontal observers remain within the unchanged 5 N resolution
for two complete 31-sample observer windows. Invalid observations or renewed
force reset settlement. Native R10 remains an unconditional pass-through and
owns a pending source.

This is not a task-clock cooldown: rearm requires measured two-channel normal
settlement. The 62 samples were fixed from two full observer windows before
simulation output, not selected from R31 prospective results.

## Known and official

- two lifecycle unit tests passed;
- 16 disclosed R10/R21/R31 known executions passed;
- authentic official 58-episode comparison passed;
- R10 remained 23/24 and R32 reached 24/24;
- the only official collision change was the disclosed empty 3.86 s +Y10 miss
  becoming a 131 ms detection;
- no-impact false rises remained zero and controls were identical.

## Disjoint validation

Seed 320032 fixed eight signed single-force/calm twins and four genuine
none/first/second/both force quartets before output (64 episodes). All rows were
retained.

- every single was strict for both R10 and R32: 8/8 versus 8/8;
- all four isolated-second events remained present in the corresponding both
  trace for both arms;
- no duplicate edge or true-event loss occurred;
- all quiet `none` twins were silent;
- R32 produced **zero candidate-only gains** over R10.

The validation runner reported eight false entries because the stock simulator
metric knew only about the first collision argument and therefore labeled the
separately instrumented, signed second force as false in `second` and `both`
traces. The retained edge/force rows show one edge inside each registered
second-force window and zero quiet-twin edges. This scorer limitation does not
rescue the candidate: the preregistered success rule required at least one
actual R10 miss recovered in the disjoint panel, and there were none.

R32 is therefore an interesting ownership implementation but not an eligible
user-facing opt-in module. No retune or additional panel was attempted.

## Pins

- candidate: `6c5cfca6b6f8d387a6b2c22a1ca41bf20c3f6df6f238add21f01a8f2b4c88d7f`
- method plan: `d0ac597bcc70454d440af7727b02f42f4359d5a45f9e023d2bd7a142a12b02dd`
- validation plan: `3227b9dee40e9f8a0432266c71248164a63ac1f7a8464445ee96b493c9a4177c`
- validation runner: `204daaada25681f1ceb9d533388a24eeba09a5c602987c8215d156b28cf241f0`

The first validation registration contained JSON-only leading-dot syntax and
failed before importing a detector or running a simulation. Its log and hash
were preserved; the corrected JSON was re-frozen before the sole physical
validation run.
