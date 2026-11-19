# R31 event channels

Status: **portable read-only telemetry adapter verified**.

`experiments.r31_event_channels.adapter:Detector` subclasses the frozen held
detector and returns its residual and Boolean without modification. After each
update it exposes an immutable `EventChannels` record and appends the record to
`channel_log`.

The separate channels are:

- final public output level/rise;
- optional notice envelope level/rise;
- raw native-evidence level/rise and active route names;
- route-completeness/native-consumed metadata; and
- source-present and source-linked-native flags when attribution is known.

The notice is **not contact duration**. Native evidence is **not physical force
truth**. This adapter is not a safety stop, controller change, new entrant, or
repair for the failed occupancy gate.

## Disclosed replay verification

Frozen R31 and the adapter were replayed on D4 rows 266, 287, 298, and 316 and
their exact registered no-force twins (16 episodes). Every physical array,
public residual sample, Boolean sample, and output interval was identical.
Every no-force twin had empty notice/native/output channels.

Examples that show why channels must remain distinct:

- row 266: four-tick notice `[2729,2733)`, followed by source-linked native
  activity `[2740,2834)`;
- row 287: four-tick notice `[3127,3131)`, followed by source-linked native
  activity `[3135,3252)`;
- row 298: four-tick notice `[3422,3426)`, later native activity
  `[3428,3561)` plus an independent native/output interval `[3818,3862)`;
- row 316: native/output activity without an optional notice.

The first verifier version compared `ctrl_time_us` and was preserved as a
timing-only diagnostic failure. The corrected verifier ignores only this
observational wall-time field; all scientific metrics remain exact.

## Portable stage files

- `__init__.py`
- `adapter.py`
- `test_adapter.py`
- `RESULT.json`
- `RESULT.md`

`verify.py` is the authenticated local replay helper and depends on the
separately supplied course simulator/kit; it is not required by the runtime
adapter.
