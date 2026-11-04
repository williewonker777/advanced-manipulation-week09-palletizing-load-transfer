import copy
import json
import os
from pathlib import Path
import tempfile

import numpy as np

from tools.r2_stability import (
    ControllerMemory, _jsonable, make_controller,
    compact_candidate_result,
    replay_candidate_operating_point,
)


ROOT = Path(__file__).resolve().parents[1]
PRIVATE = Path(os.environ.get(
    "R2_P0_DIR", ROOT / "results/v2_protocol/_private_p0_missing"
))


def snapshot(architecture="torque"):
    data = json.loads((PRIVATE / f"{architecture}_0.85kg_stability_snapshots.json").read_text())
    return data["snapshots"][1]["before_compute"]


def test_controller_memory_round_trip_and_declared_shape():
    before = snapshot("torque")
    controller = make_controller("torque", 0.85, (0.013, -0.009, 0.019))
    memory = ControllerMemory("torque", before)
    packed = memory.pack_snapshot()
    memory.restore(controller, packed, 2.5)
    assert packed.size > 100
    assert np.array_equal(memory.pack_object(controller), packed)
    assert controller.shared_estimator.last_update_t == 2.499


def test_restore_is_pure_across_repeated_calls():
    before = snapshot("servo")
    controller = make_controller("servo", 0.85, (0.013, -0.009, 0.019))
    memory = ControllerMemory("servo", before)
    packed = memory.pack_snapshot()
    memory.restore(controller, packed, 2.5)
    controller.tau_dot += 123.0
    controller.shared_estimator.r_joint -= 9.0
    memory.restore(controller, packed, 2.5)
    assert np.array_equal(memory.pack_object(controller), packed)


def test_jsonable_complex_poles_are_explicit():
    value = _jsonable(np.array([0.5 + 0.25j]))
    assert value == [{"real": 0.5, "imag": 0.25,
                      "magnitude": float(abs(0.5 + 0.25j))}]


def test_snapshot_input_is_not_mutated_by_restore():
    before = snapshot("torque")
    saved = copy.deepcopy(before)
    controller = make_controller("torque", 0.85, (0.013, -0.009, 0.019))
    memory = ControllerMemory("torque", before)
    memory.restore(controller, memory.pack_snapshot(), 2.5)
    assert before == saved


def test_candidate_declared_state_is_included():
    before = snapshot("torque")
    before["controller"]["dob_state"] = [1.0, 2.0]
    before["shared_estimator"]["prne_state"] = [3.0]
    controller = make_controller("torque", 0.85, (0.013, -0.009, 0.019))
    controller.dob_state = np.zeros(2)
    controller.R2_STABILITY_STATE = ("dob_state",)
    controller.shared_estimator.prne_state = np.zeros(1)
    controller.shared_estimator.R2_STABILITY_STATE = ("prne_state",)
    memory = ControllerMemory.for_candidate("torque", before, controller)
    schema = {(row["owner"], row["name"]) for row in memory.schema()}
    assert ("controller", "dob_state") in schema
    assert ("shared", "prne_state") in schema


def test_contact_mode_excludes_frozen_unused_states_and_replays_queue():
    from experiments import r2_contact
    data = json.loads((PRIVATE / "torque_0.85kg_stability_snapshots.json").read_text())
    cls = r2_contact.make_controller(
        "torque", {"family": "C1", "fraction": 0.85, "feedback_gain": 0.25}
    )
    point = replay_candidate_operating_point(cls, "torque", data, 2.5, PRIVATE)
    controller = make_controller("torque", 0.85, (0.013, -0.009, 0.019), cls)
    memory = ControllerMemory.for_candidate(
        "torque", point["memory_snapshot"], controller, time_s=2.5, grip=True
    )
    names = {(row["owner"], row["name"]) for row in memory.schema()}
    assert ("controller", "feedback_force") not in names
    assert np.asarray(point["pending_command"]).shape == (6,)
    assert np.all(np.isfinite(point["previous_applied_torque"]))


def test_compact_candidate_ledger_preserves_every_case():
    source = ROOT / "results/v2_protocol/R2_CANDIDATE_STABILITY.json"
    with tempfile.TemporaryDirectory() as directory:
        output = Path(directory) / "compact.json"
        compact = compact_candidate_result(source, output)
        assert len(compact["all_cases"]) == 300
        assert compact["summary"]["total_cases"] == 300
        assert len(compact["case_columns"]) == len(compact["all_cases"][0])
        assert output.stat().st_size < source.stat().st_size / 20
