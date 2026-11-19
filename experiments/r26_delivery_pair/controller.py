"""Zero-argument entrypoint for the frozen R24 G4 torque control/calibration."""
import json
from pathlib import Path

from experiments.r24_transfer.deadband import Controller as Deadband


class Controller(Deadband):
    def __init__(self):
        data = Path(__file__).resolve().parents[1] / "r24_transfer_replay" / "calibration.json"
        c = json.loads(data.read_bytes())
        super().__init__(c["height"], c["information"])
