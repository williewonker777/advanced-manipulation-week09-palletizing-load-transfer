import json
from pathlib import Path
from experiments.r22_tangent.detector import Detector as Unbounded


class Detector(Unbounded):
    bounds_tangent = tuple(json.loads(Path(__file__).with_name('model.json').read_bytes())['bounds_N'])
