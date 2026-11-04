"""Embed fixed numerical inference in the frozen v1 standalone detector."""
import argparse
import ast
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def build(model_path, output, domain_path=None, geometry_gate=False):
    payload = json.loads(model_path.read_text())
    source = (ROOT/'v1_detector.py').read_text()
    core = (ROOT/'tools/v2_learned_core.py').read_text()
    core = core[core.index('def normal_features'):]
    marker = '\nclass Detector(SharedEstimator):'
    assert source.count(marker) == 1
    source = source.replace(marker, '\n\n'+core+'\n\nMODEL_PARAMETERS = '+repr(payload['models'])+'\n'+marker)
    old = '''        else:
            self._shared = self
            super().reset()

    def __getattr__'''
    new = '''        else:
            self._shared = self
            super().reset()
        # Recognized controller-specific normal distributions only. Baseline
        # controllers retain the original v1 detector without extrapolation.
        ctrl = getattr(self, "ctrl", None)
        mode = getattr(ctrl, "MODE", None)
        known = self._shared is not self and mode in MODEL_PARAMETERS
        self.normal_model = NormalResidualModel(**MODEL_PARAMETERS[mode]) if known else None

    def __getattr__'''
    assert source.count(old) == 1
    source = source.replace(old, new)
    old_return = '        return packet.r_joint.copy(), bool(packet.detected)'
    new_return = '''        extra = False
        if self.normal_model is not None:
            extra = self.normal_model.update(t, self.box_mass, self.box_com,
                packet.last_wrench, bool(grip and not packet.last_table_allowed))
        return packet.r_joint.copy(), bool(packet.detected or extra)'''
    assert source.count(old_return) == 1
    source = source.replace(old_return, new_return)
    if domain_path is not None:
        domain = json.loads(domain_path.read_text())
        domain_code = "\nINFERENCE_DOMAIN = " + repr(domain) + "\n\n" + '''def within_training_domain(mass, com):
    c = np.asarray(com, float)
    return bool(np.isfinite(mass) and np.isfinite(c).all()
        and INFERENCE_DOMAIN["mass_min"] <= mass <= INFERENCE_DOMAIN["mass_max"]
        and np.all(c >= INFERENCE_DOMAIN["com_min"])
        and np.all(c <= INFERENCE_DOMAIN["com_max"]))

'''
        source = source.replace(marker, "\n"+domain_code+marker)
        source = source.replace("known = self._shared is not self and mode in MODEL_PARAMETERS",
            "known = (self._shared is not self and mode in MODEL_PARAMETERS\n"
            "                 and within_training_domain(self.box_mass, self.box_com))")
    if geometry_gate:
        geometry_code = '''
def outside_table_height_band(packet, grip):
    """Geometric gate, independent of payload/contact-support eligibility."""
    if not grip:
        return False
    _, point, _ = packet._point_jacobian(packet.q_prev)
    return bool(np.min(np.abs(packet.table_heights - point[2])) > packet.height_band)

'''
        source = source.replace(marker, "\n"+geometry_code+marker)
        source = source.replace("packet.last_wrench, bool(grip and not packet.last_table_allowed))",
                                "packet.last_wrench, outside_table_height_band(packet, grip))")
    ast.parse(source)
    output.write_text(source)
    return {'model_sha256': hashlib.sha256(model_path.read_bytes()).hexdigest(),
            'candidate_sha256': hashlib.sha256(source.encode()).hexdigest(),
            'core_sha256': hashlib.sha256((ROOT/'tools/v2_learned_core.py').read_bytes()).hexdigest()}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--model', type=Path, default=ROOT/'models/v2_residual.json')
    ap.add_argument('--out', type=Path, default=ROOT/'v2_detector_candidate.py')
    ap.add_argument('--domain', type=Path)
    ap.add_argument('--geometry-gate', action='store_true')
    args = ap.parse_args()
    print(json.dumps(build(args.model, args.out, args.domain, args.geometry_gate), indent=2))


if __name__ == '__main__':
    main()
