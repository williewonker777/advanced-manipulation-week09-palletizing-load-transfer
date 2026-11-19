"""R31 held identity plus flange-centered precontact inverse load."""
from experiments.r31_held_identity.candidate import Detector as Parent
from experiments.r34_precontact_inverse.inverse import FlangeInverse
from experiments.r34_precontact_inverse.witness import PrecontactInverseWitness


MODEL = {
    "schema": "r34-precontact-inverse-model.v1",
    "threshold_N": 6.084419216178501,
    "persistence": 3,
    "plan_sha256": "73386b9ca94aea324495ab7ac4d096a69025b403a77051535e60958830a690d0",
    "calibration_sha256": "f9d30921f8a34d208eeb81c79a2396a09b3c3503be9737198ff3758ff9101ad7",
}


class Detector(Parent):
    """Cooperatively OR one measured, source-owned precontact notice."""

    def reset(self):
        super().reset()
        self.precontact_inverse = FlangeInverse(self.nom, self.nom_box)
        self.precontact_source = PrecontactInverseWitness(
            MODEL["threshold_N"], MODEL["persistence"])
        self.precontact_extra = False

    def update(self, t, th, thd, tau_m, tau_s, grip):
        residual, incumbent = super().update(t, th, thd, tau_m, tau_s, grip)
        sample = self.precontact_inverse.update(t, th, tau_s, grip)
        extra = self.precontact_source.extra(sample, bool(grip), bool(incumbent))
        self.precontact_extra = bool(extra)
        return residual, bool(incumbent or extra)
