"""Read back at most two still-owned samples after detached retirement."""
from __future__ import annotations

from experiments.r30_detached_lifetime.candidate import AcquisitionEndpoint
from experiments.r31_held_identity.candidate import Detector as HeldIdentity


READBACK_SAMPLES = 2
EXISTING_LOW_QUORUM = 3


class RetirementReadback(AcquisitionEndpoint):
    """Preserve a bounded suffix without changing the original endpoint state.

    ``AcquisitionEndpoint.update`` remains authoritative and is called exactly
    once. A suffix can start only at its first real retirement transition. It
    never clears ``retired`` and cannot restart within that detached lifetime.
    ``legacy_output`` and ``readback_active`` expose the two outputs separately.
    """

    def __init__(self):
        starts = getattr(self, "readback_starts", 0)
        published = getattr(self, "readback_published", 0)
        retirements = getattr(self, "original_retirements", 0)
        super().__init__()
        self.readback_starts = starts
        self.readback_published = published
        self.original_retirements = retirements
        self.readback_armed = False
        self.readback_active = False
        self.readback_cancelled = False
        self.readback_remaining = 0
        self.readback_previous_original = False
        self.readback_previous_raw = False
        self.legacy_output = False

    def update(self, report, detached, vector, bound, mean_quiet, sole):
        previous_original = self.readback_previous_original
        previous_raw = self.readback_previous_raw
        was_retired = self.retired
        original = bool(super().update(
            report, detached, vector, bound, mean_quiet, sole))
        self.legacy_output = original
        self.readback_active = False
        transitioned = bool(not was_retired and self.retired)
        if transitioned:
            self.original_retirements += 1

        proof = bool(
            detached
            and report
            and sole
            and mean_quiet
            and self.low >= EXISTING_LOW_QUORUM
        )
        if transitioned:
            if previous_original and previous_raw and proof:
                self.readback_armed = True
                self.readback_remaining = READBACK_SAMPLES
                self.readback_starts += 1
            else:
                self.readback_armed = False
                self.readback_cancelled = True
        elif self.readback_armed and not proof:
            self.readback_armed = False
            self.readback_remaining = 0
            self.readback_cancelled = True

        # Preserve every fail-open/high result from the authoritative endpoint.
        if original:
            answer = True
        elif self.readback_armed and proof and self.readback_remaining > 0:
            answer = True
            self.readback_active = True
            self.readback_remaining -= 1
            self.readback_published += 1
            if self.readback_remaining == 0:
                self.readback_armed = False
        else:
            answer = False

        self.readback_previous_original = original
        self.readback_previous_raw = bool(report)
        return answer


class Detector(HeldIdentity):
    """Frozen R31 held identity with only its detached endpoint replaced."""

    def reset(self):
        super().reset()
        self.detached_endpoint = RetirementReadback()

