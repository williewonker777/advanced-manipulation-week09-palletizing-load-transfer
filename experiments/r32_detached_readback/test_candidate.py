import inspect
import unittest

from experiments.r31_held_identity.candidate import Detector as HeldIdentity
from experiments.r32_detached_readback.candidate import (
    Detector,
    EXISTING_LOW_QUORUM,
    READBACK_SAMPLES,
    RetirementReadback,
)


LOW = (0.0, 0.0)
HIGH = (2.0, 0.0)


def feed(endpoint, *, report=True, detached=True, vector=LOW,
         quiet=True, sole=True):
    return endpoint.update(report, detached, vector, 1.0, quiet, sole)


class RetirementReadbackTest(unittest.TestCase):
    def acquire_to_low_two(self, endpoint):
        self.assertTrue(feed(endpoint, vector=HIGH, quiet=False))
        self.assertTrue(feed(endpoint, quiet=False))
        self.assertTrue(feed(endpoint, quiet=False))

    def test_exactly_two_samples_after_authoritative_retirement(self):
        endpoint = RetirementReadback()
        self.acquire_to_low_two(endpoint)
        self.assertTrue(feed(endpoint))
        self.assertFalse(endpoint.legacy_output)
        self.assertTrue(endpoint.readback_active)
        self.assertTrue(endpoint.retired)
        self.assertEqual(endpoint.original_retirements, 1)
        self.assertEqual(endpoint.readback_starts, 1)
        self.assertEqual(endpoint.readback_published, 1)
        self.assertTrue(feed(endpoint))
        self.assertFalse(endpoint.legacy_output)
        self.assertTrue(endpoint.readback_active)
        self.assertEqual(endpoint.readback_published, READBACK_SAMPLES)
        self.assertFalse(feed(endpoint))
        self.assertFalse(endpoint.readback_active)
        self.assertTrue(endpoint.retired)

    def test_raw_gap_cancels_and_cannot_reopen(self):
        endpoint = RetirementReadback()
        self.acquire_to_low_two(endpoint)
        self.assertTrue(feed(endpoint))
        self.assertFalse(feed(endpoint, report=False))
        self.assertTrue(endpoint.readback_cancelled)
        self.assertFalse(feed(endpoint))
        self.assertEqual(endpoint.readback_starts, 1)
        self.assertEqual(endpoint.readback_published, 1)

    def test_route_or_short_low_loss_cancels(self):
        for kwargs in ({"sole": False}, {"vector": HIGH}, {"quiet": False}):
            with self.subTest(kwargs=kwargs):
                endpoint = RetirementReadback()
                self.acquire_to_low_two(endpoint)
                self.assertTrue(feed(endpoint))
                answer = feed(endpoint, **kwargs)
                # A new raw route is inherited fail-open for its current sample,
                # but it is never mislabeled as detached readback.
                self.assertEqual(answer, kwargs == {"sole": False})
                self.assertFalse(endpoint.readback_active)
                self.assertFalse(feed(endpoint))

    def test_real_detached_loss_resets_for_a_new_acquisition(self):
        endpoint = RetirementReadback()
        self.acquire_to_low_two(endpoint)
        self.assertTrue(feed(endpoint))
        self.assertFalse(feed(endpoint, detached=False, report=False))
        self.assertTrue(feed(endpoint, vector=HIGH, quiet=False))
        self.assertTrue(endpoint.legacy_output)
        self.assertFalse(endpoint.readback_active)

    def test_new_detached_acquisition_can_start_after_real_reset(self):
        endpoint = RetirementReadback()
        self.acquire_to_low_two(endpoint)
        self.assertTrue(feed(endpoint))
        self.assertTrue(feed(endpoint))
        self.assertFalse(feed(endpoint))
        self.assertFalse(feed(endpoint, report=False, detached=False))
        self.acquire_to_low_two(endpoint)
        self.assertTrue(feed(endpoint))
        self.assertEqual(endpoint.readback_starts, 2)

    def test_fail_open_original_high_is_never_suppressed(self):
        endpoint = RetirementReadback()
        for _ in range(20):
            self.assertTrue(feed(endpoint, vector=HIGH, quiet=False))
            self.assertTrue(endpoint.legacy_output)
            self.assertFalse(endpoint.readback_active)
        self.assertFalse(endpoint.retired)
        self.assertEqual(endpoint.readback_starts, 0)

    def test_parent_constants_and_no_forbidden_branch(self):
        self.assertTrue(issubclass(Detector, HeldIdentity))
        self.assertEqual(READBACK_SAMPLES, 2)
        self.assertEqual(EXISTING_LOW_QUORUM, 3)
        source = inspect.getsource(RetirementReadback.update)
        self.assertEqual(source.count("super().update"), 1)
        for forbidden in ("case", "force", "time", "mass", "collision"):
            self.assertNotIn(forbidden, source)


if __name__ == "__main__":
    unittest.main()
