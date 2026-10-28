import unittest
from tools import d4_robustness as R


class DesignTests(unittest.TestCase):
    def test_full_reference_grid(self):
        cases = R.build_cases("my_servo")
        self.assertEqual(len(cases), R.protocol()["cases_per_controller"])
        refs = [c for c in cases if c["kind"] == "reference"]
        self.assertEqual(len(refs), 189)
        self.assertEqual(len({(r["mass"], r["com"]) for r in refs}), 189)

    def test_force_coverage_and_matched_references(self):
        cases = R.build_cases("my_servo")
        refs = {(c["mass"], c["com"]) for c in cases if c["kind"] == "reference"}
        collisions = [c for c in cases if c["kind"] == "collision"]
        self.assertEqual(len(collisions), 180)
        for phase in R.TIMES:
            subset = [c for c in collisions if c["phase"] == phase]
            self.assertEqual({c["direction"] for c in subset}, set(R.DIRECTIONS))
            self.assertEqual({c["force_n"] for c in subset}, set(R.FORCES))
            self.assertEqual({c["collision"][0] for c in subset}, set(R.TIMES[phase]))
        for c in collisions:
            self.assertIn((c["mass"], c["com"]), refs)
            self.assertAlmostEqual(sum(v*v for v in c["collision"][1]), c["force_n"]**2)


if __name__ == "__main__":
    unittest.main()
