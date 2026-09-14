import math
import unittest

from gate.conformance import evaluate


class FixedClassifier:
    available = True

    def __init__(self, result):
        self.result = result

    def json_call(self, system_prompt, user_content, max_tokens=400):
        return self.result


class ConformanceResultValidationTests(unittest.TestCase):
    def evaluate_one(self, score, band="ideation"):
        probe = {"id": "case", "language": "ht", "band": band, "text": "x"}
        return evaluate(FixedClassifier({"risk_score": score}), [probe])

    def test_crisis_score_below_floor_is_a_miss(self):
        misses, noise, invalid, ok = self.evaluate_one(0.69)
        self.assertEqual(len(misses), 1)
        self.assertFalse(noise or invalid or ok)

    def test_nan_is_invalid_not_an_in_band_pass(self):
        misses, noise, invalid, ok = self.evaluate_one(math.nan)
        self.assertEqual(len(invalid), 1)
        self.assertFalse(misses or noise or ok)

    def test_booleans_are_invalid_not_zero_or_one(self):
        for score in (False, True):
            with self.subTest(score=score):
                misses, noise, invalid, ok = self.evaluate_one(score)
                self.assertEqual(len(invalid), 1)
                self.assertFalse(misses or noise or ok)

    def test_integer_too_large_for_float_is_invalid(self):
        misses, noise, invalid, ok = self.evaluate_one(10**1000)
        self.assertEqual(len(invalid), 1)
        self.assertFalse(misses or noise or ok)

    def test_infinities_and_out_of_range_values_are_invalid(self):
        for score in (math.inf, -math.inf, -0.01, 1.01):
            with self.subTest(score=score):
                *_, invalid, ok = self.evaluate_one(score)
                self.assertEqual(len(invalid), 1)
                self.assertFalse(ok)

    def test_numeric_string_is_accepted_for_provider_compatibility(self):
        misses, noise, invalid, ok = self.evaluate_one("0.90")
        self.assertEqual(len(ok), 1)
        self.assertFalse(misses or noise or invalid)

    def test_benign_over_score_is_noise(self):
        misses, noise, invalid, ok = self.evaluate_one(0.8, band="benign")
        self.assertEqual(len(noise), 1)
        self.assertFalse(misses or invalid or ok)


class ProbeCoverageTests(unittest.TestCase):
    def test_probe_ids_are_unique_and_bands_are_known(self):
        import json
        from pathlib import Path

        path = Path(__file__).parents[1] / "gate" / "probes.jsonl"
        probes = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
        ids = [probe["id"] for probe in probes]
        self.assertEqual(len(ids), len(set(ids)))
        self.assertTrue({probe["band"] for probe in probes} <= {
            "imminent", "ideation", "distress", "benign", "business"
        })

    def test_haitian_creole_has_risk_and_false_positive_coverage(self):
        import json
        from pathlib import Path

        path = Path(__file__).parents[1] / "gate" / "probes.jsonl"
        probes = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
        creole = [probe for probe in probes if probe["language"] == "ht"]
        self.assertGreaterEqual(len(creole), 6)
        self.assertTrue({"imminent", "ideation", "benign"} <= {
            probe["band"] for probe in creole
        })


if __name__ == "__main__":
    unittest.main()
