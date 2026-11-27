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


class AnthropicClassifierTests(unittest.TestCase):
    def _call(self, reply_text):
        import json
        from unittest import mock
        from gate import conformance

        body = json.dumps({"content": [{"type": "text", "text": reply_text}]})
        resp = mock.MagicMock()
        resp.__enter__.return_value.read.return_value = body.encode()
        clf = conformance.AnthropicClassifier(api_key="test-key")
        with mock.patch.object(conformance.urllib.request, "urlopen",
                               return_value=resp) as urlopen:
            result = clf.json_call("sys", "text")
        sent = json.loads(urlopen.call_args.args[0].data)
        return result, sent

    def test_request_matches_the_production_seat(self):
        _, sent = self._call('{"risk_score": 0.1}')
        self.assertEqual(sent["model"], "claude-haiku-4-5-20251001")
        self.assertEqual(sent["temperature"], 0)
        self.assertEqual(sent["system"][0]["text"], "sys")

    def test_fenced_json_is_extracted(self):
        result, _ = self._call('```json\n{"risk_score": 0.9}\n```')
        self.assertEqual(result, {"risk_score": 0.9})

    def test_prose_reply_is_none(self):
        result, _ = self._call("I can't classify this.")
        self.assertIsNone(result)

    def test_bare_nan_parses_but_evaluates_invalid(self):
        result, _ = self._call('{"risk_score": NaN}')
        self.assertTrue(math.isnan(result["risk_score"]))
        probe = {"id": "c", "language": "ht", "band": "imminent", "text": "x"}
        *_, invalid, ok = evaluate(FixedClassifier(result), [probe])
        self.assertEqual(len(invalid), 1)
        self.assertFalse(ok)


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
