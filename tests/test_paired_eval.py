import json
import tempfile
import unittest
from pathlib import Path

from training.validate_paired_eval import validate


class PairedEvalValidationTests(unittest.TestCase):
    def test_gematria_benchmark_has_complete_pairs(self):
        path = Path(__file__).parents[1] / "data" / "gematria_v1_eval.jsonl"
        cases, pairs, errors = validate(path)
        self.assertEqual((cases, pairs), (12, 6))
        self.assertEqual(errors, [])

    def test_missing_pair_mate_fails(self):
        case = {
            "id": "only", "pair_id": "p", "label": "supported_relationship",
            "domain": "test", "prompt": "x", "expected": ["x"], "avoid": ["y"],
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "eval.jsonl"
            path.write_text(json.dumps(case) + "\n", encoding="utf-8")
            _, _, errors = validate(path)
        self.assertTrue(any("expected one of each label" in error for error in errors))

    def test_non_object_rows_are_reported_and_validation_continues(self):
        valid_pair = [
            {
                "id": label, "pair_id": "p", "label": label,
                "domain": "test", "prompt": "x", "expected": ["x"],
                "avoid": ["y"],
            }
            for label in ("supported_relationship", "coincidental_pattern")
        ]
        rows = [None, "text", ["array"], *valid_pair]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "eval.jsonl"
            path.write_text(
                "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
            )
            cases, pairs, errors = validate(path)

        self.assertEqual((cases, pairs), (2, 1))
        self.assertEqual(len(errors), 3)
        self.assertIn("line 1: expected a JSON object, found NoneType", errors)
        self.assertIn("line 2: expected a JSON object, found str", errors)
        self.assertIn("line 3: expected a JSON object, found list", errors)


if __name__ == "__main__":
    unittest.main()
