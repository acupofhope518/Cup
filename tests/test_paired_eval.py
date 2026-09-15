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


if __name__ == "__main__":
    unittest.main()
