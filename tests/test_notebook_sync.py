"""The Kaggle notebook embeds repo files via %%writefile; keep them identical.

A stale embedded copy means Kaggle trains or measures something other than
what the repo says, e.g. an old probe set that predates the Haitian
Creole cases.
"""
import json
import unittest
from pathlib import Path

ROOT = Path(__file__).parents[1]
NOTEBOOK = ROOT / "kaggle" / "assistivemind_v4_qwen_qlora.ipynb"

EMBEDDED = {
    "format_data.py": ROOT / "training" / "format_data.py",
    "train.py": ROOT / "training" / "train.py",
    "validate_dataset.py": ROOT / "training" / "validate_dataset.py",
    "inference_eval.py": ROOT / "training" / "inference_eval.py",
    "assistivemind_v4_eval.jsonl": ROOT / "data" / "assistivemind_v4_eval.jsonl",
    "conformance.py": ROOT / "gate" / "conformance.py",
    "probes.jsonl": ROOT / "gate" / "probes.jsonl",
}


def _writefile_cells():
    cells = json.loads(NOTEBOOK.read_text(encoding="utf-8"))["cells"]
    out = {}
    for cell in cells:
        src = "".join(cell["source"])
        if src.startswith("%%writefile "):
            header, _, body = src.partition("\n")
            out[header.split()[1]] = body
    return out


class NotebookSyncTests(unittest.TestCase):
    def test_every_embedded_file_matches_the_repo(self):
        cells = _writefile_cells()
        self.assertEqual(set(cells), set(EMBEDDED))
        for name, path in EMBEDDED.items():
            with self.subTest(name=name):
                self.assertEqual(cells[name].rstrip("\n"),
                                 path.read_text(encoding="utf-8").rstrip("\n"))


if __name__ == "__main__":
    unittest.main()
