"""Regression tests for the maintenance entry point, without training models."""

import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import marl


class MaintenanceTests(unittest.TestCase):
    def test_accepts_existing_summary_formats(self):
        for summary in (
            {"checks_passed": 2, "checks_total": 2},
            {"check_count": 2},
            {"checks": ["shape", "termination"]},
        ):
            self.assertIs(marl.summarize({"result": summary}, 2), summary)

    def test_rejects_missing_or_incomplete_checks(self):
        for summary in ({}, {"checks": ["shape"]}, {"checks_passed": 1, "checks_total": 2}):
            with self.assertRaises(ValueError):
                marl.summarize({"summary": summary}, 2)

    def test_normalizes_editor_output_without_changing_the_file(self):
        import nbformat

        notebook = nbformat.v4.new_notebook(cells=[nbformat.v4.new_code_cell("print(1)")])
        data = json.loads(nbformat.writes(notebook))
        data["cells"][0]["source"] = ["print(1)\n"]
        data["cells"][0]["outputs"] = [
            {"output_type": "display_data", "data": {"text/plain": ["1"]},
             "metadata": {}, "jetTransient": True},
        ]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "example.ipynb"
            path.write_text(json.dumps(data), encoding="utf-8")
            before = path.read_bytes()
            loaded = marl.load_notebook(path)
            self.assertIsInstance(loaded.cells[0].source, str)
            self.assertNotIn("jetTransient", loaded.cells[0].outputs[0])
            self.assertEqual(path.read_bytes(), before)

    def test_checks_real_links_without_treating_code_as_links(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "target.md").write_text("# Target\n", encoding="utf-8")
            readme = root / "README.md"
            readme.write_text(
                "[Read](target.md)\n\n```python\nactors[i](local[i])\n```\n",
                encoding="utf-8",
            )
            with patch.object(marl, "ROOT", root):
                self.assertEqual(marl.check_links(), 1)
                readme.write_text("[Missing](missing.md)\n", encoding="utf-8")
                with self.assertRaisesRegex(ValueError, "Broken link"):
                    marl.check_links()


if __name__ == "__main__":
    unittest.main()
