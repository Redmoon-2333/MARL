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

    def test_legacy_outputs_are_validated_without_fabricating_a_check_count(self):
        namespace = {
            "return_A": 0.76, "return_B": 0.68,
            "q_learning_target": 0.62, "sarsa_target": 0.08,
            "q_learning_new": 0.51, "sarsa_new": 0.24,
        }
        result = marl.summarize(namespace, None, "greedy")
        self.assertEqual(result["q_learning_new"], 0.51)
        self.assertNotIn("checks_passed", result)
        namespace["q_learning_new"] = 0.4
        with self.assertRaisesRegex(ValueError, "unexpected q_learning_new"):
            marl.summarize(namespace, None, "greedy")

    def test_legacy_counted_checks_must_still_match(self):
        namespace = {
            "checks": 13, "trajectory": [], "returns": [],
            "before_probs": FakeTensor(), "after_probs": FakeTensor(),
            "policy_loss": FakeTensor(), "gradient": FakeTensor(),
        }
        with self.assertRaisesRegex(ValueError, "Expected 14"):
            marl.summarize(namespace, 14, "reinforce-episode")
        namespace["checks"] = 14
        self.assertEqual(marl.summarize(namespace, 14, "reinforce-episode")["checks_passed"], 14)

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
                "[Read](<target.md>)\n\n```python\nactors[i](local[i])\n```\n",
                encoding="utf-8",
            )
            with patch.object(marl, "ROOT", root):
                self.assertEqual(marl.check_links(), 1)
                readme.write_text("[Missing](missing.md)\n", encoding="utf-8")
                with self.assertRaisesRegex(ValueError, "Broken link"):
                    marl.check_links()


class FakeTensor:
    def tolist(self):
        return []

    def item(self):
        return 0.0


if __name__ == "__main__":
    unittest.main()
