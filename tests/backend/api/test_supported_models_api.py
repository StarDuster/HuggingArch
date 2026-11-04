"""End-to-end /analyze sanity test for every model in supported_models.txt.

Catches errors that only surface at the API-response-builder boundary
(diagrams, block_graph, dtype attaching, etc.) — analyzer-level smoke
tests don't exercise those code paths. Uses FastAPI's TestClient so no
external HTTP server is needed.
"""
import sys
import unittest

import pytest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from fastapi.testclient import TestClient

from backend.api import app


_SUPPORTED_LIST = PROJECT_ROOT / "backend" / "arch" / "supported_models.txt"


def _load_supported_ids() -> list:
    ids = []
    with open(_SUPPORTED_LIST) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            ids.append(line)
    return ids


@pytest.mark.regression  # every supported model
class TestSupportedModelsAnalyze(unittest.TestCase):
    """For every curated supported model, /analyze must return success=True
    with a non-zero param total. Specifically validates that response-builder
    code paths (block_graph, diagrams, dtype propagation, group accounting)
    survive the parameterized-compound-op refactor."""

    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app)

    def test_analyze_each_supported_model(self):
        model_ids = _load_supported_ids()
        self.assertGreater(len(model_ids), 0, "supported_models.txt is empty")
        for model_id in model_ids:
            with self.subTest(model_id=model_id):
                resp = self.client.post(
                    "/analyze",
                    json={"model_id": model_id, "batch_size": 1, "seq_len": 1},
                )
                self.assertEqual(
                    resp.status_code, 200,
                    f"{model_id}: HTTP {resp.status_code}: {resp.text[:500]}",
                )
                data = resp.json()
                self.assertTrue(
                    data.get("success"),
                    f"{model_id}: analyze failed: {data.get('message')}",
                )
                # Sanity: payload contains the structures the frontend renders.
                self.assertIn("dashboard", data, f"{model_id}: missing dashboard")
                self.assertIn("block_graph", data, f"{model_id}: missing block_graph")
                self.assertGreater(
                    data["dashboard"].get("total_params", 0), 0,
                    f"{model_id}: total_params <= 0",
                )


if __name__ == "__main__":
    unittest.main()
