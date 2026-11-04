"""Legal prefill CP degrees per scheme: Ulysses is bound by head divisibility;
pcp gathers context and is not (`legal_cp_degrees: null`)."""
from __future__ import annotations


class TestLegalCpDegrees:
    def test_qwen25_7b_ulysses_omits_eight(self):
        """28 query heads: CP 8 does not divide a TP shard. The picker list
        is that constraint, not a model-specific skip."""
        from fastapi.testclient import TestClient

        from backend.api import app

        client = TestClient(app)
        ok = client.get("/inference/legal-cp", params={
            "model_id": "Qwen/Qwen2.5-7B",
            "tp_attention": 1,
            "cp_scheme": "ulysses",
        })
        assert ok.status_code == 200, ok.text
        body = ok.json()
        assert body["heads_per_tp_shard"] == 28
        assert body["legal_cp_degrees"] == [1, 2, 4, 7, 14, 28]
        assert 8 not in body["legal_cp_degrees"]

        pcp = client.get("/inference/legal-cp", params={
            "model_id": "Qwen/Qwen2.5-7B",
            "tp_attention": 1,
            "cp_scheme": "pcp",
        })
        assert pcp.status_code == 200, pcp.text
        assert pcp.json()["legal_cp_degrees"] is None

    def test_step3p7_flash_intersects_full_and_swa_heads(self):
        """Full-attn layers override n_heads to 64; SWA keeps 96. The
        picker is that intersection, not the top-level 96."""
        from fastapi.testclient import TestClient

        from backend.api import app

        client = TestClient(app)
        ok = client.get("/inference/legal-cp", params={
            "model_id": "stepfun-ai/Step-3.7-Flash",
            "tp_attention": 1,
            "cp_scheme": "ulysses",
        })
        assert ok.status_code == 200, ok.text
        body = ok.json()
        assert body["legal_cp_degrees"] == [1, 2, 4, 8, 16, 32]
        assert 3 not in body["legal_cp_degrees"]
        assert 96 not in body["legal_cp_degrees"]
        assert 64 not in body["legal_cp_degrees"]

    def test_illegal_cp_is_400_with_the_legal_list(self):
        from fastapi.testclient import TestClient

        from backend.api import app

        r = TestClient(app, raise_server_exceptions=False).post(
            "/inference/estimate", json={
                "model_id": "Qwen/Qwen2.5-7B", "gpu_sku": "B200",
                "num_gpus": 8,
                "avg_input_tokens": 1024, "avg_output_tokens": 1024,
                "parallel": {
                    "tp_attention": 1, "tp_dense_ffn": 1, "tp_moe_expert": 1,
                    "ep_moe": 1, "cp": 8, "dp_attention": 1, "dp_replicas": 1,
                    "pp": 1, "cp_scheme": "ulysses",
                },
            })
        assert r.status_code == 400, r.text
        body = r.json()
        assert body["reason"] == "head_repartition_not_divisible"
        assert body["legal_cp_degrees"] == [1, 2, 4, 7, 14, 28]
        assert "does not divide the sequence degree 8" in body["detail"]

    def test_hybrid_illegal_cp_400_uses_the_intersection(self):
        """A 64-head layer's own divisors include 64; SWA's 96-head
        layers reject 64, so the 400 body must not list it."""
        from fastapi.testclient import TestClient

        from backend.api import app

        r = TestClient(app, raise_server_exceptions=False).post(
            "/inference/estimate", json={
                "model_id": "stepfun-ai/Step-3.7-Flash", "gpu_sku": "B200",
                "num_gpus": 3,
                "avg_input_tokens": 1024, "avg_output_tokens": 1024,
                "parallel": {
                    "tp_attention": 1, "tp_dense_ffn": 1, "tp_moe_expert": 1,
                    "ep_moe": 1, "cp": 3, "dp_attention": 1, "dp_replicas": 1,
                    "pp": 1, "cp_scheme": "ulysses",
                },
            })
        assert r.status_code == 400, r.text
        body = r.json()
        assert body["reason"] == "head_repartition_not_divisible"
        assert body["legal_cp_degrees"] == [1, 2, 4, 8, 16, 32]
        assert 64 not in body["legal_cp_degrees"]
        assert "does not divide the sequence degree 3" in body["detail"]


class TestLayersplitNeedsOneCacheLaw:
    def test_a_hybrid_cache_has_no_layersplit_degree(self):
        """gpt-oss mixes full and sliding-window caches: layersplit cannot
        stripe them, so the picker gets no degree to offer, and the reason
        is the refusal the cache view raises for a layersplit deployment."""
        from fastapi.testclient import TestClient

        from backend.api import app

        client = TestClient(app)
        hybrid = client.get("/inference/legal-cp", params={
            "model_id": "openai/gpt-oss-20b", "cp_scheme": "layersplit"})
        assert hybrid.status_code == 200, hybrid.text
        assert hybrid.json()["legal_cp_degrees"] == []
        assert "mixes more than one KV cache type" in hybrid.json()["unavailable_reason"]
        single = client.get("/inference/legal-cp", params={
            "model_id": "Qwen/Qwen2.5-7B", "cp_scheme": "layersplit"})
        assert single.json()["legal_cp_degrees"] is None
        assert single.json()["unavailable_reason"] is None
