"""Playground model-editor routes: the layer-group DTO and compose-model."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from backend.api import app
from backend.arch.eval.projection import project_model
from backend.arch.loader import SpecLoader
from backend.playground import compose as cm
from tests.snapshots import resolve_snapshot, snapshot_config


@pytest.fixture(scope="module")
def client():
    patch = pytest.MonkeyPatch()
    patch.setattr(cm, "fetch_config", snapshot_config)
    patch.setattr(cm, "fetch_metadata", lambda m, overlay_dir=None: {})
    yield TestClient(app)
    patch.undo()


def _skeleton(client, model_id):
    r = client.get("/api/playground/model-skeleton", params={"model_id": model_id})
    assert r.status_code == 200
    return r.json()


def _echo(group):
    return {k: group[k] for k in ("group_id", "template_name", "count",
                                  "assignment_index")}


def test_no_edit_compose_reports_decode_and_kv_sweep(client):
    r = client.post("/api/playground/compose-model", json={
        "base_model_id": "meta-llama/Llama-3.2-3B",
        "groups": [],
        "kv_cache_lens": [2048, 4096],
    }).json()
    assert r["success"], r
    assert r["composed"]["phases"] == r["base"]["phases"]
    assert r["composed"]["kv_storage_by_cache_len"] == r["base"]["kv_storage_by_cache_len"]
    assert r["contract"]["kv_cache_lens"] == [2048, 4096]
    at_L = next(p for p in r["composed"]["kv_storage_by_cache_len"]
                if p["cache_len"] == 2048)
    assert at_L["storage_bytes"] == r["composed"]["kv_bytes_at_L"]
    spec = resolve_snapshot("meta-llama/Llama-3.2-3B", SpecLoader())
    decode = project_model(spec, batch=1, input_tokens=2048, phase="decode")
    assert r["composed"]["phases"]["decode"]["flops"] == decode.total_flops
    assert r["contract"]["gpu_sku"] is None
    assert "timings" not in r["composed"]


def test_compose_echoes_gpu_sku_and_times(client, monkeypatch):
    monkeypatch.setattr(
        "backend.playground.module_bench._measurement_scope", lambda: None)
    r = client.post("/api/playground/compose-model", json={
        "base_model_id": "meta-llama/Llama-3.2-3B",
        "groups": [],
        "gpu_sku": "H100-SXM",
    }).json()
    assert r["success"], r
    assert r["contract"]["gpu_sku"] == "H100-SXM"
    t = r["composed"]["timings"]["prefill"]
    assert t["gpu_sku"] == "H100-SXM" and t["t_sol_s"] > 0


def test_compose_unknown_gpu_sku_fails(client):
    r = client.post("/api/playground/compose-model", json={
        "base_model_id": "meta-llama/Llama-3.2-3B",
        "groups": [],
        "gpu_sku": "not-a-gpu",
    }).json()
    assert r["success"] is False
    assert any("unknown gpu_sku" in (e.get("detail") or "") for e in r["errors"])


def test_the_skeleton_lists_every_layer_group(client):
    sk = _skeleton(client, "moonshotai/Kimi-K3")
    groups = sk["groups"]
    assert [g["group_id"] for g in groups] == [0, 1, 2, 3, 4]
    assert [(g["aux_kind"], g["count"], g["editable"]) for g in groups] == [
        ("vision", 1, False), (None, 1, True), (None, 24, True),
        (None, 7, True), (None, 61, True)]
    assert groups[2]["layers_label"] == "3, 7, …, 91 (every 4) · 92"
    assert groups[2]["composition"] == ["MLA", "ffn_moe"]
    assert groups[3]["layers_label"] == "12, 24, …, 84 (every 12)"
    assert groups[3]["selector"] == {"stride": "Kar", "offset": "Kar"}
    assert groups[4]["composition"] == ["KDA", "ffn_moe"]
    assert groups[4]["selector"] == {"rest": True}
    assert sk["n_layers"] == 93
    assert sorted(i for g in groups[1:] for i in g["layers"]) == list(range(93))
    attn = groups[4]["slots"][0]
    assert attn["slot_id"] == "self_attn" and attn["kind"] == "attention"
    assert "mla_output_gate_attention" in attn["candidates"]
    assert "swiglu_ffn" not in attn["candidates"]
    # "all groups bound to KDA": layer 0, the 7 restart layers, the other 61
    assert attn["bound_with"] == {
        "slots": [{"group_id": 1, "slot_id": "self_attn"},
                  {"group_id": 3, "slot_id": "self_attn"},
                  {"group_id": 4, "slot_id": "self_attn"}],
        "groups": 3, "layers": 69}
    assert "d_model" in groups[4]["local_dims"]
    assert groups[4]["local_dims"]["d_model"]["interface"] is True
    # The history length is the walk's, not an editable dim.
    assert "Sar" not in groups[4]["local_dims"]


@pytest.mark.timeout(1200)
def test_a_hybrid_group_swap_changes_that_group_only(client):
    """A hybrid Kimi-K3 edit prices both whole models and every group row.
    That legitimate request takes more than ten minutes on a loaded dev box;
    keep it in the smoke run, but at the regression workflow's longer timeout
    rather than letting the default's 600s ceiling report it as stuck."""
    sk = _skeleton(client, "moonshotai/Kimi-K3")
    r = client.post("/api/playground/compose-model", json={
        "base_model_id": "moonshotai/Kimi-K3",
        "groups": [{**_echo(sk["groups"][4]),
                    "slots": {"self_attn": "mla_output_gate_attention"}}],
        "kv_cache_lens": [2048],
    }).json()
    assert r["success"], r
    # The group's 61 layers are 8 records, one per history length they see.
    assert [row["group_id"] for row in r["rows"] if row["changed"]] == [4] * 8
    # 61 layers' KDA attention replaced by MLA: 61 x -211,544,320.
    assert r["composed"]["params"] - r["base"]["params"] == -12_904_203_520
    assert r["contract"]["kv_cache_lens"] == [2048]
    at_L = next(p for p in r["composed"]["kv_storage_by_cache_len"]
                if p["cache_len"] == 2048)
    assert at_L["storage_bytes"] == r["composed"]["kv_bytes_at_L"]
    at_base = next(p for p in r["base"]["kv_storage_by_cache_len"]
                   if p["cache_len"] == 2048)
    assert at_base["storage_bytes"] == r["base"]["kv_bytes_at_L"]
    # per-group totals + the model level = the whole model, on both sides
    for side, whole in (("base", r["base"]), ("edited", r["composed"])):
        rows = [row[side]["group_totals"] for row in r["rows"]]
        level = r["model_level"][side]["totals"]
        assert sum(t["params"] for t in rows) + level["params"] == whole["params"]
        assert (sum(t["flops"] for t in rows) + level["flops"]
                == whole["flops"])
        assert (sum(t["storage_bytes"] for t in rows) + level["storage_bytes"]
                == whole["weight_bytes"])
    (group,) = r["groups"]
    assert group["group_id"] == 4 and "d_kv_lora" in group["local_dims"]


def test_mtp_is_a_group_and_swaps_on_its_own(client):
    sk = _skeleton(client, "deepseek-ai/DeepSeek-V3")
    mtp = sk["groups"][2]
    assert mtp["aux_kind"] == "mtp" and mtp["editable"]
    assert mtp["layers_label"] == "MTP · 0"
    attn = mtp["slots"][0]
    assert attn["slot_id"] == "transformer/self_attn"
    assert attn["bound_with"]["groups"] == 3   # dense, MoE and MTP
    r = client.post("/api/playground/compose-model", json={
        "base_model_id": "deepseek-ai/DeepSeek-V3",
        "groups": [{**_echo(mtp), "slots": {attn["slot_id"]: "mla_output_gate_attention"}}],
    }).json()
    assert r["success"], r
    assert [row["group_id"] for row in r["rows"] if row["changed"]] == [2]
    # an MTP layer's cache is outside the whole-model KV, as in the Inspector
    assert r["composed"]["kv_bytes_at_L"] == r["base"]["kv_bytes_at_L"]


def test_a_stale_skeleton_echo_is_refused(client):
    sk = _skeleton(client, "deepseek-ai/DeepSeek-V3")
    for field, wrong in (("template_name", "not_the_template"), ("count", 57),
                         ("assignment_index", 0)):
        entry = {**_echo(sk["groups"][1]), field: wrong,
                 "slots": {"self_attn": "mla_output_gate_attention"}}
        r = client.post("/api/playground/compose-model", json={
            "base_model_id": "deepseek-ai/DeepSeek-V3", "groups": [entry]}).json()
        assert r["success"] is False, field
        assert [(e["kind"], e["group_id"]) for e in r["errors"]] == [
            ("stale_skeleton", 1)], field


def test_refusals_come_back_as_errors(client):
    sk = _skeleton(client, "zai-org/GLM-5.2")
    r = client.post("/api/playground/compose-model", json={
        "base_model_id": "zai-org/GLM-5.2",
        "groups": [{**_echo(sk["groups"][1]), "slots": {"self_attn": "mla_attention"}}],
    }).json()
    assert r["success"] is False
    assert [e["kind"] for e in r["errors"]] == ["stream"]
    assert "dsa_topk" in r["errors"][0]["detail"]


def test_missing_dims_come_back_as_errors(client):
    sk = _skeleton(client, "meta-llama/Llama-3.2-3B")
    r = client.post("/api/playground/compose-model", json={
        "base_model_id": "meta-llama/Llama-3.2-3B",
        "groups": [{**_echo(sk["groups"][0]), "slots": {"self_attn": "mla_attention"}}],
    }).json()
    assert r["success"] is False
    (error,) = r["errors"]
    assert (error["kind"], error["group_id"], error["slot"]) == (
        "missing_dims", 0, "self_attn")
    assert "d_kv_lora" in error["missing_dims"]


def test_a_group_dim_edit_reports_what_it_moved(client):
    sk = _skeleton(client, "deepseek-ai/DeepSeek-V3")
    r = client.post("/api/playground/compose-model", json={
        "base_model_id": "deepseek-ai/DeepSeek-V3",
        "groups": [{**_echo(sk["groups"][1]), "dims": {"n_experts": 128}}],
    }).json()
    assert r["success"], r
    (group,) = r["groups"]
    assert group["group_id"] == 1
    assert group["dim_changes"]["n_experts"] == {"global": 256, "group": 128}
    # d_model is the residual stream: one group cannot change it
    r = client.post("/api/playground/compose-model", json={
        "base_model_id": "deepseek-ai/DeepSeek-V3",
        "groups": [{**_echo(sk["groups"][0]), "dims": {"d_model": 4096}}],
    }).json()
    assert [e["kind"] for e in r["errors"]] == ["interface_dim"]
