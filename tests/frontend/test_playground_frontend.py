"""Headless test for the Playground module-comparison page.

Self-contained harness (per the React headless-harness convention): serve the
built `web/dist` from a throwaway threaded HTTP server with SPA fallback, drive
it in headless chromium, and stub the three `/api` calls the page makes — so
the test needs neither the backend nor the shared :2053 dev server. Skips
cleanly if the app is not built or playwright/chromium is absent.

Asserts the page's contract with the backend DTO: it picks components, POSTs the
contract to `/api/playground/module-bench`, and renders a card per result plus a
gap notice per error — computing nothing itself.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import threading
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[2]
_WEB = _ROOT / "web"
_DIST = _WEB / "dist"


def _agents_answer() -> dict:
    import asyncio
    from backend.api.routes.meta import agent_backends
    return asyncio.run(agent_backends())


# `GET /api/agents` as the backend answers it (its own registry), for the
# stubs: the pages name the coding agents from it (`web/src/agents.ts`).
# Read at import — a route handler runs inside playwright's event loop.
AGENTS = _agents_answer()


def _sources_newer_than_dist() -> bool:
    """True if any web/src source is newer than the built index.html (or dist is
    absent) — i.e. the build is stale for what we're about to test."""
    index = _DIST / "index.html"
    if not index.is_file():
        return True
    built = index.stat().st_mtime
    return any(p.stat().st_mtime > built for p in (_WEB / "src").rglob("*")
               if p.is_file())


def _ensure_built() -> None:
    """Make web/dist current for THIS page — a stale dist (no Playground chunk,
    or built before the latest source edit, e.g. right after a merge) would
    serve an old app and the test would time out on a false negative. Rebuild
    when the built chunk is missing OR any source is newer than the build; reuse
    otherwise (fast); skip if node/npm is absent."""
    fresh = bool(list((_DIST / "web-assets").glob("Playground-*.js"))) \
        and not _sources_newer_than_dist()
    if fresh:
        return
    if shutil.which("npm") is None:
        pytest.skip("npm not available to build web/dist")
    try:
        subprocess.run(["npm", "run", "build"], cwd=str(_WEB), check=True,
                       capture_output=True, timeout=600)
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
        pytest.skip(f"web build unavailable: {exc}")

# Canned backend responses (the page renders these verbatim — zero client math).
_COMPONENTS = {
    "success": True,
    "components": [
        {"name": "mla_attention", "role": "attention", "attn_display": "MLA", "model_ids": ["deepseek-ai/DeepSeek-V3"]},
        {"name": "mla_dsa_attention", "role": "attention", "attn_display": "DSA", "model_ids": ["deepseek-ai/DeepSeek-V3.2", "zai-org/GLM-5.2", "zai-org/GLM-5.1"]},
        {"name": "gqa_attention", "role": "attention", "attn_display": "GQA", "model_ids": []},
        {"name": "swiglu_ffn", "role": "ffn_dense", "attn_display": None, "model_ids": []},
    ],
}
_MODELS = {"model_ids": ["deepseek-ai/DeepSeek-V3.2"]}
_BENCH_RESULTS = [
        {
            "component": "mla_attention", "entry": "mla_attention", "model_id": None, "role": "attention",
            "params": 100, "active_params": 100, "weight_bytes": 374214656,
            "units": {"weight_bytes": "per layer (one block copy) · stored once"},
            "dims_used": {"d_kv_lora": 512, "d_model": 7168, "n_experts": 256},
            "editable_dims": {"d_kv_lora": 512},
            "model_ids": ["deepseek-ai/DeepSeek-V3"],
            "kv": {"display": "MLA", "family": "MLA",
                   "unit": "bytes/layer/request @ cache_len = 4,096 tokens", "cache_len": 4096,
                   "storage_bytes_at_L": 4718592,
                   "nodes": [{"name": "kv_a_layernorm", "dtype": "bfloat16",
                              "law": "growing", "growing": True,
                              "elements_per_token": 512, "storage_bytes_at_L": 4194304,
                              "formula": {"symbolic": "d_kv_lora",
                                          "substituted": "= 512 el/tok"}},
                             {"name": "k_pe", "dtype": "bfloat16",
                              "law": "growing", "growing": True,
                              "elements_per_token": 64, "storage_bytes_at_L": 524288,
                              "formula": {"symbolic": "d_qk_rope",
                                          "substituted": "= 64 el/tok"}}]},
            "_prefill": (2225718689792, 10, 3, 4718592),
            "_decode_kv": 4718592,
            "block_graph": {
                "nodes": [
                    {"id": "x", "op": "input", "label": "input"},
                    {"id": "mla_attention|q_a_proj", "op": "linear", "label": "q_a_proj", "params": 11010048, "flops": 90194313216,
                     "flops_symbolic": "2 * B * S * in * out", "flops_values": "2 * 1 * 4096 * 7168 * 1536"},
                    {"id": "mla_attention|kv_b_proj", "op": "linear", "label": "kv_b_proj", "params": 37748736, "flops": 309237645312,
                     "flops_symbolic": "2 * B * S * in * out", "flops_values": "2 * 1 * 4096 * 7168 * 5248"},
                    # a param-less but flops-bearing op — must still show in the
                    # breakdown (it's what distinguishes attention variants).
                    {"id": "mla_attention|attention_score", "op": "attention_score", "label": "attention_score", "params": 0, "flops": 500000000000,
                     "flops_symbolic": "4 * B * S * S * n_heads * d_head", "flops_values": "4 * 1 * 4096 * 4096 * 128 * 128"},
                ],
                "edges": [
                    {"src": "x", "dst": "mla_attention|q_a_proj"},
                    {"src": "mla_attention|q_a_proj", "dst": "mla_attention|kv_b_proj"},
                    {"src": "mla_attention|kv_b_proj", "dst": "mla_attention|attention_score"},
                ],
            },
            # The same architecture after the runtime rewrite. Unlike
            # block_graph, its live kernel nodes carry the priced runtime row.
            "fused_graph": {
                "nodes": [
                    {"id": "x", "op": "input", "label": "input"},
                    {"id": "mla_attention|flash_attention", "op": "flash_attention",
                     "label": "attn_apply", "style": "fused", "fused": "flash_attention",
                     "badge": "3→1", "inputs": ["x"], "input_ports": [""],
                     "params": 0, "flops": 500_000_000_000,
                     "access_bytes": 200_000_000,
                     "arithmetic_intensity": 2.5, "priced_runtime": True},
                    {"id": "mla_attention|o_proj", "op": "linear", "label": "o_proj",
                     "style": "op", "inputs": ["mla_attention|flash_attention"],
                     "input_ports": [""], "params": 117_440_512,
                     "flops": 120_000_000_000, "access_bytes": 258_881_024,
                     "arithmetic_intensity": 0.46, "priced_runtime": True},
                ],
                "edges": [
                    {"src": "x", "dst": "mla_attention|flash_attention",
                     "shape": "[1,4096,7168]"},
                    {"src": "mla_attention|flash_attention", "dst": "mla_attention|o_proj",
                     "shape": "[1,4096,16384]"},
                ],
            },
            "fused_l1": {
                "available": True,
                "reason": None,
                "weight_semantics": "weight_read_step",
                "boxes": [
                    {"id": "input", "members": ["x"], "flops": 0, "params": 0,
                     "weight_bytes": 0, "ops": {}, "label": "input", "rank": 0},
                    {"id": "core", "members": ["mla_attention|flash_attention"],
                     "flops": 500_000_000_000, "params": 0, "weight_bytes": 0,
                     "ops": {"flash_attention": 1}, "label": "Attention core",
                     "rank": 1},
                    {"id": "output", "members": ["mla_attention|o_proj"],
                     "flops": 120_000_000_000, "params": 117_440_512,
                     "weight_bytes": 234_881_024, "ops": {"linear": 1},
                     "label": "Output", "rank": 2},
                ],
                "node_box": {
                    "x": "input",
                    "mla_attention|flash_attention": "core",
                    "mla_attention|o_proj": "output",
                },
                "totals": {"flops": 620_000_000_000, "params": 117_440_512,
                           "weight_bytes": 234_881_024},
                "core": "attention",
                "other_merges": False,
            },
        },
        {
            "component": "mla_dsa_attention", "entry": "mla_dsa_attention", "model_id": None, "role": "attention",
            "params": 100, "active_params": 100, "weight_bytes": 374214656,
            "units": {"weight_bytes": "per layer (one block copy) · stored once"},
            "dims_used": {"d_kv_lora": 512, "d_indexer_head": 128, "d_model": 7168},
            "editable_dims": {"d_kv_lora": 512, "d_indexer_head": 128},
            "model_ids": ["deepseek-ai/DeepSeek-V3.2", "zai-org/GLM-5.2", "zai-org/GLM-5.1"],
            "kv": {"display": "DSA", "family": "MLA",
                   "unit": "bytes/layer/request @ cache_len = 4,096 tokens", "cache_len": 4096,
                   "storage_bytes_at_L": 4980736,
                   "nodes": [{"name": "kv_a_layernorm", "dtype": "bfloat16",
                              "law": "growing", "growing": True,
                              "elements_per_token": 512, "storage_bytes_at_L": 4194304,
                              "formula": {"symbolic": "d_kv_lora",
                                          "substituted": "= 512 el/tok"}},
                             {"name": "indexer_keys_fp8", "dtype": "float8_e4m3fn",
                              "law": "growing", "growing": True,
                              "elements_per_token": 192, "storage_bytes_at_L": 786432,
                              "formula": {"symbolic": "d_indexer_head",
                                          "substituted": "= 192 el/tok"}}]},
            "_prefill": (2131724863488, 10, 3, 4980736),
            "_decode_kv": 4587520,
            "block_graph": {
                "nodes": [
                    {"id": "x", "op": "input", "label": "input"},
                    {"id": "mla_dsa_attention|q_a_proj", "op": "linear", "label": "q_a_proj", "params": 11010048, "flops": 90194313216},
                    {"id": "mla_dsa_attention|indexer_logits", "op": "matmul", "label": "indexer_logits", "params": 0, "flops": 5000000000},
                ],
                "edges": [
                    {"src": "x", "dst": "mla_dsa_attention|q_a_proj"},
                    {"src": "mla_dsa_attention|q_a_proj", "dst": "mla_dsa_attention|indexer_logits"},
                ],
            },
        },
        {
            "component": "gqa_attention", "entry": "gqa_attention", "model_id": None, "role": "attention",
            "params": 100, "active_params": 100, "weight_bytes": 200000000,
            "units": {"weight_bytes": "per layer (one block copy) · stored once"},
            "dims_used": {"d_head": 56, "d_model": 7168},
            "editable_dims": {"d_head": 56},
            "model_ids": [],
            "kv": {"display": "GQA", "family": "GQA",
                   "unit": "bytes/layer/request @ cache_len = 4,096 tokens", "cache_len": 4096,
                   "storage_bytes_at_L": 1835008, "nodes": []},
            "_prefill": (1000000000000, 10, 3, 1835008),
            "_decode_kv": 1835008,
        },
]



def _phase(flops, act, act_fused, kv, weight_read, runs_as, decode):
    step = ("one decode step @ batch 1 × 1 new token, cache_len = 4,096 tokens"
            if decode else "one forward @ batch 1 × 4,096 tokens")
    units = {"flops": f"per layer · {step}", "act_bytes": f"per layer · {step}, bf16",
             "traffic": f"per layer · {step} · KV × batch",
             "weight_read": f"per layer · {step} · weights + scales (routed experts: as far as tokens reach)"}
    row = {"flops": flops, "act_read_bytes": act, "act_write_bytes": act,
           "act_read_bytes_fused": act_fused, "act_write_bytes_fused": act_fused,
           "act_bytes_naive": act + act, "act_bytes_fused": act_fused + act_fused,
           "weight_read_bytes": weight_read,
           "kv_read_bytes": kv if decode else 0,
           "memory_dominant": "kv" if kv > weight_read else "weights",
           "fused_exceeds_naive": act_fused > act,
           "runs_as": runs_as, "units": units}
    if not decode:
        row["kv_step_bytes"] = kv
    return row


# Per-phase numbers (backend `phases`): prefill mirrors the row's top-level
# fields; decode is one new token against the cache. mla_dsa_attention runs as
# its declared decode form at decode. gqa is given a big cache so its decode
# step is KV-read dominated while the MLA ones are weight-read dominated.
_DECODE = {
    "mla_attention": (1517736320, 3000000, 2000000, "mla_attention"),
    # A deliberately asymmetric split: the chart must read the backend total,
    # not recompute it from read/write fields.
    "mla_dsa_attention": (1041293952, 2500000, 300000000, "mla_dsa_attention_absorbed"),
    "gqa_attention": (151681024, 900000, 800000, "gqa_attention"),
}
_SWEEP = [1024, 2048, 4096, 8192, 16384, 32768, 131072, 262144, 1048576]


def _sweep(per_token, state, layers):
    def whole(n):
        total = (per_token * n + state) * layers
        return {"symbolic": f"0/self_attn × {layers}",
                "substituted": f"= {total:,d} B"}
    return [{"cache_len": n, "storage_bytes": per_token * n + state,
             "model_storage_bytes": (per_token * n + state) * layers,
             "model_formula": whole(n)} for n in _SWEEP]


def _storage(weight_bytes, precision, note):
    return {"bytes": weight_bytes, "parameter_bytes": weight_bytes, "metadata_bytes": 0,
            "buffer_bytes": 0, "by_dtype": [], "precision": precision,
            "source": {"model_id": "deepseek-ai/DeepSeek-V3.2", "bound": False, "invoked": True,
                       "role": "attention", "address": "model.layers.X.self_attn.", "note": note}}


def _op_table_for(row, phase):
    """A compact representation of the backend's post-fusion payload.

    The stub's `block_graph` remains unfused for the structure diff; runtime
    rows are intentionally separate so the test can exercise fused origins and
    phase-dependent kernels without replaying the evaluator in the browser.
    """
    decode = phase == "decode"

    def make(kernel, label, origins, params, active, flops, act_read,
             act_write, weight=0, kv=0):
        total = act_read + act_write + weight + kv
        return {
            "node_id": f"{kernel}:{label}:{phase}",
            "kernel": kernel, "label": label, "origins": origins,
            "params": params, "active_params": active, "flops": flops,
            "act_read_bytes": act_read, "act_write_bytes": act_write,
            "weight_read_bytes": weight, "kv_read_bytes": kv,
            "total_bytes": total,
            "arithmetic_intensity": flops / total if total else None,
            "formulas": {
                "params": {"symbolic": "params", "substituted": f"= {params:,d}"},
                "flops": {"symbolic": "flops", "substituted": f"= {flops:,d} FLOPs"},
                "total_bytes": {
                    "symbolic": "act_read + act_write + weights + kv_read",
                    "substituted": f"= {total:,d} B",
                },
            },
        }

    if row["component"] == "mla_attention":
        out = [
            make("flash_attention", "attn_apply",
                ["qk_score", "softmax", "attn_apply_latent" if decode else "attn_apply"],
                0, 0, 80_000 if decode else 500_000_000_000,
                40_000 if decode else 120_000_000, 20_000 if decode else 80_000_000,
                kv=8192 if decode else 0),
            make("o_proj", "o_proj", [], 117_440_512, 117_440_512,
                20_000 if decode else 120_000_000_000,
                16_384 if decode else 16_000_000,
                8_192 if decode else 8_000_000, weight=234_881_024),
        ]
    elif row["component"] == "mla_dsa_attention":
        out = [
            make("indexer_logits", "indexer_logits", [], 0, 0,
                8_000_000 if decode else 800_000_000_000,
                16_000 if decode else 16_000_000,
                2_000_000 if decode else 2_000_000_000, kv=65_536 if decode else 0),
            make("dsa_topk_mask", "dsa_topk_mask", [], 0, 0,
                512 if decode else 500_000_000,
                2_000_000 if decode else 2_000_000_000,
                4_096 if decode else 4_096_000),
            make("flash_dsa", "attn_apply",
                ["qk_score", "softmax", "attn_apply_latent" if decode else "attn_apply"],
                0, 0, 80_000 if decode else 84_000_000_000,
                40_000 if decode else 40_000_000, 20_000 if decode else 20_000_000,
                kv=8192 if decode else 0),
        ]
    elif row["component"] == "gated_delta_net_fused_attention":
        out = [
            make("merged_column_parallel_linear", "in_proj_ba",
                ["in_proj_qkvz", "in_proj_ba"], 25_296_896, 25_296_896,
                50_000 if decode else 50_000_000_000,
                4_096 if decode else 4_000_000, 2_048 if decode else 2_000_000,
                weight=50_593_792),
            make("gated_delta_rule", "delta_rule", ["delta_rule"], 0, 0,
                20_000 if decode else 20_000_000_000,
                8_192 if decode else 8_000_000, 4_096 if decode else 4_000_000),
        ]
    else:
        out = []

    out.sort(key=lambda r: (-r["flops"], -r["total_bytes"], -r["params"]))
    total_params = sum(r["params"] for r in out)
    total_active = sum(r["active_params"] for r in out)
    total_flops = sum(r["flops"] for r in out)
    total_bytes = sum(r["total_bytes"] for r in out)
    for r in out:
        r["params_share"] = (r["params"] / total_params) if total_params else 0.0
        r["flops_share"] = (r["flops"] / total_flops) if total_flops else 0.0
        r["bytes_share"] = (r["total_bytes"] / total_bytes) if total_bytes else 0.0
    return {
        "contract": {"phase": phase, "batch": 1, "input_tokens": 4096,
                     "fusion": "shipped", "parallel": "single_rank_unsharded",
                     "activation_dtype": "BF16",
                     "precision": {"profile_key": "k", "label": "As shipped",
                                   "preset": "as_shipped", "canonical": {}}},
        "rows": out,
        "totals": {"params": total_params, "active_params": total_active,
                   "flops": total_flops, "total_bytes": total_bytes,
                   "arithmetic_intensity": total_flops / total_bytes if total_bytes else None},
    }


for _r in _BENCH_RESULTS:
    _df, _da, _dfa, _runs = _DECODE[_r["component"]]
    _pf, _pa, _pfa, _pkv = _r.pop("_prefill")
    _dkv = _r.pop("_decode_kv")
    _r["weight_storage"] = _storage(
        _r["weight_bytes"], "checkpoint precision: FP8 (E4M3), some BF16",
        "unbound: priced at deepseek-ai/DeepSeek-V3.2's quant context")
    _kv = _r["kv"]
    _kv["bytes_per_token"] = _kv["storage_bytes_at_L"] // 4096
    _kv["state_bytes"] = 0
    _kv["storage_by_cache_len"] = _sweep(_kv["bytes_per_token"], 0, 61)
    _kv["model_layers"] = {"model_id": "deepseek-ai/DeepSeek-V3.2", "layers": 61}
    _r["phases"] = {
        "prefill": _phase(_pf, _pa, _pfa, _pkv, _r["weight_bytes"], _r["component"], False),
        "decode": _phase(_df, _da, _dfa, _dkv, _r["weight_bytes"], _runs, True),
    }
    _r["op_tables"] = {phase: _op_table_for(_r, phase)
                       for phase in ("prefill", "decode")}

# A linear-attention entry: no per-token KV, a constant recurrent state.
_GDN_ROW = json.loads(json.dumps(_BENCH_RESULTS[0]))
_GDN_ROW.update({"component": "gated_delta_net_fused_attention", "entry": "gated_delta_net_fused_attention",
                 "model_ids": ["Qwen/Qwen3-Coder-Next"], "weight_bytes": 67436928})
_GDN_ROW["kv"] = {"display": "GatedDeltaNet", "family": "GatedDeltaNet",
                  "unit": "bytes/layer/request @ cache_len = 4,096 tokens", "cache_len": 4096,
                  "storage_bytes_at_L": 1097728,
                  "bytes_per_token": 0, "state_bytes": 1097728,
                  "storage_by_cache_len": _sweep(0, 1097728, 36),
                  "model_layers": {"model_id": "Qwen/Qwen3-Coder-Next", "layers": 36},
                  "nodes": [{"name": "recurrent_state", "dtype": "bfloat16", "law": "constant",
                             "growing": False, "elements_per_token": None,
                             "elements_per_request": 548864, "storage_bytes_at_L": 1097728,
                             "formula": {"symbolic": "n_v_heads * d_head * d_state",
                                         "substituted": "= 548864 el/req"}}]}
_GDN_ROW["weight_storage"] = _storage(67436928, "checkpoint precision: BF16",
                                      "as stored in Qwen/Qwen3-Coder-Next's checkpoint")
_GDN_ROW["phases"] = {
    "prefill": _phase(900000000000, 10, 3, 1097728, 67436928, "gated_delta_net_fused_attention", False),
    "decode": _phase(548244576, 10, 3, 1097728, 67436928, "gated_delta_net_fused_attention", True),
}
_GDN_ROW.pop("block_graph", None)
_GDN_ROW.pop("fused_graph", None)
_GDN_ROW.pop("fused_l1", None)
_GDN_ROW["op_tables"] = {"prefill": _op_table_for(_GDN_ROW, "prefill"),
                         "decode": _op_table_for(_GDN_ROW, "decode")}

# Gaps carry FACTS ONLY — which dims nothing binds (there are no suggested
# values anywhere in the contract; repair is binding the column to a model
# that parameterizes the component).
_GAP_DSA = {"component": "mla_dsa_attention", "entry": "mla_dsa_attention",
            "error": "missing dim: d_indexer_head — the base model does not bind it",
            "missing_dims": ["d_indexer_head"]}
_GAP_GQA = {"component": "gqa_attention", "entry": "gqa_attention",
            "error": "missing dims: d_head, n_kv_heads — the base model does not bind them",
            "missing_dims": ["d_head", "n_kv_heads"]}

def _graph_diff(results):
    """A stand-in for the backend's compute-graph diff over the canned rows:
    nodes pair when their ids agree past the component prefix (the real diff
    pairs from declarations; the page only draws what it is handed). Every
    node not in all rows is its own one-class path."""
    n = len(results)
    key = lambda nid: nid.split("|", 1)[-1]
    order, members = [], {}
    for i, r in enumerate(results):
        for node in r["block_graph"]["nodes"]:
            k = key(node["id"])
            if k not in members:
                order.append(k)
                members[k] = [None] * n
            members[k][i] = node["id"]
    cid = {k: f"c{m}" for m, k in enumerate(order)}
    classes = [{"id": cid[k], "members": members[k], "basis": [], "boxes": [None] * n,
                "diff": {}, "view": False, "partition": 0} for k in order]
    edges = {}
    for i, r in enumerate(results):
        for e in r["block_graph"]["edges"]:
            edges.setdefault((cid[key(e["src"])], cid[key(e["dst"])]), []).append(i)
    paths = []
    for c in classes:
        own = [i for i, m in enumerate(c["members"]) if m]
        if len(own) < n:
            paths.append({"id": f"p{len(paths)}", "classes": [c["id"]], "owners": own, "from": [],
                          "into": [], "boxes": [], "chain": ["op"], "group": None, "partition": 0,
                          "sums": [{"params": 1, "flops": 1, "weight_bytes": 2} if i in own else None
                                   for i in range(n)],
                          "kv": [None] * n})
    return {"available": True, "reason": None, "entries": [r["entry"] for r in results],
            "grouping": "topological", "classes": classes,
            "edges": [{"src": s, "dst": d, "port": 0, "owners": o} for (s, d), o in edges.items()],
            "paths": paths, "groups": [], "skeleton_boxes": [], "reads": [[] for _ in results],
            "boxes": [], "totals": {"delta": [{"params": 0, "flops": 0, "weight_bytes": 0}] * n,
                                    "regrouped": False}}


# component-dims without a model_id → every binding of the REQUESTED component
# with the dims each resolves to (the bindings table), plus `failed` for
# bindings that didn't resolve (reported, never dropped).
def _call(component, call="1/self_attn", label="backbone · layers 3–60 · self_attn",
          model_calls=1, model_parameterizations=None):
    """One call of `component` in a model, as the backend lists it, with how
    many calls of it the model makes and how many of those evaluate
    differently (all of them unless said), and which parameterization this
    call is (calls that differ are numbered by their call index)."""
    params = model_calls if model_parameterizations is None else model_parameterizations
    return {"call": call, "label": label, "component": component,
            "formals": [["B", "s_ctx", "d_model"]], "model_calls": model_calls,
            "model_parameterizations": params,
            "parameterization": 0 if params == 1 else int(call.split("/", 1)[0])}


#: One row per call of the component in each binding model (each model here
#: calls it once).
_BINDINGS_BY_COMPONENT = {
    "mla_attention": [
        {"model_id": "deepseek-ai/DeepSeek-V3", **_call("mla_attention"),
         "dims": {"d_kv_lora": 512}},
        {"model_id": "deepseek-ai/DeepSeek-V3.2", **_call("mla_attention"),
         "dims": {"d_kv_lora": 512, "d_rope": 64}},
        # a TWIN of V3 (identical dims) — the bindings table marks it "≡ DeepSeek-V3"
        {"model_id": "deepseek-ai/DeepSeek-V3.1", **_call("mla_attention"),
         "dims": {"d_kv_lora": 512}},
    ],
    "mla_dsa_attention": [
        {"model_id": "deepseek-ai/DeepSeek-V3.2", **_call("mla_dsa_attention"),
         "dims": {"d_kv_lora": 512, "d_indexer_head": 128}},
        {"model_id": "zai-org/GLM-5.2", **_call("mla_dsa_attention"),
         "dims": {"d_kv_lora": 512, "d_indexer_head": 192}},
    ],
}


def _row_for(name, entry_key, model_id, entry_dims=None, call=None, calls=()):
    """One canned result row stamped with the REQUESTING entry's identity and
    resolved dims — the real backend derives entry/model_id/editable_dims per
    entry, so two entries of one component are two rows with their own dims.
    Entry dims only override dims the component actually exposes (the canned
    editable_dims allowlist) — like the backend's own-knobs filter."""
    row = json.loads(json.dumps(next(r for r in _BENCH_RESULTS if r["component"] == name)))
    row["entry"] = entry_key
    row["model_id"] = model_id
    if call is not None:
        row["call"] = next(c for c in calls if c["call"] == call)
        row["calls"] = list(calls)
    if entry_dims:
        own = {k: v for k, v in entry_dims.items() if k in row["editable_dims"]}
        row["editable_dims"] = {**row["editable_dims"], **own}
        row["dims_used"] = {**row["dims_used"], **own}
    return row


def _bench_payload(body):
    """A module-bench stand-in honoring the real entries contract:
    - one result row PER ENTRY (same component twice → two rows);
    - an entry lands in results XOR errors, never both;
    - an unseeded bound entry is seeded inline from its binding — and a binding
      that can't resolve is an ERROR (never a silent base-dims fallback);
    - an unbound entry resolves with base dims when the BASE itself binds the
      component (the stub base is DeepSeek-V3.2);
    - the echo folds seeds in (empty applied dims omit the key), and names
      each seed apart (`seed_dims`) as the baseline of the entry's edits;
    - graph_diff whenever there are ≥2 entries (two bindings of one component
      count as two): drawn when every entry benched, else declined with the
      failing entries named (same gate as the backend)."""
    entries = body.get("entries") or [{"name": n} for n in body.get("components", [])]
    base = "deepseek-ai/DeepSeek-V3.2"
    applied: dict = {}
    seeds: dict = {}
    results, errors = [], []
    for e in entries:
        name = e["name"]
        dims = dict(e.get("dims") or {})
        # identity: explicit label (a saved variant) or the component×binding×call key
        key = e.get("label") or (f"{name}@{e['model_id']}" if e.get("model_id") else name)
        if e.get("call") and not e.get("label"):
            key = f"{key}#{e['call']}"
        rows = [b for b in _BINDINGS_BY_COMPONENT.get(name, [])
                if b["model_id"] == e.get("model_id")]
        calls = [{k: b[k] for k in ("call", "label", "component", "formals")} for b in rows]
        if e.get("call") and any(b.get("error") for b in rows if b["call"] == e["call"]):
            errors.append({"component": name, "entry": key, "model_id": e.get("model_id"),
                           "error": next(b["error"] for b in rows if b["call"] == e["call"]),
                           "missing_dims": [], "calls": calls})
            continue
        if e.get("model_id") and len(rows) > 1 and not e.get("call"):
            errors.append({"component": name, "entry": key, "model_id": e["model_id"],
                           "error": f"{e['model_id']} calls {name!r} {len(rows)} times; "
                                    f"name the call to evaluate it at (`call`)",
                           "missing_dims": [], "calls": calls})
            continue
        if e.get("model_id") and not e.get("seeded"):
            seed = {b["call"]: b["dims"] for b in rows}.get(
                e.get("call") or (rows[0]["call"] if rows else None))
            if seed is None:
                errors.append({"component": name, "entry": key, "model_id": e["model_id"],
                               "error": f"cannot resolve {name} as {e['model_id']} parameterizes it",
                               "missing_dims": []})
                continue
            seeds[key] = dict(seed)
            for k, v in seed.items():
                dims.setdefault(k, v)
        if dims:
            applied[key] = dims
        if name == "gqa_attention" and "d_head" not in dims:
            err = dict(_GAP_GQA)
            err["entry"] = key
            errors.append(err)
            continue
        results.append(_row_for(name, key, e.get("model_id"), dims,
                                call=e.get("call"), calls=calls))
    payload = {
        "success": True,
        "base_model_id": base,
        "contract": {"batch": 1, "input_tokens": 4096,
                     "per_component_dims": applied,
                     "seed_dims": seeds,
                     "phases": {"prefill": {"batch": 1, "s_query": 4096, "s_kv": 4096},
                                "decode": {"batch": 1, "s_query": 1, "s_kv": 4096}}},
        "results": results,
        "errors": errors,
    }
    if len(entries) >= 2:
        payload["graph_diff"] = (
            {"available": False,
             "reason": f"{len(errors)} of {len(entries)} entries did not bench: "
                       + ", ".join(e["entry"] for e in errors)}
            if errors else _graph_diff(results))
    return payload


def _ensure_picker(page):
    """The chip catalog hides after the first results; + Add opens it."""
    groups = page.locator("#pg-groups")
    if not groups.is_visible():
        page.locator("#pg-add-pick").click()
        groups.wait_for(state="visible")


def _wait_picks(page, n):
    """The sticky config row shows exactly `n` entries."""
    page.wait_for_function(
        "n => document.querySelectorAll('#pg-picks [data-entry]').length === n",
        arg=n, timeout=15000)


def _run_picks(page):
    """Selection changes batch in the sticky row; this submits the whole set."""
    button = page.locator("#pg-run-picks")
    button.wait_for(state="visible", timeout=15000)
    button.click(timeout=15000)  # waits for the button to be enabled


def _ensure_details(page):
    """#pg-details is one fold; open it so dims / runtime table / provenance show."""
    el = page.locator("#pg-details")
    el.wait_for(state="attached", timeout=15000)
    if el.get_attribute("open") is None:
        el.locator("summary").first.click()
        page.wait_for_function("() => document.querySelector('#pg-details').open")


def _ensure_dims(page):
    """The editable-dims table is a fold inside #pg-details."""
    _ensure_details(page)
    summary = page.locator("#pg-details summary:has-text('internal dims')")
    summary.wait_for(state="visible", timeout=5000)
    box = summary.locator("xpath=..")
    if box.get_attribute("open") is None:
        summary.click()
        page.wait_for_function(
            """() => {
                const s = [...document.querySelectorAll('#pg-details summary')]
                    .find(e => e.textContent.includes('internal dims'));
                return s && s.parentElement.open;
            }""")


def _ensure_bound_models(page):
    page.locator("#pg-bound-models").wait_for(state="visible")
    heading = page.locator("#pg-bound-models >> text=as each shipped model parameterizes it")
    heading.first.wait_for(state="visible")


def _serve_dist():
    """Threaded static server for web/dist with SPA fallback (any non-file path
    → index.html), so /playground.html resolves to the app."""
    class Handler(SimpleHTTPRequestHandler):
        def do_GET(self):
            rel = self.path.split("?", 1)[0].lstrip("/")
            target = _DIST / rel
            if not rel or not target.is_file():
                self.path = "/index.html"
            return super().do_GET()

        def log_message(self, *a):
            pass

    httpd = ThreadingHTTPServer(("127.0.0.1", 0), partial(Handler, directory=str(_DIST)))
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    return httpd


def test_playground_renders_and_compares():
    pytest.importorskip("playwright.sync_api")
    from playwright.sync_api import sync_playwright

    _ensure_built()
    if not (_DIST / "index.html").exists():
        pytest.skip("web/dist not built (run `npm run build` in web/)")

    httpd = _serve_dist()
    port = httpd.server_address[1]
    base = f"http://127.0.0.1:{port}"
    try:
        with sync_playwright() as p:
            try:
                browser = p.chromium.launch()
            except Exception as exc:  # chromium not installed
                pytest.skip(f"chromium unavailable: {exc}")
            page = browser.new_page()

            def _json(route, payload):
                route.fulfill(status=200, content_type="application/json",
                              body=json.dumps(payload))

            captured = {"bodies": []}

            def _bench(route):
                body = json.loads(route.request.post_data or "{}")
                captured["bodies"].append(body)
                payload = _bench_payload(body)
                captured["payload"] = payload
                _json(route, payload)

            def _component_dims(route):
                # with model_id → one binding's seed (only REAL bindings resolve;
                # the base-capability probe uses this to gate the "base dims"
                # option); without → the full table for the REQUESTED component
                from urllib.parse import urlparse, parse_qs
                q = parse_qs(urlparse(route.request.url).query)
                comp = q.get("component", [""])[0]
                if "model_id" in q:
                    mid = q["model_id"][0]
                    seed = {b["model_id"]: b["dims"]
                            for b in _BINDINGS_BY_COMPONENT.get(comp, [])}.get(mid)
                    if seed is None:
                        _json(route, {"success": False,
                                      "message": f"{comp} does not resolve under {mid}"})
                    else:
                        _json(route, {"success": True, "dims": seed})
                else:
                    _json(route, {"success": True, "component": comp,
                                  "bindings": _BINDINGS_BY_COMPONENT.get(comp, []),
                                  "failed": {}})

            page.route("**/api/playground/components", lambda r: _json(r, _COMPONENTS))
            page.route("**/supported-models", lambda r: _json(r, _MODELS))
            page.route("**/api/playground/module-bench", _bench)
            page.route("**/api/playground/component-dims**", _component_dims)

            # "my modules": a signed-in session — the cookie flow's session
            # probe (/api/auth/me) answers with a user — two saved variants
            # of mla_attention, and a POST hook that records save bodies.
            captured["module_posts"] = []
            module_rows = [{"id": "v1", "component": "mla_attention", "name": "big-kv",
                            "dims": {"d_kv_lora": 640}, "base_model_id": "deepseek-ai/DeepSeek-V3.2"}]

            def _modules(route):
                if route.request.method == "POST":
                    body = json.loads(route.request.post_data or "{}")
                    captured["module_posts"].append(body)
                    module_rows.append({"id": "v2", "component": body["component"],
                                        "name": body["name"], "dims": body["dims"],
                                        "base_model_id": body.get("base_model_id")})
                    _json(route, {"success": True, "variant": module_rows[-1]})
                else:
                    from urllib.parse import urlparse, parse_qs
                    q = parse_qs(urlparse(route.request.url).query)
                    comp = q.get("component", [""])[0]
                    variants = [v for v in module_rows if not comp or v["component"] == comp]
                    _json(route, {"success": True, "variants": variants})
            page.route("**/api/auth/me", lambda r: _json(r, {
                "authEnabled": True,
                "user": {"id": "u1", "email": "e2e@example.com", "user_metadata": {}},
                "profile": {"username": None},
                "isAdmin": False,
            }))
            page.route("**/api/account/modules**", _modules)

            page.goto(f"{base}/playground.html")
            # family groups render collapsed: header rows visible, chips hidden
            page.wait_for_selector("button.uppercase:has-text('MLA')", timeout=15000)
            headers = {h.inner_text().replace("▸", "").replace("▾", "").split("·")[0].strip()
                       for h in page.query_selector_all("button.uppercase")}
            assert {"MLA", "DSA", "GQA"} <= headers, headers
            assert page.query_selector("#pg-groups >> text=mla_attention") is None  # collapsed
            # expand the groups to reach their chips
            for fam in ("MLA", "DSA", "GQA"):
                page.click(f"button.uppercase:has-text('{fam}')")
            page.wait_for_selector("#pg-groups >> text=mla_attention")

            # search filter narrows to matching chips and auto-expands their group
            page.fill("input[placeholder='Filter components…']", "dsa")
            page.wait_for_selector("#pg-groups >> text=mla_dsa_attention")
            assert page.query_selector("#pg-groups >> text=gqa_attention") is None
            page.fill("input[placeholder='Filter components…']", "")
            page.wait_for_selector("#pg-groups >> text=gqa_attention")

            # Picking a component adds one entry per DISTINCT binding — the
            # component × parameterization matrix is the default, and every
            # column carries a known model provenance (no mystery base column).
            page.click("#pg-groups >> text=mla_attention")
            _wait_picks(page, 2)
            # gqa binds nothing in this stub: the chip does not invent an
            # unbound column, and the note says so. Gap cards are the
            # seed-failure path below (rebind to an unresolvable model).
            _ensure_picker(page)
            page.click("#pg-groups >> text=gqa_attention")
            page.wait_for_timeout(1500)
            assert page.locator("#pg-picks [data-entry]").count() == 2
            # switching role clears the selection so an attention pick can't
            # bleed into an ffn comparison
            _ensure_picker(page)
            page.select_option("#pg-role", "ffn_dense")
            page.select_option("#pg-role", "attention")
            assert page.locator("#pg-picks [data-entry]").count() == 0  # cleared

            # role switch re-collapses the groups
            for fam in ("MLA", "DSA"):
                page.click(f"button.uppercase:has-text('{fam}')")
            page.click("#pg-groups >> text=mla_attention")
            _ensure_picker(page)
            page.click("#pg-groups >> text=mla_dsa_attention")
            _wait_picks(page, 4)
            assert captured["bodies"] == []  # staging: neither chip fired a bench
            # mixed architectures keep Bound models — one table, a section per component
            _ensure_bound_models(page)
            page.wait_for_selector("#pg-bound-models tbody[data-component='mla_attention']")
            page.wait_for_selector("#pg-bound-models tbody[data-component='mla_dsa_attention']")
            bound_txt = page.inner_text("#pg-bound-models")
            assert "mla_attention" in bound_txt and "mla_dsa_attention" in bound_txt
            assert bound_txt.count("as each shipped model parameterizes it") == 1
            assert page.query_selector("#pg-bound-models th:has-text('component')") is None
            _run_picks(page)

            # 2 architectures × 2 bindings each: four entries, one
            # compute-graph diff, one colour per entry
            page.wait_for_selector("#pg-graph-diff .dag-node-slot", timeout=15000)
            assert page.query_selector("#pg-graph-diff .dag-node-box.s-diff-shared") is not None
            # the legend names every entry
            legend = page.inner_text("#pg-graph-diff [data-diff-legend]")
            assert "shared by every entry" in legend
            assert legend.count("mla_dsa_attention @") == 2 and legend.count("mla_attention @") == 2
            main_txt = page.inner_text("main")
            assert "@ GLM-5.2" in main_txt and "@ DeepSeek-V3" in main_txt

            # Each entry's own graph is symmetric: Declared / Fused, each with
            # Structure / Detailed. Fused Structure boxes close to runtime rows;
            # Fused Detailed shows those kernels with FLOPs/HBM on each node.
            page.click("#pg-structure summary:has-text(\"each entry's own graph\")")
            page.click("#pg-structure button:has-text('mla_attention @') >> nth=0")
            page.wait_for_timeout(300)
            overlay_legend = page.inner_text("#pg-graph-diff [data-diff-legend]")
            page.click("#pg-structure button:has-text('Fused')")
            page.wait_for_selector(
                "#pg-structure [data-node] >> text=Attention core", timeout=15000)
            assert page.inner_text("#pg-structure").find("box weights are this step's reads") >= 0
            page.click("#pg-structure button:has-text('Detailed')")
            page.wait_for_selector(
                "#pg-structure [data-node] .dag-node-box.s-fused", timeout=15000)
            assert page.locator("#pg-structure [data-fused-resource]").count() >= 2
            assert page.inner_text("#pg-structure").find("after shipped fusions") >= 0
            page.click("#pg-structure button:has-text('Declared')")
            page.wait_for_selector(
                "#pg-structure [data-node*='attention_score']", timeout=15000)
            page.click("#pg-structure button:has-text('Fused')")
            page.wait_for_selector(
                "#pg-structure [data-node] >> text=Attention core", timeout=15000)
            assert page.inner_text("#pg-graph-diff [data-diff-legend]") == overlay_legend
            assert page.query_selector("#pg-graph-diff .dag-node-box.s-diff-shared") is not None
            _ensure_dims(page)
            dims_txt = page.inner_text("#pg-details")
            assert "d_kv_lora" in dims_txt and "d_indexer_head" in dims_txt

            # the kv tensor table lives in #pg-kv as a fold: union of per-column
            # tensors, with law/dtype per cell; rows highlight on ANY visible
            # difference (elements, dtype, law) — and NOT when all columns agree
            assert page.query_selector("#pg-kv") is not None
            page.click("#pg-kv summary:has-text('kvcache tensors')")
            page.wait_for_selector("#pg-kv tr:has-text('indexer_keys_fp8')", timeout=5000)
            assert "kv_a_layernorm" in page.inner_text("#pg-kv") and "indexer_keys_fp8" in page.inner_text("#pg-kv")
            kv_diff_row = page.query_selector("#pg-kv tr:has-text('indexer_keys_fp8')")
            assert kv_diff_row is not None and kv_diff_row.query_selector("td.bg-amber-50") is not None
            kv_same_row = page.query_selector("#pg-kv tr:has-text('kv_a_layernorm')")
            assert kv_same_row is not None and kv_same_row.query_selector("td.bg-amber-50") is None
            # element counts pop over the same formulaHTML identity the KV panel uses
            assert kv_same_row.query_selector(".underline") is not None

            # editing a bound column's dim marks it as a customization of that
            # binding (dsa@GLM-5.2's d_indexer_head 192 → 128)
            _ensure_dims(page)
            row_inputs = page.query_selector_all("tr:has-text('d_indexer_head') input[type=number]")
            assert len(row_inputs) == 2  # both dsa columns
            row_inputs[1].fill("128")
            row_inputs[1].press("Tab")
            page.wait_for_selector("text=· edited", timeout=15000)

            # seed-failure path: rebind the dsa@GLM-5.2 column (nth=3) to
            # GLM-5.1 — listed as a binding in this stub but UNRESOLVABLE. The
            # entry becomes an honest error (never a silent base-dims fallback
            # under an @ model label), and the card's bind-to select repairs it
            captured["bodies"].clear()
            _ensure_dims(page)
            page.select_option("#pg-details select[title^='Bind this column'] >> nth=3", "zai-org/GLM-5.1")
            _run_picks(page)  # a rebind is a selection change: it stages
            page.wait_for_selector("text=Gaps (1)", timeout=15000)
            gaps_txt = page.inner_text("#pg-gaps")
            assert "cannot resolve mla_dsa_attention as zai-org/GLM-5.1" in gaps_txt
            assert page.query_selector("#pg-gaps input") is None  # facts only — no repair form
            page.select_option("#pg-gaps select", "zai-org/GLM-5.2")
            _run_picks(page)
            page.wait_for_selector("#pg-gaps", state="detached", timeout=15000)
            page.wait_for_selector("text=mla_dsa_attention @ GLM-5.2", timeout=15000)

            # after repairing the gap, the four bound columns remain; there is
            # no empty "base dims" option on the bind select
            _ensure_dims(page)
            page.wait_for_timeout(1200)
            assert page.query_selector("#pg-gaps") is None

            # Details paints Inference's PerOpTable over each entry's runtime
            # kernels for the selected phase — the backend's
            # `op_tables[phase].rows`, one leaf row per kernel, with formula
            # hover on FLOPs.
            _ensure_details(page)
            page.wait_for_selector("#pg-op-table", timeout=5000)
            assert page.query_selector("#pg-op-table [data-formula-pop]") is not None
            payload = captured["payload"]

            def shown_rows(phase):
                """Per entry, the kernel ids its table paints — and what the
                backend's phase table holds for it."""
                painted = {
                    h.get_attribute("data-group-header"): sorted(
                        tr.get_attribute("data-aggregation-id")
                        for tr in h.query_selector_all("[data-leaf-row]"))
                    for h in page.query_selector_all("#pg-op-table [data-group-header]")
                }
                expected = {
                    row["entry"]: sorted(op["node_id"] for op in row["op_tables"][phase]["rows"])
                    for row in payload["results"] if row["op_tables"][phase]["rows"]
                }
                return painted, expected

            painted, expected = shown_rows("prefill")
            assert painted == expected and expected
            page.click("#pg-op-table [aria-label='Runtime table phase'] button:has-text('Decode')")
            page.wait_for_timeout(100)
            painted, expected = shown_rows("decode")
            assert painted == expected and expected
            page.click("#pg-op-table [aria-label='Runtime table phase'] button:has-text('Prefill')")
            assert "via kernel fusion" not in page.inner_text("#pg-details")

            # back to ONE architecture: drop the two dsa entries; the two mla
            # bindings stay as two entries of one component — still a diff
            # (every node paired). Bound models is one section.
            _ensure_picker(page)
            page.locator('#pg-picks [data-entry*="GLM-5.2"] button[title="Remove"]').click()
            page.locator('#pg-picks [data-entry*="mla_dsa_attention"] button[title="Remove"]').click()
            _run_picks(page)
            _ensure_bound_models(page)
            bound_txt = page.inner_text("#pg-bound-models")
            assert "mla_attention" in bound_txt
            assert "mla_dsa_attention" not in bound_txt
            assert bound_txt.count("as each shipped model parameterizes it") == 1
            assert page.query_selector("#pg-bound-models th:has-text('component')") is None
            page.wait_for_selector("#pg-graph-diff [data-legend-item='entry:2']", state="detached", timeout=15000)
            main_txt = page.inner_text("main")
            assert "mla_attention @ DeepSeek-V3" in main_txt
            assert "mla_attention @ DeepSeek-V3.2" in main_txt
            assert page.query_selector("#pg-graph-diff") is not None
            assert "as each shipped model parameterizes it" in main_txt
            # a twin binding (V3.1, dims identical to V3) is marked as such and
            # not addable as a duplicate column
            assert "≡ DeepSeek-V3" in main_txt

            # my modules: the saved variants appear under "mine", toggle in as
            # LABELED entries (name~my:<variant> rides the request, the results
            # and the echo), and a customized column saves its FULL resolved
            # own dims — never a fragment
            _ensure_bound_models(page)
            page.wait_for_selector("text=my:big-kv", timeout=15000)
            captured["bodies"].clear()
            page.click("button:has-text('my:big-kv')")
            _wait_picks(page, 3)
            _run_picks(page)
            for _ in range(60):
                if any(any(e.get("label") == "mla_attention~my:big-kv"
                           for e in b.get("entries", []))
                       for b in captured["bodies"]):
                    break
                page.wait_for_timeout(100)
            assert any(any(e.get("label") == "mla_attention~my:big-kv"
                           and (e.get("dims") or {}).get("d_kv_lora") == 640
                           for e in b.get("entries", []))
                       for b in captured["bodies"]), \
                "a variant entry rides its label and its DB dims"
            page.wait_for_selector("text=mla_attention · big-kv", timeout=15000)
            # save flow: edit mla@V3's d_kv_lora, then ☆ save posts the FULL
            # resolved own dims of that column (not just the edited key)
            inp = page.query_selector("tr:has-text('d_kv_lora') input[type=number]")
            inp.fill("640")
            inp.press("Tab")
            page.wait_for_selector("button:has-text('☆ save')", timeout=15000)
            page.click("button:has-text('☆ save')")
            # the name is asked in the site's own dialog, not the browser's
            page.fill("[data-dialog=prompt] input", "big-kv-2")
            page.click("[data-dialog=prompt] button[type=submit]")
            for _ in range(60):
                if captured["module_posts"]:
                    break
                page.wait_for_timeout(100)
            assert captured["module_posts"], "save should POST /api/account/modules"
            posted = captured["module_posts"][-1]
            assert posted["component"] == "mla_attention"
            assert posted["name"] == "big-kv-2"
            assert posted["dims"] == {"d_kv_lora": 640}, posted["dims"]
            # the save refetches the variants — the new one appears under mine
            page.wait_for_selector("text=my:big-kv-2", timeout=15000)
            # variant entries toggle off from the same chip
            page.click("button:has-text('✓ my:big-kv ×')")
            _wait_picks(page, 2)

            # a binding row toggles OFF too — the table is not a one-way door
            _ensure_bound_models(page)
            page.click("#pg-bound-models >> tr:has-text('DeepSeek-V3.2') >> button:has-text('✓ added')")
            _wait_picks(page, 1)
            _run_picks(page)
            # the stale two-column response is replaced after the debounced rerun
            page.wait_for_selector("text=mla_attention @ DeepSeek-V3.2", state="detached", timeout=15000)
            browser.close()
    finally:
        httpd.shutdown()


def _open_compare(p, port, stub_bench):
    """Launch the page with the playground stubs, pick MLA + DSA (four bound
    entries) and wait for the summary and KV sections. `stub_bench(body)`
    answers the module-bench POST."""
    try:
        browser = p.chromium.launch()
    except Exception as exc:  # chromium not installed
        pytest.skip(f"chromium unavailable: {exc}")
    page = browser.new_page(viewport={"width": 1400, "height": 1600})

    def _json(route, payload):
        route.fulfill(status=200, content_type="application/json", body=json.dumps(payload))

    def _component_dims(route):
        from urllib.parse import urlparse, parse_qs
        q = parse_qs(urlparse(route.request.url).query)
        comp = q.get("component", [""])[0]
        if "model_id" in q:
            seed = {b["model_id"]: b["dims"]
                    for b in _BINDINGS_BY_COMPONENT.get(comp, [])}.get(q["model_id"][0])
            _json(route, {"success": seed is not None, "dims": seed or {}})
        else:
            _json(route, {"success": True, "component": comp,
                          "bindings": _BINDINGS_BY_COMPONENT.get(comp, [])})

    page.route("**/api/playground/components", lambda r: _json(r, _COMPONENTS))
    page.route("**/supported-models", lambda r: _json(r, _MODELS))
    page.route("**/api/playground/module-bench",
               lambda r: _json(r, stub_bench(json.loads(r.request.post_data or "{}"))))
    page.route("**/api/playground/component-dims**", _component_dims)
    page.route("**/dim-glossary", lambda r: _json(r, {"items": _dim_meanings()}))
    page.route("**/precision/presets", lambda r: _json(r, _precision_presets()))

    base = f"http://127.0.0.1:{port}"
    page.goto(f"{base}/playground.html")
    page.wait_for_selector("button.uppercase:has-text('MLA')", timeout=15000)
    for fam in ("MLA", "DSA"):
        page.click(f"button.uppercase:has-text('{fam}')")
    page.click("#pg-groups >> text=mla_attention")
    _wait_picks(page, 2)
    _ensure_picker(page)
    page.click("#pg-groups >> text=mla_dsa_attention")
    _wait_picks(page, 4)
    _run_picks(page)
    page.wait_for_selector("#pg-summary", timeout=15000)
    page.wait_for_selector("#pg-kv", timeout=15000)
    page.wait_for_timeout(1200)  # settle the debounced rerun
    return browser, page


def _precision_presets():
    """The real `/precision/presets` payload."""
    from backend.arch.eval.precision import presets
    return {"presets": presets()}


def _dim_meanings():
    """The real `/dim-glossary` payload: the registry's meanings."""
    from backend.arch import dims
    return dims.meanings()


def test_playground_phase_toggle_lives_on_the_runtime_table():
    """The summary table shows both phases side by side and does not follow
    the runtime table's phase toggle, which lives in #pg-details. FLOPs live
    in the summary, scaled in SI units. At decode, entries evaluated in a
    declared decode form are named above the table."""
    pytest.importorskip("playwright.sync_api")
    from playwright.sync_api import sync_playwright

    _ensure_built()
    if not (_DIST / "index.html").exists():
        pytest.skip("web/dist not built")

    httpd = _serve_dist()
    try:
        with sync_playwright() as p:
            payloads = []

            def bench(body):
                payloads.append(_bench_payload(body))
                return payloads[-1]

            browser, page = _open_compare(p, httpd.server_address[1], bench)

            # names live on the summary table's entry cells
            labels = page.inner_text("#pg-summary")
            assert "mla_attention" in labels and "mla_dsa_attention" in labels
            # both phase groups sit side by side in the summary
            head = page.inner_text("#pg-summary table thead")
            assert "Stored" in head and "Prefill / step" in head and "Decode / step" in head
            # the toggle belongs to the runtime table, not to KV / summary
            _ensure_details(page)
            assert page.query_selector("#pg-op-table [aria-label='Runtime table phase']") is not None
            assert page.query_selector("#pg-kv [aria-label=Phase]") is None
            assert page.query_selector("#pg-summary [aria-label=Phase]") is None
            summary = page.inner_text("#pg-summary")
            assert "KV write / step" in summary and "KV read / step" in summary
            # A column has one shared unit; its cells are unitless and align.
            flops_cell = page.query_selector("#pg-summary table tbody tr [data-col=prefill_flops]")
            assert page.inner_text(
                "#pg-summary table thead th[data-col=prefill_flops]"
            ).split()[-1] == "TFLOPs"
            assert flops_cell.inner_text().strip() == "2.23"
            assert "2225718689792" not in flops_cell.inner_text()
            dec_flops = page.query_selector("#pg-summary table tbody tr [data-col=decode_flops]")
            assert page.get_attribute(
                "#pg-summary table thead th[data-col=decode_flops]", "data-unit"
            ) == "GFLOPs"
            assert dec_flops.inner_text().strip() == "1.52"

            def painted_and_expected(phase):
                """Per entry, the kernel ids its runtime table paints, and the
                backend's rows for that phase (`op_tables[phase].rows`)."""
                painted = {
                    h.get_attribute("data-group-header"): sorted(
                        tr.get_attribute("data-aggregation-id")
                        for tr in h.query_selector_all("[data-leaf-row]"))
                    for h in page.query_selector_all("#pg-op-table [data-group-header]")
                }
                expected = {
                    row["entry"]: sorted(op["node_id"] for op in row["op_tables"][phase]["rows"])
                    for row in payloads[-1]["results"] if row["op_tables"][phase]["rows"]
                }
                return painted, expected

            # the runtime table's rows are the backend's rows for the phase
            painted, expected = painted_and_expected("prefill")
            assert painted == expected and expected

            # the declared decode form is named above the table, only at decode
            assert "declared decode form" not in page.inner_text("#pg-op-table")
            page.click("#pg-op-table [aria-label='Runtime table phase'] button:has-text('Decode')")
            page.wait_for_selector("#pg-op-table >> text=declared decode form", timeout=5000)
            painted, expected = painted_and_expected("decode")
            assert painted == expected and expected
            hint = page.inner_text("#pg-op-table")
            assert "decode runs mla_dsa_attention as its declared decode form" in hint
            assert "mla_dsa_attention_absorbed" in hint
            page.click("#pg-op-table [aria-label='Runtime table phase'] button:has-text('Prefill')")
            page.wait_for_timeout(200)
            assert "declared decode form" not in page.inner_text("#pg-op-table")
            browser.close()
    finally:
        httpd.shutdown()


_SUMMARY_CELLS = (
    ("params", lambda r: r["params"]),
    ("active_params", lambda r: r["active_params"]),
    ("weight_bytes", lambda r: r["weight_bytes"]),
    ("kv_storage", lambda r: r["kv"]["storage_bytes_at_L"]),
    ("prefill_flops", lambda r: r["phases"]["prefill"]["flops"]),
    ("prefill_weight_read", lambda r: r["phases"]["prefill"]["weight_read_bytes"]),
    ("prefill_kv_write", lambda r: r["phases"]["prefill"]["kv_step_bytes"]),
    ("prefill_act_fused", lambda r: r["phases"]["prefill"]["act_bytes_fused"]),
    ("decode_flops", lambda r: r["phases"]["decode"]["flops"]),
    ("decode_weight_read", lambda r: r["phases"]["decode"]["weight_read_bytes"]),
    ("decode_kv_read", lambda r: r["phases"]["decode"]["kv_read_bytes"]),
    ("decode_act_fused", lambda r: r["phases"]["decode"]["act_bytes_fused"]),
)


def test_playground_summary_table_matches_the_payload():
    """Every summary cell's data-value is the backend number; prefill and
    decode sit side by side; toggling the runtime table's phase leaves the
    table unchanged. The decode KV-read 'dominant' mark is the backend's
    memory_dominant, not a frontend comparison."""
    pytest.importorskip("playwright.sync_api")
    from playwright.sync_api import sync_playwright

    _ensure_built()
    if not (_DIST / "index.html").exists():
        pytest.skip("web/dist not built")

    captured = {}

    def _bench(body):
        payload = _bench_payload(body)
        payload["results"][0]["phases"]["decode"]["memory_dominant"] = "kv"
        captured["payload"] = payload
        return payload

    httpd = _serve_dist()
    try:
        with sync_playwright() as p:
            browser, page = _open_compare(p, httpd.server_address[1], _bench)
            page.wait_for_selector("#pg-summary table tbody tr[data-entry]", timeout=15000)
            assert page.query_selector("#pg-jump a[href='#pg-summary']") is not None
            assert page.query_selector("#pg-jump a[href='#pg-kv']") is not None
            assert page.query_selector("#pg-jump a[href='#pg-step']") is None
            assert page.query_selector("#pg-jump a[href='#pg-details']") is not None
            results = captured["payload"]["results"]
            for row in results:
                tr = page.query_selector(
                    f"#pg-summary table tbody tr[data-entry='{row['entry']}']")
                assert tr is not None, row["entry"]
                for col, get in _SUMMARY_CELLS:
                    cell = tr.query_selector(f"[data-col='{col}']")
                    assert cell is not None, (row["entry"], col)
                    assert cell.get_attribute("data-value") == str(get(row)), (
                        row["entry"], col, cell.get_attribute("data-value"), get(row))
            first = page.query_selector(
                f"#pg-summary table tbody tr[data-entry='{results[0]['entry']}'] "
                "[data-col=decode_kv_read] .pg-dominant")
            assert first is not None and first.inner_text() == "(dominant)"
            for row in results[1:]:
                assert page.query_selector(
                    f"#pg-summary table tbody tr[data-entry='{row['entry']}'] "
                    "[data-col=decode_kv_read] .pg-dominant") is None
            r0 = results[0]
            stored = page.inner_text("#pg-summary [data-footnote=stored]")
            assert r0["units"]["weight_bytes"] in stored
            assert r0["kv"]["unit"] in stored
            # Each phase's contract is written once, then what each column
            # adds to it.
            for phase in ("prefill", "decode"):
                units = r0["phases"][phase]["units"]
                note = page.inner_text(f"#pg-summary [data-footnote={phase}]")
                assert note.count(units["flops"]) == 1, note
                for key in ("weight_read", "traffic"):
                    assert units[key].startswith(units["flops"] + " · ")
                    assert units[key][len(units["flops"]) + 3:] in note, (key, note)
            before = page.inner_text("#pg-summary")
            _ensure_details(page)
            page.click("#pg-op-table [aria-label='Runtime table phase'] button:has-text('Decode')")
            page.wait_for_selector("#pg-op-table >> text=declared decode form", timeout=5000)
            assert page.inner_text("#pg-summary") == before
            browser.close()
    finally:
        httpd.shutdown()


def test_playground_kv_headline_shows_recurrent_state_as_state():
    """A linear-attention entry has no per-token KV: the sweep names its
    cache a flat state, constant at every length, while a dense entry grows."""
    pytest.importorskip("playwright.sync_api")
    from playwright.sync_api import sync_playwright

    _ensure_built()
    if not (_DIST / "index.html").exists():
        pytest.skip("web/dist not built")

    def _bench(body):
        payload = _bench_payload(body)
        payload["results"] = [_GDN_ROW, *payload["results"][:1]]
        payload.pop("graph_diff", None)
        return payload

    httpd = _serve_dist()
    try:
        with sync_playwright() as p:
            browser, page = _open_compare(p, httpd.server_address[1], _bench)
            page.wait_for_selector("#pg-kv-sweep >> text=gated_delta_net_fused_attention", timeout=15000)
            gdn = page.locator("#pg-kv-sweep tbody tr").first.inner_text()
            assert "state · flat" in gdn
            # state column + every sweep length (a flat recurrent state)
            assert gdn.count("1.05 MiB") == len(_SWEEP) + 1
            other = page.locator("#pg-kv-sweep tbody tr").nth(1).inner_text()
            assert "state · flat" not in other
            browser.close()
    finally:
        httpd.shutdown()


def test_playground_names_kda_by_its_mechanism():
    """Kimi-K3's KDA keeps a GatedDeltaNet-shaped state, but the mechanism
    is KDA: the KV sweep entry subtitle names KDA first and the storage
    family second; where the two agree (MLA) only one name shows."""
    pytest.importorskip("playwright.sync_api")
    from playwright.sync_api import sync_playwright

    _ensure_built()
    if not (_DIST / "index.html").exists():
        pytest.skip("web/dist not built")

    kda = json.loads(json.dumps(_GDN_ROW))
    kda.update({"component": "kimi_k3_kda_attention", "entry": "kimi_k3_kda_attention",
                "model_ids": ["moonshotai/Kimi-K3"]})
    kda["kv"]["display"] = "KDA"

    def _bench(body):
        payload = _bench_payload(body)
        payload["results"] = [kda, *payload["results"][:1]]
        payload.pop("graph_diff", None)
        return payload

    httpd = _serve_dist()
    try:
        with sync_playwright() as p:
            browser, page = _open_compare(p, httpd.server_address[1], _bench)
            page.wait_for_selector("#pg-kv-sweep >> text=kimi_k3_kda_attention", timeout=15000)
            kda_row = page.locator("#pg-kv-sweep tbody tr").first
            mech = kda_row.locator(".pg-kv-mech").inner_text()
            assert mech.startswith("KDA · state shaped as GatedDeltaNet"), mech
            mla_row = page.locator("#pg-kv-sweep tbody tr").nth(1)
            mla = mla_row.locator(".pg-kv-mech").inner_text()
            assert "MLA" in mla and "shaped as" not in mla
            browser.close()
    finally:
        httpd.shutdown()


def test_playground_kv_sweep_table_covers_every_sweep_length():
    """The KV storage table asks the backend for every Scaling-sweep length
    and renders one column per length, each cell in scaled units; the smallest
    cell of each column is marked; × layers switches to the whole model."""
    pytest.importorskip("playwright.sync_api")
    from playwright.sync_api import sync_playwright

    _ensure_built()
    if not (_DIST / "index.html").exists():
        pytest.skip("web/dist not built")

    bodies = []

    def _bench(body):
        bodies.append(body)
        payload = _bench_payload(body)
        payload["results"] = [_GDN_ROW, *payload["results"]]
        payload.pop("graph_diff", None)
        return payload

    httpd = _serve_dist()
    try:
        with sync_playwright() as p:
            browser, page = _open_compare(p, httpd.server_address[1], _bench)
            assert bodies and bodies[-1]["kv_cache_lens"] == _SWEEP
            page.wait_for_selector("#pg-kv-sweep table", timeout=15000)
            heads = [h.inner_text() for h in page.query_selector_all("#pg-kv-sweep thead th")]
            assert "per token" in heads and "state" in heads, heads
            assert heads[-len(_SWEEP):] == ["1K", "2K", "4K", "8K", "16K", "32K", "128K", "256K", "1M"], heads
            assert page.query_selector("#pg-kv-sweep th.pg-kv-at-L") is not None
            assert page.query_selector("#pg-kv-sweep th.pg-kv-at-L").inner_text() == "4K"
            rows = page.query_selector_all("#pg-kv-sweep tbody tr")
            assert len(rows) == 5
            for row in rows:
                cells = [c.inner_text() for c in row.query_selector_all(".pg-kv-cell")]
                assert len(cells) == len(_SWEEP)
                assert all(c.endswith("B") and " " in c for c in cells), cells
            gdn = rows[0].inner_text()
            assert "state · flat" in gdn and gdn.count("1.05 MiB") == len(_SWEEP) + 1
            # the flat state is the smallest at every length
            assert len(rows[0].query_selector_all(".pg-kv-min")) == len(_SWEEP)
            page.click("#pg-kv-scope-model")
            page.wait_for_selector("#pg-kv-sweep >> text=× 36", timeout=5000)
            assert "whole model" in page.inner_text("#pg-kv-sweep")
            assert "37.69 MiB" in rows[0].inner_text()  # 1,097,728 B × 36 layers
            # cache_len runs on a log2 axis across the sweep, so the middle of
            # the plot is 32K (a linear axis would put 256K there). The axis
            # tooltip names the length it snapped to.
            page.click("#pg-kv-scope-layer")
            chart = page.locator("#pg-kv-sweep canvas").first
            chart.scroll_into_view_if_needed()
            box = chart.bounding_box()
            page.mouse.move(box["x"] + 72 + (box["width"] - 88) * 0.5, box["y"] + box["height"] * 0.5)
            page.wait_for_timeout(500)
            tips = page.evaluate("""() => [...document.querySelectorAll('#pg-kv-sweep div')]
                .filter(d => (d.getAttribute('style') || '').includes('position: absolute') && d.innerText)
                .map(d => d.innerText)""")
            assert tips and tips[-1].startswith("32K"), tips
            assert "36.00 MiB" in tips[-1], tips  # 1,152 B/token × 32,768 tokens
            browser.close()
    finally:
        httpd.shutdown()


def test_playground_editable_dims_carry_their_meaning():
    """Every row of the editable internal-dims table names its symbol with the
    registry's meaning on hover — the routing knobs a bound MoE entry seeds
    (`n_expert_copies`, `n_route_targets`, `n_experts_active`) included,
    never a bare name."""
    pytest.importorskip("playwright.sync_api")
    from playwright.sync_api import sync_playwright

    _ensure_built()
    if not (_DIST / "index.html").exists():
        pytest.skip("web/dist not built")

    routing = {"n_expert_copies": 64, "n_route_targets": 64,
               "n_experts_active": 6}

    def _bench(body):
        payload = _bench_payload(body)
        for row in payload["results"]:
            row["editable_dims"] = {**row["editable_dims"], **routing}
        return payload

    meanings = _dim_meanings()
    httpd = _serve_dist()
    try:
        with sync_playwright() as p:
            browser, page = _open_compare(p, httpd.server_address[1], _bench)
            _ensure_dims(page)
            page.wait_for_selector("td.pg-dim-name >> text=n_route_targets", timeout=15000)
            for name in routing:
                assert meanings.get(name), name
                label = page.locator(f"td.pg-dim-name span:text-is('{name}')")
                assert "dotted" in (label.get_attribute("style") or ""), name
                page.mouse.move(0, 0)
                label.hover()
                page.wait_for_selector("#ha-tip:not(.hidden)", timeout=5000)
                tip = " ".join(page.inner_text("#ha-tip").split())
                expected = " ".join(meanings[name].replace("**", "").replace("`", "").split())
                assert tip == expected, (name, tip)
            browser.close()
    finally:
        httpd.shutdown()


def test_playground_weight_cells_are_phase_independent():
    """Weight storage is a summary cell in the stored group, with the
    checkpoint precision under the value and no phase in that group; weight
    read sits in both per-step groups of the summary."""
    pytest.importorskip("playwright.sync_api")
    from playwright.sync_api import sync_playwright

    _ensure_built()
    if not (_DIST / "index.html").exists():
        pytest.skip("web/dist not built")

    httpd = _serve_dist()
    try:
        with sync_playwright() as p:
            browser, page = _open_compare(p, httpd.server_address[1], _bench_payload)
            cell = page.query_selector("#pg-summary table tbody [data-col=weight_bytes]")
            assert cell is not None
            note = cell.inner_text()
            assert "(FP8 (E4M3), BF16 · " in note
            assert "unbound: priced at deepseek-ai/DeepSeek-V3.2's quant context" in note
            foot = page.inner_text("#pg-summary [data-footnote=stored]")
            assert "stored once" in foot
            assert page.query_selector("#pg-summary table [data-col=prefill_weight_read]") is not None
            assert page.query_selector("#pg-summary table [data-col=decode_weight_read]") is not None
            assert page.query_selector("#pg-kv #pg-kv-sweep") is not None

            _ensure_details(page)
            page.click("#pg-op-table [aria-label='Runtime table phase'] button:has-text('Decode')")
            page.wait_for_selector("#pg-op-table >> text=declared decode form", timeout=5000)
            assert page.query_selector(
                "#pg-summary table tbody [data-col=weight_bytes]").inner_text() == note
            browser.close()
    finally:
        httpd.shutdown()


def test_playground_storage_is_the_same_in_both_phases():
    """The summary table (stored group and both step groups) and the KV sweep
    are independent of the runtime table's phase toggle: they read the same,
    word for word and pixel for pixel, whichever phase is selected. The sweep
    names no phase."""
    pytest.importorskip("playwright.sync_api")
    from playwright.sync_api import sync_playwright

    _ensure_built()
    if not (_DIST / "index.html").exists():
        pytest.skip("web/dist not built")

    def _snapshot(page):
        page.mouse.move(0, 0)
        page.wait_for_timeout(1500)  # let any chart animation finish
        return page.evaluate("""() => {
            const s = document.querySelector('#pg-summary');
            const g = document.querySelector('#pg-kv');
            return {summary: s.innerText, storage: g.innerText,
                    charts: [...g.querySelectorAll('canvas')].map(c => c.toDataURL())};
        }""")

    httpd = _serve_dist()
    try:
        with sync_playwright() as p:
            browser, page = _open_compare(p, httpd.server_address[1], _bench_payload)
            prefill = _snapshot(page)
            assert "Weight storage" in prefill["summary"]
            assert "KV / request @L" in prefill["summary"]
            assert "prefill" not in prefill["storage"].lower()
            assert "decode" not in prefill["storage"].lower()
            assert len(prefill["charts"]) == 1
            _ensure_details(page)
            page.click("#pg-op-table [aria-label='Runtime table phase'] button:has-text('Decode')")
            page.wait_for_selector("#pg-op-table >> text=declared decode form", timeout=5000)
            decode = _snapshot(page)
            assert decode["summary"] == prefill["summary"]
            assert decode["storage"] == prefill["storage"]
            assert decode["charts"] == prefill["charts"]
            browser.close()
    finally:
        httpd.shutdown()


def test_serving_contract_lives_on_summary_and_kv():
    """B / L are not page-level: they sit on the sections that consume them,
    share one state, and still ride the existing request body. Precision is
    the page's profile switch: a preset rides the body and the URL."""
    pytest.importorskip("playwright.sync_api")
    from playwright.sync_api import sync_playwright

    _ensure_built()
    if not (_DIST / "index.html").exists():
        pytest.skip("web/dist not built")

    httpd = _serve_dist()
    captured = []

    def stub(body):
        captured.append(body)
        return _bench_payload(body)

    try:
        with sync_playwright() as p:
            browser, page = _open_compare(p, httpd.server_address[1], stub)
            bar = page.locator("#pg-picks")
            assert bar.is_visible()
            assert page.locator("#pg-edit-picks").count() == 0
            txt = bar.inner_text()
            assert "mla_attention" in txt and "DeepSeek-V3.2" in txt
            assert "B1" not in txt
            assert "bfloat16" not in txt
            assert "Edit" not in txt
            _ensure_picker(page)
            picker = page.locator("#pg-groups").locator("xpath=ancestor::section[1]")
            picker.wait_for(state="visible")
            assert picker.locator(".pg-serving").count() == 0
            assert "Batch (B)" not in picker.inner_text()
            assert page.locator("#pg-structure .pg-serving").count() == 0
            for sec in ("#pg-summary", "#pg-kv"):
                assert page.locator(f"{sec} .pg-serving").count() == 1
                assert page.locator(f"{sec} .pg-serving-batch").input_value() == "1"
                assert page.locator(f"{sec} .pg-serving-seq").input_value() == "4096"
            # As shipped is the absence of a precision field.
            assert captured and all("precision" not in b and "kv_dtype" not in b for b in captured)
            n = len(captured)
            page.locator("#pg-summary .pg-serving-batch").fill("8")
            page.locator("#pg-kv .pg-serving-seq").fill("2048")
            page.click("#pg-precision [data-preset=kv_fp8]")
            for _ in range(80):
                if any(b.get("batch_size") == 8 and b.get("seq_len") == 2048
                       and b.get("precision") == {"preset": "kv_fp8"} for b in captured[n:]):
                    break
                page.wait_for_timeout(100)
            else:
                raise AssertionError(f"serving change did not POST: {captured[n:]}")
            body = next(b for b in captured[n:]
                        if b.get("batch_size") == 8 and b.get("seq_len") == 2048
                        and b.get("precision") == {"preset": "kv_fp8"})
            assert "kv_cache_lens" in body and "kv_dtype" not in body
            assert "prec=kv_fp8" in page.evaluate("location.search")
            for sec in ("#pg-summary", "#pg-kv"):
                assert page.locator(f"{sec} .pg-serving-batch").input_value() == "8"
                assert page.locator(f"{sec} .pg-serving-seq").input_value() == "2048"
            browser.close()
    finally:
        httpd.shutdown()


def test_compare_config_row_stays_sticky_without_edit_fold():
    """After results, every selected entry stays on a sticky config row
    under the page header — no Edit collapse. Scrolling to the summary
    leaves the row on screen with the same entries."""
    pytest.importorskip("playwright.sync_api")
    from playwright.sync_api import sync_playwright

    _ensure_built()
    if not (_DIST / "index.html").exists():
        pytest.skip("web/dist not built")

    httpd = _serve_dist()
    try:
        with sync_playwright() as p:
            browser, page = _open_compare(p, httpd.server_address[1], _bench_payload)
            bar = page.locator("#pg-picks")
            bar.wait_for(state="visible")
            assert page.locator("#pg-edit-picks").count() == 0
            assert bar.evaluate("e => getComputedStyle(e).position") == "sticky"
            assert bar.locator("[data-entry]").count() == 4
            assert "mla_attention" in bar.inner_text()
            assert "mla_dsa_attention" in bar.inner_text()
            page.evaluate("window.scrollTo(0, document.documentElement.scrollHeight)")
            page.wait_for_timeout(200)
            box = bar.bounding_box()
            nav = page.locator("#top-nav").bounding_box()
            assert box is not None and nav is not None
            assert box["y"] + box["height"] > 0, "config row scrolled off the top"
            assert box["y"] >= nav["y"], (box["y"], nav["y"])
            assert box["y"] <= nav["y"] + nav["height"] + 8, (
                "config row is not sitting under the page header", box["y"], nav)
            assert bar.locator("[data-entry]").count() == 4
            browser.close()
    finally:
        httpd.shutdown()


def test_a_model_whose_calls_evaluate_alike_takes_its_first_call(monkeypatch):
    """A model that calls the component in several places that differ in
    nothing the evaluation reads (Kimi-K3's KDA attention across layer groups
    rebinding only the AttnRes history) has one parameterization: the
    bound-models table shows it once, and the picked column is evaluated at
    its first call — no "evaluate at which call" gap."""
    pytest.importorskip("playwright.sync_api")
    from playwright.sync_api import sync_playwright

    _ensure_built()
    if not (_DIST / "index.html").exists():
        pytest.skip("web/dist not built")
    v3 = "deepseek-ai/DeepSeek-V3"
    first = _call("mla_attention", "1/self_attn", "backbone · layer 0 · self_attn",
                  model_calls=2, model_parameterizations=1)
    second = _call("mla_attention", "10/self_attn", "backbone · layers 1–10 · self_attn",
                   model_calls=2, model_parameterizations=1)
    monkeypatch.setitem(_BINDINGS_BY_COMPONENT, "mla_attention", [
        {"model_id": v3, **first, "dims": {"d_kv_lora": 512}},
        {"model_id": v3, **second, "dims": {"d_kv_lora": 512}},
    ])
    bodies = []

    def bench(body):
        bodies.append(body)
        return _bench_payload(body)

    httpd = _serve_dist()
    try:
        with sync_playwright() as p:
            try:
                browser = p.chromium.launch()
            except Exception as exc:  # chromium not installed
                pytest.skip(f"chromium unavailable: {exc}")
            page = browser.new_page(viewport={"width": 1400, "height": 1600})

            def _json(route, payload):
                route.fulfill(status=200, content_type="application/json",
                              body=json.dumps(payload))

            page.route("**/api/playground/components", lambda r: _json(r, _COMPONENTS))
            page.route("**/supported-models", lambda r: _json(r, _MODELS))
            page.route("**/api/playground/module-bench",
                       lambda r: _json(r, bench(json.loads(r.request.post_data or "{}"))))
            page.route("**/api/playground/component-dims**", lambda r: _json(r, {
                "success": True, "component": "mla_attention",
                "bindings": _BINDINGS_BY_COMPONENT["mla_attention"]}))
            page.route("**/dim-glossary", lambda r: _json(r, {"items": {}}))
            page.goto(f"http://127.0.0.1:{httpd.server_address[1]}/playground.html")
            page.wait_for_selector("button.uppercase:has-text('MLA')", timeout=15000)
            page.click("button.uppercase:has-text('MLA')")
            page.click("#pg-groups >> text=mla_attention")
            _wait_picks(page, 1)

            _ensure_bound_models(page)
            rows = page.locator("#pg-bound-models tr[data-call]")
            assert rows.count() == 1
            assert "2 calls alike" in rows.nth(0).inner_text()

            _run_picks(page)
            page.wait_for_selector("#pg-summary", timeout=15000)
            assert bodies[-1]["entries"][0]["call"] == first["call"]
            assert page.locator("#pg-gaps").count() == 0
            browser.close()
    finally:
        httpd.shutdown()


def test_a_model_whose_calls_differ_brings_a_column_per_call(monkeypatch):
    """A model that calls the component in two places that evaluate
    differently parameterizes it per call: the bound-models table lists each
    call, and the pick brings one column per parameterization, each at its
    own call with its dims — never a column with no call for the Gaps panel
    to ask about. The call rides each column's request and stays visible on
    it."""
    pytest.importorskip("playwright.sync_api")
    from playwright.sync_api import sync_playwright

    _ensure_built()
    if not (_DIST / "index.html").exists():
        pytest.skip("web/dist not built")
    v3 = "deepseek-ai/DeepSeek-V3"
    dense = _call("mla_attention", "0/self_attn", "backbone · layers 0–2 · self_attn",
                  model_calls=2)
    mtp = _call("mla_attention", "2/transformer.self_attn",
                "mtp · layer 0 · transformer.self_attn", model_calls=2)
    monkeypatch.setitem(_BINDINGS_BY_COMPONENT, "mla_attention", [
        {"model_id": v3, **dense, "dims": {"d_kv_lora": 512}},
        {"model_id": v3, **mtp, "dims": {"d_kv_lora": 256}},
    ])
    bodies = []

    def bench(body):
        bodies.append(body)
        return _bench_payload(body)

    httpd = _serve_dist()
    try:
        with sync_playwright() as p:
            try:
                browser = p.chromium.launch()
            except Exception as exc:  # chromium not installed
                pytest.skip(f"chromium unavailable: {exc}")
            page = browser.new_page(viewport={"width": 1400, "height": 1600})

            def _json(route, payload):
                route.fulfill(status=200, content_type="application/json",
                              body=json.dumps(payload))

            page.route("**/api/playground/components", lambda r: _json(r, _COMPONENTS))
            page.route("**/supported-models", lambda r: _json(r, _MODELS))
            page.route("**/api/playground/module-bench",
                       lambda r: _json(r, bench(json.loads(r.request.post_data or "{}"))))
            page.route("**/api/playground/component-dims**", lambda r: _json(r, {
                "success": True, "component": "mla_attention",
                "bindings": _BINDINGS_BY_COMPONENT["mla_attention"]}))
            page.route("**/dim-glossary", lambda r: _json(r, {"items": {}}))
            page.goto(f"http://127.0.0.1:{httpd.server_address[1]}/playground.html")
            page.wait_for_selector("button.uppercase:has-text('MLA')", timeout=15000)
            page.click("button.uppercase:has-text('MLA')")
            page.click("#pg-groups >> text=mla_attention")
            _wait_picks(page, 2)

            # Each call is its own row of the bound-models table.
            _ensure_bound_models(page)
            rows = page.locator("#pg-bound-models tr[data-call]")
            assert rows.count() == 2
            assert "layers 0–2" in rows.nth(0).inner_text()
            assert "mtp · layer 0" in rows.nth(1).inner_text()

            _run_picks(page)
            page.wait_for_selector("#pg-summary", timeout=15000)
            assert [e["call"] for e in bodies[-1]["entries"]] == [dense["call"], mtp["call"]]
            assert page.locator("#pg-gaps").count() == 0
            chips = page.locator("#pg-picks select[data-call-pick]")
            chips.first.wait_for(state="visible")
            assert [chips.nth(n).input_value() for n in range(chips.count())] == [
                dense["call"], mtp["call"]]
            browser.close()
    finally:
        httpd.shutdown()


def test_a_column_rebound_to_a_model_whose_calls_differ_takes_its_first_call(
        monkeypatch):
    """Rebinding a column to a model that calls the component in places that
    evaluate differently starts it at that model's first call — a column
    with a call, benched on the next run — not a column the Gaps panel asks
    about."""
    pytest.importorskip("playwright.sync_api")
    from playwright.sync_api import sync_playwright

    _ensure_built()
    if not (_DIST / "index.html").exists():
        pytest.skip("web/dist not built")
    v3, v32 = "deepseek-ai/DeepSeek-V3", "deepseek-ai/DeepSeek-V3.2"
    dense = _call("mla_attention", "0/self_attn", "backbone · layers 0–2 · self_attn",
                  model_calls=2)
    mtp = _call("mla_attention", "2/transformer.self_attn",
                "mtp · layer 0 · transformer.self_attn", model_calls=2)
    monkeypatch.setitem(_BINDINGS_BY_COMPONENT, "mla_attention", [
        {"model_id": v32, **_call("mla_attention"), "dims": {"d_kv_lora": 640}},
        {"model_id": v3, **dense, "dims": {"d_kv_lora": 512}},
        {"model_id": v3, **mtp, "dims": {"d_kv_lora": 256}},
    ])
    components = json.loads(json.dumps(_COMPONENTS))
    components["components"][0]["model_ids"] = [v3, v32]
    bodies = []

    def bench(body):
        bodies.append(body)
        return _bench_payload(body)

    httpd = _serve_dist()
    try:
        with sync_playwright() as p:
            try:
                browser = p.chromium.launch()
            except Exception as exc:  # chromium not installed
                pytest.skip(f"chromium unavailable: {exc}")
            page = browser.new_page(viewport={"width": 1400, "height": 1600})

            def _json(route, payload):
                route.fulfill(status=200, content_type="application/json",
                              body=json.dumps(payload))

            page.route("**/api/playground/components", lambda r: _json(r, components))
            page.route("**/supported-models", lambda r: _json(r, _MODELS))
            page.route("**/api/playground/module-bench",
                       lambda r: _json(r, bench(json.loads(r.request.post_data or "{}"))))
            page.route("**/api/playground/component-dims**", lambda r: _json(r, {
                "success": True, "component": "mla_attention",
                "bindings": _BINDINGS_BY_COMPONENT["mla_attention"]}))
            page.route("**/dim-glossary", lambda r: _json(r, {"items": {}}))
            page.goto(f"http://127.0.0.1:{httpd.server_address[1]}/playground.html")
            page.wait_for_selector("button.uppercase:has-text('MLA')", timeout=15000)
            page.click("button.uppercase:has-text('MLA')")
            page.click("#pg-groups >> text=mla_attention")
            _wait_picks(page, 3)

            # Drop V3's first column, then bind V3.2's column to V3.
            page.click(f"#pg-picks [data-entry='mla_attention@{v3}#{dense['call']}'] "
                       f"button[title=Remove]")
            _wait_picks(page, 2)
            page.select_option(
                f"#pg-picks [data-entry='mla_attention@{v32}#1/self_attn'] "
                f"select[title^='Bind this column']", v3)
            page.wait_for_selector(
                f"#pg-picks [data-entry='mla_attention@{v3}#{dense['call']}']",
                timeout=15000)
            _run_picks(page)
            page.wait_for_selector("#pg-summary", timeout=15000)
            assert sorted(e["call"] for e in bodies[-1]["entries"]) == sorted(
                [dense["call"], mtp["call"]])
            assert page.locator("#pg-gaps").count() == 0
            browser.close()
    finally:
        httpd.shutdown()


def test_playground_url_carries_the_comparison():
    """The URL is the comparison on screen, so a link shares it: opening one
    restores its columns, a bound column's edits to its binding and the
    contract, and after the first run the page writes the same state back —
    the edit measured against the binding's seed, not swallowed into it."""
    pytest.importorskip("playwright.sync_api")
    from urllib.parse import parse_qs, urlencode, urlsplit

    from playwright.sync_api import sync_playwright

    _ensure_built()
    if not (_DIST / "index.html").exists():
        pytest.skip("web/dist not built")
    bodies = []

    def bench(body):
        bodies.append(body)
        return _bench_payload(body)

    query = urlencode([("pick", "mla_attention@deepseek-ai/DeepSeek-V3#1/self_attn|d_kv_lora=640"),
                       ("pick", "mla_attention@deepseek-ai/DeepSeek-V3.2#1/self_attn"),
                       ("role", "attention"), ("batch", "4"), ("phase", "decode")])
    httpd = _serve_dist()
    try:
        with sync_playwright() as p:
            try:
                browser = p.chromium.launch()
            except Exception as exc:  # chromium not installed
                pytest.skip(f"chromium unavailable: {exc}")
            page = browser.new_page(viewport={"width": 1400, "height": 1600})

            def _json(route, payload):
                route.fulfill(status=200, content_type="application/json",
                              body=json.dumps(payload))

            page.route("**/api/playground/components", lambda r: _json(r, _COMPONENTS))
            page.route("**/supported-models", lambda r: _json(r, _MODELS))
            page.route("**/api/playground/module-bench",
                       lambda r: _json(r, bench(json.loads(r.request.post_data or "{}"))))
            page.route("**/api/playground/component-dims**", lambda r: _json(r, {
                "success": True, "component": "mla_attention",
                "bindings": _BINDINGS_BY_COMPONENT["mla_attention"]}))
            page.route("**/dim-glossary", lambda r: _json(r, {"items": {}}))
            page.goto(f"http://127.0.0.1:{httpd.server_address[1]}/playground.html?{query}")
            page.wait_for_selector("#pg-summary", timeout=15000)

            first = bodies[0]
            assert first["batch_size"] == 4
            assert [(e["model_id"], e["call"], e["dims"]) for e in first["entries"]] == [
                ("deepseek-ai/DeepSeek-V3", "1/self_attn", {"d_kv_lora": 640}),
                ("deepseek-ai/DeepSeek-V3.2", "1/self_attn", {}),
            ]
            assert page.locator("#pg-picks [data-entry]").count() == 2

            # the echo seeded both columns; the URL still says only the edit
            page.wait_for_timeout(300)  # the adopt's re-render writes the URL
            page.wait_for_function(
                "() => new URLSearchParams(location.search).getAll('pick').length === 2")
            q = parse_qs(urlsplit(page.url).query)
            assert q["pick"] == ["mla_attention@deepseek-ai/DeepSeek-V3#1/self_attn|d_kv_lora=640",
                                 "mla_attention@deepseek-ai/DeepSeek-V3.2#1/self_attn"]
            assert q["batch"] == ["4"] and q["phase"] == ["decode"]
            assert "role" not in q and "gpu" not in q  # page defaults stay out
            browser.close()
    finally:
        httpd.shutdown()


def test_a_link_without_a_role_opens_on_the_role_it_compares():
    """The page leaves its default role out of the URL, so a shared link of
    attention columns carries none: the catalog opens on the role of what
    the link compares, and the URL is not rewritten to another role."""
    pytest.importorskip("playwright.sync_api")
    from urllib.parse import parse_qs, urlencode, urlsplit

    from playwright.sync_api import sync_playwright

    _ensure_built()
    if not (_DIST / "index.html").exists():
        pytest.skip("web/dist not built")
    query = urlencode([("pick", "mla_attention@deepseek-ai/DeepSeek-V3#1/self_attn")])
    httpd = _serve_dist()
    try:
        with sync_playwright() as p:
            try:
                browser = p.chromium.launch()
            except Exception as exc:  # chromium not installed
                pytest.skip(f"chromium unavailable: {exc}")
            page = browser.new_page(viewport={"width": 1400, "height": 1600})

            def _json(route, payload):
                route.fulfill(status=200, content_type="application/json",
                              body=json.dumps(payload))

            page.route("**/api/playground/components", lambda r: _json(r, _COMPONENTS))
            page.route("**/supported-models", lambda r: _json(r, _MODELS))
            page.route("**/api/playground/module-bench",
                       lambda r: _json(r, _bench_payload(json.loads(r.request.post_data or "{}"))))
            page.route("**/api/playground/component-dims**", lambda r: _json(r, {
                "success": True, "component": "mla_attention",
                "bindings": _BINDINGS_BY_COMPONENT["mla_attention"]}))
            page.route("**/dim-glossary", lambda r: _json(r, {"items": {}}))
            page.goto(f"http://127.0.0.1:{httpd.server_address[1]}/playground.html?{query}")
            page.wait_for_selector("#pg-summary", timeout=15000)
            page.wait_for_timeout(300)
            assert page.eval_on_selector("#pg-role", "e => e.value") == "attention"
            assert "role" not in parse_qs(urlsplit(page.url).query)
            browser.close()
    finally:
        httpd.shutdown()


def test_a_pick_past_the_bench_cap_says_so():
    """The bench holds 8 columns. Picking a component when it is full adds
    nothing and says why, instead of ignoring the click."""
    pytest.importorskip("playwright.sync_api")
    from urllib.parse import urlencode

    from playwright.sync_api import sync_playwright

    _ensure_built()
    if not (_DIST / "index.html").exists():
        pytest.skip("web/dist not built")
    query = urlencode([("pick", f"mla_attention@deepseek-ai/DeepSeek-V3#{i}/self_attn")
                       for i in range(8)])
    httpd = _serve_dist()
    try:
        with sync_playwright() as p:
            try:
                browser = p.chromium.launch()
            except Exception as exc:  # chromium not installed
                pytest.skip(f"chromium unavailable: {exc}")
            page = browser.new_page(viewport={"width": 1400, "height": 1600})

            def _json(route, payload):
                route.fulfill(status=200, content_type="application/json",
                              body=json.dumps(payload))

            page.route("**/api/playground/components", lambda r: _json(r, _COMPONENTS))
            page.route("**/supported-models", lambda r: _json(r, _MODELS))
            page.route("**/api/playground/module-bench",
                       lambda r: _json(r, _bench_payload(json.loads(r.request.post_data or "{}"))))
            page.route("**/api/playground/component-dims**", lambda r: _json(r, {
                "success": True, "component": "mla_dsa_attention",
                "bindings": _BINDINGS_BY_COMPONENT["mla_dsa_attention"]}))
            page.route("**/dim-glossary", lambda r: _json(r, {"items": {}}))
            page.goto(f"http://127.0.0.1:{httpd.server_address[1]}/playground.html?{query}")
            page.wait_for_selector("#pg-summary", timeout=15000)
            assert page.locator("#pg-picks [data-entry]").count() == 8
            page.fill("#pg-catalog input[placeholder='Filter components…']", "mla_dsa")
            page.click("#pg-groups >> text=mla_dsa_attention")
            page.get_by_text("the bench holds 8 columns and is full").wait_for(timeout=10000)
            assert page.locator("#pg-picks [data-entry]").count() == 8
            browser.close()
    finally:
        httpd.shutdown()


def test_bound_models_line_up_a_dim_two_of_three_components_carry(monkeypatch):
    """Bound models gives a dim carried by two or more of the selected
    components one column in every section, most widely carried first: with
    MLA, DSA and GQA picked, `n_heads` (all three) leads and `d_kv_lora`
    (MLA and DSA) sits in the same column in both, left blank in GQA's
    section; each component's own dims fill the columns after those."""
    pytest.importorskip("playwright.sync_api")
    from urllib.parse import parse_qs, urlencode, urlsplit

    from playwright.sync_api import sync_playwright

    _ensure_built()
    if not (_DIST / "index.html").exists():
        pytest.skip("web/dist not built")
    llama = "meta-llama/Llama-3.2-3B"
    monkeypatch.setitem(_BINDINGS_BY_COMPONENT, "mla_attention", [
        {"model_id": "deepseek-ai/DeepSeek-V3", **_call("mla_attention"),
         "dims": {"d_kv_lora": 512, "d_rope": 64, "n_heads": 128}}])
    monkeypatch.setitem(_BINDINGS_BY_COMPONENT, "mla_dsa_attention", [
        {"model_id": "deepseek-ai/DeepSeek-V3.2", **_call("mla_dsa_attention"),
         "dims": {"d_kv_lora": 512, "d_indexer_head": 128, "n_heads": 128}}])
    monkeypatch.setitem(_BINDINGS_BY_COMPONENT, "gqa_attention", [
        {"model_id": llama, **_call("gqa_attention"),
         "dims": {"d_head": 128, "n_heads": 24, "n_kv_heads": 8}}])
    query = urlencode([("pick", "mla_attention@deepseek-ai/DeepSeek-V3#1/self_attn"),
                       ("pick", "mla_dsa_attention@deepseek-ai/DeepSeek-V3.2#1/self_attn"),
                       ("pick", f"gqa_attention@{llama}#1/self_attn"),
                       ("role", "attention")])

    def dims_of(route):
        name = parse_qs(urlsplit(route.request.url).query)["component"][0]
        route.fulfill(status=200, content_type="application/json", body=json.dumps({
            "success": True, "component": name, "bindings": _BINDINGS_BY_COMPONENT[name]}))

    httpd = _serve_dist()
    try:
        with sync_playwright() as p:
            try:
                browser = p.chromium.launch()
            except Exception as exc:  # chromium not installed
                pytest.skip(f"chromium unavailable: {exc}")
            page = browser.new_page(viewport={"width": 1400, "height": 1600})

            def _json(route, payload):
                route.fulfill(status=200, content_type="application/json",
                              body=json.dumps(payload))

            page.route("**/api/playground/components", lambda r: _json(r, _COMPONENTS))
            page.route("**/supported-models", lambda r: _json(r, _MODELS))
            # the bench is not under test here (the canned rows carry no
            # graph for a three-way diff); Bound models is drawn without it
            page.route("**/api/playground/module-bench",
                       lambda r: _json(r, {"success": False, "message": "not under test"}))
            page.route("**/api/playground/component-dims**", dims_of)
            page.route("**/dim-glossary", lambda r: _json(r, {"items": {}}))
            page.goto(f"http://127.0.0.1:{httpd.server_address[1]}/playground.html?{query}")
            _ensure_bound_models(page)
            for name in ("mla_attention", "mla_dsa_attention", "gqa_attention"):
                page.wait_for_selector(f"#pg-bound-models tbody[data-component='{name}'] tr[data-call]",
                                       timeout=15000)
            heads = {
                name: page.eval_on_selector(
                    f"#pg-bound-models tbody[data-component='{name}']",
                    "b => [...b.rows[0].cells].slice(1, -1).map(c => c.innerText.trim())")
                for name in ("mla_attention", "mla_dsa_attention", "gqa_attention")}
            # the padding after a narrower section's last dim is not a column
            for cells in heads.values():
                while cells and cells[-1] == "":
                    cells.pop()
            assert heads["mla_attention"] == ["n_heads", "d_kv_lora", "d_rope"], heads
            assert heads["mla_dsa_attention"] == ["n_heads", "d_kv_lora", "d_indexer_head"], heads
            assert heads["gqa_attention"] == ["n_heads", "", "d_head", "n_kv_heads"], heads
            browser.close()
    finally:
        httpd.shutdown()
