#!/usr/bin/env python3
"""Reproducible health sweep for the Inspector `/analyze` path and the
Inference APIs that page (and the sibling Inference tab) actually POST.

The Inspector lives at `/inspector.html` (there is no `/analyze` route).
Anonymous Analyze clicks POST `/analyze`; a signed-in session POSTs
`/api/spec/analyze` with the same body. The Inference tab's default
Estimate click POSTs `/inference/estimate` (and then GETs `/inference/compute-graph`
for prefill + decode). This script hits those endpoints with the
frontend's real request bodies.

Usage (from repo root; do not use `uv run`):

    PYTHONPATH=. \\
      OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 NUMEXPR_NUM_THREADS=8 \\
      ./.venv/bin/python tools/analyze_health.py --out /tmp/analyze_health

    ./.venv/bin/python tools/analyze_health.py --browser --out /tmp/analyze_health

`--api` (default) uses FastAPI TestClient, no listening port.
`--browser` builds `web/dist` if needed, starts this copy on :2099,
drives headless Chromium against `/inspector.html`, then shuts down.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import socket
import subprocess
import sys
import time
import traceback
import urllib.error
import urllib.request
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import get_context
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple


# ---------------------------------------------------------------------------
# Thread caps — set before any numeric runtime is imported in workers.
# ---------------------------------------------------------------------------
for _k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_k, "8")

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
MODELS_DIR = ROOT / "backend" / "arch" / "models"
SNAPSHOTS_DIR = ROOT / "backend" / "arch" / "snapshots"
SUPPORTED_TXT = ROOT / "backend" / "arch" / "supported_models.txt"
USER_SPECS_ROOT = Path.home() / ".huggingarch" / "user_specs"
MODEL_INFO_CACHE = Path.home() / ".cache" / "huggingarch" / "model_info"

# Inspector.tsx initial state / Advanced defaults.
INSPECTOR_DEFAULTS = {
    "batch_size": 1,
    "seq_len": 2048,
}

# Inference.tsx initial state (EMPTY_PARALLEL + B200 + 1024/1024).
INFER_GPU = "B200"
INFER_FRAC = 0.9
INFER_IN_TOK = 1024
INFER_OUT_TOK = 1024
EMPTY_PARALLEL = {
    "pp": 1,
    "tp_attention": 1,
    "tp_dense_ffn": 1,
    "tp_moe_expert": 1,
    "ep_moe": 1,
    "cp": 1,
    "dp_attention": 1,
    "dp_replicas": 1,
}
# backend/inference/sweep_grid.py AUDIT_PARALLEL — MoE picker seed.
AUDIT_PARALLEL = {"ep_moe": 8, "dp_attention": 8}

# Models the browser pass walks with Inspector defaults. Dense / MoE / MLA /
# DSA / linear-attn / multimodal / fp8 / int4 — one of each family that the
# shared tree actually ships.
BROWSER_MODELS = [
    "meta-llama/Llama-3.2-1B",          # dense, tiny
    "Qwen/Qwen3-8B",                    # dense GQA + QK-norm
    "Qwen/Qwen2.5-7B",                  # dense qwen2
    "google/gemma-2-9b",                # gemma2
    "THUDM/glm-4-9b",                   # glm4 dense
    "deepseek-ai/DeepSeek-V3",          # MLA MoE fp8
    "deepseek-ai/DeepSeek-V3.2",        # DSA
    "deepseek-ai/DeepSeek-V4-Pro",      # HCA / CSA
    "Qwen/Qwen3-30B-A3B",               # MoE
    "Qwen/Qwen3-Coder-Next",            # linear attention + MoE
    "Qwen/Qwen3.5-35B-A3B",             # hybrid linear + MoE
    "google/gemma-4-12B-it",            # gemma4 unified / multimodal
    "moonshotai/Kimi-K2.5",             # int4
    "openai/gpt-oss-20b",               # mxfp4
    "zai-org/GLM-5",                    # DSA MoE
    "XiaomiMiMo/MiMo-V2-Flash",         # flash / MLA-family
    "meituan-longcat/LongCat-Flash-Thinking-2601",
    "MiniMaxAI/MiniMax-M2.7",
]

_MODEL_ID_LINE = re.compile(
    r"\bmodel_id[ \t]*:[ \t]*(.+)",
)
_BASE_LINE = re.compile(r"^base:[ \t]*(\S+)", re.M)


# ---------------------------------------------------------------------------
# Model collection
# ---------------------------------------------------------------------------

def _tokens_from_model_id_value(value: str) -> List[str]:
    value = value.strip()
    if value.startswith("["):
        value = value[1:].split("]", 1)[0]
    out = []
    for tok in value.split(","):
        t = tok.strip().strip("'\"")
        if t and not t.startswith("#"):
            out.append(t)
    return out


def parse_match_model_ids(raw_text: str) -> List[str]:
    """Best-effort `match.model_id` values from a spec YAML (one-line form)."""
    m = _MODEL_ID_LINE.search(raw_text)
    if not m:
        return []
    return _tokens_from_model_id_value(m.group(1))


def _follow_base_text(path: Path, seen: Optional[set] = None) -> str:
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return ""
    seen = seen or set()
    if path in seen:
        return text
    seen.add(path)
    m = _BASE_LINE.search(text)
    if not m:
        return text
    base = m.group(1).strip().strip("'\"")
    bp = path.parent / f"{base}.yaml"
    if not bp.is_file():
        bp = MODELS_DIR / f"{base}.yaml"
    if bp.is_file():
        return text + "\n" + _follow_base_text(bp, seen)
    return text


def model_flags_from_yaml(path: Path) -> Dict[str, bool]:
    blob = _follow_base_text(path)
    return {
        "moe": bool(re.search(r"\bffn_moe\s*:", blob)),
        "fp8": bool(re.search(r"\bmethod:\s*fp8\b", blob)),
        "int4": bool(re.search(r"\bnum_bits:\s*4\b", blob)),
        "linear_attn": bool(re.search(
            r"linear[_-]attn|gated_delta|qwen35_linear|gla_|delta_net",
            blob, re.I,
        )),
        "has_attention": bool(re.search(r"\battention\s*:", blob)),
    }


def _add(models: Dict[str, Dict[str, Any]], model_id: str, source: str,
         flags: Optional[Dict[str, bool]] = None, yaml_path: Optional[str] = None) -> None:
    mid = (model_id or "").strip()
    if not mid:
        return
    rec = models.setdefault(mid, {
        "model_id": mid,
        "sources": [],
        "moe": False,
        "fp8": False,
        "int4": False,
        "linear_attn": False,
        "has_attention": True,
        "yaml_path": None,
    })
    if source not in rec["sources"]:
        rec["sources"].append(source)
    if flags:
        for k, v in flags.items():
            rec[k] = rec.get(k) or bool(v)
    if yaml_path and not rec.get("yaml_path"):
        rec["yaml_path"] = yaml_path


def collect_models() -> List[Dict[str, Any]]:
    """Shared + cached + user-overlay model ids, offline, read-only."""
    models: Dict[str, Dict[str, Any]] = {}

    if MODELS_DIR.is_dir():
        for path in sorted(MODELS_DIR.glob("*.yaml")):
            try:
                raw = path.read_text(encoding="utf-8")
            except OSError:
                continue
            ids = parse_match_model_ids(raw)
            if not ids:
                continue
            flags = model_flags_from_yaml(path)
            for mid in ids:
                _add(models, mid, "shared_yaml", flags, str(path))

    if SUPPORTED_TXT.is_file():
        for line in SUPPORTED_TXT.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            _add(models, line, "supported_txt")

    if SNAPSHOTS_DIR.is_dir():
        for path in sorted(SNAPSHOTS_DIR.glob("*.json")):
            mid = path.stem.replace("__", "/")
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                data = {}
            if isinstance(data, dict) and data.get("model_id"):
                mid = str(data["model_id"])
            _add(models, mid, "snapshot")

    if MODEL_INFO_CACHE.is_dir():
        for path in sorted(MODEL_INFO_CACHE.glob("*.json")):
            mid = path.stem.replace("__", "/")
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                data = {}
            if isinstance(data, dict) and data.get("model_id"):
                mid = str(data["model_id"])
            _add(models, mid, "model_info_cache")

    if USER_SPECS_ROOT.is_dir():
        for yml in USER_SPECS_ROOT.glob("*/*/models/*.yaml"):
            try:
                raw = yml.read_text(encoding="utf-8")
            except OSError:
                continue
            ids = parse_match_model_ids(raw)
            flags = model_flags_from_yaml(yml)
            if not ids:
                # overlay dir name is owner__name
                ids = [yml.stem.replace("__", "/")]
            for mid in ids:
                _add(models, mid, "user_overlay", flags, str(yml))

    # supported_txt first so `--limit` hits real Inspector models, not a
    # leftover model_info cache entry (e.g. attacker/evil).
    _rank = {
        "supported_txt": 0,
        "shared_yaml": 1,
        "snapshot": 2,
        "model_info_cache": 3,
        "user_overlay": 4,
    }
    return sorted(
        models.values(),
        key=lambda r: (
            min(_rank.get(s, 9) for s in r["sources"]),
            r["model_id"].lower(),
        ),
    )


# ---------------------------------------------------------------------------
# Request matrix
# ---------------------------------------------------------------------------

def _par(**overrides) -> dict:
    p = dict(EMPTY_PARALLEL)
    p.update(overrides)
    p["cp_scheme"] = "ulysses"
    return p


def cluster_size(p: dict) -> int:
    per_stage = max(
        int(p.get("tp_attention") or 1) * int(p.get("cp") or 1) * int(p.get("dp_attention") or 1),
        int(p.get("tp_dense_ffn") or 1) * int(p.get("cp") or 1) * int(p.get("dp_attention") or 1),
        int(p.get("tp_moe_expert") or 1) * int(p.get("ep_moe") or 1),
    )
    return int(p.get("pp") or 1) * per_stage * int(p.get("dp_replicas") or 1)


def estimate_body(model_id: str, parallel: dict, *,
                  preset: str = "as_shipped") -> dict:
    """Inference.tsx `run()` body: the page's precision preset as
    `precision` (omitted as shipped, as the page omits it)."""
    return {
        "model_id": model_id,
        "gpu_sku": INFER_GPU,
        "gpu_memory_fraction": INFER_FRAC,
        **({} if preset == "as_shipped"
           else {"precision": {"preset": preset}}),
        "parallel": parallel,
        "prefix_cached_tokens": 0,
        "prefill_chunk_size": 8192,
        "include_aux_weights": False,
        "avg_input_tokens": INFER_IN_TOK,
        "avg_output_tokens": INFER_OUT_TOK,
        "num_gpus": cluster_size(parallel),
        "selection_mode": "envelope",
    }


def compute_graph_query(model_id: str, parallel: dict, phase: str) -> dict:
    """ComputeGraph.tsx `computeGraphQuery` — query params as a dict."""
    q = {
        "model_id": model_id,
        "pp": parallel.get("pp", 1),
        "tp_attention": parallel.get("tp_attention", 1),
        "tp_dense_ffn": parallel.get("tp_dense_ffn", 1),
        "tp_moe_expert": parallel.get("tp_moe_expert", 1),
        "ep_moe": parallel.get("ep_moe", 1),
        "cp": parallel.get("cp", 1),
        "dp_attention": parallel.get("dp_attention", 1),
        "dp_replicas": parallel.get("dp_replicas", 1),
        "cp_scheme": parallel.get("cp_scheme", "ulysses"),
        "phase": phase,
        "avg_input_tokens": INFER_IN_TOK,
        "avg_output_tokens": INFER_OUT_TOK,
        "prefix_cached_tokens": 0,
        "speculative_draft_tokens": 0,
        "batch_size": 1,
    }
    return q


def jobs_for_model(rec: Dict[str, Any], *, with_inference: bool = True) -> List[dict]:
    mid = rec["model_id"]
    jobs = [
        {
            "kind": "analyze",
            "model_id": mid,
            "config": "inspector_default",
            "method": "POST",
            "path": "/analyze",
            "json": {
                "model_id": mid,
                **INSPECTOR_DEFAULTS,
            },
        },
    ]
    if not with_inference:
        return jobs

    jobs.append({
        "kind": "precision_effective",
        "model_id": mid,
        "config": "inspector_default",
        "method": "POST",
        "path": "/precision/effective",
        "json": {"model_id": mid},
    })

    # The precision presets the Inference page offers: as shipped, and the
    # FP8 cache profile every checkpoint can run.
    presets: List[str] = ["as_shipped", "kv_fp8"]

    parallels: List[Tuple[str, dict]] = [
        ("single", _par()),
        ("tp8", _par(tp_attention=8, tp_dense_ffn=8, tp_moe_expert=8)),
    ]
    if rec.get("moe"):
        parallels.append(("ep8", _par(ep_moe=8, dp_attention=8)))
    if rec.get("has_attention", True):
        parallels.append(("cp8", _par(cp=8)))

    # Frontend default: dense stays 单卡; MoE auto-seeds AUDIT_PARALLEL.
    if rec.get("moe"):
        default_par = _par(**AUDIT_PARALLEL)
        default_name = "frontend_default_moe_seed"
    else:
        default_par = _par()
        default_name = "frontend_default"

    seen = set()

    def _add_est(cfg_name: str, parallel: dict, preset: str) -> None:
        key = (cfg_name, json.dumps(parallel, sort_keys=True), preset)
        if key in seen:
            return
        seen.add(key)
        jobs.append({
            "kind": "estimate",
            "model_id": mid,
            "config": cfg_name,
            "method": "POST",
            "path": "/inference/estimate",
            "json": estimate_body(mid, parallel, preset=preset),
        })

    # Frontend-default group first: the page opens as shipped.
    _add_est(default_name, default_par, "as_shipped")

    for pname, parallel in parallels:
        for preset in presets:
            _add_est(f"{pname}_{preset}", parallel, preset)

    # Compute graph: default deployment, both phases (the drill-down the
    # Inference page loads after a successful estimate).
    jobs.extend({
        "kind": "compute_graph",
        "model_id": mid,
        "config": f"{default_name}_{phase}",
        "method": "GET",
        "path": "/inference/compute-graph",
        "params": compute_graph_query(mid, default_par, phase),
    } for phase in ("prefill", "decode"))
    return jobs


def meta_jobs() -> List[dict]:
    return [
        {"kind": "meta", "model_id": "", "config": "supported-models",
         "method": "GET", "path": "/supported-models"},
        {"kind": "meta", "model_id": "", "config": "dim-glossary",
         "method": "GET", "path": "/dim-glossary"},
        {"kind": "meta", "model_id": "", "config": "inspector-examples",
         "method": "GET", "path": "/inspector/examples"},
        {"kind": "meta", "model_id": "", "config": "inference-gpus",
         "method": "GET", "path": "/inference/gpus"},
        {"kind": "meta", "model_id": "", "config": "inference-defaults",
         "method": "GET", "path": "/inference/defaults"},
        # Truncated repo id — a model lookup must answer it as a JSON 4xx.
        {"kind": "probe", "model_id": "glm-5.", "config": "truncated_repo_id",
         "method": "POST", "path": "/precision/effective",
         "json": {"model_id": "glm-5."}},
    ]


# ---------------------------------------------------------------------------
# Bad-value scan
# ---------------------------------------------------------------------------

_POS_HINT = re.compile(
    r"(flops|params|bytes|latency|throughput|tokens|time_s|t_sol|t_op|"
    r"weight|hidden_size|num_layers|cluster_size|gflops)",
    re.I,
)


def scan_bad_values(obj: Any, path: str = "$") -> List[str]:
    """Walk a JSON-decoded payload for NaN / Inf / obviously broken values."""
    hits: List[str] = []

    def _walk(v: Any, p: str) -> None:
        if isinstance(v, bool):
            pass
        elif isinstance(v, float):
            if math.isnan(v):
                hits.append(f"{p}: NaN")
            elif math.isinf(v):
                hits.append(f"{p}: Inf")
            elif v < 0 and _POS_HINT.search(p):
                hits.append(f"{p}: negative {v}")
        elif isinstance(v, int):
            if v < 0 and _POS_HINT.search(p):
                hits.append(f"{p}: negative {v}")
        elif isinstance(v, str):
            if v in ("NaN", "Infinity", "-Infinity", "nan", "inf", "-inf"):
                hits.append(f"{p}: string {v}")
        elif isinstance(v, dict):
            for k, child in v.items():
                _walk(child, f"{p}.{k}")
        elif isinstance(v, list):
            if p.endswith(".layer_groups") and not v:
                hits.append(f"{p}: empty list")
            for i, child in enumerate(v[:40]):
                _walk(child, f"{p}[{i}]")

    _walk(obj, path)
    return hits[:40]


def _first_line(msg: str) -> str:
    return (msg or "").strip().splitlines()[0][:400]


# ---------------------------------------------------------------------------
# Worker (imported in the child process)
# ---------------------------------------------------------------------------

def run_one(job: dict) -> dict:
    """Hit one API with TestClient. Imported backend only in the child."""
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    t0 = time.perf_counter()
    rec: Dict[str, Any] = {
        "kind": job.get("kind"),
        "model_id": job.get("model_id"),
        "config": job.get("config"),
        "method": job.get("method"),
        "path": job.get("path"),
        "status": None,
        "ok": False,
        "exc_type": None,
        "exc_msg": None,
        "elapsed_s": None,
        "bad_values": [],
        "success": None,
        "detail": None,
    }
    try:
        from fastapi.testclient import TestClient
        from backend.api import app

        client = TestClient(app, raise_server_exceptions=False)
        method = job["method"].upper()
        path = job["path"]
        if method == "GET":
            resp = client.request("GET", path, params=job.get("params") or None)
        else:
            resp = client.request(method, path, json=job.get("json"))
        rec["status"] = resp.status_code
        rec["elapsed_s"] = round(time.perf_counter() - t0, 3)
        body: Any
        try:
            body = resp.json()
        except Exception:
            body = None
            rec["exc_type"] = "JSONDecodeError"
            rec["exc_msg"] = _first_line(resp.text or "")
            rec["ok"] = False
            return rec
        if isinstance(body, dict):
            rec["success"] = body.get("success")
            detail = body.get("detail") or body.get("message") or body.get("error")
            if detail is not None:
                rec["detail"] = _first_line(str(detail))
            rec["bad_values"] = scan_bad_values(body)
            # Inspector treats success:false as a rendered failure.
            if path == "/analyze" and body.get("success") is False:
                rec["ok"] = False
                rec["exc_type"] = body.get("error_kind") or "analyze_failed"
                rec["exc_msg"] = rec["detail"]
                return rec
        rec["ok"] = 200 <= resp.status_code < 400 and not rec["bad_values"]
        if rec["status"] and rec["status"] >= 400 and not rec["exc_type"]:
            rec["exc_type"] = f"HTTP{rec['status']}"
            rec["exc_msg"] = rec.get("detail")
    except Exception as e:
        rec["elapsed_s"] = round(time.perf_counter() - t0, 3)
        rec["ok"] = False
        rec["exc_type"] = type(e).__name__
        rec["exc_msg"] = _first_line(str(e) or traceback.format_exc())
    return rec


# ---------------------------------------------------------------------------
# API sweep
# ---------------------------------------------------------------------------

def _worker_count(requested: int) -> int:
    cpu = os.cpu_count() or 4
    return max(1, min(requested, 16, max(1, cpu)))


def run_api_sweep(out_dir: Path, *, workers: int, limit: Optional[int],
                  with_inference: bool) -> Dict[str, Any]:
    models = collect_models()
    if limit:
        models = models[:limit]
    jobs: List[dict] = meta_jobs()
    for rec in models:
        jobs.extend(jobs_for_model(rec, with_inference=with_inference))

    out_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = out_dir / "api_results.jsonl"
    summary_path = out_dir / "api_summary.json"
    print(f"models={len(models)} jobs={len(jobs)} workers={workers}", flush=True)

    results: List[dict] = []
    n_ok = 0
    with ProcessPoolExecutor(
        max_workers=workers, mp_context=get_context("spawn"),
    ) as pool:
        futs = {pool.submit(run_one, job): i for i, job in enumerate(jobs)}
        done = 0
        with jsonl_path.open("w", encoding="utf-8") as fh:
            for fut in as_completed(futs):
                try:
                    rec = fut.result(timeout=240)
                except Exception as e:
                    rec = {
                        "ok": False,
                        "exc_type": type(e).__name__,
                        "exc_msg": _first_line(str(e)),
                        "status": None,
                    }
                results.append(rec)
                fh.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")
                fh.flush()
                done += 1
                if rec.get("ok"):
                    n_ok += 1
                if done % 10 == 0 or not rec.get("ok"):
                    flag = "ok" if rec.get("ok") else "FAIL"
                    print(
                        f"[{done}/{len(jobs)} {flag}] "
                        f"{rec.get('method')} {rec.get('path')} "
                        f"{rec.get('model_id')} {rec.get('config')} "
                        f"status={rec.get('status')} "
                        f"{rec.get('exc_type') or ''} "
                        f"{(rec.get('exc_msg') or '')[:120]}",
                        flush=True,
                    )

    by_kind: Dict[str, Dict[str, int]] = {}
    by_exc: Dict[str, int] = {}
    matrix: Dict[str, Dict[str, int]] = {}
    for rec in results:
        k = rec.get("kind") or "?"
        bucket = by_kind.setdefault(k, {"n": 0, "ok": 0, "fail": 0})
        bucket["n"] += 1
        bucket["ok" if rec.get("ok") else "fail"] += 1
        if not rec.get("ok"):
            et = rec.get("exc_type") or f"HTTP{rec.get('status')}"
            by_exc[et] = by_exc.get(et, 0) + 1
        mid = rec.get("model_id") or "(meta)"
        cfg = rec.get("config") or "?"
        cell = matrix.setdefault(mid, {})
        cell[cfg] = 1 if rec.get("ok") else 0

    summary = {
        "n_models": len(models),
        "n_jobs": len(jobs),
        "n_ok": n_ok,
        "n_fail": len(results) - n_ok,
        "pass_rate": (n_ok / len(results)) if results else 0.0,
        "by_kind": by_kind,
        "by_exc": dict(sorted(by_exc.items(), key=lambda kv: -kv[1])),
        "models": [
            {"model_id": m["model_id"], "sources": m["sources"],
             "moe": m.get("moe"), "fp8": m.get("fp8")}
            for m in models
        ],
        "matrix": matrix,
    }
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False, default=str))
    print(f"wrote {jsonl_path}  {summary_path}  "
          f"ok={n_ok}/{len(results)}", flush=True)
    return summary


# ---------------------------------------------------------------------------
# Browser sweep
# ---------------------------------------------------------------------------

def _port_free(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(0.3)
        return s.connect_ex(("127.0.0.1", port)) != 0


def _wait_health(base: str, timeout: float = 90.0) -> None:
    t0 = time.time()
    last = ""
    while time.time() - t0 < timeout:
        try:
            with urllib.request.urlopen(base + "/health", timeout=2) as r:
                if r.status == 200:
                    return
                last = f"HTTP {r.status}"
        except Exception as e:
            last = str(e)
        time.sleep(0.4)
    raise RuntimeError(f"backend at {base} never became healthy: {last}")


def _ensure_web_dist() -> None:
    index = ROOT / "web" / "dist" / "index.html"
    src = ROOT / "web" / "src"
    if index.is_file():
        built = index.stat().st_mtime
        if not any(p.stat().st_mtime > built for p in src.rglob("*") if p.is_file()):
            return
    print("building web/dist …", flush=True)
    subprocess.run(
        ["npm", "run", "build"],
        cwd=str(ROOT / "web"),
        check=True,
        timeout=600,
    )


def _start_backend(port: int) -> subprocess.Popen:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT)
    for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
              "NUMEXPR_NUM_THREADS"):
        env[k] = env.get(k, "8")
    py = ROOT / ".venv" / "bin" / "python"
    cmd = [str(py), "-m", "backend.api", "--host", "127.0.0.1", "--port", str(port)]
    # The child keeps its own copy of the descriptor; ours closes here.
    with open("/tmp/analyze_health/backend_2099.log", "w", encoding="utf-8") as log:
        proc = subprocess.Popen(
            cmd, cwd=str(ROOT), env=env,
            stdout=log, stderr=subprocess.STDOUT,
        )
    return proc


def run_browser_sweep(out_dir: Path, *, port: int, models: Sequence[str],
                      timeout_s: float = 120.0) -> Dict[str, Any]:
    from playwright.sync_api import sync_playwright

    out_dir.mkdir(parents=True, exist_ok=True)
    shots = out_dir / "shots"
    shots.mkdir(parents=True, exist_ok=True)
    _ensure_web_dist()
    if not _port_free(port):
        raise RuntimeError(f"port {port} is already in use")
    proc = _start_backend(port)
    base = f"http://127.0.0.1:{port}"
    rows: List[dict] = []
    try:
        _wait_health(base)
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            page = browser.new_page(viewport={"width": 1440, "height": 900})
            console_buf: List[str] = []
            page_errors: List[str] = []
            page.on("console", lambda msg: console_buf.append(
                f"{msg.type}: {msg.text}") if msg.type in ("error", "warning") else None)
            page.on("pageerror", lambda exc: page_errors.append(str(exc)))

            for mid in models:
                console_buf.clear()
                page_errors.clear()
                slug = mid.replace("/", "__")
                rec: Dict[str, Any] = {
                    "model_id": mid,
                    "ok": False,
                    "error_text": "",
                    "console": [],
                    "page_errors": [],
                    "nan": False,
                    "blank": False,
                    "overlay": None,
                    "shot": None,
                    "elapsed_s": None,
                }
                t0 = time.perf_counter()
                try:
                    # Full navigation (not a hash-only change): a hash swap on
                    # the same /inspector.html document leaves the previous
                    # model's #results visible and the wait returns in 100ms.
                    page.goto(
                        f"{base}/inspector.html",
                        wait_until="domcontentloaded",
                        timeout=30_000,
                    )
                    inp = page.locator("#model-id")
                    inp.wait_for(timeout=15_000)
                    inp.fill(mid)
                    page.locator('[data-action="analyze"]').click()
                    page.wait_for_function(
                        """(mid) => {
                          const gen = document.querySelector('#generate-overlay');
                          if (gen && !gen.classList.contains('hidden')) return true;
                          const err = document.querySelector('#error');
                          if (err && !err.classList.contains('hidden')) return true;
                          const id = document.querySelector('#result-model-id');
                          const results = document.querySelector('#results');
                          return !!(results && !results.classList.contains('hidden')
                                    && id && id.textContent.trim() === mid);
                        }""",
                        arg=mid,
                        timeout=int(timeout_s * 1000),
                    )
                    # Stop a spec-gen/refine overlay so we don't wait on an agent.
                    for ov in ("#generate-overlay", "#refine-overlay"):
                        el = page.locator(ov)
                        if el.count() and el.first.is_visible():
                            rec["overlay"] = ov
                            stop = page.locator(
                                f"{ov} button[data-action='stop-spec-generation'], "
                                f"{ov} button[data-action='stop-spec-refinement'], "
                                f"{ov} button"
                            )
                            if stop.count():
                                try:
                                    stop.first.click(timeout=2000)
                                except Exception:
                                    pass
                    err = page.locator("#error-message")
                    if err.count() and page.locator("#error").first.is_visible():
                        rec["error_text"] = (err.inner_text() or "").strip()
                    body_text = page.inner_text("main") or ""
                    rec["nan"] = bool(re.search(r"\bNaN\b|\bInfinity\b", body_text))
                    results = page.locator("#results")
                    rec["blank"] = (
                        results.count() > 0
                        and results.first.is_visible()
                        and not rec["error_text"]
                        and ("—" in (page.locator("#total-params").inner_text()
                                     if page.locator("#total-params").count() else "—")
                             or not (page.locator("#total-params").inner_text() or "").strip())
                    )
                    rec["ok"] = (
                        not rec["error_text"]
                        and not rec["nan"]
                        and not page_errors
                        and rec["overlay"] != "#generate-overlay"
                        and results.count() > 0
                        and results.first.is_visible()
                    )
                except Exception as e:
                    rec["error_text"] = _first_line(str(e))
                    rec["ok"] = False
                rec["elapsed_s"] = round(time.perf_counter() - t0, 3)
                rec["console"] = list(console_buf)
                rec["page_errors"] = list(page_errors)
                shot = shots / f"{slug}.png"
                try:
                    page.screenshot(path=str(shot), full_page=True)
                    rec["shot"] = str(shot)
                except Exception:
                    pass
                rows.append(rec)
                flag = "ok" if rec["ok"] else "FAIL"
                print(f"[browser {flag}] {mid} {rec['error_text'][:160]} "
                      f"overlay={rec['overlay']} nan={rec['nan']}", flush=True)

            browser.close()
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()

    path = out_dir / "browser_results.json"
    path.write_text(json.dumps(rows, indent=2, ensure_ascii=False, default=str))
    n_ok = sum(1 for r in rows if r.get("ok"))
    print(f"wrote {path}  ok={n_ok}/{len(rows)}", flush=True)
    return {"n": len(rows), "n_ok": n_ok, "rows": rows}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, default=Path("/tmp/analyze_health"))
    p.add_argument("--workers", type=int, default=8,
                   help="process-pool size, capped at 16")
    p.add_argument("--limit", type=int, default=None,
                   help="cap the number of models (debug)")
    p.add_argument("--api", action="store_true", help="run TestClient sweep")
    p.add_argument("--browser", action="store_true",
                   help="run headless Inspector sweep on :2099")
    p.add_argument("--all", action="store_true", help="api + browser")
    p.add_argument("--no-inference", action="store_true",
                   help="only POST /analyze")
    p.add_argument("--port", type=int, default=2099)
    p.add_argument("--list-models", action="store_true")
    args = p.parse_args(argv)

    if args.list_models:
        for rec in collect_models():
            print(f"{rec['model_id']}\t{','.join(rec['sources'])}\t"
                  f"moe={int(rec['moe'])} fp8={int(rec['fp8'])}")
        return 0

    run_api = args.api or args.all or not (args.browser or args.api or args.all)
    run_br = args.browser or args.all
    args.out.mkdir(parents=True, exist_ok=True)

    if run_api:
        workers = _worker_count(args.workers)
        run_api_sweep(
            args.out, workers=workers, limit=args.limit,
            with_inference=not args.no_inference,
        )
    if run_br:
        models = list(BROWSER_MODELS)
        if args.limit:
            models = models[: args.limit]
        run_browser_sweep(args.out, port=args.port, models=models)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
