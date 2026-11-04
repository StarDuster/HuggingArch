"""The per-op table's fold (`web/src/moduleFold.ts`): a row is hidden for
costing nothing, never for its op name."""
from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest

WEB = Path(__file__).resolve().parents[2] / "web"


def _missing(reason: str) -> None:
    if os.environ.get("CI"):
        pytest.fail(f"module fold test cannot run in CI: {reason}")
    pytest.skip(reason)


def _visible(tmp_path: Path, rows: list, folded: list) -> list:
    if shutil.which("node") is None:
        _missing("node not on PATH")
    esbuild = WEB / "node_modules" / ".bin" / "esbuild"
    if not esbuild.exists():
        _missing("web/node_modules missing; run `npm ci` in web/")
    bundle = tmp_path / "moduleFold.mjs"
    built = subprocess.run(
        [str(esbuild), str(WEB / "src" / "moduleFold.ts"), "--bundle",
         "--format=esm", "--platform=neutral", f"--outfile={bundle}"],
        capture_output=True, text=True, timeout=60, check=False)
    assert built.returncode == 0, built.stderr
    runner = tmp_path / "run.mjs"
    runner.write_text(
        "import { visibleFold } from './moduleFold.mjs';\n"
        f"const rows = {json.dumps(rows)};\n"
        f"const out = visibleFold(rows, new Set({json.dumps(folded)}));\n"
        "console.log(JSON.stringify(out.map((v) => v.row.name)));\n")
    run = subprocess.run(["node", str(runner)], cwd=tmp_path,
                         capture_output=True, text=True, timeout=30, check=False)
    assert run.returncode == 0, run.stderr
    return json.loads(run.stdout)


def _row(name, op, depth, **cost):
    return {"name": name, "op": op, "depth": depth,
            "aggregation_id": f"blk|{name}", **cost}


def test_a_view_that_moves_bytes_stays_and_a_costless_one_goes(tmp_path):
    rows = [
        _row("attn", "mla_attention", 1, t_op_agg=3e-3, access_bytes_agg=10),
        _row("query_concat", "concat", 2, t_op_s=1e-4, access_bytes_per_rank=6e8),
        _row("permute", "permute", 2, t_op_s=5e-5, access_bytes_per_rank=2e8),
        _row("gather", "gather", 2, t_op_s=1e-5, access_bytes_per_rank=1e6),
        _row("q_split_heads", "head_split", 2, t_op_agg=0),
        _row("unflatten", "view_split", 3),
        _row("attn_apply", "flash_attention", 2, flops_per_rank=100, t_op_s=2e-3),
        _row("all_reduce", "all_reduce", 2, comm_bytes_per_rank=4096),
    ]
    assert _visible(tmp_path, rows, []) == [
        "attn", "query_concat", "permute", "gather", "attn_apply", "all_reduce"]


def test_a_named_view_op_with_cost_is_never_hidden_by_name(tmp_path):
    """Every op the old name list hid, once it costs time, shows."""
    ops = ["transpose", "view_split", "view_merge", "broadcast", "slice",
           "head_split", "head_merge", "head_broadcast", "concat", "permute",
           "gather"]
    rows = [_row(op, op, 1, t_op_s=1e-6) for op in ops]
    assert _visible(tmp_path, rows, []) == ops


def test_a_folded_row_hides_its_subtree_not_itself(tmp_path):
    rows = [
        _row("mlp", "swiglu_ffn", 1, t_op_agg=1e-3),
        _row("gate_proj", "linear", 2, flops_per_rank=5, t_op_s=5e-4),
        _row("down_proj", "linear", 2, flops_per_rank=5, t_op_s=5e-4),
        _row("out", "add", 1, t_op_s=1e-5),
    ]
    assert _visible(tmp_path, rows, ["blk|mlp"]) == ["mlp", "out"]
