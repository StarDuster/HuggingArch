"""A table the checkpoint stores in parts is, in the weight breakdown, the
runtime table: its on-disk bytes are the parts' bytes, summed.

`stored_split` declares that one runtime tensor is written as `parts` files
`<module>.<segment(i)>.weight` (transformers concatenates them at load —
Qwen3.8-Flash-Next's n-gram table, `conversion_mapping.py:1826-1828`). The
deployment holds one tensor; the file index holds the parts. The breakdown
joins the leaf to the parts the declaration names.
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from backend.arch.eval.program import Program
from backend.arch.eval.transforms import Deployment
from backend.arch.loader import SpecLoader
from backend.inference.parallel_config import ParallelConfig
from backend.inference.weight_breakdown import per_rank_weight_breakdown
from tests.snapshots import resolve_snapshot, snapshot_header

BF16 = 2
TABLE, WIDTH, ROWS, PARTS = 4096, 32, 4, 8

SPEC = f"""\
match: {{ model_type: [split_table], model_id: [split_table] }}
dims:
  n_vocab: config.vocab_size
  d_model: config.hidden_size
  n_ple_vocab: "{TABLE}"
  n_ngram_heads: "{ROWS}"
blocks:
  blk:
    graph:
      n: {{ op: rmsnorm, in: x, dim: d_model, weight_axes: [{{ replicated: true }}] }}
    outputs: [n]
model_inputs:
  input_ids: {{ kind: tokens }}
model:
  - {{ name: embed_tokens, op: embedding, inputs: [input_ids], boundary_role: input_embedding,
      dims: {{ n_vocab: n_vocab, d_model: d_model }},
      weight_axes: [{{ replicated: true }}, {{ replicated: true }}] }}
  - {{ name: ngram, op: hash_embedding, inputs: [input_ids],
      dims: {{ n_vocab: n_ple_vocab, d_model: d_model, rows: n_ngram_heads }},
      weight_axes: [{{ replicated: true }}, {{ replicated: true }}],
      stored_split: {{ axis: 0, parts: "{PARTS}", segment: "shard_{{i}}" }} }}
  - {{ name: layers, template: blk, repeat: "1", inputs: [embed_tokens],
      streams: {{ hidden: {{ carries: true }} }} }}
  - {{ name: norm, op: rmsnorm, dim: d_model, inputs: [layers], weight_axes: [{{ replicated: true }}] }}
  - {{ name: head, op: linear, boundary_role: output_head, inputs: [norm],
      dims: {{ in: d_model, out: n_vocab }},
      weight_axes: [{{ replicated: true }}, {{ replicated: true }}] }}
"""


def _resolved():
    with tempfile.TemporaryDirectory() as d:
        base = Path(d)
        for sub in ("components/library", "components/adapters",
                    "components/drafts", "models"):
            (base / sub).mkdir(parents=True)
        (base / "models" / "split_table.yaml").write_text(SPEC)
        return SpecLoader(base_dir=str(base)).match(
            {"model_type": "split_table", "torch_dtype": "bfloat16",
             "vocab_size": 1000, "hidden_size": WIDTH},
            model_id="split_table")


def _leaves(node):
    yield node
    for child in node.get("children") or ():
        yield from _leaves(child)


def _leaf_bytes(breakdown, name):
    nodes = [n for top in breakdown["groups"] + breakdown["model_level"]
             for n in _leaves(top) if n.get("name") == name]
    assert len(nodes) == 1, (name, len(nodes))
    return int(nodes[0]["bytes_full"])


def test_a_split_table_is_the_sum_of_its_parts_on_disk():
    resolved = _resolved()
    analysis = Deployment(Program.of(resolved)).at(batch=1, input_tokens=8).analyze()
    parts = [r for r in analysis.checkpoint_tensors() if r.name.startswith("ngram.")]
    assert [r.name for r in parts] == [f"ngram.shard_{i}.weight" for i in range(PARTS)]
    on_disk = sum(int(r.storage_bytes) for r in parts)
    assert on_disk == TABLE * WIDTH * BF16
    breakdown = per_rank_weight_breakdown(
        resolved=resolved, parallel=ParallelConfig())
    assert _leaf_bytes(breakdown, "ngram") == on_disk


def test_flash_next_ngram_table_is_its_128_checkpoint_shards():
    """Qwen3.8-Flash-Next: 128 header tensors `…ngram_embedding.shard_{i}.weight`,
    each [2500012, 160] BF16, are the one n-gram table the PLE layer reads."""
    model_id = "Qwen/Qwen3.8-Flash-Next"
    shards = [g for g in snapshot_header(model_id)
              if ".ngram_embedding.shard_" in g.template]
    assert sum(g.count for g in shards) == 128
    header = 0
    for group in shards:
        assert group.dtype == "BF16"
        rows, width = group.shape
        header += rows * width * BF16 * group.count
    assert header == 128 * 2500012 * 160 * BF16
    breakdown = per_rank_weight_breakdown(
        resolved=resolve_snapshot(model_id), parallel=ParallelConfig())
    assert _leaf_bytes(breakdown, "ple_embedding.ngram_embedding") == header
