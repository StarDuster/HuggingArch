"""A lookup table stored in a quantized format, with its scales.

DeepSeek-V4.1-Flash's Engram table (`ParallelEngramEmbedding`, model.py:
296-325) is `weight: [rows, d] float8_e4m3fn` with `scale: [rows, d / 32]
float8_e8m0fnu` — one scale per 32 elements of a row — row-sharded over the
ranks. A lookup gathers the fp8 rows and, with the same ids, the scale rows
(`F.embedding(local_indices, self.scale)`), then multiplies them in.

So the table is a `(rows, width)` tensor a quant scheme tiles exactly as it
tiles a linear's `(out, in)`, with a `1 x 32` block:

  - storage: the table at the scheme's dtype, the `.scale` sibling the scheme
    implies at `[rows, width / 32]`, booked as quantization metadata;
  - reads: a step reads the rows it gathers and those rows' scales, not the
    table;
  - sharding: a row-split table splits its scales along the same rows;
  - FLOPs: the format moves no FLOPs, as for every quantized weight;
  - a quant-method row naming a tensor no scheme describes (a gain vector) is
    a validate finding, not a silent fall-back to the default dtype.
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT))

from backend.arch.eval.program import Program
from backend.arch.eval.storage import StorageKind
from backend.arch.eval.transforms import Deployment
from backend.arch.layout.persistent import physical_artifacts
from backend.arch.layout.physical_cost import physical_traffic
from backend.arch.loader import SpecLoader
from backend.arch.validate._packing_consistency import run_quant_override_check

VOCAB, HEADS, HEAD_DIM = 4096, 4, 32
WIDTH = HEADS * HEAD_DIM
BLOCK = 32

TABLE_FP8 = """\
  overrides:
    - { tensors_re: '^table\\.weight$', dtype: F8_E4M3, method: fp8_ue8m0, block_h: 1, block_w: 32 }
"""
GAIN_FP8 = """\
  overrides:
    - { tensors_re: '^norm\\.weight$', dtype: F8_E4M3, method: fp8_ue8m0, block_h: 1, block_w: 32 }
"""


def _program(quant: str = "") -> Program:
    spec = f"""\
match: {{ model_type: [lutq], model_id: [lutq] }}
dims:
  n_vocab: config.vocab_size
  n_heads: config.num_attention_heads
  d_head: config.head_dim
  d_model: config.hidden_size
{"quant_context:" + chr(10) + quant if quant else ""}
blocks:
  blk:
    graph:
      n: {{ op: rmsnorm, in: x, dim: d_model, weight_axes: [{{ replicated: true }}] }}
    outputs: [n]
model_inputs:
  input_ids: {{ kind: tokens }}
model:
  - {{ name: table, op: embedding, inputs: [input_ids], boundary_role: input_embedding,
      dims: {{ n_vocab: n_vocab, d_model: d_model }},
      weight_axes: [{{ world: attention, parts: n_vocab, fit: pad }}, {{ replicated: true }}] }}
  - {{ name: layers, template: blk, repeat: "1", inputs: [table],
      streams: {{ hidden: {{ carries: true }} }} }}
  - {{ name: norm, op: rmsnorm, dim: d_model, inputs: [layers], weight_axes: [{{ replicated: true }}] }}
  - {{ name: head, op: linear, boundary_role: output_head, inputs: [norm],
      dims: {{ in: d_model, out: n_vocab }},
      weight_axes: [{{ replicated: true }}, {{ replicated: true }}] }}
"""
    with tempfile.TemporaryDirectory() as d:
        base = Path(d)
        for sub in ("components/library", "components/adapters",
                    "components/drafts", "models"):
            (base / sub).mkdir(parents=True)
        (base / "models" / "lutq.yaml").write_text(spec)
        resolved = SpecLoader(base_dir=str(base)).match(
            {"model_type": "lutq", "vocab_size": VOCAB,
             "num_attention_heads": HEADS, "head_dim": HEAD_DIM,
             "hidden_size": WIDTH, "torch_dtype": "bfloat16"},
            model_id="lutq")
    return Program.of(resolved)


def _rows(analysis, module: str):
    return {row.name: row for row in analysis.iter_checkpoint()
            if row.name.startswith(f"{module}.")}


def test_the_table_is_stored_in_the_scheme_with_row_block_scales():
    analysis = Deployment(_program(TABLE_FP8)).at(batch=1, input_tokens=8).analyze()
    rows = _rows(analysis, "table")
    assert set(rows) == {"table.weight", "table.scale"}
    weight, scale = rows["table.weight"], rows["table.scale"]
    assert weight.tensor.kind is StorageKind.TABLE
    assert weight.storage_dtype == "F8_E4M3"
    assert int(weight.storage_bytes) == VOCAB * WIDTH
    assert scale.is_metadata
    assert tuple(int(x) for x in scale.shape) == (VOCAB, WIDTH // BLOCK)
    assert scale.storage_dtype == "F8_E8M0FNU"
    # The scale is quantization metadata, not a parameter: params are the
    # table's elements whatever its format.
    native = Deployment(_program()).at(batch=1, input_tokens=8).analyze()
    assert analysis.total().params == native.total().params
    assert int(analysis.total().quant_metadata_params) == VOCAB * WIDTH // BLOCK


def test_a_step_reads_the_gathered_rows_and_their_scales():
    lookups = 64
    analysis = Deployment(_program(TABLE_FP8)).at(
        batch=1, input_tokens=lookups).analyze()
    traffic = physical_traffic(analysis)
    (leaf,) = [n for n in analysis.program.nodes() if n.op == "embedding"]
    read = traffic.values[leaf.uid]
    assert float(read.weight_reads) == lookups * WIDTH * 1
    assert float(read.metadata_reads) == lookups * (WIDTH // BLOCK) * 1


def test_a_row_split_table_splits_its_scales_along_the_same_rows():
    tp = 4
    analysis = Deployment(_program(TABLE_FP8)).at(
        batch=1, input_tokens=8).on(tp=tp).analyze()
    shapes = {a.artifact_name: a.runtime_shape
              for a in physical_artifacts(analysis)
              if a.artifact_name.startswith("table.")}
    assert shapes["table.weight"] == (VOCAB // tp, WIDTH)
    assert shapes["table.scale"] == (VOCAB // tp, WIDTH // BLOCK)


def test_the_format_moves_no_flops():
    quantized = Deployment(_program(TABLE_FP8)).at(batch=2, input_tokens=16).analyze()
    native = Deployment(_program()).at(batch=2, input_tokens=16).analyze()
    assert quantized.total().flops == native.total().flops


def test_a_method_row_on_a_tensor_no_scheme_describes_is_a_finding():
    analysis = Deployment(_program(GAIN_FP8)).at(batch=1, input_tokens=8).analyze()
    findings = run_quant_override_check(analysis)
    assert [f.kind for f in findings] == ["quant_override_undescribed"]
    assert "norm.weight" in findings[0].detail
    assert "dtype:" in findings[0].detail
    # The table row is described; it raises nothing.
    analysis = Deployment(_program(TABLE_FP8)).at(batch=1, input_tokens=8).analyze()
    assert run_quant_override_check(analysis) == []


def test_a_scale_shape_override_lays_out_the_table_scales():
    """`scale_shape` is resolved against the table's own `(out, in)` as it
    is against a linear's: rows and width."""
    analysis = Deployment(_program("""\
  overrides:
    - tensors_re: '^table\\.weight$'
      dtype: F8_E4M3
      method: fp8_ue8m0
      block_h: 1
      block_w: 32
      scale_shape: ["out", "in // (2 * block_w)", "2"]
""")).at(batch=1, input_tokens=8).analyze()
    scale = _rows(analysis, "table")["table.scale"]
    assert tuple(int(x) for x in scale.shape) == (VOCAB, WIDTH // 64, 2)
