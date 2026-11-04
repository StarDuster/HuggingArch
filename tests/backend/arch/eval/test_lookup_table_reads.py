"""A lookup table is gathered, not streamed: a step reads the rows it looks up.

`OpSpec.lookup` declares a row gather from a stored table (vLLM
`vocab_parallel_embedding.py` `F.embedding(input_, weight)`; sglang
`fused_vocab_parallel_embedding`). Its weight read is the rank's table bytes
times `min(1, N / R)` — N lookups, R the table's global rows — and the two ways
a table can be split give that figure two different meanings:

  - split along the row WIDTH (or not split): every rank reads one narrow row
    per lookup. Exact per rank, whatever the token ids are;
  - split along the ROWS (vocab-parallel): the rank a lookup hits depends on the
    token id, which no workload carries. Under the balanced-lookup convention
    the rank serves the share of lookups its share of real rows is, so the
    figure is the LONG-RUN AVERAGE expected useful bytes per rank — not any one
    step's physical bytes on the busiest rank.
"""
from __future__ import annotations

import math
import sys
import tempfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT))

from backend.arch.eval.program import Program
from backend.arch.eval.transforms import DECODE, Deployment
from backend.arch.layout.physical_cost import (
    physical_traffic)
from backend.arch.loader import SpecLoader
from tests.snapshots import shared_program, shared_walk

SNAPSHOTS = PROJECT_ROOT / "backend" / "arch" / "snapshots"
BF16 = 2

# A table whose rows are split along their width: `d_model` is cut over the
# attention world through `n_heads`, the way Qwen4Exp's n-gram table is cut
# (`configuration_qwen4_exp.py` `colwise_gather_output`).
WIDTH_SPLIT = """\
match: { model_type: [lut], model_id: [lut] }
dims:
  n_vocab: config.vocab_size
  n_heads: config.num_attention_heads
  d_head: config.head_dim
  d_model: config.hidden_size
blocks:
  blk:
    graph:
      n: { op: rmsnorm, in: x, dim: d_model, weight_axes: [{ replicated: true }] }
    outputs: [n]
model_inputs:
  input_ids: { kind: tokens }
model:
  - { name: table, op: embedding, inputs: [input_ids], boundary_role: input_embedding,
      dims: { n_vocab: n_vocab, d_model: "n_heads * d_head" },
      weight_axes: [{ replicated: true }, { world: attention, parts: n_heads, fit: exact }] }
  - { name: layers, template: blk, repeat: "1", inputs: [table],
      streams: { hidden: { carries: true } } }
  - { name: norm, op: rmsnorm, dim: d_model, inputs: [layers], weight_axes: [{ replicated: true }] }
  - { name: head, op: linear, boundary_role: output_head, inputs: [norm],
      dims: { in: d_model, out: n_vocab },
      weight_axes: [{ replicated: true }, { replicated: true }] }
"""
VOCAB, HEADS, HEAD_DIM = 1000, 4, 8


def _width_split_program() -> Program:
    with tempfile.TemporaryDirectory() as d:
        base = Path(d)
        for sub in ("components/library", "components/adapters",
                    "components/drafts", "models"):
            (base / sub).mkdir(parents=True)
        (base / "models" / "lut.yaml").write_text(WIDTH_SPLIT)
        resolved = SpecLoader(base_dir=str(base)).match(
            {"model_type": "lut", "vocab_size": VOCAB,
             "num_attention_heads": HEADS, "head_dim": HEAD_DIM,
             "hidden_size": HEADS * HEAD_DIM, "torch_dtype": "bfloat16"},
            model_id="lut")
    return Program.of(resolved)


def _llama() -> Program:
    return shared_program('meta-llama/Llama-3.2-1B')


def _table_read(deployment: Deployment) -> float:
    """The per-rank weight read of the program's one lookup leaf."""
    analysis = shared_walk(deployment)
    traffic = physical_traffic(analysis)
    (leaf,) = [n for n in analysis.program.nodes() if n.op == "embedding"]
    return float(traffic.values[leaf.uid].weight_reads)


def test_width_split_table_reads_one_narrow_row_per_lookup_exactly():
    """Split along its width, every rank reads every lookup's row, at the
    rank's width: B·S·(d/tp)·w, with no dependence on the table's length."""
    program = _width_split_program()
    batch, tokens = 2, 16
    width = HEADS * HEAD_DIM
    for tp in (1, 2, 4):
        read = _table_read(Deployment(program).at(
            batch=batch, input_tokens=tokens).on(tp=tp))
        assert read == batch * tokens * (width // tp) * BF16, tp


def test_width_split_table_read_is_capped_at_the_table():
    """Each row is read at most once: more lookups than rows read the table."""
    program = _width_split_program()
    tokens = 4 * VOCAB
    read = _table_read(Deployment(program).at(batch=1, input_tokens=tokens).on(tp=2))
    assert read == VOCAB * (HEADS * HEAD_DIM // 2) * BF16


def test_vocab_split_table_is_the_long_run_average_expected_useful_bytes():
    """Split along its rows, a lookup hits the one rank that holds its row.
    Under the balanced-lookup convention a rank serves its share of real rows'
    share of the lookups — the fullest rank holds ceil(R/tp) rows — so its
    figure is N · ceil(R/tp)/R rows: the long-run average over steps, not the
    bytes any single step moves on the busiest rank (at B=1, tp=2 one rank
    reads a whole row and the other none; the figure says half a row each)."""
    program = _llama()
    analysis = shared_walk(Deployment(program).at(batch=1, input_tokens=8))
    n_vocab = int(analysis.binding["n_vocab"])
    d_model = int(analysis.binding["d_model"])
    lookups = 4096
    for tp in (1, 2):
        read = _table_read(Deployment(program).at(
            batch=1, input_tokens=lookups).on(tp=tp))
        rows_on_rank = math.ceil(n_vocab / tp)
        assert read == lookups * rows_on_rank * d_model * BF16 / n_vocab, tp
    # Decode, B=1, tp=2: half a row each — the expectation, not the step.
    read = _table_read(Deployment(program).at(batch=1, input_tokens=lookups)
                       .during(DECODE, cache_len=lookups).on(tp=2))
    assert read == d_model * BF16 / 2


def test_a_projection_reading_the_same_table_streams_it_whole():
    """Llama ties its head to the embedding table: the gather reads rows, the
    projection that shares the stored copy reads all of it
    (`Program.read_aliases`)."""
    program = _llama()
    analysis = shared_walk(Deployment(program).at(batch=1, input_tokens=1).during(
        DECODE, cache_len=16))
    traffic = physical_traffic(analysis)
    table = int(analysis.binding["n_vocab"]) * int(analysis.binding["d_model"]) * BF16
    (head,) = [uid for uid in program.read_aliases]
    assert float(traffic.values[head].weight_reads) == table
