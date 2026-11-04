"""A row gather's declaration (`OpSpec.lookup`) and what is refused.

One declaration covers a lookup into the op's own weight (an embedding) and
a lookup into a stored parameter handed in on an input (DeepSeek-V4
`hash_route`, model.py `Gate.forward`: `self.tid2eid[input_ids]`). Every
reader asks `Program.lookups`, so the facts below hold for both:

  * the gathered tensor is stored as a TABLE, whoever owns it;
  * a step reads the rows its ids select, `min(T, T x lookups / rows)`;
  * a declaration or a graph that cannot say where the table is, or a table
    without the `[rows, ..., row width]` geometry, is refused.
"""
from __future__ import annotations

import sys
import tempfile
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT))

from backend.arch.eval import traffic as T
from backend.arch.eval.program import Program
from backend.arch.eval.refusal import SpecRefusal
from backend.arch.eval.storage import StorageKind
from backend.arch.eval.transforms import Deployment
from backend.arch.loader import SpecLoader, SpecLoadError
from backend.arch.ops import Lookup, OpSpec

VOCAB, WIDTH, ACTIVE, EXPERTS = 1000, 64, 4, 16


def _zero(d):
    return 0


def _spec(table_in: str) -> str:
    return f"""\
match: {{ model_type: [hr], model_id: [hr] }}
dims:
  n_vocab: config.vocab_size
  d_model: config.hidden_size
  n_experts_active: config.num_experts_per_tok
  n_experts: config.n_routed_experts
blocks:
  blk:
    inputs: [x, input_ids]
    graph:
      tbl: {{ op: parameter_tensor, shape: [n_vocab, n_experts_active], weight_axes: [{{ replicated: true }}, {{ replicated: true }}] }}
      sc: {{ op: linear, in: x, dims: {{ in: d_model, out: n_experts }}, weight_axes: [{{ replicated: true }}, {{ replicated: true }}] }}
      route: {{ op: hash_route, in: [{table_in}], dims: {{ n_experts_active: n_experts_active, n_vocab: n_vocab }} }}
      g: {{ op: gather, in: [sc, route], dims: {{ in: n_experts, k: n_experts_active }} }}
      up: {{ op: linear, in: g, dims: {{ in: n_experts_active, out: d_model }}, weight_axes: [{{ replicated: true }}, {{ replicated: true }}] }}
      res: {{ op: add, in: [x, up], dim: d_model }}
    outputs: [res]
model_inputs:
  input_ids: {{ kind: tokens }}
model:
  - {{ name: table, op: embedding, inputs: [input_ids], boundary_role: input_embedding,
      dims: {{ n_vocab: n_vocab, d_model: d_model }},
      weight_axes: [{{ replicated: true }}, {{ replicated: true }}] }}
  - {{ name: layers, template: blk, repeat: "1", inputs: [table, input_ids],
      streams: {{ hidden: {{ carries: true }} }} }}
  - {{ name: head, op: linear, boundary_role: output_head, inputs: [layers],
      dims: {{ in: d_model, out: n_vocab }},
      weight_axes: [{{ replicated: true }}, {{ replicated: true }}] }}
"""


def _program(table_in: str = "input_ids, tbl") -> Program:
    with tempfile.TemporaryDirectory() as d:
        base = Path(d)
        for sub in ("components/library", "components/adapters",
                    "components/drafts", "models"):
            (base / sub).mkdir(parents=True)
        (base / "models" / "hr.yaml").write_text(_spec(table_in))
        resolved = SpecLoader(base_dir=str(base)).match(
            {"model_type": "hr", "vocab_size": VOCAB, "hidden_size": WIDTH,
             "num_experts_per_tok": ACTIVE, "n_routed_experts": EXPERTS,
             "torch_dtype": "bfloat16"},
            model_id="hr")
    return Program.of(resolved)


def _analysis():
    return Deployment(_program()).at(batch=2, input_tokens=8).analyze()


def _node(analysis, op):
    (node,) = [n for n in analysis.program.nodes() if n.op == op]
    return node


# -- the two placements of the table, one fact --------------------------------

def test_both_gathered_tables_are_stored_as_tables():
    analysis = _analysis()
    for op in ("embedding", "parameter_tensor"):
        (tensor,) = analysis.storage.values[_node(analysis, op).uid]
        assert tensor.kind is StorageKind.TABLE, op


def test_both_lookups_read_the_rows_their_ids_select():
    analysis = _analysis()
    layer = T.derive(analysis)
    tokens = 2 * 8
    embed = layer.values[_node(analysis, "embedding").uid]
    route = layer.values[_node(analysis, "hash_route").uid]
    # BF16 rows: WIDTH x 2 bytes for the embedding, ACTIVE x 2 for tid2eid.
    assert float(embed.weight_reads) == tokens * WIDTH * 2
    assert float(route.borrowed_weight_reads) == tokens * ACTIVE * 2
    assert float(route.weight_reads) == 0


# -- the declaration ----------------------------------------------------------

@pytest.mark.parametrize("lookup, weight, why", [
    (Lookup(ids=-1), ("n_vocab", "d_model"), "ids port -1"),
    (Lookup(ids=2), ("n_vocab", "d_model"), "ids port 2"),
    (Lookup(ids=0, table=-1), (), "table port -1"),
    (Lookup(ids=0, table=2), (), "table port 2"),
    (Lookup(ids=0, table=0), (), "share port 0"),
    (Lookup(ids=0, table=1), ("n_vocab", "d_model"), "stores a weight"),
    (Lookup(ids=0), (), "declares the table as its weight"),
])
def test_a_lookup_declaration_says_where_the_table_is(lookup, weight, why):
    with pytest.raises(ValueError, match=why):
        OpSpec("lookup_probe", flops_fn=_zero, params_fn=_zero,
               shape_in=(("B", "s_ctx"), ("n_vocab", "d_model")),
               shape_out=("B", "s_ctx", "d_model"), shape_kind="explicit",
               weight_shape=weight, lookup=lookup)


# -- the graph ----------------------------------------------------------------

def test_a_lookup_wired_without_its_table_is_refused():
    with pytest.raises(SpecLoadError, match="requires at least 2"):
        _program(table_in="input_ids")


@pytest.mark.parametrize("table_in", ["input_ids, sc", "input_ids, x"])
def test_a_table_port_that_reads_no_stored_parameter_is_refused(table_in):
    with pytest.raises(SpecRefusal, match="not a stored parameter"):
        _ = _program(table_in=table_in).lookups


# -- the table ----------------------------------------------------------------

def _fractions(analysis, storage):
    shaped = SimpleNamespace(program=analysis.program, shapes=analysis.shapes,
                             storage=SimpleNamespace(values=storage))
    return T._table_read_fractions(shaped, analysis.program, analysis.shapes,
                                   {})


def test_a_table_that_stores_nothing_is_refused():
    analysis = _analysis()
    storage = dict(analysis.storage.values)
    del storage[_node(analysis, "parameter_tensor").uid]
    with pytest.raises(ValueError, match="stores nothing"):
        _fractions(analysis, storage)


@pytest.mark.parametrize("shape", [(VOCAB * ACTIVE,), (VOCAB, ACTIVE + 1)])
def test_a_table_without_rows_of_the_gathered_width_is_refused(shape):
    analysis = _analysis()
    uid = _node(analysis, "parameter_tensor").uid
    storage = dict(analysis.storage.values)
    (tensor,) = storage[uid]
    storage[uid] = (replace(tensor, shape=shape),)
    with pytest.raises(ValueError, match=r"a table is \[rows"):
        _fractions(analysis, storage)


def test_a_fused_kernel_that_absorbs_a_lookup_gathers_its_rows():
    """DeepSeek-V4's hash router runs `tid2eid[input_ids]` inside the top-k
    launch: the rows read are the lookup's own, billed on the launch."""
    analysis = _analysis()
    route = _node(analysis, "hash_route").uid
    alone = T._table_read_fractions(analysis, analysis.program,
                                    analysis.shapes, {})
    fused = T._table_read_fractions(analysis, analysis.program,
                                    analysis.shapes, {-1: (route,)})
    assert fused[-1] == alone[route]
    assert route not in fused


def test_a_fused_kernel_bills_one_lookup():
    analysis = _analysis()
    route = _node(analysis, "hash_route").uid
    table = _node(analysis, "embedding").uid
    with pytest.raises(ValueError, match="absorbed 2 lookups"):
        T._table_read_fractions(analysis, analysis.program, analysis.shapes,
                                {-1: (route, table)})
