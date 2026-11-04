"""A row measured before an identity change — the dim-symbol rename
(59db3827 / bb1db576, 08-04), the grouped GEMM's routed `n_tokens` (schema
v6), the per-expert mean's move off `s_query` onto `s_expert_rows` (10-24) —
has an old shape. Its annotations — physical shape and workload — are
the ones a current row of the same kernel problem has."""
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pytest

from backend.inference.benchmark_identity import current_shape
from backend.service.bench.routes import row_annotations

# (op, retired shape, the same shape as a current row names it) — one per
# kind the shared corpus held before the rename.
CASES = [
    ("linear", {"B": 1, "K": 4096, "N": 6144, "Sq": 128},
     {"B": 1, "d_reduce": 4096, "d_out": 6144, "s_query": 128}),
    ("grouped_gemm", {"E": 8, "K": 7168, "N": 4096, "Sq": 32},
     {"n_experts": 8, "n_tokens": 256, "d_reduce": 7168, "d_out": 4096, "s_expert_rows": 32}),
    # Current letters, before schema v6: n_experts × s_query routed rows —
    # and that `s_query` was the per-expert mean, now `s_expert_rows`.
    ("grouped_gemm", {"n_experts": 8, "d_reduce": 7168, "d_out": 4096, "s_query": 32},
     {"n_experts": 8, "n_tokens": 256, "d_reduce": 7168, "d_out": 4096, "s_expert_rows": 32}),
    # After v6, before the mean's rename.
    ("grouped_gemm", {"n_experts": 8, "n_tokens": 256, "d_reduce": 7168,
                      "d_out": 4096, "s_query": 32},
     {"n_experts": 8, "n_tokens": 256, "d_reduce": 7168, "d_out": 4096, "s_expert_rows": 32}),
    ("flash_attention", {"B": 1, "D": 128, "Lq": 4096, "N": 32, "Nk": 8, "Sq": 4096},
     {"B": 1, "d_head": 128, "s_kv": 4096, "n_heads": 32, "n_kv_heads": 8, "s_query": 4096}),
    ("mla_attention", {"B": 2, "D": 192, "Dv": 128, "Lq": 4096, "N": 16, "Sq": 1},
     {"B": 2, "d_head": 192, "d_v_head": 128, "s_kv": 4096, "n_heads": 16, "s_query": 1}),
    ("rmsnorm", {"B": 1, "H": 7168, "Sq": 16}, {"B": 1, "d_model": 7168, "s_query": 16}),
    ("silu_and_mul", {"B": 4, "I": 2048, "Sq": 1}, {"B": 4, "d_ffn": 2048, "s_query": 1}),
]


@pytest.mark.parametrize("op, retired, current", CASES, ids=[f"{c[0]}-{i}" for i, c in enumerate(CASES)])
@pytest.mark.parametrize("phase", ["prefill", "decode"])
def test_a_retired_shape_annotates_as_its_current_spelling(op, retired, current, phase):
    def row(shape):
        return {"op": op, "shape_json": shape, "phase": phase, "kernel_impl": "torch"}
    assert row_annotations(row(retired)) == row_annotations(row(current))


def test_an_old_dispatch_boundary_shape_is_not_rekeyed():
    """The boundary signatures' current cells name BOTH sides of the
    dispatch — a measured-before row (whose `s_query` was the per-expert
    mean) cannot supply the token side, and in a current row `s_query` IS
    the token side, so no rename can run: the old shape is left spelled as
    it was and matches no current key, which is what "re-measure" means."""
    for op, old in (
            ("permute", '{"d_model":7168,"n_experts":8,"s_query":32}'),
            ("combine", '{"d_model":7168,"n_experts":8,"s_query":32}')):
        assert current_shape(op, old) is old


def test_a_boundary_shape_before_its_routed_total_gains_it():
    """A boundary row measured between the token side's arrival and the
    routed total's (both 10-24) states both factors of the total at the
    padded geometry, so its total is their product."""
    for op in ("permute", "combine"):
        got = json.loads(current_shape(
            op, '{"B":1,"d_model":64,"n_experts":8,"s_expert_rows":4,"s_query":128}'))
        assert got["n_tokens"] == 8 * 4


def test_a_current_shape_is_unchanged():
    for op, text in [("rmsnorm", '{"B":1,"d_model":64,"s_query":4}'),
                     ("permute", '{"B":1,"d_model":64,"n_experts":8,'
                                 '"n_tokens":32,"s_expert_rows":4,"s_query":128}'),
                     ("combine", '{"B":1,"d_model":64,"n_experts":8,'
                                 '"n_tokens":32,"s_expert_rows":4,"s_query":128}'),
                     ("grouped_gemm", '{"d_out":64,"d_reduce":64,"n_experts":8,"n_tokens":125,"s_expert_rows":16}')]:
        assert current_shape(op, text) is text


def test_a_retired_axis_beside_its_current_name_is_refused():
    with pytest.raises(ValueError, match="d_reduce"):
        current_shape("linear", '{"B":1,"K":64,"d_reduce":128,"N":8,"Sq":1}')
    # The same value twice is the same fact.
    assert current_shape("linear", '{"B":1,"K":64,"d_reduce":64,"N":8,"Sq":1}') == \
        '{"B":1,"d_out":8,"d_reduce":64,"s_query":1}'


def test_call_token_roles_are_a_closed_vocabulary():
    from backend.arch.dims import DimSymbol, call_token_axes
    assert call_token_axes() == {"factor": ("B", "s_expert_rows", "s_query"),
                                 "total": ("n_tokens",), "context": ("s_kv",)}
    with pytest.raises(ValueError, match="call_tokens"):
        DimSymbol(name="x", kind="whole", meaning="m", group="g", call_tokens="batch")
