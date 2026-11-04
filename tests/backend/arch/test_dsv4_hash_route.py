"""DeepSeek-V4's hash-routed MoE layers look experts up by token id.

Source (`inference/model.py`): `Block.forward` hands every ffn the token ids
(`x = self.ffn(x, input_ids)`), and the hash `Gate` holds a whole
`tid2eid = nn.Parameter(vocab_size, n_activated_experts)` and indexes it
locally, `indices = self.tid2eid[input_ids]` (vLLM deepseek_v4
nvidia/model.py does the same with the whole table on every rank). So
`hash_route` reads the ids, under tensor parallelism the table is neither
cut nor gathered, and a step reads the rows its ids select.
"""
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

MODEL = "deepseek-ai/DeepSeek-V4-Flash"


def _rows(tp: int):
    from tests.snapshots import resolve_snapshot, shared_loader
    from backend.inference.parallel_config import ParallelConfig
    from backend.inference.pipeline import analyze
    resolved = resolve_snapshot(MODEL, shared_loader())
    parallel = ParallelConfig(tp_attention=tp, tp_dense_ffn=tp,
                              tp_moe_expert=tp)
    return analyze(resolved, parallel=parallel, batch_size=4, avg_input_tokens=1, avg_output_tokens=0,
                   phase="decode", breakdown=False)


def test_hash_route_reads_the_token_ids_from_a_table_held_whole_at_tp8():
    from backend.arch.ops import get_op
    from backend.arch.eval.dim import value_of
    result = _rows(8)
    program = result.program
    routes = [n for n in program.nodes() if n.op == "hash_route"]
    assert routes
    for node in routes:
        ids, table = node.inputs_obj
        assert ids.external_sid == "input_ids" and ids.src is None
        assert table.src.name == "gate.tid2eid"
        # [B, s_ctx] token ids in, one expert id per active slot out.
        ins = result.shapes.values[node.uid].ins
        assert [int(value_of(d)) for d in ins[0]] == [4, 1]
    rows = [r for r in result.per_op_timing
            if "hash_route" in r["aggregation_id"]
            or "tid2eid" in r["aggregation_id"]]
    assert rows
    assert not [r for r in rows if get_op(r["op"]).is_comm_op], rows
    # Every rank holds the whole table, so a rank's row reads are the same
    # at TP8 as at TP1.
    one = {r["aggregation_id"]: r["weight_read_bytes_per_rank"]
           for r in _rows(1).per_op_timing if r["op"] == "hash_route"}
    eight = {r["aggregation_id"]: r["weight_read_bytes_per_rank"]
             for r in rows if r["op"] == "hash_route"}
    assert eight == one and all(eight.values())


def test_hash_route_reads_the_rows_its_token_ids_select():
    """`tid2eid[input_ids]` reads one table row per token — n_activated_experts
    I64 ids, 6 x 8 = 48 bytes (checkpoint `layers.X.ffn.gate.tid2eid`:
    [129280, 6] I64) — and never more than the whole table."""
    from tests.snapshots import resolve_snapshot, shared_loader
    from backend.inference.parallel_config import ParallelConfig
    from backend.inference.pipeline import analyze
    row_bytes, table_bytes = 6 * 8, 129280 * 6 * 8
    resolved = resolve_snapshot(MODEL, shared_loader())
    parallel = ParallelConfig(tp_attention=8, tp_dense_ffn=8, tp_moe_expert=8)
    for batch, seq_len, phase in ((4, 1, "decode"), (4, 4096, "prefill"),
                                  (2, 65536, "prefill")):
        result = analyze(resolved, parallel=parallel, batch_size=batch,
                         avg_input_tokens=seq_len, avg_output_tokens=0, phase=phase, breakdown=False)
        rows = [r for r in result.per_op_timing if r["op"] == "hash_route"]
        assert sum(r["count_per_step"] for r in rows) == 3
        tokens = batch * seq_len
        for r in rows:
            assert r["weight_read_bytes_per_rank"] == r["count_per_step"] * min(
                tokens * row_bytes, table_bytes), (phase, tokens, r)
