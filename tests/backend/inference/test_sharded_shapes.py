"""Physical shape layers used by graph, table, and kernel consumers."""
from __future__ import annotations

import re
import sys
import unittest
from pathlib import Path

import sympy as sp

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from tests.snapshots import resolve_snapshot
from backend.arch.eval.dim import value_of
from backend.inference.framework_bindings import get_binding
from backend.inference.parallel_config import ParallelConfig
from backend.inference.pipeline import analyze




def _resolved(model_name):
    return resolve_snapshot(model_name)


def _analyze(model_name: str, parallel: ParallelConfig, *,
             framework=None):
    resolved = _resolved(model_name)
    fb = None
    if framework is not None:
        fb = get_binding(framework)
        if fb is None:
            return None
    return analyze(
        resolved_spec=resolved,
        parallel=parallel,
        framework_binding=fb,
        batch_size=1,
        avg_input_tokens=64, avg_output_tokens=0,
        breakdown=True,
    )


def _find_sid(result, suffix: str) -> str:
    """First spec_id ending with `suffix` (template_name agnostic)."""
    for program in (result.program, result.pre_program):
        for node in program.nodes():
            if node.aggregation_id.endswith(suffix):
                return node.aggregation_id
    raise AssertionError(f"no spec_id ending with {suffix!r}")


def _shape(result, sid, *, full=False):
    for program, layer in (
        (result.program, result.full_shapes if full else result.shapes),
        (result.pre_program,
         result.pre_full_shapes if full else result.pre_shapes),
    ):
        for node in program.nodes():
            if node.aggregation_id == sid and node.uid in layer.values:
                return layer.values[node.uid]
    raise AssertionError(f"no shape for {sid!r}")


def _axes(dims):
    return [int(value_of(axis)) for axis in dims] if dims is not None else None


def _input(shape):
    values = [_axes(dims) for dims in shape.ins]
    return values[0] if len(values) == 1 else values


def _access(result):
    return {
        result.program.by_uid(uid).aggregation_id: row
        for uid, row in result.access.values.items()
    }


class TestColumnParallelOutSharded(unittest.TestCase):
    """A column-parallel linear (output axis carries a sharded symbol)
    divides its `shape_out` last dim by the TP world. DSv3 MLA's
    `q_b_proj` is the canonical case — its `out` is `N*(D + P)`, where
    `N` (num_heads) is in `mla_attention`'s shard_symbols, so TP=8
    shrinks the output from 24576 to 3072.
    """

    def test_dsv3_q_b_proj_tp8(self):
        result = _analyze("deepseek-ai__DeepSeek-V3",
                          ParallelConfig(tp_attention=8))
        sid = _find_sid(result, "self_attn.q_b_proj")
        prn = _shape(result, sid)
        g = _shape(result, sid, full=True)
        self.assertIsNotNone(prn.out)
        # Column-parallel: out dim shards by tp_attention=8 (global/8).
        self.assertEqual(_axes(prn.out)[-1], 3072)
        self.assertEqual(_axes(g.out)[-1], 24576)
        self.assertEqual(_axes(g.out)[-1] // 8, _axes(prn.out)[-1])
        # Per-rank weight dims derived from shape: in = shape_in[-1],
        # out = shape_out[-1].
        in_ = _input(prn)[-1]
        out_ = _axes(prn.out)[-1]
        self.assertEqual((in_, out_), (1536, 3072))


class TestRowParallelInShardedOutReplicated(unittest.TestCase):
    """A row-parallel linear (input axis sharded; output sums via AR)
    divides its `shape_in` but its `shape_out` stays at the unsharded
    width. DSv3 `o_proj`'s `in` is `Dv*N` (heads sharded), so TP=8
    halves shape_in from 16384 to 2048 while shape_out stays at 7168.
    """

    def test_dsv3_o_proj_tp8(self):
        result = _analyze("deepseek-ai__DeepSeek-V3",
                          ParallelConfig(tp_attention=8))
        sid = _find_sid(result, "self_attn.o_proj")
        prn = _shape(result, sid)
        g = _shape(result, sid, full=True)
        # in: Dv*N sharded → 16384/8 = 2048
        in_shape = _input(prn)
        self.assertEqual(int(in_shape[-1]), 2048)
        g_in = _input(g)
        self.assertEqual(int(g_in[-1]), 16384)
        # out: H NOT sharded → stays at 7168 (partial-sum, AR follows).
        self.assertEqual(_axes(prn.out)[-1], 7168)
        self.assertEqual(_axes(g.out)[-1], 7168)
        # Per-rank weight dims: in = shape_in[-1], out = shape_out[-1].
        self.assertEqual((int(in_shape[-1]), _axes(prn.out)[-1]), (2048, 7168))


class TestReplicatedLeafUntouched(unittest.TestCase):
    """A norm leaf whose dims (e.g. `dim=1536`) don't intersect any
    parent shard_symbols stays at divisor=1 — its shape is
    byte-identical to the engine's global shape.
    """

    def test_dsv3_q_a_layernorm_tp8(self):
        result = _analyze("deepseek-ai__DeepSeek-V3",
                          ParallelConfig(tp_attention=8))
        sid = _find_sid(result, "self_attn.q_a_layernorm")
        prn = _shape(result, sid)
        g = _shape(result, sid, full=True)
        # Replicated (divisor was 1): per-rank shape == global shape.
        self.assertEqual(_axes(prn.out), _axes(g.out))
        self.assertEqual(_axes(prn.out)[-1], 1536)


class TestIdentityModeNoOp(unittest.TestCase):
    """When TP=EP=CP=1, physical and semantic shape layers are byte-equal."""

    def test_dsv3_tp1_identity(self):
        result = _analyze("deepseek-ai__DeepSeek-V3", ParallelConfig())
        # Pick a known leaf and confirm equality with the engine shape.
        sid = _find_sid(result, "self_attn.q_b_proj")
        prn = _shape(result, sid)
        semantic = _shape(result, sid, full=True)
        self.assertEqual(_axes(prn.out), _axes(semantic.out))


class TestPerOpTimingShape(unittest.TestCase):
    """The breakdown table's "Shape" column reads `row.shape_out` from
    `compute_per_op_timing`. Verify it picks up the sharded shape rather
    than the global one.
    """

    def test_dsv3_breakdown_shape_sharded(self):
        result = _analyze("deepseek-ai__DeepSeek-V3",
                          ParallelConfig(tp_attention=8))
        for row in result.per_op_timing:
            if row["aggregation_id"].endswith("self_attn.q_b_proj"):
                self.assertEqual(row["shape_out"][-1], 3072)
                return
        self.fail("q_b_proj row not in per_op_timing")


class TestComputeOnlyLeafSharded(unittest.TestCase):
    """Computation-only leaves (concat, split_heads, transpose, rope, …)
    must be sharded too — the physical shape layer covers every leaf whose axes
    carry a sharded symbol, not just the weight-bearing linears. The head
    axis `N` on `query_concat` / `k_split_heads` divides by tp_attention
    exactly as the projection leaves do, so their per-rank shape shrinks
    while the global shape stays full.
    """

    def test_dsv3_query_concat_tp8(self):
        result = _analyze("deepseek-ai__DeepSeek-V3",
                          ParallelConfig(tp_attention=8))
        # q_b is head-major: split its real H axis first, then concatenate
        # Q_nope and Q_pe within each head. TP therefore divides axis 1 while
        # the per-head trailing width remains whole.
        try:
            sid = _find_sid(result, "self_attn.query_concat__mla_attention")
        except AssertionError:
            self.skipTest("query_concat not in this MLA spec variant")
        prn = _shape(result, sid)
        g = _shape(result, sid, full=True)
        self.assertIsNotNone(prn.out)
        self.assertEqual(_axes(prn.out)[1], 16)
        self.assertEqual(_axes(g.out)[1], 128)
        self.assertEqual(_axes(prn.out)[-1], 192)
        self.assertEqual(_axes(g.out)[-1], 192)

    def test_dsv3_k_split_heads_tp8(self):
        result = _analyze("deepseek-ai__DeepSeek-V3",
                          ParallelConfig(tp_attention=8))
        try:
            sid = _find_sid(result, "self_attn.k_split_heads__mla_attention")
        except AssertionError:
            self.skipTest("k_split_heads not in this MLA spec variant")
        prn = _shape(result, sid)
        g = _shape(result, sid, full=True)
        # Heads axis (N=128) divides by tp_attention=8 → 16.
        self.assertIn(16, _axes(prn.out))
        # ... and the global (unsharded) head count stays at 128.
        self.assertIn(128, _axes(g.out))


class TestPerRankFlopsFormula(unittest.TestCase):
    """`_flops_formula_for_row` renders both lines from the per-rank
    forward-transform node, so neither the symbolic nor the substituted
    line carries a trailing divisor — the shape values are already
    per-rank (the symbol N already means per-rank N).
    """

    def test_dsv3_q_b_proj_per_rank_formula(self):
        result = _analyze("deepseek-ai__DeepSeek-V3",
                          ParallelConfig(tp_attention=16))
        for row in result.per_op_timing:
            if not row["aggregation_id"].endswith("self_attn.q_b_proj"):
                continue
            ff = row.get("flops_formula") or {}
            sym = ff.get("symbolic") or ""
            sub = ff.get("substituted") or ""
            # Neither line carries `/ 16` — the symbol N is per-rank N
            # on both sides, so there is nothing left to divide by.
            self.assertNotIn("/ 16", sym)
            self.assertNotIn("/ 16", sub)
            # Sanity: the formula renders something (not empty).
            self.assertTrue(sym, "symbolic line should not be empty")
            self.assertTrue(sub, "substituted line should not be empty")
            return
        self.fail("q_b_proj row not in per_op_timing")


class TestFlopsFormulaValuesLine(unittest.TestCase):
    """The popover's values line shows the numeric substitution
    (`2*4*8192*3072*1024`), not the folded product — the long-standing
    display was `values 541165879296` beside
    `formula 2*B*d_ffn*d_model*s_ctx`, a number nobody could check. The
    bound form must also reproduce the row's per-rank FLOPs: a binding that
    cannot be shown to equal the engine's value falls back to the folded
    number rather than display a wrong formula.
    """

    _IDENT = re.compile(r"[A-Za-z_][A-Za-z_0-9]*")

    def _check_every_row(self, model, parallel):
        result = _analyze(model, parallel)
        checked = 0
        for row in result.per_op_timing:
            ff = row.get("flops_formula") or {}
            sym, sub = ff.get("symbolic") or "", ff.get("substituted") or ""
            if not sym or not self._IDENT.search(sym):
                continue  # no formula, or a bare number on both sides
            checked += 1
            sid = row["aggregation_id"]
            self.assertIn(
                "*", sub,
                f"{sid}: values line folded to a bare number: {sub!r}")
            leftover = self._IDENT.findall(sub)
            self.assertEqual(
                leftover, [],
                f"{sid}: dim symbols survived on the values line: {sub!r}")
            cnt = int(row.get("count_per_step") or 1)
            bound = sp.sympify(sub)
            self.assertFalse(
                bound.free_symbols,
                f"{sid}: values line did not parse to a number: {sub!r}")
            # flops_per_rank is round(value * cnt) — allow the rounding.
            self.assertAlmostEqual(
                float(bound) * cnt, float(row["flops_per_rank"]),
                delta=max(1.0, cnt),
                msg=f"{sid}: values line {sub!r} does not reproduce the "
                    f"row's FLOPs")
        self.assertGreater(checked, 0, "no per-op rows carried a formula")

    def test_llama_tp1(self):
        self._check_every_row("meta-llama__Llama-3.2-3B", ParallelConfig())

    def test_llama_tp2(self):
        self._check_every_row(
            "meta-llama__Llama-3.2-3B",
            ParallelConfig(tp_attention=2, tp_dense_ffn=2))

    def test_llama_tp2_cp2(self):
        # Under Ulysses CP the env's `s_ctx` is the shard while the query
        # axis still carries the full length — the record must bind the
        # axis (what the evaluation read), not the env entry.
        self._check_every_row(
            "meta-llama__Llama-3.2-3B",
            ParallelConfig(tp_attention=2, tp_dense_ffn=2, cp=2))

    def test_dsv3_tp16(self):
        # MoE + MLA: the dispatch zone rebinds B/s_ctx scope-locally and
        # head counts live only inside composite axes — the two cases the
        # folded display used to hide.
        self._check_every_row(
            "deepseek-ai__DeepSeek-V3", ParallelConfig(tp_attention=16))

    def test_dsv3_q_b_proj_weight_term_is_substituted(self):
        result = _analyze("deepseek-ai__DeepSeek-V3",
                          ParallelConfig(tp_attention=8))
        for row in result.per_op_timing:
            if not row["aggregation_id"].endswith("self_attn.q_b_proj"):
                continue
            sub = (row.get("access_bytes_formula") or {}).get(
                "substituted") or ""
            weight_term = sub.split("+")[0]
            self.assertIn(
                "*", weight_term,
                f"weight term folded to a bare number: {weight_term!r}")
            self.assertEqual(self._IDENT.findall(weight_term), [],
                             f"symbols survived: {weight_term!r}")
            return
        self.fail("q_b_proj row not in per_op_timing")


class TestAccessBytesPerComponentSplit(unittest.TestCase):
    """Per-component per-rank access bytes — `LeafAccessBytes.*_bytes_per_rank`
    must reflect the actual sharding behavior of each tensor, not a
    uniform `full // divisor` split. Column-parallel linears keep the
    input activation REPLICATED (each rank reads the full input); the
    earlier uniform formula under-counted that by `divisor`.
    Row-parallel linears keep the output activation replicated
    (partial-sum before AR), similarly under-counted by the legacy
    formula.
    """

    def test_dsv3_kv_b_proj_act_in_replicated_tp16(self):
        # kv_b_proj is column-parallel: in axis carries the unsharded
        # latent (R = kv_lora_rank = 512), out axis carries N*(D+Dv).
        # Per rank: weight reads /16, act_in stays full, act_out /16.
        result = _analyze("deepseek-ai__DeepSeek-V3",
                          ParallelConfig(tp_attention=16))
        ab = None
        for sid, entry in _access(result).items():
            if sid.endswith("self_attn.kv_b_proj"):
                ab = entry
                break
        # Activation IN is replicated on column-parallel — full bytes.
        self.assertEqual(ab.act_in_bytes_per_rank, ab.act_in_bytes,
                         "kv_b_proj act_in must be replicated (column-parallel)")
        # Weights and act_out are sharded by tp_attention=16.
        self.assertEqual(ab.weight_bytes_per_rank * 16, ab.weight_bytes)
        self.assertEqual(ab.act_out_bytes_per_rank * 16, ab.act_out_bytes)
        # per_rank == sum of per-component PR (not full // divisor).
        self.assertEqual(
            ab.per_rank,
            ab.weight_bytes_per_rank + ab.act_in_bytes_per_rank
            + ab.act_out_bytes_per_rank + ab.metadata_bytes_per_rank
            + ab.convert_bytes_per_rank,
        )
        # Sanity: per_rank > full // 16 — uniform divide would have
        # under-counted act_in by 16×.
        self.assertGreater(ab.per_rank, ab.full // 16)

    def test_dsv3_o_proj_act_out_replicated_tp16(self):
        # o_proj is row-parallel: in axis (H = Dv*N) sharded, out axis
        # (H = hidden) replicated (partial-sum, AR follows). Per rank:
        # weight reads /16, act_in /16, act_out FULL.
        result = _analyze("deepseek-ai__DeepSeek-V3",
                          ParallelConfig(tp_attention=16))
        ab = None
        for sid, entry in _access(result).items():
            if sid.endswith("self_attn.o_proj"):
                ab = entry
                break
        self.assertIsNotNone(ab, "o_proj must be in post_access_bytes")
        self.assertEqual(ab.act_out_bytes_per_rank, ab.act_out_bytes,
                         "o_proj act_out must be replicated (row-parallel)")
        self.assertEqual(ab.weight_bytes_per_rank * 16, ab.weight_bytes)
        self.assertEqual(ab.act_in_bytes_per_rank * 16, ab.act_in_bytes)
        self.assertEqual(
            ab.per_rank,
            ab.weight_bytes_per_rank + ab.act_in_bytes_per_rank
            + ab.act_out_bytes_per_rank + ab.metadata_bytes_per_rank
            + ab.convert_bytes_per_rank,
        )

    def test_identity_mode_per_rank_equals_full(self):
        # TP=EP=1 → every divisor is 1 → per_rank == full per leaf.
        result = _analyze("deepseek-ai__DeepSeek-V3", ParallelConfig())
        for sid, ab in _access(result).items():
            self.assertEqual(ab.per_rank, ab.full,
                             f"identity mode: per_rank != full for {sid}")


class TestAccessBytesFormulaNoSpuriousDivisor(unittest.TestCase):
    """Per-rank substitution + per-component divisor: the rendered
    `access_bytes` formula's substituted line must NOT carry a trailing
    `/ <divisor>` when every term that's actually sharded has its shard
    symbol absorbed into per-rank values. The legacy renderer applied
    one uniform `/ div` to the sum, which the substitution helper had
    to bail on whenever any term was shard-symbol-free (e.g. kv_b_proj's
    replicated act_in). After the fix each component carries its own
    `/ div` (or none).

    Both lines follow the same per-rank convention — when substitution
    absorbs the divisor for a term, neither side carries `/ 16` for
    that term.
    """

    def test_dsv3_kv_b_proj_substituted_no_trailing_div(self):
        result = _analyze("deepseek-ai__DeepSeek-V3",
                          ParallelConfig(tp_attention=16))
        for row in result.per_op_timing:
            if not row["aggregation_id"].endswith("self_attn.kv_b_proj"):
                continue
            af = row.get("access_bytes_formula") or {}
            sub = af.get("substituted") or ""
            sym = af.get("symbolic") or ""
            self.assertNotIn("/ 16", sub,
                             f"unexpected /16 in kv_b_proj substituted: {sub!r}")
            self.assertNotIn("/ 16", sym,
                             f"unexpected /16 in kv_b_proj symbolic: {sym!r}")
            return
        self.fail("kv_b_proj row not in per_op_timing")


class TestFusedLeafFormulaShardSubstituted(unittest.TestCase):
    """Fused leaves (e.g. sglang's flash_attention chain-fuses
    qk_score + softmax + attn_apply) carry a multi-term symbolic with
    per-sibling `<name>:` labels. The shard-substitute pass must:
      - strip those labels for parsing
      - parse the values side with sympy `local_dict` so `S` doesn't
        collide with the sympy singleton registry
      - count shard symbols structurally via sympy expr-tree (each
        summand contains exactly one factor of the shard symbol; the
        total expression divides cleanly by `divisor`)
      - rewrite the substituted line with sharded shard-symbol values
        and drop the trailing `/ <divisor>` factor on BOTH lines

    Per the sharded-formula convention: both symbolic and substituted
    lines use per-rank shape — when substitution absorbs the divisor,
    neither side carries `/ 16` (the symbol N is per-rank N).
    """

    def _flash_row(self, result):
        for row in result.per_op_timing:
            if row.get("op") == "flash_attention":
                return row
        return None

    def test_dsv3_sglang_flash_attention_tp16_no_trailing_div(self):
        result = _analyze("deepseek-ai__DeepSeek-V3",
                          ParallelConfig(tp_attention=16),
                          framework="sglang")
        row = self._flash_row(result)
        # Both formulas drop `/ 16` on both lines after shard substitution
        # — values line carries N=8 inline, symbolic line uses per-rank N.
        for key in ("flops_formula", "access_bytes_formula"):
            f = row.get(key) or {}
            sym = f.get("symbolic") or ""
            sub = f.get("substituted") or ""
            self.assertNotIn("/ 16", sym,
                             f"{key}.symbolic must drop /16 after shard sub: {sym!r}")
            self.assertNotIn("/ 16", sub,
                             f"{key}.substituted must drop /16 after shard sub: {sub!r}")
            self.assertNotIn("/16", sub,
                             f"{key}.substituted has spaceless /16: {sub!r}")
            # Per-rank N (heads) must appear as 8 = 128/16 in the
            # values line.
            self.assertIn("8", sub)

    def test_dsv3_sglang_flash_attention_tp1_unchanged(self):
        # Identity mode (divisor=1) — no sharding happens, no `/1` slot
        # at all (the engine doesn't render trailing `/ 1`).
        result = _analyze("deepseek-ai__DeepSeek-V3", ParallelConfig(),
                          framework="sglang")
        row = self._flash_row(result)
        for key in ("flops_formula", "access_bytes_formula"):
            f = row.get(key) or {}
            sub = f.get("substituted") or ""
            # In identity mode N=128 (heads) stays full — pre-rank
            # equals full.
            self.assertNotIn("/ 1", sub)


class TestAllReduceShapePassthrough(unittest.TestCase):
    """An `all_reduce` injected after a row-parallel linear is
    shape-preserving: its sharded shape_in == shape_out == producer's
    sharded shape_out.
    """

    def test_dsv3_ar_after_o_proj(self):
        result = _analyze("deepseek-ai__DeepSeek-V3",
                          ParallelConfig(tp_attention=8))
        # Find any spec_id whose post-tree op is all_reduce and whose
        # producer is `*.o_proj`.
        ar_found = False
        for node in result.pre_program.nodes():
            if node.op != "all_reduce":
                continue
            if node.uid not in result.pre_shapes.values:
                continue
            prn = result.pre_shapes.values[node.uid]
            # shape-preserving: in == out (both lists)
            self.assertIsNotNone(prn.out)
            self.assertEqual(_input(prn), _axes(prn.out))
            ar_found = True
            break
        self.assertTrue(ar_found, "no all_reduce node in the physical shape layer")


class TestCpAttentionShardMatchesCommScheme(unittest.TestCase):
    """The declared CP plan owns both compute geometry and communication.

    PCP keeps query rows sequence-sharded and gathers K/V context. Ulysses
    moves the sequence cut onto heads. layersplit uses the same
    context-gather kernel geometry. Decode PCP still replicates the token
    stream, while DCP requires a valid context-parallel mesh.

    The decode row is `dcp` and NOT `cp`, which is the same distinction vLLM
    draws: PCP replicates the decode token stream and the whole cache over its
    group (`attention/pcp.py::_gather_prefill_cache_inputs`), so it divides no
    decode shape at all, while DCP re-cuts the TP ranks and splits exactly this
    axis. Two knobs, and reading one as the other divides a decode step by a
    degree nothing in it was split by.

    These are structural plan contracts rather than numeric golden exceptions.
    """

    def _score_in(self, cp, phase, scheme="ulysses", dcp=1):
        resolved = _resolved("deepseek-ai__DeepSeek-V3")
        if resolved is None:
            return None
        res = analyze(
            resolved_spec=resolved,
            parallel=ParallelConfig(cp=cp, cp_scheme=scheme, dcp=dcp),
            batch_size=1,
            phase=phase, avg_input_tokens=64, avg_output_tokens=0, breakdown=True)
        for node in res.program.nodes():
            shape = res.shapes.values.get(node.uid)
            inputs = [_axes(dims) for dims in shape.ins] if shape else []
            if (node.op == "attention_score" and not node.children
                    and len(inputs) >= 2):
                q = inputs[0]      # [B, N, Sq, D]
                k = inputs[1]      # [B, N, Lq, D]
                return {"n_heads": int(q[1]), "s_query": int(q[2]), "s_kv": int(k[2])}
        return None

    def test_prefill_pcp_keeps_query_rows_and_gathers_context(self):
        pcp = self._score_in(2, "prefill", "pcp")
        ulysses = self._score_in(2, "prefill", "ulysses")
        self.assertEqual(pcp["s_query"], 32)
        self.assertEqual(pcp["s_kv"], 64)
        self.assertEqual(ulysses["s_query"], 64)
        self.assertEqual(ulysses["s_kv"], 64)

    def test_prefill_layersplit_matches_pcp_kernel_shapes(self):
        pcp = self._score_in(2, "prefill", "pcp")
        split = self._score_in(2, "prefill", "layersplit")
        self.assertEqual(split, pcp)

    def test_cp_scheme_string_coerces_to_enum(self):
        from backend.arch.layout.parallel_config import CpScheme

        self.assertIs(
            ParallelConfig(cp_scheme="pcp").cp_scheme, CpScheme.PCP)
        self.assertEqual(str(CpScheme.PCP), "pcp")

    def test_unknown_cp_scheme_is_rejected(self):
        for scheme in ("ring", "kv_gather"):
            errs = ParallelConfig(cp_scheme=scheme).validate()
            self.assertTrue(errs, scheme)
            self.assertIn("cp_scheme", errs[0])

    def test_decode_rejects_dcp_without_a_context_parallel_mesh(self):
        with self.assertRaisesRegex(ValueError, "dcp = 2"):
            self._score_in(1, "decode", dcp=2)

    def test_decode_shards_nothing_by_prefill_cp(self):
        """The control for the test above, and the one that costs money if it
        is wrong: `cp` alone leaves the decode score leaf whole."""
        full = self._score_in(1, "decode")
        self.assertEqual(self._score_in(2, "decode"), full,
                         "decode: PCP replicates, so no axis may shard")


if __name__ == "__main__":
    unittest.main()
