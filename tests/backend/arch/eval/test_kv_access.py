"""`cache.kv_access`: which node writes and which reads each cached tensor.

Every shipped snapshot, both phases, no allow-list. The graph marks drawn on
every compute graph are projections of this layer, so a hole here is a
missing cache mark there.
"""
from __future__ import annotations

import sys
from functools import lru_cache
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT))

from backend.arch.eval import cache as cache_mod
from backend.arch.eval import traffic as traffic_mod
from backend.arch.eval.dim import value_of
from backend.arch.eval.program import Program
from backend.arch.eval.transforms import Deployment, analyze_isolated
from backend.arch.ops_base import OP_REGISTRY
from backend.arch.snapshots import iter_snapshot_ids
from tests.snapshots import resolve_snapshot, shared_loader

SNAPSHOTS = sorted(iter_snapshot_ids())
PHASES = ("prefill", "decode")


@lru_cache(maxsize=None)
def _analysis(model_id: str, phase: str):
    resolved = resolve_snapshot(model_id, shared_loader())
    return (Deployment(Program.of(resolved))
            .at(batch=1, input_tokens=64).during(phase).analyze())


def _holders(analysis):
    """(component node, Cache) for every PERSISTING cache — the scope the
    marks cover."""
    return [(holder, entry)
            for holder, entry in cache_mod._holders(analysis, None)
            if entry.persists]


def _leaves(node):
    stack = list(node.children)
    while stack:
        current = stack.pop()
        if current.children:
            stack.extend(current.children)
        else:
            yield current


def _declares(leaf, field: str):
    spec = OP_REGISTRY.get(leaf.op or "")
    return spec is not None and bool(getattr(spec, field))


def _readers(access):
    return {uid for uid, value in access.values.items() if value.reads}


def _billed(analysis):
    """Leaves `traffic` bills a cache read to, outside a cache that does not
    persist between steps — such a cache is not marked."""
    transient = {leaf.uid
                 for holder, entry in cache_mod._holders(analysis, None)
                 if not entry.persists for leaf in _leaves(holder)}
    return {uid for uid, value in traffic_mod.derive(analysis).values.items()
            if value.kv_reads and uid not in transient
            and not analysis.program.by_uid(uid).children}


@pytest.mark.regression
@pytest.mark.parametrize("model_id", SNAPSHOTS)
def test_authoritative_context_spans_have_no_uncovered_excess(model_id):
    """Every K/V reader's declared span is fresh rows plus covered cache rows.

    The scan is marked regression because it walks all 62 snapshots; the
    DSpark focused cases are default-run in `test_step_kv.py`.
    """
    resolved = resolve_snapshot(model_id, shared_loader())
    prefill = Deployment(Program.of(resolved))
    cases = (
        prefill.at(batch=1, input_tokens=64,
                   s_prefix_cached=32).during("prefill").analyze(),
        prefill.at(batch=1, input_tokens=64,
                   s_prefix_cached=0).during("prefill").analyze(),
        prefill.at(batch=1, input_tokens=32).during(
            "decode", cache_len=32).analyze(),
    )
    for analysis in cases:
        assert cache_mod.context_span_gaps(analysis) == ()


def _from_cache(leaf, access, inside, program):
    """Whether `leaf` reads the cache, or reads a tensor computed from a
    cache read inside the same component. The second is a kernel scoring
    keys recomputed from the cache every step — an indexer that re-pools
    cached raw keys before it scores them, as HF's QSA forward does
    (modeling_qwen4_exp.py Qwen4ExpTextQSAIndexer.forward): the pool is the
    cache read, and the score reads what the pool made of it. (The specs
    cache the pooled keys, as serving does, so their indexers read the cache
    directly.) Another attention kernel's output
    is not a cache read of this one: a sparse attention reading the
    indexer's selection still reads its own keys from the cache, so the
    walk does not pass through an upstream kernel."""
    seen, stack = set(), [leaf]
    while stack:
        node = stack.pop()
        if node.children:  # a ref may name a wrapper; it IS its output leaf
            node, _port = program.output_leaf(node)
        if node.uid in seen or node.uid not in inside:
            continue
        seen.add(node.uid)
        if node is not leaf and _declares(node, "attention_kernel"):
            continue
        value = access.get(node)
        if value is not None and value.reads:
            return True
        stack.extend(ref.src for ref in (node.inputs_obj or ())
                     if ref.src is not None)
    return False


@pytest.mark.parametrize("model_id", SNAPSHOTS)
def test_every_attention_kernel_reads_the_cache_at_decode(model_id):
    """G2. At decode every attention kernel in a persisting cache component
    reads it — directly, or through what the component computes from its
    cache read — and every recurrent kernel reads and writes its state."""
    analysis = _analysis(model_id, "decode")
    access = cache_mod.kv_access(analysis)
    missing = []
    for holder, _entry in _holders(analysis):
        inside = {leaf.uid for leaf in _leaves(holder)}
        for leaf in _leaves(holder):
            value = access.get(leaf)
            if _declares(leaf, "attention_kernel") and not _from_cache(
                    leaf, access, inside, analysis.program):
                missing.append(leaf.display_path)
            if _declares(leaf, "carries_recurrent_state") and not (
                    value is not None and value.reads and value.writes):
                missing.append(leaf.display_path)
    assert not missing, missing


@pytest.mark.parametrize("model_id", SNAPSHOTS)
@pytest.mark.parametrize("phase", PHASES)
def test_every_cached_tensor_has_a_writer_and_a_reader(model_id, phase):
    """G3. A persisting cached tensor nobody reads, or nobody writes, is a
    hole in the marks."""
    analysis = _analysis(model_id, phase)
    access = cache_mod.kv_access(analysis)
    written = {ref for value in access.values.values() for ref in value.writes}
    read = {ref for value in access.values.values() for ref in value.reads}
    kept = set(cache_mod.resident(analysis))
    assert kept - written == set()
    assert kept - read == set()
    assert (written | read) <= kept


@pytest.mark.parametrize("model_id", SNAPSHOTS)
@pytest.mark.parametrize("phase", PHASES)
def test_traffic_cache_readers_are_structural_readers(model_id, phase):
    """G5. `traffic` bills `kv_reads` only on nodes `kv_access` marks as
    readers — it clips by span on top of the same pedigree. A fresh prefill
    attends what it just wrote, so nothing there reads the cache as such —
    except a kernel that gathers a per-query index list
    (`Analysis.gathered`): its rows land in the cache and each query loads
    its own selection back from it."""
    analysis = _analysis(model_id, phase)
    readers = _readers(cache_mod.kv_access(analysis))
    billed = _billed(analysis)
    assert billed <= readers
    if phase == "prefill":
        recurrent = {tensor.node for _holder, entry in _holders(analysis)
                     for tensor in entry.tensors if tensor.recurrent}
        assert billed <= recurrent | analysis.gathered


def test_prefix_cached_prefill_reads_the_prefix_through_marked_readers():
    """G5, chunked form: a prefill behind a 2048-token prefix reads that
    prefix from the cache, on the same readers the structure names."""
    resolved = resolve_snapshot("Qwen/Qwen3-8B", shared_loader())
    analysis = (Deployment(Program.of(resolved))
                .at(batch=1, input_tokens=4096, s_prefix_cached=2048)
                .during("prefill").analyze())
    readers = _readers(cache_mod.kv_access(analysis))
    billed = _billed(analysis)
    assert billed
    assert billed <= readers
    ops = {analysis.program.by_uid(uid).op for uid in billed}
    assert ops == {"attention_score", "attention_apply"}


def test_mla_prefix_prefill_decompresses_the_cached_prefix():
    """G5 for a cache stored in a different form than attention reads.
    Behind a cached prefix a decompressed-prefill MLA layer attends `s_kv`
    rows while it wrote `s_query`: the prefix rows are read from the latent
    cache and re-derived (`kv_b_proj`, the rope half's broadcast), as vLLM's
    `_compute_prefill_context` and sglang's prefix-extend path do. The
    re-derivation then runs over `s_kv` rows, and the cache read is the
    whole latent row of every attended position in every layer. With no
    prefix nothing is read: the step's own rows are the attended ones."""
    resolved = resolve_snapshot("deepseek-ai/DeepSeek-V2-Lite", shared_loader())
    for prefix in (0, 2048):
        analysis = (Deployment(Program.of(resolved))
                    .at(batch=1, input_tokens=4096, s_prefix_cached=prefix)
                    .during("prefill").analyze())
        binding = analysis.binding
        s_query, s_kv = (int(value_of(binding[k])) for k in ("s_query", "s_kv"))
        edges = cache_mod.Edges(analysis)
        rederived = [
            node for node in analysis.program.nodes()
            if not node.children and node.op != "kv_cache_view"
            and any(source.op == "kv_cache_view"
                    for index in range(len(node.inputs_obj or ()))
                    for source in edges.sources(node, index))]
        assert rederived
        for node in rederived:
            rows = int(value_of(analysis.shapes.values[node.uid].ins[0][-2]))
            assert rows == s_kv, (node.display_path, rows, s_kv, s_query)
        traffic = traffic_mod.derive(analysis)
        read = sum(value_of(value.kv_reads)
                   * value_of(analysis.instances.values.get(uid, 1))
                   for uid, value in traffic.values.items()
                   if value.kv_reads
                   and not analysis.program.by_uid(uid).children)
        layers, latent, rope = (int(value_of(binding[k]))
                                for k in ("n_layers", "d_kv_lora", "d_qk_rope"))
        expected = 0 if prefix == 0 else layers * s_kv * (latent + rope) * 2
        assert read == expected, (prefix, read, expected)


def _declares_a_cache(program, node):
    if not node.is_component:
        return False
    cache = program.spec.get_block(node.op).cache
    return cache is not None and bool(cache.entries)


def test_an_encoder_cache_is_not_marked():
    """G7. A tower runs once per media item, in the encode phase, and keeps
    no cache between steps — whatever its attention declares. Step-3.7-Flash's
    vision attention is the corpus's encoder that declares a cache, so it must
    be among the models checked; MiMo-V2.5's audio attention is block-local
    and declares none. The language model's decode marks name no encoder
    component, and the encoder's own encode walk marks nothing."""
    checked = []
    for model_id in SNAPSHOTS:
        analysis = _analysis(model_id, "decode")
        program = analysis.program
        encoders = program.encoder_uids
        if not encoders:
            continue
        refs = {ref for value in cache_mod.kv_access(analysis).values.values()
                for ref in value.writes | value.reads}
        assert not {ref for ref in refs if ref.component in encoders}, model_id
        if any(_declares_a_cache(program, program.by_uid(uid))
               for uid in encoders):
            checked.append(model_id)
    assert "stepfun-ai/Step-3.7-Flash" in checked, checked

    program = _analysis("stepfun-ai/Step-3.7-Flash", "decode").program
    encode = (Deployment(program).at(batch=1,
                                     media={"pixel_values": {"n_img": 1},
                                            "patch_pixel_values": {"n_crops": 0}})
              .during("encode", stack="vision").analyze())
    assert {uid for uid in program.encoder_uids if uid in encode.cost.values}
    assert not [value for value in cache_mod.kv_access(encode).values.values()
                if value.reads or value.writes]


@pytest.mark.parametrize("model_id", SNAPSHOTS)
def test_mla_prefill_attention_reads_the_decompressed_keys(model_id):
    """G10. In prefill an MLA layer with an absorbed decode form attends
    over the up-projected K, not over the cache: the cache's readers are the
    nodes that decompress it (`kv_b_proj`, the positional half's concat, the
    sparse indexer), never the attention kernels."""
    analysis = _analysis(model_id, "prefill")
    access = cache_mod.kv_access(analysis)
    wrong = []
    for holder, entry in _holders(analysis):
        graph = analysis.program.spec.get_block(holder.op)
        if not (graph.decode_variant and graph.kv_label == "MLA"):
            continue
        # The sparse indexer's own cache (`kind: index_key`) is scored
        # directly by the indexer's kernel — GLM-5.3-Flash's pooled keys, as
        # DSA's `indexer_logits` reads its keys — and is not the MLA cache.
        indexer = {t.name for t in entry.tensors if t.kind == "index_key"}
        for leaf in _leaves(holder):
            value = access.get(leaf)
            if (_declares(leaf, "attention_kernel")
                    and value is not None
                    and {r.tensor for r in value.reads} - indexer):
                wrong.append(leaf.display_path)
    assert not wrong, wrong


def test_the_mla_prefill_check_covers_the_mla_corpus():
    """G10 is not vacuous: the corpus has MLA layers with an absorbed decode
    form, and the check above visits their attention kernels."""
    kernels = 0
    for model_id in SNAPSHOTS:
        analysis = _analysis(model_id, "prefill")
        for holder, _entry in _holders(analysis):
            graph = analysis.program.spec.get_block(holder.op)
            if graph.decode_variant and graph.kv_label == "MLA":
                kernels += sum(1 for leaf in _leaves(holder)
                               if _declares(leaf, "attention_kernel"))
    assert kernels >= 20


def test_the_marked_readers_of_known_layers():
    """The readers the design worked through by hand."""
    def readers(model_id, phase):
        analysis = _analysis(model_id, phase)
        access = cache_mod.kv_access(analysis)
        return {(analysis.program.by_uid(uid).op,
                 tuple(sorted(ref.tensor for ref in value.reads)))
                for uid, value in access.values.items() if value.reads}

    assert readers("Qwen/Qwen3-8B", "decode") == {
        ("attention_score", ("k_rope",)),
        ("attention_apply", ("v_split_heads",)),
    }
    assert readers("deepseek-ai/DeepSeek-V3", "decode") == {
        ("attention_score", ("k_pe_rope", "kv_a_layernorm")),
        ("attention_apply", ("kv_a_layernorm",)),
    }
    assert readers("deepseek-ai/DeepSeek-V3", "prefill") == {
        ("linear", ("kv_a_layernorm",)),
        ("concat", ("k_pe_rope",)),
    }
    # DeepSeek-V4 reads each compressor through the wrapper's name while the
    # cache is declared on its `pooled_out` leaf, and each fp32 block state
    # is read by the kernel that carries it.
    assert readers("deepseek-ai/DeepSeek-V4-Pro", "decode") == {
        ("attention_score", ("compressor.pooled_out", "kv_concat")),
        ("attention_apply", ("compressor.pooled_out", "kv_concat")),
        ("attention_score", ("indexer.compressor.pooled_out",)),
        ("mul", ("compressor.kv_state",)),
        ("mul", ("indexer.compressor.kv_state",)),
        ("softmax_act", ("compressor.score_state",)),
        ("softmax_act", ("indexer.compressor.score_state",)),
        ("swa_attention_score", ("kv_concat",)),
        ("swa_attention_apply", ("kv_concat",)),
    }
    # GLM-5.3-Flash decodes its MLA absorbed: the sparse score and apply
    # read the latent itself. Its k-pool indexer caches one pooled key per
    # pool, as vLLM and sglang do, and the indexer score reads it; the
    # pending pool's K and gate rings are read by the kernels that pool them
    # (as DeepSeek-V4's compressor states above).
    assert readers("zai-org/GLM-5.3-Flash", "decode") == {
        ("dsa_attention_score", ("kv_a_layernorm",)),
        ("dsa_attention_apply", ("kv_a_layernorm",)),
        ("attention_score", ("index_kpool.pooled_out",)),
        ("mul", ("index_kpool.key_state",)),
        ("softmax_act", ("index_kpool.gate_state",)),
        ("gated_delta_rule", ("recurrent_state",)),
        ("conv1d", ("q_conv_state",)),
        ("conv1d", ("k_conv_state",)),
        ("conv1d", ("v_conv_state",)),
    }


def test_a_mixed_stream_union_publishes_and_the_attention_reads_it():
    """A positional publisher is not the cache reader; the kernels that span
    its authoritative rows are."""
    resolved = resolve_snapshot(
        "deepseek-ai/DeepSeek-V4.1-Flash", shared_loader())
    analysis = (Deployment(Program.of(resolved))
                .at(batch=1, input_tokens=64, s_prefix_cached=32)
                .during("prefill").analyze())
    access = cache_mod.kv_access(analysis)
    nodes = [node for node in analysis.program.nodes()
             if "dsv41_dspark_attention" in node.display_path
             and node.op in ("kv_stream_concat", "attention_score",
                             "attention_apply")]
    assert {node.op for node in nodes} == {
        "kv_stream_concat", "attention_score", "attention_apply"}
    for node in nodes:
        value = access[node.uid]
        if node.op == "kv_stream_concat":
            assert value.publishes and not value.reads
        else:
            assert not value.publishes
            assert {ref.tensor for ref in value.reads} == {
                "main_kv_concat"}
            assert value.key_slots == frozenset(("kv",))


def test_isolation_keeps_the_recurrent_states():
    """An isolated linear-attention component holds the same states the
    full-model walk bills — the Playground's graph is drawn on isolation —
    and each is written and read by its own carrier: the scan's matrix by
    the delta rule, the conv window by the conv."""
    resolved = resolve_snapshot("Qwen/Qwen3-Coder-Next", shared_loader())
    name = "gated_delta_net_fused_attention"
    analysis = analyze_isolated(resolved, name)
    isolated = cache_mod.of_component(analysis, name)
    assert [(t.name, t.recurrent) for t in isolated] == [
        ("recurrent_state", True), ("conv_state", True)]
    full = _analysis("Qwen/Qwen3-Coder-Next", "decode")
    states = {(t.name, int(t.elements))
              for entry in cache_mod.derive(full).values.values()
              for t in entry.tensors if t.recurrent}
    assert states == {(t.name, int(t.elements)) for t in isolated}
    access = cache_mod.kv_access(analysis)
    carried = {analysis.program.by_uid(uid).op:
               sorted(ref.tensor for ref in value.writes)
               for uid, value in access.values.items()
               if value.writes and value.writes == value.reads}
    assert carried == {"gated_delta_rule": ["recurrent_state"],
                       "conv1d": ["conv_state"]}
