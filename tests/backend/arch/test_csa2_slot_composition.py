"""CSA2's four per-layer modes are one attention body with its slots bound.

DeepSeek-V4.1-Flash's backbone runs four shapes of the same attention
(`Attention.forward`, inference/model.py:765-789): a layer that pools its own
KV and indexes it, the one that also publishes the candidate blocks, one that
only re-indexes the shared keys, and one that reuses both. What differs
between them is `Attention._compress_kv` — which compressor it runs, which
indexer, and whether it keeps the compressed buffer — so the spec says the
attention once, as `library/csa2_attention` with a `compressor` / `indexer` /
`compressed` param, and each layer group binds the form it runs.

What this pins is that the sharing is real and that it did not move any
checkpoint name: every CSA2 layer invokes the SAME attention template, the
per-layer difference is exactly the set of components its slots bind, and the
compressor / indexer tensors keep the module paths (and the per-layer counts)
the private V4.1 checkpoint has.
"""
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from backend.arch.eval.program import Program
from backend.arch.eval.transforms import PREFILL, Deployment
from tests.backend.arch.spec_tree.test_inner_stream_reads import _v41

# Which layers run which form of `_compress_kv`, from the config lists the
# spec's `per_layer.assign` reads: `kv_source_layer_ids` [2, 8, 14, 20],
# `index_source_layer_ids` + those = {24, 28, 32, 36}, `compress_ratios`
# 0 on layers 0-1 and the ratio split at 20.
KV_INDEX_SOURCE = (2, 8, 14)
CANDIDATE_SOURCE = (20,)
INDEX_SOURCE_ONLY = (24, 28, 32, 36)
SWA = (0, 1)


def _attention_calls(program):
    """`{layer: (component name, {local path: component name})}` for every
    backbone layer's attention call and the components under it."""
    out = {}
    for template in program.blocks:
        if template.meta.aux_kind is not None or template.meta.stack_name != "layers":
            continue
        for root in template.roots:
            pending = [(root, "")]
            while pending:
                node, path = pending.pop()
                if node.role == "attention" or (
                        node.is_component and node.name == "attn"):
                    inner = {}
                    sub = [(c, c.name) for c in node.children or ()]
                    while sub:
                        child, cpath = sub.pop()
                        if child.is_component:
                            inner[cpath] = child.op
                            sub.extend((g, f"{cpath}.{g.name}")
                                       for g in child.children or ())
                    for layer in template.layer_indices:
                        out[layer] = (node.op, inner)
                    continue
                pending.extend((c, f"{path}{c.name}.")
                               for c in node.children or ())
    return out


def test_every_csa2_layer_runs_one_attention_body():
    """One template name for all 38 CSA2 layers (`csa2_attention` with its
    `compress` binding appended), and the two SWA layers run the DSv4 body."""
    resolved, _text = _v41()
    calls = _attention_calls(Program.of(resolved))
    assert set(calls) == set(range(40))
    bodies = {}
    for layer, (name, _inner) in calls.items():
        bodies.setdefault(name.split("__")[0], set()).add(layer)
    assert bodies == {
        "csa2_attention": set(range(40)) - set(SWA),
        "dsv4_swa_only_attention": set(SWA),
    }


def test_the_per_layer_difference_is_which_components_the_slots_bind():
    """Each CSA2 layer group binds its own `_compress_kv` form, and nothing
    else about the attention changes."""
    resolved, _text = _v41()
    calls = _attention_calls(Program.of(resolved))
    bound = {}
    for layer, (_name, inner) in calls.items():
        if layer in SWA:
            continue
        # The shape utilities (`head_split` / `head_merge` /
        # `head_broadcast`) are the same in every form; what varies is the
        # csa2 components the slots name.
        bound.setdefault(
            tuple(sorted(op.split("__")[0] for op in inner.values()
                         if op.startswith("csa2_"))),
            set()).add(layer)
    assert bound == {
        ("csa2_compress_store", "csa2_compressor_gated", "csa2_index_keys",
         "csa2_index_score", "csa2_indexer"): set(KV_INDEX_SOURCE),
        ("csa2_compress_store", "csa2_compressor_plain", "csa2_index_keys",
         "csa2_index_score", "csa2_indexer_candidate_source"):
            set(CANDIDATE_SOURCE),
        ("csa2_compress_borrowed", "csa2_index_score",
         "csa2_indexer_shared_keys", "csa2_shared_latent"):
            set(INDEX_SOURCE_ONLY),
        ("csa2_compress_borrowed", "csa2_shared_latent",
         "csa2_shared_selection"):
            set(range(40)) - set(SWA) - set(KV_INDEX_SOURCE)
            - set(CANDIDATE_SOURCE) - set(INDEX_SOURCE_ONLY),
    }


def test_the_compressor_and_indexer_keep_their_checkpoint_paths():
    """`pattern_counts` of the private V4.1 record (spec header, snapshot
    2333-2339): compressor.wkv x4, wgate x3 (the ratio-1 kv source has no
    gate), norm x4; indexer.wq_b x8, weights_proj x8, wk x4, k_norm x4. The
    nesting the slots introduce contributes no path segment of its own, so
    the names and the counts are the checkpoint's."""
    resolved, _text = _v41()
    analysis = Deployment(Program.of(resolved)).at(
        batch=1, input_tokens=8).during(PREFILL).analyze()
    counts = {}
    for uid, stored in analysis.storage.values.items():
        copies = int(analysis.instances.values.get(uid, 0))
        for tensor in stored:
            if ".compressor." in tensor.name or ".indexer." in tensor.name:
                counts[tensor.name] = counts.get(tensor.name, 0) + copies
    assert counts == {
        "layers.X.attn.compressor.wkv.weight": 4,
        "layers.X.attn.compressor.wgate.weight": 3,
        "layers.X.attn.compressor.norm.weight": 4,
        "layers.X.attn.indexer.wq_b.weight": 8,
        "layers.X.attn.indexer.weights_proj.weight": 8,
        "layers.X.attn.indexer.wk.weight": 4,
        "layers.X.attn.indexer.k_norm.weight": 4,
    }


def test_the_swa_layers_run_no_per_head_query_norm():
    """V4.1's `Attention.forward` goes straight from `wq_b` to
    `apply_rotary_emb` (source/model.py:767-768), so the fixture binds
    `n_q_head_norm: "0"` and `dsv4_swa_only_attention`'s `q_head_norm` step
    is not built."""
    resolved, _text = _v41()
    program = Program.of(resolved)
    assert program.resolved.dim_values["n_q_head_norm"] == 0
    assert not [n for n in program.nodes() if n.name == "q_head_norm"]
    assert [n for n in program.nodes() if n.op == "dsv4_swa_only_attention"]
