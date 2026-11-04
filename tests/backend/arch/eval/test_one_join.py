"""Three facts, each from one declaration.

Weight bytes, leaf HBM access, tied aliases. Equality
against the join — not against a regex of the old spelling.
"""
from __future__ import annotations

import sympy as sp

from backend.analyzer import _build_layer_groups
from backend.arch.eval.program import Program
from backend.arch.eval.projection import display_formulas, project_model
from backend.arch.eval.traffic import derive
from backend.arch.eval.transforms import Deployment
from backend.rendering.builders import build_block_graph_data


def _analysis(resolved):
    return Deployment(Program.of(resolved)).at(
        batch=1, input_tokens=128).analyze()


def test_dashboard_weights_are_tensors_join(resolved):
    spec = resolved("meta-llama/Llama-3.2-1B")
    analysis = _analysis(spec)
    payload = display_formulas(spec, batch_size=1, seq_len=128)
    want = analysis.checkpoint_param_bytes()
    got = int(payload["memory"]["weights"]["result"].replace(",", ""))
    assert got == want
    # The formula's terms, each bound where it was walked, come to it.
    assert int(sp.sympify(payload["memory"]["weights"]["with_values"])) == want
    # Dense BF16: the original `params × 2` line is still the formula,
    # and with_values still shows the expanded params (not one integer).
    assert want == analysis.total().params * 2
    assert "*" in payload["memory"]["weights"]["symbolic"]
    plugged = payload["memory"]["weights"]["with_values"]
    assert "*" in plugged and "+" in plugged
    assert "d_model" not in plugged
    assert "2048" in plugged  # Llama-3.2-1B hidden size, still a factor


def test_mixed_precision_weights_are_not_params_times_two(resolved):
    spec = resolved("openai/gpt-oss-20b")
    analysis = _analysis(spec)
    payload = display_formulas(spec, batch_size=1, seq_len=128)
    want = analysis.checkpoint_param_bytes()
    got = int(payload["memory"]["weights"]["result"].replace(",", ""))
    assert got == want
    assert want != analysis.total().params * 2
    # Mixed: still a formula (sum of widths), not a bare total.
    # Packed widths live in the numeric join; the symbolic half still
    # prices some expert rows at storage-slot width. Binding the formula
    # therefore does not yet equal checkpoint_param_bytes — that is the
    # dtype-name cut, not this one.
    assert "*" in payload["memory"]["weights"]["symbolic"]


def test_layer_group_bytes_are_tensors_for_that_block(resolved):
    spec = resolved("openai/gpt-oss-20b")
    report = project_model(spec, batch=1, input_tokens=128)
    groups = _build_layer_groups(report, spec, 1, 128)
    blocks = [g for g in groups if g.get("layer_type") == "transformer_block"]
    assert blocks
    from collections import defaultdict
    per_bucket = defaultdict(list)
    for g in blocks:
        per_bucket[g["bucket_id"]].append(g)
    for i, (_tmpl, _count, br) in enumerate(report.block_results):
        cards = per_bucket[i]
        assert cards
        assert cards[0]["total_memory_bytes"] == br.total_storage_bytes


def test_dashboard_tie_badge_follows_spec_not_json(resolved):
    from backend.rendering.builders import build_model_config
    spec = resolved("google/gemma-2-9b")
    cfg = spec.config or {}
    assert not cfg.get("tie_word_embeddings")
    mc = build_model_config({}, spec)
    assert mc["tied_weight"] is True


def test_mtp_head_sharing_the_lm_head_does_not_tie_embeddings(resolved):
    # GLM-5's MTP head reads the LM head's weight (a shared weight), but
    # the LM head has its own table: the embeddings are not tied.
    spec = resolved("zai-org/GLM-5")
    program = Program.of(spec)
    assert program.read_aliases
    assert not (spec.config or {}).get("tie_word_embeddings")
    assert program.embeddings_tied is False


def test_gemma2_tied_head_reads_the_embedding(resolved):
    spec = resolved("google/gemma-2-9b")
    program = Program.of(spec)
    assert program.read_aliases, (
        "Gemma-2 omits tie_word_embeddings in JSON; spec ?? true must still tie")
    analysis = Deployment(program).at(batch=1, input_tokens=128).analyze()
    traffic = derive(analysis)
    head_uid, sources = next(iter(program.read_aliases.items()))
    assert traffic.values[head_uid].weight_reads > 0
    (embed_uid,) = sources
    embed_bytes = sum(
        int(row.storage_bytes) * int(row.count)
        for row in analysis.tensors()
        if row.uid == embed_uid and not row.is_metadata and not row.is_buffer)
    # Traffic is per copy; the alias bills one copy of the table.
    per_copy = embed_bytes / int(analysis.instances.values.get(embed_uid, 1))
    assert int(traffic.values[head_uid].weight_reads) == int(per_copy)


def test_leaf_tooltip_access_matches_traffic(resolved):
    spec = resolved("meta-llama/Llama-3.2-1B")
    analysis = _analysis(spec)
    traffic = derive(analysis)
    graph, _unmatched = build_block_graph_data(spec, batch_size=1, seq_len=128)
    checked = 0

    def walk(node):
        nonlocal checked
        for child in node.get("children") or []:
            walk(child)
        uid = node.get("uid")
        if uid is None or node.get("children"):
            return
        row = traffic.values.get(uid)
        assert row is not None, node.get("id")
        assert node["access_bytes"] == int(row.total or 0)
        checked += 1

    for block in graph.values():
        for node in block.get("nodes") or []:
            walk(node)
    assert checked > 0
