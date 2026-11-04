"""Every MTP stack ends in a vocabulary projection.

An MTP layer is a draft model: each step it predicts a token, so each step
it projects its hidden state onto the vocabulary. vLLM's MTP models all do
so in `compute_logits` — with a private head where the checkpoint ships one
(DeepSeek-V3 / GLM-4.5 `shared_head.head`, Step `shared_head.output`) and
with the target's lm_head otherwise (`llm_base_proposer._maybe_share_lm_head`).
A stack that stops at its final norm leaves the largest matmul of the step
out of the draft's cost. Where the head is the target's, the stack reads the
target's weight rather than storing a second copy."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from backend.arch.spec_tree import iter_tree
from tests.snapshots import SNAPSHOT_DIR, resolve_snapshot

pytestmark = pytest.mark.regression  # walks every snapshot


def _walk(nodes):
    for node in nodes:
        yield node
        yield from _walk(node.children or ())


def _mtp_heads(resolved):
    """(has an MTP stack, [(head node, owns its weight)])."""
    tree = iter_tree(resolved)
    vocab = resolved.dim_values["n_vocab"]
    groups = [layer for layer in resolved.spec.model
              if getattr(layer, "aux_kind", None) == "mtp"]
    rows = {sub.name for group in groups for sub in group.layers or ()}
    heads = [node for node in _walk(tree.top_level)
             if node.name in rows and node.op == "linear"
             and node.dims.get("out") == vocab]
    for meta, roots in tree.blocks:
        if meta.aux_kind == "mtp":
            heads += [node for node in _walk(roots)
                      if node.op == "linear" and node.dims.get("out") == vocab]
    return bool(groups), heads


def test_every_mtp_stack_projects_onto_the_vocabulary():
    checked = 0
    for path in sorted(SNAPSHOT_DIR.glob("*.json")):
        model_id = path.stem.replace("__", "/", 1)
        resolved = resolve_snapshot(model_id)
        has_mtp, heads = _mtp_heads(resolved)
        if not has_mtp:
            continue
        checked += 1
        assert heads, f"{model_id}: MTP stack has no logits projection"
    assert checked, "no snapshot carries an MTP stack"


def test_a_shared_head_reads_the_target_weight():
    # MiMo-7B's checkpoint has no MTP head; vLLM mimo_mtp.py projects with
    # the main lm_head (compute_logits).
    _has, heads = _mtp_heads(resolve_snapshot("XiaomiMiMo/MiMo-7B-RL"))
    (head,) = heads
    assert head.tied_to == "lm_head"
    config = json.loads(
        (SNAPSHOT_DIR / "XiaomiMiMo__MiMo-7B-RL.json").read_text())["config"]
    assert config["tie_word_embeddings"] is False


def test_the_model_has_one_lm_head_and_the_draft_head_is_its_stacks():
    # An MTP head is an output head on its own stack. The model's LM head is
    # the backbone's one: the analyzer draws one "LM Head" group and the
    # LM-head FLOPs are that head's, not the draft's added in.
    from backend.arch.eval.program import Program
    from backend.arch.eval.projection import project_model

    resolved = resolve_snapshot("XiaomiMiMo/MiMo-7B-RL")
    program = Program.of(resolved)
    heads = [n for n in _walk(program.ir.top_level)
             if n.boundary_role == "output_head"]
    assert sorted(n.name for n in heads) == ["lm_head", "mtp_logits"]
    assert [n.name for n in heads if program.is_lm_head(n)] == ["lm_head"]

    model = project_model(resolved, batch=1, input_tokens=16)
    lm = [name for name, node in model.model_level_nodes.items()
          if node.is_lm_head]
    assert lm == ["lm_head"]
    assert model.lm_head_flops == model.model_level_nodes["lm_head"].flops


def test_a_conditionally_tied_head_chain_ends_at_the_embedding():
    # Qwen3.5's lm_head is tied to the embedding when tie_word_embeddings is
    # set, and its MTP head is tied to the lm_head. With the tie on, both
    # heads read the embedding table: one owner, no second copy anywhere.
    from tests.snapshots import shared_loader
    from tests.snapshots import snapshot_config

    model_id = "Qwen/Qwen3.5-35B-A3B"
    config = snapshot_config(model_id)
    for tied in (False, True):
        cfg = json.loads(json.dumps(config))
        (cfg.get("text_config") or cfg)["tie_word_embeddings"] = tied
        cfg["tie_word_embeddings"] = tied
        resolved = shared_loader().match(cfg, model_id=model_id)
        tree = iter_tree(resolved)
        by_name = {n.name: n for n in _walk(tree.top_level)}
        embed = by_name["model.language_model.embed_tokens"]
        lm_head = by_name["lm_head"]
        mtp = by_name["mtp_logits"]
        if tied:
            assert lm_head.tied_to_uid == embed.uid
            assert mtp.tied_to_uid == embed.uid
        else:
            assert lm_head.tied_to_uid is None
            assert mtp.tied_to_uid == lm_head.uid


def test_deepseek_v4_draft_mixes_every_position_then_samples():
    # The reference MTP head mixes the copies and norms every position, then
    # takes the logits at the last one: `ParallelHead.forward` runs `hc_head`
    # and `norm` on [b,s,hc,d] and `get_logits` reads `x[:, -1]`
    # (DeepSeek-V4 inference/model.py:716-721), the backbone's order too.
    from backend.arch.eval.program import Program
    from backend.arch.eval.transforms import Deployment

    resolved = resolve_snapshot("deepseek-ai/DeepSeek-V4-Flash")
    analysis = Deployment(Program.of(resolved)).at(
        batch=1, input_tokens=64, logit_positions=1).analyze()
    program = analysis.program
    positions, sampled = {}, {}
    for node in program.nodes():
        stack = program.stack_of(node) or "backbone"
        if node.op == "hc_head":
            positions[stack] = int(analysis.shapes[node].out[1])
        if node.op == "sample":
            sampled[stack] = int(analysis.shapes[node].out[1])
    assert positions == {"backbone": 64, "mtp": 64}
    assert sampled.get("mtp") == 1


def test_flash_next_draft_projects_each_copy_and_adds_the_embedding_to_all():
    # Qwen3.8-Flash-Next's HF file drops the MTP head; vLLM's Qwen4Exp MTP
    # (vllm/models/qwen4_exp/nvidia/mtp.py:293-306) normalises the widened
    # backbone stream as one hc*d vector, projects each copy with the one
    # d->d fc_hidden, and adds fc_embedding's single lane to every copy
    # (sglang qwen4_exp_mtp.py:116) before the draft layer.
    from backend.arch.eval.program import Program
    from backend.arch.eval.transforms import Deployment

    resolved = resolve_snapshot("Qwen/Qwen3.8-Flash-Next")
    analysis = Deployment(Program.of(resolved)).at(
        batch=1, input_tokens=16).analyze()
    nodes = {n.name: n for n in analysis.program.nodes()}
    out = {name: tuple(int(x) for x in analysis.shapes[nodes[name]].out)
           for name in ("mtp.pre_fc_norm_hidden", "mtp.fc_hidden",
                        "mtp.fc_embedding", "mtp.fuse")}
    assert out == {
        "mtp.pre_fc_norm_hidden": (1, 16, 4 * 2560),
        "mtp.fc_hidden": (1, 16, 4, 2560),
        "mtp.fc_embedding": (1, 16, 2560),
        "mtp.fuse": (1, 16, 4, 2560),
    }
