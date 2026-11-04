"""Parameter-free steps of the forward that no weight or shape check sees.

Each case is a gate, scoring function, normalization axis or rotation the
real forward applies and whose absence (or wrong choice) leaves every tensor
and every byte count unchanged. The source for each is cited at the test.
"""
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from backend.arch.spec_tree import iter_tree
from tests.snapshots import shared_loader, snapshot_config


def _tree(model_id, config=None):
    config = snapshot_config(model_id) if config is None else config
    return iter_tree(shared_loader().match(config, model_id=model_id))


def _walk(nodes, prefix=""):
    for node in nodes:
        path = prefix + node.name
        yield path, node
        yield from _walk(node.children or (), path + "/")


def _block_nodes(model_id, template_part):
    """`path -> node` for the first block whose template name contains
    `template_part`."""
    for meta, roots in _tree(model_id).blocks:
        if template_part in meta.template_name:
            return dict(_walk(roots))
    raise AssertionError(f"{model_id}: no block template like {template_part!r}")


def _ending(nodes, suffix):
    (hit,) = [node for path, node in nodes.items() if path.endswith(suffix)]
    return hit


def _reads(node):
    return [ref.src.name for ref in node.inputs_obj]


# Qwen3.5 / Qwen3-Next full attention: `attn_output * torch.sigmoid(gate)`
# (modeling_qwen3_5_moe.py Qwen3_5MoeAttention.forward, modeling_qwen3_next.py
# Qwen3NextAttention.forward; vLLM qwen3_next.py). Gated DeltaNet closes with
# RMSNormGated: norm first, then × silu(z) (Qwen3_5MoeRMSNormGated,
# Qwen3NextRMSNormGated; vLLM RMSNormGated(norm_before_gate=True)).

QWEN_HYBRIDS = (("Qwen/Qwen3.5-35B-A3B", "qwen35_full_moe_block",
                 "qwen35_linear_moe_block"),
                ("Qwen/Qwen3-Coder-Next", "sigmoid_gated_attention",
                 "qwen3_next_linear_moe_block"))


def test_qwen_hybrid_attention_output_gate_is_sigmoid():
    for model_id, full, _linear in QWEN_HYBRIDS:
        nodes = _block_nodes(model_id, full)
        gate = _ending(nodes, "self_attn/gate_act")
        assert gate.op == "sigmoid", model_id
        assert "gate_act" in _reads(_ending(nodes, "self_attn/gated_attn"))


def test_gated_deltanet_norm_is_gated_by_silu_of_z():
    for model_id, _full, linear in QWEN_HYBRIDS:
        nodes = _block_nodes(model_id, linear)
        assert _ending(nodes, "z_act").op == "silu"
        assert _reads(_ending(nodes, "norm_gated")) == ["norm", "z_act"]


# Qwen3.5 / Qwen3-Next sparse MoE: softmax → top-k → divide by the sum
# (Qwen3_5MoeTopKRouter / Qwen3NextTopKRouter; vLLM renormalize=
# getattr(config, "norm_topk_prob", True)), and the shared expert is scaled by
# sigmoid(shared_expert_gate(x)).

def test_qwen_hybrid_moe_scores_selects_and_renormalizes():
    for model_id, full, _linear in QWEN_HYBRIDS:
        nodes = _block_nodes(model_id, full)
        router = _ending(nodes, "mlp/gate")
        assert router.output_names == ("topk_norm",), model_id
        assert _ending(nodes, "mlp/gate/softmax_scores").op == "softmax_act"
        assert _reads(_ending(nodes, "mlp/gate/topk_norm")) == ["topk_gather"]
        gate_act = _ending(nodes, "mlp/shared_expert_gate_act")
        assert gate_act.op == "sigmoid"
        assert _reads(_ending(nodes, "mlp/shared_expert_scaled")) == [
            "shared_expert", "shared_expert_gate_act"]


# MiniMax-M2: the correction bias only chooses; the weights are the
# un-biased sigmoid scores at the chosen experts, renormalized
# (MiniMaxM2TopKRouter.forward; vLLM minimax_m2.py renormalize=True).

def test_minimax_m2_weights_are_unbiased_renormalized_top_k():
    nodes = _block_nodes("MiniMaxAI/MiniMax-M2.7", "minimax_m2")
    assert _reads(_ending(nodes, "moe_router_topk_indices")) == [
        "moe_router_scores_biased"]
    assert _reads(_ending(nodes, "moe_router_topk_gather")) == [
        "moe_router_score", "moe_router_topk_indices"]
    assert _reads(nodes["combine"])[1] == "moe_router_topk_norm"


# DeepSeek-V4 hash layers replace only where the indices come from; the
# weights are still gathered, divided by their sum and scaled by route_scale
# (official inference/model.py Gate.forward).

def test_dsv4_hash_layers_normalize_and_scale_the_weights():
    nodes = _block_nodes("deepseek-ai/DeepSeek-V4-Flash", "dsv4_moe_hash")
    assert _reads(_ending(nodes, "ffn/topk_gather")) == ["scores", "hash_route"]
    assert _reads(_ending(nodes, "ffn/topk_scale")) == ["topk_norm"]
    assert _reads(_ending(nodes, "ffn/combine"))[1] == "topk_scale"


# DeepSeek-V4 compressor: `score.softmax(dim=2)` over the window positions of
# the unflattened [b, s/ratio, ratio, d] view (official inference/model.py
# Compressor.forward): ratio = 4 for the overlapping CSA compressor, 128 for HCA.

def test_dsv4_compressor_softmax_runs_over_the_window():
    for template, ratio in (("dsv4_csa_attention", 4), ("dsv4_hca_attention", 128)):
        nodes = _block_nodes("deepseek-ai/DeepSeek-V4-Flash", template)
        smax = _ending(nodes, "attn/compressor/smax")
        assert smax.op == "softmax_act" and smax.axis == 2
        fold = _ending(nodes, "attn/compressor/gate_win")
        assert fold.op == "view_split" and fold.axis == 1
        assert fold.dims["window"] == ratio


# DSA lightning indexer: RoPE on the first qk_rope_head_dim dims of each
# query head and of the key before scoring (DeepseekV32Indexer /
# GlmMoeDsaIndexer.forward; vLLM deepseek_v2.py Indexer indexer_rope_emb).

def test_dsa_indexer_scores_rotated_query_and_key():
    for model_id in ("deepseek-ai/DeepSeek-V3.2", "zai-org/GLM-5"):
        nodes = _block_nodes(model_id, "")
        logits = _ending(nodes, "indexer_logits")
        assert _reads(logits) == [
            "indexer.query_concat", "indexer.key_concat", "indexer.weights_proj"]
        assert _reads(_ending(nodes, "indexer.key_concat")) == [
            "indexer.k_pe_rope", "indexer.k_nope_slice"]
        assert _ending(nodes, "indexer.q_pe_rope").op == "rope"


# Gemma 2: logits = c·tanh(logits / c) when final_logit_softcapping is not
# None (Gemma2ForCausalLM.forward; vLLM gemma2.py soft_cap). A config that
# sets no cap keeps the head's logits; the cap's value never switches the
# steps off.

def _softcap(config=None):
    tree = _tree("google/gemma-2-9b", config)
    (cap,) = [n for n in tree.top_level if n.name == "final_logit_softcap"]
    return cap


def test_gemma2_soft_caps_its_logits():
    cap = _softcap()
    assert snapshot_config("google/gemma-2-9b")["final_logit_softcapping"] == 30.0
    assert _reads(cap) == ["lm_head"]
    assert [c.op for c in cap.children] == ["scale_const", "tanh", "scale_const"]


def test_gemma_soft_cap_is_built_for_any_set_value():
    """The source's condition is `is not None`: an explicit 0 (the source
    divides by it) and a fractional cap build the steps like 30.0 does. A
    Gemma 2 config that omits the field gets Gemma2Config's default 30.0; a
    Gemma 4 one gets its config class's None."""
    capped = snapshot_config("google/gemma-2-9b")
    for value in (0, 0.0, 0.5, 30.5):
        cap = _softcap(dict(capped, final_logit_softcapping=value))
        assert [c.op for c in cap.children] == ["scale_const", "tanh", "scale_const"], value
    omitted = {k: v for k, v in capped.items() if k != "final_logit_softcapping"}
    assert _softcap(omitted).children

    g4 = snapshot_config("google/gemma-4-12B-it")
    text = {k: v for k, v in g4["text_config"].items() if k != "final_logit_softcapping"}
    (cap,) = [n for n in _tree("google/gemma-4-12B-it", dict(g4, text_config=text)).top_level
              if n.name == "final_logit_softcap"]
    assert not cap.children


def test_gemma2_without_a_cap_keeps_the_logits():
    """final_logit_softcapping = None: no soft-cap step is built and the analysis prices
    nothing past lm_head — the same FLOPs as the capped model minus exactly
    its soft-cap steps."""
    from backend.arch.eval.program import Program
    from backend.arch.eval.transforms import Deployment

    capped = snapshot_config("google/gemma-2-9b")
    config = dict(capped, final_logit_softcapping=None)
    assert not _softcap(config).children

    def analyze(cfg):
        resolved = shared_loader().match(cfg, model_id="google/gemma-2-9b")
        a = Deployment(Program.of(resolved)).at(
            batch=1, input_tokens=64, logit_positions=1).analyze()
        cap_flops = sum(a.cost.values[n.uid].flops for n in a.program.nodes()
                        if not n.children and n.uid in a.cost.values
                        and "final_logit_softcap" in (n.display_path or ""))
        return a.total().flops, cap_flops

    total_off, cap_off = analyze(config)
    total_on, cap_on = analyze(capped)
    assert cap_off == 0 and cap_on > 0
    assert total_off == total_on - cap_on


# Gemma 2 / Gemma 4 FFN: ACT2FN[hidden_activation] with every config setting
# `gelu_pytorch_tanh` — the tanh approximation, not the erf GELU (vLLM
# gemma2.py builds GeluAndMul(approximate="tanh")).

def test_gemma_ffn_uses_tanh_approximate_gelu():
    for model_id, block in (("google/gemma-2-9b", "gemma2_4norm_block"),
                            ("google/gemma-4-12B-it", "gemma4_4norm_scaled_block")):
        config = snapshot_config(model_id)
        config = config.get("text_config", config)
        assert config["hidden_activation"] == "gelu_pytorch_tanh"
        nodes = _block_nodes(model_id, block)
        (act,) = [n for p, n in nodes.items()
                  if p.startswith("mlp/") and p.endswith("/act")]
        assert act.op == "gelu_tanh", model_id


# Kimi-K3 KDA decay gate: the source branches on `gate_lower_bound is not
# None` (modeling_kimi_linear.py KimiDeltaAttention.forward passes
# `safe_gate=`/`lower_bound=` to chunk_kda / fused_recurrent_kda; sglang
# fused_sigmoid_gating_recurrent.py and vLLM kimi_k3 kda/fused_recurrent.py
# compute `lower_bound * sigmoid(exp(A_log) * x)` when set,
# `-exp(A_log) * softplus(x)` otherwise). Kimi-K3 sets -5.0.

def _kda_decay_chain(config=None):
    """Ops on the path into delta_rule's decay operand, nearest first, as
    `(name, op)` down each node's first input, plus the rate operand's op."""
    model_id = "moonshotai/Kimi-K3"
    for meta, roots in _tree(model_id, config).blocks:
        if "block_kda_moe" in meta.template_name:
            nodes = dict(_walk(roots))
            break
    delta = _ending(nodes, "self_attn/delta_rule")
    node = delta.inputs_obj[3].src
    chain = []
    while node.op not in ("mul",):
        chain.append(node.op)
        node = node.inputs_obj[0].src
    gate, rate = (ref.src for ref in node.inputs_obj)
    return chain, gate, rate


def _head_split_source(node):
    """What a head split (`view_split` then `transpose`, the `head_split`
    component's body, spelled inline in kimi_k3_kda_attention) reads."""
    assert node.op == "transpose", node.op
    unflatten = node.inputs_obj[0].src
    assert unflatten.op == "view_split", unflatten.op
    return unflatten.inputs_obj[0].src


def test_kimi_k3_kda_gate_is_the_lower_bounded_sigmoid():
    config = snapshot_config("moonshotai/Kimi-K3")
    assert config["text_config"]["linear_attn_config"]["gate_lower_bound"] == -5.0
    chain, gate, rate = _kda_decay_chain()
    assert chain == ["scale_const", "sigmoid"]
    # exp(A_log) multiplies (f + dt_bias) itself — no softplus, no negation.
    assert _head_split_source(gate).op == "add"
    assert rate.op == "exp"


def test_kda_gate_without_a_lower_bound_is_the_softplus_form():
    config = snapshot_config("moonshotai/Kimi-K3")
    text = dict(config["text_config"])
    text["linear_attn_config"] = {k: v for k, v in text["linear_attn_config"].items()
                                  if k != "gate_lower_bound"}
    chain, gate, rate = _kda_decay_chain(dict(config, text_config=text))
    assert chain == []
    assert _head_split_source(gate).op == "softplus"
    assert rate.op == "scale_const" and rate.inputs_obj[0].src.op == "exp"
