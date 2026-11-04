"""LongCat-Flash's router renormalizes its top-k weights only when
`config.norm_topk_prob` is set, and the config class defaults it to False
(modeling_longcat_flash.py::LongcatFlashTopkRouter.forward;
configuration_longcat_flash.py `norm_topk_prob=False`; vLLM
longcat_flash.py passes `renormalize=False`). The shipped LongCat configs do
not carry the field, so their router has no renormalization step."""
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from backend.arch.spec_tree import iter_tree
from tests.snapshots import shared_loader, snapshot_config

MODEL = "meituan-longcat/LongCat-Flash-Thinking-2601"


def _router(config):
    resolved = shared_loader().match(config, model_id=MODEL)
    tree = iter_tree(resolved)

    def walk(nodes):
        for node in nodes:
            if node.name == "router" and node.children:
                return node
            found = walk(node.children or ())
            if found is not None:
                return found
        return None

    for _meta, roots in tree.blocks:
        router = walk(roots)
        if router is not None:
            return {child.name: child for child in router.children}
    raise AssertionError("no router in the LongCat block")


def _producer(node):
    (ref,) = node.inputs_obj
    return ref.src.name


def test_softmax_scores_and_no_renormalization_by_default():
    config = snapshot_config(MODEL)
    assert "norm_topk_prob" not in config
    router = _router(config)
    assert router["softmax"].op == "softmax_act"
    assert "sigmoid" not in router
    assert "topk_norm" not in router
    assert _producer(router["topk_scale"]) == "topk_gather"


def test_renormalization_when_the_config_asks_for_it():
    config = dict(snapshot_config(MODEL), norm_topk_prob=True)
    router = _router(config)
    assert router["topk_norm"].op == "normalize_sum_one"
    assert _producer(router["topk_scale"]) == "topk_norm"


# The same switch on the plain softmax router. Qwen3-MoE renormalizes the
# gathered top-k weights when `norm_topk_prob` is set (modeling_qwen3_moe.py
# `Qwen3MoeTopKRouter.forward`; vLLM qwen3_moe.py `renormalize=
# config.norm_topk_prob`) and both shipped Qwen3-MoE configs set it; the
# DeepSeek-V2 configs set it False. The step is the router's output, so the
# MoE reads whichever of the two the switch leaves standing.

def _router_of(model_id, router_name="gate", config=None):
    config = snapshot_config(model_id) if config is None else config
    resolved = shared_loader().match(config, model_id=model_id)
    tree = iter_tree(resolved)

    def walk(nodes):
        for node in nodes:
            if node.name == router_name and node.children:
                return node
            found = walk(node.children or ())
            if found is not None:
                return found
        return None

    for _meta, roots in tree.blocks:
        router = walk(roots)
        if router is not None:
            return router
    raise AssertionError(f"no {router_name} in {model_id}")


def test_qwen3_moe_renormalizes_its_top_k_weights():
    config = snapshot_config("Qwen/Qwen3-30B-A3B")
    assert config["norm_topk_prob"] is True
    router = _router_of("Qwen/Qwen3-30B-A3B")
    children = {child.name: child for child in router.children}
    assert children["topk_norm"].op == "normalize_sum_one"
    assert _producer(children["topk_norm"]) == "topk_gather"
    assert router.output_names == ("topk_norm",)


def test_deepseek_v2_keeps_the_gathered_scores():
    """No renormalization; the gathered scores go straight to the
    `routed_scaling_factor` multiply (`DeepseekV2TopkRouter.forward`)."""
    config = snapshot_config("deepseek-ai/DeepSeek-V2-Lite")
    assert config["norm_topk_prob"] is False
    router = _router_of("deepseek-ai/DeepSeek-V2-Lite")
    children = {child.name: child for child in router.children}
    assert "topk_norm" not in children
    assert _producer(children["topk_scale"]) == "topk_gather"
    assert router.output_names == ("topk_scale",)


def test_deepseek_v2_scales_its_weights_and_qwen3_moe_does_not():
    """DeepseekV2TopkRouter multiplies the top-k weights by
    routed_scaling_factor (V2 16.0; vLLM deepseek_v2.py passes it to
    FusedMoE); Qwen3MoeTopKRouter has no such factor."""
    assert snapshot_config("deepseek-ai/DeepSeek-V2")["routed_scaling_factor"] == 16.0
    router = _router_of("deepseek-ai/DeepSeek-V2")
    assert {c.name: c for c in router.children}["topk_scale"].op == "scale_const"
    assert router.output_names == ("topk_scale",)
    qwen = _router_of("Qwen/Qwen3-30B-A3B")
    assert "topk_scale" not in {child.name for child in qwen.children}


def test_deepseek_v2_scales_whatever_the_factor_is():
    """The multiply has no condition in the source, so the step does not
    read the factor's value: 0 (weights all zero), a fractional 2.5, and a
    config that omits the field (class default 1.0) all build it."""
    base = snapshot_config("deepseek-ai/DeepSeek-V2")
    omitted = {k: v for k, v in base.items() if k != "routed_scaling_factor"}
    for config in (dict(base, routed_scaling_factor=0.0),
                   dict(base, routed_scaling_factor=2.5), omitted):
        router = _router_of("deepseek-ai/DeepSeek-V2", config=config)
        assert {c.name: c for c in router.children}["topk_scale"].op == "scale_const"
        assert router.output_names == ("topk_scale",)


def test_gpt_oss_softmax_runs_over_the_top_k_logits():
    """`GptOssTopKRouter.forward`: top-k over the router logits, then a
    softmax over the k picked values (modeling_gpt_oss.py:133-134)."""
    resolved = shared_loader().match(
        snapshot_config("openai/gpt-oss-20b"), model_id="openai/gpt-oss-20b")
    moe = next(node for _meta, roots in iter_tree(resolved).blocks
               for root in roots for node in (root, *root.children)
               if any(c.name == "topk_softmax" for c in node.children))
    children = {child.name: child for child in moe.children}
    assert _producer(children["topk_indices"]) == "router"
    assert _producer(children["topk_softmax"]) == "topk_gather"
    assert children["topk_softmax"].op == "softmax_act"


# DeepSeek-V2's `topk_method: group_limited_greedy` keeps the best
# `topk_group` of `n_group` expert groups, ranked by their best expert, before
# the token-level top-k (modeling_deepseek_v2.py
# `DeepseekV2TopkRouter.forward`; vLLM deepseek_v2.py calls `grouped_topk`
# with num_expert_group=n_group, topk_group=topk_group). V2 ships 3 of 8;
# V2-Lite ships 1 of 1, which restricts nothing, so its router has no step.

def test_deepseek_v2_limits_top_k_to_the_best_groups():
    config = snapshot_config("deepseek-ai/DeepSeek-V2")
    assert (config["n_group"], config["topk_group"]) == (8, 3)
    router = _router_of("deepseek-ai/DeepSeek-V2")
    children = {child.name: child for child in router.children}
    assert children["group_max"].op == "group_max"
    assert _producer(children["group_max"]) == "softmax_scores"
    assert _producer(children["group_topk"]) == "group_max"
    assert [ref.src.name for ref in children["group_mask"].inputs_obj] == [
        "softmax_scores", "group_topk"]
    assert _producer(children["topk_indices"]) == "group_mask"


def test_deepseek_v2_lite_selects_over_all_experts():
    config = snapshot_config("deepseek-ai/DeepSeek-V2-Lite")
    assert (config["n_group"], config["topk_group"]) == (1, 1)
    router = _router_of("deepseek-ai/DeepSeek-V2-Lite")
    children = {child.name: child for child in router.children}
    assert not {"group_max", "group_topk", "group_mask"} & set(children)
    assert _producer(children["topk_indices"]) == "softmax_scores"
