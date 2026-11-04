"""Cross-model render invariant: the backend-expressed compute graph
must be fully renderable by the frontend — every edge that the engine
draws from a shaped producer must carry that shape. A `None`-shape edge
is dataflow the backend knows but the frontend can't label, i.e. a graph
the UI can't faithfully express.

This gate runs the invariant over EVERY shipped snapshot config on BOTH
render paths:

  • Inspector path (`build_block_graph_data`) — per-block DAG.
  • Inference post-tree path (`analyze` → `serialize_post_tree`) — the
    whole-model graph with kernel fusion + comm injection applied, the
    one the inference page actually renders.

Backstops the three root causes fixed alongside it:
  1. secondary-port shape drift (`_resolve_ref_nr` unification) — the
     V4-Pro `hc_post` post/comb edges,
  2. unknown ops silently evaluating to shapeless no-op nodes (engine
     now raises) — stale `activation`, unregistered `hash_route`,
  3. synthetic `block_stack` backbone wrappers carrying no NR — the
     block→block / block→norm edges (residual-stream propagation).

If a new model or a refactor reintroduces ANY of these, this test names
the exact (model, edge) instead of letting it ship as an unrenderable
edge.
"""
from __future__ import annotations

import glob
import json
import pathlib
import sys
import unittest
import pytest

pytestmark = pytest.mark.regression  # walks every snapshot


PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

SNAPSHOTS = sorted(glob.glob(
    str(PROJECT_ROOT / "backend/arch/snapshots/*.json")))

#: The framework the inference page asks for, so the gate renders the
#: graph the page renders.
_RENDERED_BINDING = ("vllm",)
#: Kernels this binding is known to fuse. The corpus union must keep
#: them — a model with nothing to fuse is fine; losing one of these
#: is a silent un-fuse.
_FUSION_FLOOR = frozenset({
    "fused_add_rmsnorm",
    "flash_attention",
    "qkv_parallel_linear",
})
#: Per snapshot, kernels this binding fused last time we locked the
#: gate. A model may grow this set; it may not shrink. Snapshots not
#: listed are new and only have to keep the corpus floor.
_FUSED_AT_LEAST = {
    k: frozenset(v.split()) for k, v in (
        ("MiniMaxAI__MiniMax-M2.7",
         "flash_attention fused_add_rmsnorm fused_qk_rope_partial "
         "merged_column_parallel_linear qkv_parallel_linear silu_and_mul"),
        ("Qwen__Qwen2.5-72B",
         "flash_attention fused_add_rmsnorm fused_qk_rope "
         "merged_column_parallel_linear qkv_parallel_linear silu_and_mul"),
        ("Qwen__Qwen2.5-7B",
         "flash_attention fused_add_rmsnorm fused_qk_rope "
         "merged_column_parallel_linear qkv_parallel_linear silu_and_mul"),
        ("Qwen__Qwen3-235B-A22B",
         "flash_attention fused_add_rmsnorm fused_qk_rope "
         "merged_column_parallel_linear qkv_parallel_linear silu_and_mul"),
        ("Qwen__Qwen3-30B-A3B",
         "flash_attention fused_add_rmsnorm fused_qk_rope "
         "merged_column_parallel_linear qkv_parallel_linear silu_and_mul"),
        ("Qwen__Qwen3-32B",
         "flash_attention fused_add_rmsnorm fused_qk_rope "
         "merged_column_parallel_linear qkv_parallel_linear silu_and_mul"),
        ("Qwen__Qwen3-8B",
         "flash_attention fused_add_rmsnorm fused_qk_rope "
         "merged_column_parallel_linear qkv_parallel_linear silu_and_mul"),
        ("Qwen__Qwen3-Coder-Next",
         "flash_attention fused_add_rmsnorm fused_qk_rope_partial "
         "merged_column_parallel_linear qkv_parallel_linear silu_and_mul"),
        ("THUDM__glm-4-9b-chat",
         "flash_attention fused_add_rmsnorm fused_qk_rope silu_and_mul"),
        ("THUDM__glm-4-9b",
         "flash_attention fused_add_rmsnorm fused_qk_rope silu_and_mul"),
        ("XiaomiMiMo__MiMo-V2-Flash",
         "flash_attention fused_add_rmsnorm fused_qk_rope_partial "
         "merged_column_parallel_linear moe_fused_gate "
         "qkv_parallel_linear silu_and_mul"),
        ("XiaomiMiMo__MiMo-V2.5-Pro",
         "flash_attention fused_add_rmsnorm fused_qk_rope_partial "
         "merged_column_parallel_linear moe_fused_gate silu_and_mul"),
        # Its audio encoder's separate q/k/v projections were the
        # qkv_parallel_linear; the encoder runs in the encode walk, not the
        # step this graph draws.
        ("XiaomiMiMo__MiMo-V2.5",
         "flash_attention fused_add_rmsnorm fused_qk_rope_partial "
         "merged_column_parallel_linear moe_fused_gate silu_and_mul"),
        ("deepseek-ai__DeepSeek-V2-Lite",
         "flash_attention fused_add_rmsnorm fused_qk_rope_partial_q "
         "merged_column_parallel_linear silu_and_mul"),
        ("deepseek-ai__DeepSeek-V2",
         "flash_attention fused_add_rmsnorm fused_qk_rope_partial_q "
         "merged_column_parallel_linear silu_and_mul"),
        ("deepseek-ai__DeepSeek-V3.1",
         "flash_attention fused_add_rmsnorm fused_qk_rope_partial_q "
         "merged_column_parallel_linear moe_fused_gate silu_and_mul"),
        ("deepseek-ai__DeepSeek-V3.2-Exp",
         "flash_dsa fused_add_rmsnorm fused_qk_rope "
         "merged_column_parallel_linear moe_fused_gate silu_and_mul"),
        ("deepseek-ai__DeepSeek-V3.2",
         "flash_dsa fused_add_rmsnorm fused_qk_rope "
         "merged_column_parallel_linear moe_fused_gate silu_and_mul"),
        ("deepseek-ai__DeepSeek-V3",
         "flash_attention fused_add_rmsnorm fused_qk_rope_partial_q "
         "merged_column_parallel_linear moe_fused_gate silu_and_mul"),
        # The V4 attention's q/kv RoPE runs inside
        # fused_qnorm_rope_kv_insert, not fused_qk_rope; its experts' SwiGLU
        # is the clamped one (SiluAndMulWithClamp).
        ("deepseek-ai__DeepSeek-V4-Pro",
         "flash_attention fused_compress_norm_rope_store fused_inv_rope "
         "fused_overlap_compress_norm_rope_store "
         "fused_qnorm_rope_kv_insert indexer_logits "
         "merged_column_parallel_linear silu_and_mul_with_clamp"),
        ("google__gemma-2-27b",
         "flash_attention fused_add_rmsnorm fused_qk_rope gelu_and_mul "
         "merged_column_parallel_linear qkv_parallel_linear"),
        ("google__gemma-2-9b",
         "flash_attention fused_add_rmsnorm fused_qk_rope gelu_and_mul "
         "merged_column_parallel_linear qkv_parallel_linear"),
        ("google__gemma-4-12B-it",
         "flash_attention fused_add_rmsnorm fused_qk_rope gelu_and_mul "
         "merged_column_parallel_linear qkv_parallel_linear"),
        ("inclusionAI__Ling-2.6-1T",
         "flash_attention fused_add_rmsnorm fused_qk_rope_partial fused_qk_rope_partial_q "
         "merged_column_parallel_linear moe_fused_gate silu_and_mul"),
        # No moe_fused_gate: that kernel is sigmoid grouped top-k, and
        # LongCat's router scores with softmax and does not group
        # (LongcatFlashTopkRouter; vLLM FusedMoE softmax, renormalize=False).
        ("meituan-longcat__LongCat-Flash-Thinking-2601",
         "flash_attention fused_add_rmsnorm fused_qk_rope_partial_q "
         "merged_column_parallel_linear silu_and_mul"),
        ("meta-llama__Llama-2-7b-hf",
         "flash_attention fused_add_rmsnorm fused_qk_rope "
         "merged_column_parallel_linear qkv_parallel_linear silu_and_mul"),
        ("meta-llama__Llama-3.2-1B",
         "flash_attention fused_add_rmsnorm fused_qk_rope "
         "merged_column_parallel_linear qkv_parallel_linear silu_and_mul"),
        ("meta-llama__Llama-3.2-3B",
         "flash_attention fused_add_rmsnorm fused_qk_rope "
         "merged_column_parallel_linear qkv_parallel_linear silu_and_mul"),
        ("mistralai__Mistral-7B-v0.1",
         "flash_attention fused_add_rmsnorm fused_qk_rope "
         "merged_column_parallel_linear qkv_parallel_linear silu_and_mul"),
        ("mistralai__Mistral-Nemo-Base-2407",
         "flash_attention fused_add_rmsnorm fused_qk_rope "
         "merged_column_parallel_linear qkv_parallel_linear silu_and_mul"),
        ("moonshotai__Kimi-K2-Base",
         "flash_attention fused_add_rmsnorm fused_qk_rope_partial_q "
         "merged_column_parallel_linear moe_fused_gate silu_and_mul"),
        ("moonshotai__Kimi-K2-Instruct",
         "flash_attention fused_add_rmsnorm fused_qk_rope_partial_q "
         "merged_column_parallel_linear moe_fused_gate silu_and_mul"),
        ("moonshotai__Kimi-K2.5",
         "flash_attention fused_add_rmsnorm fused_qk_rope_partial_q "
         "merged_column_parallel_linear moe_fused_gate silu_and_mul"),
        # Its vision blocks' residual norms were the fused_add_rmsnorm; the
        # tower runs in the encode walk, not the step this graph draws. No
        # fused_qk_rope_partial_q: K3's MLA layers are NoPE (60d646b2f —
        # neither framework rotates them), so there is no rotary to fuse.
        ("moonshotai__Kimi-K3",
         "flash_attention merged_column_parallel_linear moe_fused_gate "
         "qkv_parallel_linear situ_and_mul"),
        # No fused_qk_rope: Nemotron-H attention applies no rotary embedding.
        ("nvidia__NVIDIA-Nemotron-3-Ultra-550B-A55B-BF16",
         "flash_attention fused_add_rmsnorm "
         "moe_fused_gate qkv_parallel_linear"),
        ("openai__gpt-oss-20b",
         "flash_attention fused_add_rmsnorm fused_qk_rope "
         "qkv_parallel_linear silu_and_mul"),
        ("stepfun-ai__Step-3.7-Flash",
         "flash_attention fused_add_rmsnorm fused_qk_rope "
         "merged_column_parallel_linear silu_and_mul"),
        ("zai-org__GLM-5.1",
         "flash_dsa fused_add_rmsnorm fused_qk_rope "
         "merged_column_parallel_linear moe_fused_gate silu_and_mul"),
        ("zai-org__GLM-5.2",
         "flash_dsa fused_add_rmsnorm fused_qk_rope "
         "merged_column_parallel_linear moe_fused_gate silu_and_mul"),
    )
}


def _shapeless_edges(graph_or_blocks, *, whole_model: bool):
    """Collect (src, dst, port) for every edge with a `None` shape,
    recursing into nested `child_edges`. Works on both the Inspector
    block-dict (`{template: {nodes, edges}}`) and the post-tree graph
    (`{nodes, edges}`)."""
    out = []

    def _collect(edges):
        out.extend((e.get("src", "").split("|")[-1],
                    e.get("dst", "").split("|")[-1],
                    e.get("src_port"))
                   for e in edges or [] if e.get("shape") is None)

    def _walk(nodes):
        for n in nodes:
            _collect(n.get("child_edges"))
            if n.get("children"):
                _walk(n["children"])

    if whole_model:
        _walk(graph_or_blocks.get("nodes") or [])
        _collect(graph_or_blocks.get("edges"))
    else:
        for block in graph_or_blocks.values():
            nodes = block["nodes"] if isinstance(block, dict) else block
            _walk(nodes)
            if isinstance(block, dict):
                _collect(block.get("edges"))
    return out


class TestAllModelsRenderWithoutShapelessEdges(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from tests.snapshots import shared_loader
        cls.loader = shared_loader()

    def _configs(self):
        for sp in SNAPSHOTS:
            raw = json.loads(pathlib.Path(sp).read_text())
            yield pathlib.Path(sp).stem, raw.get("config", raw)

    def test_inspector_path_no_shapeless_edges(self):
        from backend.rendering.builders import build_block_graph_data
        offenders = {}
        checked = 0
        for name, config in self._configs():
            resolved = self.loader.match(config, model_id=name.replace("__", "/", 1))
            if resolved is None:
                continue
            bg, _ = build_block_graph_data(resolved, batch_size=2, seq_len=128)
            bad = _shapeless_edges(bg, whole_model=False)
            checked += 1
            if bad:
                offenders[name] = bad[:8]
        self.assertGreater(checked, 0, "no snapshot configs matched")
        self.assertEqual(
            offenders, {},
            f"Inspector path drew shapeless edges (checked {checked} models): "
            f"{offenders}",
        )

    def test_post_tree_path_no_shapeless_edges(self):
        from backend.inference.framework_bindings import get_binding
        from backend.inference.pipeline import analyze
        from backend.rendering._compute_graph import serialize_post_tree
        from backend.rendering.kv_marks import KVMarks
        from backend.inference.parallel_config import ParallelConfig
        binding = get_binding(*_RENDERED_BINDING)
        self.assertIsNotNone(
            binding, f"no binding for {_RENDERED_BINDING} — the gate would "
                     f"silently fall back to the unfused graph")
        offenders = {}
        checked = 0
        fused_ops: set[str] = set()
        fused_by_model: dict[str, set[str]] = {}
        for name, config in self._configs():
            resolved = self.loader.match(config, model_id=name.replace("__", "/", 1))
            if resolved is None:
                continue
            res = analyze(resolved, parallel=ParallelConfig(),
                          framework_binding=binding,
                          batch_size=1, avg_input_tokens=1, avg_output_tokens=0, phase="prefill", breakdown=False)
            here = {m["fused_op"] for m in res.fusion_matches}
            fused_ops.update(here)
            fused_by_model[name] = here
            graph = serialize_post_tree(
                res.post_tree, resolved, pre_tree=res.comm_tree,
                fusion_matches=res.fusion_matches,
                program=res.program,
                shapes=res.shapes,
                full_shapes=res.full_shapes,
                pre_program=res.pre_program,
                pre_shapes=res.pre_shapes,
                pre_full_shapes=res.pre_full_shapes,
                kv=KVMarks.of_serving(res),
            )
            bad = _shapeless_edges(graph, whole_model=True)
            checked += 1
            if bad:
                offenders[name] = bad[:8]
        self.assertGreater(checked, 0, "no snapshot configs matched")
        self.assertTrue(
            fused_ops,
            "no model fused under the rendered binding — the gate is "
            "not exercising the path the inference page renders",
        )
        missing = _FUSION_FLOOR - fused_ops
        self.assertFalse(
            missing,
            f"binding {_RENDERED_BINDING} lost fused kernel(s) {sorted(missing)}",
        )
        lost = {
            name: sorted(expected - fused_by_model.get(name, set()))
            for name, expected in _FUSED_AT_LEAST.items()
            if expected - fused_by_model.get(name, set())
        }
        self.assertFalse(
            lost,
            f"binding {_RENDERED_BINDING} lost fused kernel(s) on "
            f"already-fused models: {lost}",
        )
        self.assertEqual(
            offenders, {},
            f"Inference post-tree path drew shapeless edges "
            f"(checked {checked} models): {offenders}",
        )


if __name__ == "__main__":
    unittest.main()
