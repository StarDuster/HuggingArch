"""What every MoE variant's work and storage are, and what its dispatch
boundary moves.

The corpus declares an expert bank in at least four weight layouts — one wide
`gate_up_proj`, separate `gate_proj` / `up_proj` leaves, `w1`/`w3`/`w2`, and
variants that inherit one of those through another component — crossed with
`repeat:` sitting on a wrapping component versus on each leaf, crossed with
shared experts, routing bias and routing scale. Three big models share one of
those layouts, so a gate built from three big models says nothing about the
other three. This module is one model per MoE component the corpus reaches,
with the numbers written out.

Two halves, and they are opposite claims:

* **work and storage do not move.** Re-spelling where the expert count lives
  re-factors one product (`2·R·K·N == E · 2·M·K·N` with `R = M·E`), so a
  model's FLOPs and params are the same integer before and after. Those are
  the first table.
* **the dispatch boundary's bytes do.** `permute` writes, and `combine`
  reads, the whole routed buffer once per layer, and both were billed one
  expert's slice of it. The
  second table is those rows, per model, hand-closed in the comment that
  carries them. It moved once, in the commit that made the two ops declare
  the buffer they touch.

Every number here is per *whole model* at one deployment (no parallelism) and
one workload (prefill, batch 8, 2048 tokens), summed over every instance the
program counts — so a reader can multiply a layer's closed form by the layer
count and check it.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, Tuple

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT))

from tests.snapshots import (
    SNAPSHOT_DIR,
    shared_program,
    shared_walk)

#: The one workload every number below is taken at.
BATCH, TOKENS = 8, 2048

#: `MoE component -> (snapshot stem, whole-model FLOPs, whole-model params)`.
#:
#: The key is the INNERMOST component holding the dispatch pair, which is what
#: `SpecIRNode.component_path` ends in — the thing whose expert-bank layout is
#: under test. `TestEveryVariantIsCovered` asserts this table names every such
#: component the checked-out corpus reaches, so admitting a new MoE component
#: without a row here is red rather than silently unmeasured.
#:
#: The four `target_*` rows are not variants: they are the models whose
#: dispatch-boundary bytes `_BOUNDARY` writes out, kept here so their work and
#: storage are pinned by the same walk.
#:
#: `dsv41_moe_vl` moved once, 620,875,385,815,040 -> 627,834,818,281,472
#: (+1.12%), when DeepSeek-V4.1-Flash's three DSpark draft stages began
#: dispatching over their own 128 destinations instead of inheriting the
#: backbone's 384. Each stage's bank went from 128 to 384 rows per expert, so
#: its three projections and its activation are 3x: 3 x 2,319,810,822,144.
#: Params do not move — the stages always held 128 experts.
#:
#: Main's sparse-kv merge (3f3ab12de) moved five of these FLOPs, params
#: unchanged: bd1f40626 ("gated_delta_rule counts the recurrent kernel's
#: arithmetic") raised every gated-delta / KDA model — Kimi-K3
#: 3,444,403,798,492,416 -> 3,453,335,946,872,064, Qwen3.5-35B-A3B
#: 118,954,967,894,848 -> 120,249,482,152,768, GLM-5.3-Flash
#: 597,501,194,064,128 -> 600,435,426,382,080, Qwen3-Coder-Next
#: 120,806,685,805,952 -> 122,360,102,915,456 — and 7525ea535 ("DeepSeek-V4.1's
#: candidate-consumer indexers score only the candidates") took 167,772,160
#: off DeepSeek-V4.1-Flash, which composes additively with the DSpark move
#: above (627,834,818,281,472 -> 627,834,650,509,312). Each was measured at
#: the commit before and after it.
#:
#: DeepSeek-V4-Flash moved 489,848,067,883,008 -> 489,845,953,953,792
#: (-2,113,929,216, params unchanged) when the indexer's score stopped
#: restating `s_kv // compress_ratio` in its `dims:` and took its key length
#: from the compressor stream it reads. The restatement had also kept the
#: stream's live pairs out (a query at context `c` scores the ⌊c/4⌋ complete
#: blocks: 523,776 pairs per head at 2,048 tokens), so the score was priced
#: at the dense causal fraction of its 2,048 x 512 matrix (524,544): 768
#: pairs x 2·8·64·128 x 21 CSA layers.
#:
#: The same exact count reached the readers of a pooled cache with no
#: `read_topk` (`kv_cache_view`'s pairs, `⌊c / compress_ratio⌋` blocks a query
#: at context `c` sees), which had been priced at the phase's dense fraction
#: of their pooled matrix: DeepSeek-V4.1-Flash 627,834,650,509,312 ->
#: 627,834,549,846,016 and GLM-5.3-Flash 613,636,008,366,336 ->
#: 613,635,404,386,560, params unchanged.
#:
#: 004f23e7e ("Describe hc_split_sinkhorn as V4's kernel computes it") priced
#: the hyper-connection mix the way DeepSeek-V4's kernel splits it: sigmoid on
#: the pre/post segments, Sinkhorn on the n² comb block only —
#: B·s·(2·d_hc_mix + 80·n²) instead of 80·B·s·d_hc_mix, 1,328 vs 1,920 per
#: token at n = 4. That is 592 x 16,384 tokens off per hc_split_sinkhorn call,
#: params unchanged: DeepSeek-V4-Flash 88 calls (44 layers incl. MTP x 2)
#: 489,845,953,953,792 -> 489,845,100,412,928, DeepSeek-V4.1-Flash 86 calls
#: 627,834,549,846,016 -> 627,833,715,703,808, GLM-5.3-Flash 90 calls
#: 613,635,404,386,560 -> 613,634,531,447,040.
#:
#: Re-pinned again after the cast one-copy-per-tensor and small-launch
#: billing series (the same moves sol_surface's re-pin, e3bd743a3,
#: attributed: quantise_activation FLOPs, silu epilogues taking the cast
#: in, Kimi-K3/KDA small launches, DSv4.1's Engram wkv split). Params
#: moved nowhere: DeepSeek-V4-Flash 489,845,100,412,928 ->
#: 489,867,737,071,616; Kimi-K3 3,453,335,946,872,064 ->
#: 3,453,328,623,617,280; Ling-2.6-1T 2,116,749,368,492,032 ->
#: 2,116,758,763,732,992; LongCat-Flash-Omni 928,592,776,396,800 ->
#: 928,581,502,107,648.
_VARIANTS: Dict[str, Tuple[str, int, int]] = {
    "dsv41_moe_vl": ("deepseek-ai__DeepSeek-V4.1-Flash",
                     627833715703808, 762720047314),
    "dsv4_moe_hash": ("deepseek-ai__DeepSeek-V4-Flash",
                      489867737071616, 290944616402),
    "dsv4_moe_sigmoid": ("deepseek-ai__DeepSeek-V4-Flash",
                         489867737071616, 290944616402),
    "kimi_k3_latent_moe": ("moonshotai__Kimi-K3",
                           3453328623617280, 2779484478208),
    "minimax_m2_decoder_block": ("MiniMaxAI__MiniMax-M2.7",
                                 367241551675392, 228689764864),
    "moe_ffn": ("Qwen__Qwen3-235B-A22B", 759186768527360, 235093634560),
    "moe_ffn_fused_packed": ("Qwen__Qwen3-VL-235B-A22B-Instruct",
                             759186768527360, 235093634560),
    "moe_ffn_fused_packed_bias": ("openai__gpt-oss-20b",
                                  122027127472128, 20914757184),
    "moe_ffn_fused_scaled_shared": ("Qwen__Qwen3.5-35B-A3B",
                                    120249482152768, 35505251456),
    "moe_ffn_gate_expert_bias": ("inclusionAI__Ling-2.6-1T",
                                 2116758763732992, 1025414425344),
    "moe_ffn_gate_routing_bias": ("XiaomiMiMo__MiMo-V2-Flash",
                                  589331472760832, 309766601088),
    # GLM-5.3-Flash and Hy4: their DSA attention prefills absorbed
    # (`transforms.phase_form`), scoring at the latent width — GLM-5.3-Flash
    # 600,435,426,382,080 -> 613,636,008,366,336 on top of bd1f40626's move
    # above, Hy4 1,721,315,717,627,904 -> 1,819,082,527,948,800.
    "moe_ffn_gate_routing_bias_clamped": ("zai-org__GLM-5.3-Flash",
                                          613634531447040, 320759404382),
    "moe_ffn_gate_routing_bias_packed_clamped": ("tencent__Hy4-preview",
                                                 1819082527948800,
                                                 779960992733),
    "moe_ffn_packed_debias_no_shared": ("stepfun-ai__Step-5-Preview",
                                        999501837631488, 600991912512),
    "moe_ffn_relu2_shared": ("nvidia__NVIDIA-Nemotron-3-Nano-30B-A3B-BF16",
                             108262003654656, 31577940288),
    "moe_ffn_router_classifier_routing_bias": (
        "meituan-longcat__LongCat-Flash-Omni",
        928581502107648, 560664980480),
    "moe_ffn_router_expert_bias_shared": ("tencent__Hy3",
                                          731867905523712, 298786155776),
    "moe_ffn_scaled_shared_gate": ("Qwen__Qwen3-Coder-Next",
                                   122360102915456, 79674391296),
    "nemotron_h_latent_moe": (
        "nvidia__NVIDIA-Nemotron-3-Ultra-550B-A55B-BF16",
        1881930874011648, 560524603904),
    "step3p7_moe_packed_no_shared": ("stepfun-ai__Step-3.7-Flash",
                                     474981667176448, 199384301376),
    "target_deepseek_v2_lite": ("deepseek-ai__DeepSeek-V2-Lite",
                                85040125247488, 15706484224),
    "target_deepseek_v3": ("deepseek-ai__DeepSeek-V3",
                           1339178383572992, 684489845504),
    "target_kimi_k25": ("moonshotai__Kimi-K2.5",
                        1080774370459648, 1026408232448),
    "target_qwen3_30b_a3b": ("Qwen__Qwen3-30B-A3B",
                             113064015822848, 30532122624),
}

#: The dispatch boundary's bytes, whole-model, at the same workload:
#: `stem -> {display path: (instances, act_in, act_out, access)}`.
#:
#: Each is written out with its closed form below, so the table can be
#: checked against a model's config without this engine.
#:
#: Qwen3-30B-A3B (d_model 2048, BF16, 48 layers, E = n_route_targets = 128,
#: k = 8). The routed buffer is `k` copies of the token stream — the expert
#: count cancels, because `R = M·E` and `M = B·s_ctx·k/E`:
#:
#:   M = B·s_ctx·k/E = 8·2048·8/128 = 1024 rows, the per-expert MEAN
#:   R = M·E         = 131,072 rows, the whole dispatch
#:   permute act_out  = R·d_model·bpe·L = 131072·2048·2·48 = 25,769,803,776
#:                      ( = B·s_ctx·k·d_model·bpe·L, 128x the 201,326,592
#:                        one expert's slice was billed )
#:   permute act_in   = token read B·s_ctx·d_model·bpe·L
#:                      + routing read B·s_ctx·k·bpe·L
#:                    = 3,221,225,472 + 12,582,912    =  3,233,808,384
#:                      ( unchanged: neither term is routed )
#:   combine act_in   = permute's act_out + the routing read
#:                    = 25,769,803,776 + 12,582,912   = 25,782,386,688
#:   combine act_out  = B·s_ctx·d_model·bpe·L          =  3,221,225,472
#:   access (both)    = act_in + act_out               = 29,003,612,160
#:                      ( 8.44x the 3,435,134,976 billed before, because the
#:                        unrouted terms do not scale )
#:
#: The other three, same arithmetic at their own counts — all four `R` come
#: to the same 131,072 rows per layer because all four route 8 of their
#: experts per token:
#:
#:   DeepSeek-V2-Lite  E=64,  k=6,  d_model 2048, 26 layers, M=1536
#:     combine act_in = 1536·64·2048·2·26 + 5,111,808  = 10,474,094,592
#:   DeepSeek-V3       E=256, k=8,  d_model 7168, 58 layers + 1 MTP
#:     combine act_in = 131072·7168·2·58 + 15,204,352  = 108,999,999,488
#:   Kimi-K2.5         E=384, k=8,  d_model 7168, 60 layers, M=341⅓
#:     combine act_in = 131072·7168·2·60 + 15,728,640  = 112,758,620,160
#:
#: Kimi-K2.5's `M` is not an integer and is not rounded: a per-expert mean
#: times the expert count has to come back to a whole buffer, which is why
#: `_tokens_per_expert` keeps it an exact `Rational`. Rounding it would have
#: left this row 5·60 = 300 bytes short.
_BOUNDARY: Dict[str, Dict[str, Tuple[int, int, int, int]]] = {
    "Qwen__Qwen3-30B-A3B": {
        "mlp.permute__moe_ffn":
            (48, 3233808384, 25769803776, 29003612160),
        "mlp.combine__moe_ffn":
            (48, 25782386688, 3221225472, 29003612160),
    },
    "deepseek-ai__DeepSeek-V3": {
        "mlp.permute__moe_ffn_gate_routing_bias":
            (58, 13638303744, 108984795136, 122623098880),
        "mlp.combine__moe_ffn_gate_routing_bias":
            (58, 108999999488, 13623099392, 122623098880),
        "transformer.mlp.permute__moe_ffn_gate_routing_bias":
            (1, 235143168, 1879048192, 2114191360),
        "transformer.mlp.combine__moe_ffn_gate_routing_bias":
            (1, 1879310336, 234881024, 2114191360),
    },
    "deepseek-ai__DeepSeek-V2-Lite": {
        "mlp.permute__moe_ffn":
            (26, 1749942272, 10468982784, 12218925056),
        "mlp.combine__moe_ffn":
            (26, 10474094592, 1744830464, 12218925056),
    },
    "moonshotai__Kimi-K2.5": {
        "mlp.permute__moe_ffn_gate_routing_bias":
            (60, 14108590080, 112742891520, 126851481600),
        "mlp.combine__moe_ffn_gate_routing_bias":
            (60, 112758620160, 14092861440, 126851481600),
    },
}


def _analysis(stem: str) -> Any:
    from backend.arch.eval.transforms import Deployment

    program = shared_program(stem)
    return (shared_walk(Deployment(program)
            .at(batch=BATCH, input_tokens=TOKENS)))


def _boundary_rows(analysis: Any) -> Dict[str, Tuple[int, int, int, int]]:
    """Every dispatch-boundary leaf's whole-model bytes.

    The ops are asked for by declaration (`OpSpec.is_dispatch_boundary`), not
    by name, so a boundary op added later is measured here without this file
    learning its name.
    """
    from backend.arch.layout.persistent import physical_artifacts
    from backend.arch.layout.physical_cost import physical_access
    from backend.arch.ops import is_dispatch_boundary

    access = physical_access(analysis,
                             artifacts=physical_artifacts(analysis),
                             scale_dtype_bytes=4.0)
    out: Dict[str, Tuple[int, int, int, int]] = {}
    for uid, row in access.values.items():
        node = analysis.program.by_uid(uid)
        if not is_dispatch_boundary(node.declaration):
            continue
        count = int(analysis.instances.values[uid])
        out[node.display_path] = (count,
                                  row.act_in_bytes * count,
                                  row.act_out_bytes * count,
                                  row.full * count)
    return out


class TestWorkAndStorageDoNotMove:
    """One model per MoE component, FLOPs and params written out.

    These are the integers a re-spelling of the expert count may not change.
    They are whole-model totals at one workload, so each is checkable from the
    model's own config without this engine.
    """

    @pytest.mark.parametrize("variant", sorted(_VARIANTS))
    def test_the_models_totals_are_the_pinned_ones(self, variant):
        stem, flops, params = _VARIANTS[variant]
        total = _analysis(stem).total()
        assert (int(total.flops), int(total.params)) == (flops, params), (
            f"{variant} ({stem}): FLOPs/params moved")


class TestTheDispatchBoundarysBytes:
    """What `permute` writes and `combine` reads, per model.

    Written out rather than derived, because the point of the table is to be
    checkable against the closed forms in `_BOUNDARY`'s comment: a law
    compared against itself moves silently.
    """

    @pytest.mark.parametrize("stem", sorted(_BOUNDARY))
    def test_the_boundary_rows_are_the_pinned_ones(self, stem):
        assert _boundary_rows(_analysis(stem)) == _BOUNDARY[stem]


class TestEveryVariantIsCovered:
    """The table names every MoE component the corpus reaches.

    Enumerated from the walk, not from a list of files: a component is
    "reached" when some snapshot's program has a dispatch node inside it. A
    dispatch declared inline in a block (MiniMax-M2.7) is keyed on that
    block.
    """

    def test_no_reachable_moe_component_is_unmeasured(self):
        from backend.arch.ops import is_dispatch_boundary

        reached: Dict[str, str] = {}
        for path in sorted(SNAPSHOT_DIR.glob("*.json")):
            program = shared_program(path.stem)
            for node in program.nodes():
                if not is_dispatch_boundary(node.declaration):
                    continue
                owner = node.component_path
                reached.setdefault(
                    owner[-1][1] if owner else node.template, path.stem)
        assert reached, "no snapshot in the corpus declares a dispatch"
        missing = sorted(set(reached) - set(_VARIANTS))
        assert missing == [], (
            "MoE components with no row in `_VARIANTS`: "
            + ", ".join(f"{name} (e.g. {reached[name]})" for name in missing))
