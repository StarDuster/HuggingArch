"""Regression tests for the rendering/ pipeline (response builders).

These are end-to-end tests over committed snapshot files (no network):
spec match → engine → block_graph + architecture_view + model_config.
Catches regressions in node id composition, dtype attachment, badge
derivation, and quant-precision string."""
import json
import glob
import sys
import unittest
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

SNAPSHOTS_DIR = PROJECT_ROOT / "backend" / "arch" / "snapshots"

from tests.snapshots import shared_loader
from backend.arch import roles
from backend.arch.structure import layer_configs
from backend.rendering import (
    build_architecture_view,
    build_block_graph_data,
    build_model_config,
)

from backend.hf.header import SafetensorsHeader

pytestmark = pytest.mark.regression  # walks every snapshot


def _load(snap_name: str):
    snap = json.loads((SNAPSHOTS_DIR / snap_name).read_text())
    cfg = snap["config"]
    patterns = SafetensorsHeader.from_json(snap["metadata"]["header"])
    resolved = shared_loader().match(cfg, model_id=snap_name.removesuffix(".json").replace("__", "/", 1))
    return cfg, patterns, resolved


def _walk_nodes(nodes, fn):
    for n in nodes:
        fn(n)
        if n.get("children"):
            _walk_nodes(n["children"], fn)


def _keys_anywhere(payload, keys):
    """Every (path, key) at which one of `keys` appears in a JSON payload."""
    found = []

    def walk(x, path):
        if isinstance(x, dict):
            for k, v in x.items():
                if k in keys:
                    found.append((path, k))
                walk(v, f"{path}.{k}")
        elif isinstance(x, list):
            for i, v in enumerate(x):
                walk(v, f"{path}[{i}]")
    walk(payload, "")
    return found


class TestTransportKeysStayInside(unittest.TestCase):
    """The declaration's answers that `build_edges` and the absorption pass
    read off the node dict (`_node_build.strip_transport_keys`) are
    transport between passes, not wire format: the block graph the /analyze
    payload embeds carries none of them, on a dense model and on a MoE model
    with bare parameters to absorb."""

    TRANSPORT_KEYS = ("port_labels", "output_kind", "identity_source")

    def test_block_graph_carries_no_transport_keys(self):
        TRANSPORT_KEYS = self.TRANSPORT_KEYS
        for snap in ("Qwen__Qwen3-8B.json", "deepseek-ai__DeepSeek-V3.json"):
            _cfg, _patterns, r = _load(snap)
            self.assertIsNotNone(r)
            bg, _ = build_block_graph_data(r, batch_size=1, seq_len=1)
            self.assertEqual(_keys_anywhere(bg, TRANSPORT_KEYS), [], snap)


class TestBlockGraphData(unittest.TestCase):
    def test_qwen3_block_graph_shape(self):
        _cfg, _patterns, r = _load("Qwen__Qwen3-8B.json")
        self.assertIsNotNone(r)
        bg, unmatched = build_block_graph_data(
            r, batch_size=1, seq_len=1,
        )
        self.assertEqual(list(bg.keys()), ["block"])
        block = bg["block"]
        # Structural invariants
        self.assertGreater(len(block["nodes"]), 5)
        self.assertGreater(len(block["edges"]), 0)
        self.assertEqual(unmatched, [])
        # No mermaid field on any node (deleted in mermaid cleanup)
        leaks = []
        _walk_nodes(block["nodes"], lambda n: "mermaid" in n and leaks.append(n["id"]))
        self.assertEqual(leaks, [])
        # Tensor matching populated weight_dtype on leaf weights
        with_dtype = []
        _walk_nodes(block["nodes"],
                    lambda n: n.get("weight_dtype") and with_dtype.append(n["id"]))
        self.assertGreater(len(with_dtype), 5)

    def test_dsv3_three_templates(self):
        """DSv3 has dense + MoE assignments + MTP = 3 templates."""
        _cfg, _patterns, r = _load("deepseek-ai__DeepSeek-V3.json")
        self.assertIsNotNone(r)
        bg, _ = build_block_graph_data(
            r, batch_size=1, seq_len=1,
        )
        self.assertEqual(len(bg), 3)
        # Each template has nodes + edges
        for name, block in bg.items():
            self.assertGreater(len(block["nodes"]), 0, name)
            self.assertGreater(len(block["edges"]), 0, name)

    def test_dsv4_compressor_pooling_renders_shape_arrow(self):
        """The compressor's sequence pooling must be visible on the card.

        The pooling lives on `sum_reduce` (the window axis collapsing);
        without the `view` style the in→out shape arrow never renders and
        the reduction — the whole point of the component — is invisible.
        """
        _cfg, _patterns, r = _load("deepseek-ai__DeepSeek-V4-Flash.json")
        self.assertIsNotNone(r)
        bg, _ = build_block_graph_data(r, batch_size=1, seq_len=64)
        pools = []
        for block in bg.values():
            _walk_nodes(
                block["nodes"],
                lambda n: n.get("op") == "sum_reduce"
                and n.get("label") == "pooled"
                and pools.append(n))
        self.assertGreater(len(pools), 0)
        for n in pools:
            self.assertEqual(n.get("style"), "view", n.get("id"))
            def _axes(s: str) -> list:
                # The card factors packed feature axes ("2·512"), so plain
                # int() is not enough.
                return [eval(ax.replace("·", "*")) for ax in s.strip("[]").split(",")]
            s_in = _axes(n["tensor_in"])
            s_out = _axes(n["tensor_out"])
            self.assertEqual(s_in[0], s_out[0], n.get("id"))   # batch kept
            self.assertEqual(s_in[1], s_out[1], n.get("id"))   # slot kept
            self.assertEqual(s_in[3], s_out[2], n.get("id"))   # feature kept
            # The window axis is the one that collapses: rank 4 -> 3.
            self.assertEqual(len(s_in), 4, n.get("id"))
            self.assertEqual(len(s_out), 3, n.get("id"))

    def test_dsv4_compressor_scores_card_shows_local_formula(self):
        """A call-site `formula:` override must replace the registry's
        generic caption on the card. The generic `position_table_add`
        formula ("map(position_table, reference_positions)") is true of
        every use but shows none of them; the compressor's use site states
        the actual shape story (unflatten by window, ape broadcast on the
        window axis)."""
        _cfg, _patterns, r = _load("deepseek-ai__DeepSeek-V4-Flash.json")
        self.assertIsNotNone(r)
        bg, _ = build_block_graph_data(r, batch_size=1, seq_len=64)
        scores = []
        for block in bg.values():
            _walk_nodes(
                block["nodes"],
                lambda n: n.get("op") == "position_table_add"
                and n.get("label") == "scores" and scores.append(n))
        self.assertGreater(len(scores), 0)
        for n in scores:
            # The window is a real axis now: the add runs in the folded
            # [B, slot, window, D] frame with ape broadcasting over batch
            # and slot.
            self.assertIn("ape[p, d]", n["op_formula"], n.get("id"))
            self.assertIn("gate_win", n["op_formula"], n.get("id"))

    def test_dsv4_compressor_smax_names_the_reduction_axis(self):
        _cfg, _patterns, r = _load("deepseek-ai__DeepSeek-V4-Flash.json")
        self.assertIsNotNone(r)
        bg, _ = build_block_graph_data(r, batch_size=1, seq_len=64)
        smax = []
        for block in bg.values():
            _walk_nodes(
                block["nodes"],
                lambda n: n.get("op") == "softmax_act"
                and n.get("label") == "smax" and smax.append(n))
        self.assertGreater(len(smax), 0)
        for n in smax:
            self.assertIn("window axis", n["op_formula"], n.get("id"))
            self.assertIn("axis: 2", n["op_formula"], n.get("id"))

    def test_dsv4_compressor_window_is_a_real_axis(self):
        """The thorough window modeling: gate_win folds the sequence into
        (slot, window), the softmax/weighted pair runs in that rank-4
        frame, and sum_reduce collapses the window axis back — the
        pooling is explicit in the graph, not a caption."""
        _cfg, _patterns, r = _load("deepseek-ai__DeepSeek-V4-Flash.json")
        self.assertIsNotNone(r)
        bg, _ = build_block_graph_data(r, batch_size=1, seq_len=4096)
        nodes: list = []
        for block in bg.values():
            _walk_nodes(block["nodes"], nodes.append)
        folds = [n for n in nodes if n.get("op") == "view_split"
                 and n.get("label") in ("gate_win", "kv_win")]
        sums = [n for n in nodes if n.get("op") == "sum_reduce"
                and n.get("label") == "pooled"]
        self.assertGreater(len(folds), 0)
        self.assertGreater(len(sums), 0)
        for n in folds:
            self.assertEqual(n["style"], "view", n.get("id"))
            self.assertEqual(n["tensor_out"].count(","), 3, n.get("id"))  # rank-4
        for n in sums:
            # [1, S/m, m, D] -> [1, S/m, D]: rank 3, sequence already pooled.
            self.assertEqual(n["tensor_out"].count(","), 2, n.get("id"))
            self.assertNotEqual(n["tensor_in"], n["tensor_out"], n.get("id"))


    def test_an_activation_rule_reaches_its_roled_leaves(self):
        from backend.arch.eval.precision import PrecisionProfile
        _, _patterns, r = _load("deepseek-ai__DeepSeek-V3.json")
        self.assertIsNotNone(r)
        bg, _ = build_block_graph_data(
            r, batch_size=1, seq_len=1,
            precision=PrecisionProfile.canonicalize({"activations": [
                {"select": {"roles": ["ffn_moe"]},
                 "format": {"dtype": "F32"}}]}),
        )
        moe_hit = False
        other_hit = False

        def _check(n):
            nonlocal moe_hit, other_hit
            # `ffn_moe_shared` refines `ffn_moe`: the rule reaches it.
            in_moe = "ffn_moe" in roles.lineage(n.get("role"))
            if in_moe and n.get("out_bytes"):
                moe_hit = True
                self.assertEqual(n.get("act_dtype_bytes"), 4.0, n.get("id"))
            elif n.get("role") and not in_moe and n.get("out_bytes"):
                other_hit = True
                self.assertEqual(n.get("act_dtype_bytes"), 2.0, n.get("id"))

        for block in bg.values():
            _walk_nodes(block["nodes"], _check)
        self.assertTrue(moe_hit, "no ffn_moe leaf picked up the act override")
        self.assertTrue(other_hit, "override leaked onto every roled leaf")

    def test_first_node_is_virtual_input(self):
        """Block graphs prepend a virtual input node with shape stamped."""
        _, _patterns, r = _load("Qwen__Qwen3-8B.json")
        bg, _ = build_block_graph_data(
            r, batch_size=1, seq_len=1,
        )
        first = bg["block"]["nodes"][0]
        self.assertEqual(first["op"], "input")
        self.assertEqual(first["style"], "io")
        self.assertTrue(first.get("is_source"))
        self.assertIn("tensor_out", first)


class TestArchitectureView(unittest.TestCase):
    def _badges(self, snap_name: str):
        # The analyzer's summary, as `/analyze` passes it: the quant badge
        # reads its `weight_methods` (the analysis' bytes per scheme).
        from backend.analyzer import analyze_model
        cfg, _, r = _load(snap_name)
        summary = analyze_model(
            model_id=snap_name.removesuffix(".json").replace("__", "/", 1))["summary"]
        av = build_architecture_view(
            summary, [], r, lambda x: f"{x:,}", cfg,
        )
        return av["badges"]

    def _layers(self, snap_name: str):
        return self._view(snap_name)["layers"]

    def _view(self, snap_name: str):
        from backend.arch.eval.program import Program
        cfg, _, r = _load(snap_name)
        summary = {
            "attention_info": {"attention_type": "GQA"},
            "ffn_info": {"ffn_type": "moe", "num_experts": 0, "top_k": 0},
            "hidden_size": cfg.get("hidden_size", 0),
            "vocab_size": Program.of(r).headline.vocab,
            "num_layers": cfg.get("num_hidden_layers", 0),
            "total_params": 0,
        }
        return build_architecture_view(
            summary, [], r, lambda x: f"{x:,}", cfg,
        )

    def test_the_output_head_card_names_its_vocab_whatever_it_is_called(self):
        """The declared output head (`boundary_role: output_head`) shows the
        model's published vocab (`Headline.vocab`) whatever the layer is
        called: GLM-4's head is `transformer.output_layer` and once showed
        none, because the card looked for "head" in the name."""
        from backend.arch.eval.program import Program
        snap = "THUDM__glm-4-9b.json"
        vocab = Program.of(_load(snap)[2]).headline.vocab
        self.assertEqual(vocab, 151_552)  # config.padded_vocab_size
        heads = [l for l in self._layers(snap) if l["type"] == "lm_head"]
        self.assertEqual([(h["name"], h["detail"]) for h in heads],
                         [("transformer.output_layer", f"vocab = {vocab:,}")])

    def _incoming(self, view, name):
        """The names of the cards an edge carries into `name`."""
        by_name = {l["name"]: l for l in view["layers"]}
        return [next(n for n, l in by_name.items() if l["id"] == e["from"])
                for e in view["edges"]
                if e["to"] == by_name[name]["id"]]

    def test_every_model_level_op_leaf_has_a_card(self):
        """A model-level op leaf that is none of the named kinds is a card of
        its own kind — not a second `lm_head`, not absent. Kimi-K3's output
        AttnRes combine and V4-Pro's hc_head are the corpus's two; V4-Pro's
        hyper-connection broadcast recurs with the layers and folds into the
        cycle container. A zero-input leaf (the combine's scoring weights,
        V4-Pro's hc_head parameters, Llama-2's rotary buffer) is an operand
        of a card, not a card."""
        k3_view = self._view("moonshotai__Kimi-K3.json")
        k3 = k3_view["layers"]
        by_name = {l["name"]: l for l in k3}
        combine = by_name[
            "language_model.model.output_attn_res (attn_res_combine)"]
        self.assertEqual(combine["type"], "op")
        # Edges name cards: both of the combine's streams come from the
        # block stack's card (`block_mla_moe` twice — layers and
        # block_residual), and the zero-input scoring weights are operands,
        # not cards, so no edge names them.
        self.assertEqual(self._incoming(k3_view, combine["name"]),
                         ["block_mla_moe", "block_mla_moe"])
        self.assertEqual(
            [n for n in by_name if "output_attn_res_" in n], [])
        self.assertEqual(
            self._incoming(k3_view, "language_model.model.norm"),
            [combine["name"]])
        self.assertEqual([l["name"] for l in k3 if l["type"] == "lm_head"],
                         ["language_model.lm_head"])
        v4 = {l["name"]: l["type"] for l in self._layers("deepseek-ai__DeepSeek-V4-Pro.json")}
        self.assertIsNone(v4.get("hc_expand (broadcast)"))
        self.assertIn("× 30 cycles", v4)
        self.assertEqual(v4.get("hc_head (hc_head)"), "op")
        self.assertEqual([n for n in v4 if n.startswith("hc_head_")], [])
        llama = [l["name"] for l in self._layers("meta-llama__Llama-2-7b-hf.json")]
        self.assertEqual([n for n in llama if "rotary" in n], [])

    def test_kda_block_cards_name_the_variant_not_the_cache_family(self):
        """KDA's recurrent state is GatedDeltaNet-shaped, but architecture
        cards name the user-facing variant (`KDA`) — cache implementation
        family is a separate fact."""
        layers = self._layers("moonshotai__Kimi-K3.json")
        kda = next(x for x in layers if x.get("template") == "block_kda_moe")
        self.assertIn("KDA", kda["detail"])
        self.assertNotIn("GatedDeltaNet", kda["detail"])

    def test_a_hand_written_adapter_is_named_by_the_wrapper_it_binds(self):
        """Qwen3-Coder-Next's linear-attention layer is a hand-written
        one-node adapter (`qwen3_next_linear_moe_block`: `pre_norm` binding
        a GatedDeltaNet attention and a MoE FFN), not loader-synthesized.
        The block graph names the wrapper and its bindings, and the
        architecture card and the per-op tables show the short name the
        bindings give, one name on every page."""
        from backend.arch.eval.program import Program
        _, _, r = _load("Qwen__Qwen3-Coder-Next.json")
        adapter = "qwen3_next_linear_moe_block"
        graph, _ = build_block_graph_data(r, batch_size=1, seq_len=1)
        self.assertEqual(graph[adapter]["template"], "pre_norm")
        self.assertEqual(graph[adapter]["params"],
                         {"attn": "gated_delta_net_fused_attention",
                          "ffn": "moe_ffn_scaled_shared_gate"})
        program = Program.of(r)
        (template, *_) = [t for t in program.blocks if t.name == adapter]
        self.assertEqual(program.block_label(template),
                         "GatedDeltaNet + MoE FFN")
        names = {b["name"] for card in self._layers("Qwen__Qwen3-Coder-Next.json")
                 for b in card.get("cycle_blocks", [card])
                 if b.get("template") == adapter}
        self.assertEqual(names, {"GatedDeltaNet + MoE FFN"})

    def test_a_block_is_named_by_what_its_arguments_name(self):
        """A bound argument reads as what it names, through nested wrappers:
        DeepSeek-V3's MTP block wraps a synthesized block and reads as what
        it calls, not its wrapper's or its generated template id. And two
        templates of one stack never share a name: DeepSeek-V4-Pro's hash-
        and score-routed MoE are both "CSA + MoE FFN", so where each runs
        tells them apart — never a component id."""
        from backend.arch.eval.program import Program

        def labels(snapshot):
            _, _, r = _load(snapshot)
            program = Program.of(r)
            return {t.name: program.block_label(t) for t in program.blocks}

        v3 = labels("deepseek-ai__DeepSeek-V3.json")
        mtp = next(n for n in v3 if n.startswith("mtp_block_with_head"))
        self.assertEqual(v3[mtp], "MLA + MoE FFN")
        v4 = labels("deepseek-ai__DeepSeek-V4-Pro.json")
        backbone = [v for n, v in v4.items() if n.startswith("dsv4_block__")]
        self.assertEqual(len(set(backbone)), len(backbone), backbone)
        for v in backbone:
            self.assertNotIn("__", v)
            self.assertNotIn("dsv4_moe", v)
        self.assertIn("CSA + MoE FFN · layer 2", backbone)

    def test_hybrid_stacks_lay_out_in_layer_order(self):
        """An `assignments:` stack shows its layers in index order, a
        periodic interleave folded into one cycle card: gpt-oss alternates
        sliding-window and full attention over all 24 layers; MiMo-V2.5's
        vision tower puts a full-attention block every 9th of 28, so three
        whole cycles cover 0–26 and layer 27 follows on its own."""
        cycles = [c for c in self._layers("openai__gpt-oss-20b.json")
                  if c["type"] == "cycle_container"]
        self.assertEqual(
            [(c["cycle_period"], c["cycle_count"], c["cycle_layer_range"])
             for c in cycles], [(2, 12, [0, 23])])
        from backend.arch.eval.program import Program
        cfg, _, r = _load("XiaomiMiMo__MiMo-V2.5.json")
        # The published vocab, as the analyzer passes it: the head card
        # refuses a summary without one.
        summary = {"attention_info": {}, "ffn_info": {"top_k": 0,
                                                      "num_experts": 0},
                   "hidden_size": 0,
                   "vocab_size": Program.of(r).headline.vocab,
                   "num_layers": 0,
                   "total_params": 0}
        view = build_architecture_view(summary, [], r, lambda x: f"{x:,}",
                                       cfg)
        (visual,) = [s for s in view["extra_sections"]
                     if s["name"].startswith("visual ")]
        blocks = [c for c in visual["sub_layers"] if c["type"] != "op"]
        self.assertEqual(
            [(c["type"], c.get("cycle_period"), c.get("cycle_layer_range"),
              c.get("template")) for c in blocks],
            [("cycle_container", 9, [0, 26], None),
             ("dense_block", None, None, "mimo_v2_5_vision_block_full")])
        # Kimi-K3's MLA layer recurs every 4, but every third stretch opens
        # with `block_kda_moe_start`: the cycle is 12 long, and every layer
        # it covers is the block its card names.
        layers = self._layers("moonshotai__Kimi-K3.json")
        (cycle,) = [c for c in layers if c["type"] == "cycle_container"]
        self.assertEqual((cycle["cycle_period"], cycle["cycle_layer_range"]),
                         (12, [3, 86]))
        self.assertIn("block_kda_moe_start",
                      [b["template"] for b in cycle["cycle_blocks"]])

    def test_qwen3_dense_gqa_badge(self):
        # Qwen3 is dense GQA — single attention badge, no MoE / no quant
        self.assertEqual(self._badges("Qwen__Qwen3-8B.json"), ["GQA"])

    def test_dsv3_moe_mla_fp8(self):
        b = self._badges("deepseek-ai__DeepSeek-V3.json")
        self.assertIn("MoE", b)
        self.assertIn("MLA", b)
        self.assertIn("FP8", b)
        # MTP badge from aux_kind: mtp on the spec's group layer
        self.assertIn("MTP", b)

    def test_kimi25_w4_compressed_tensors(self):
        # compressed-tensors INT4 → "W4" via quant_schemes/compressed-tensors.yaml
        b = self._badges("moonshotai__Kimi-K2.5.json")
        self.assertIn("W4", b)
        self.assertIn("MLA", b)

    def test_gpt_oss_hybrid_mxfp4(self):
        # gpt-oss has hybrid GQA+SWA + MXFP4 quant
        b = self._badges("openai__gpt-oss-20b.json")
        self.assertIn("Hybrid", b)
        self.assertIn("GQA", b)
        self.assertIn("SWA", b)
        self.assertIn("MXFP4", b)

    def test_aux_kind_badge_covers_every_schema_kind(self):
        """`_AUX_KIND_BADGE` keys are the schema's aux_kind Literal — a
        fifth kind added there must get a badge here, not vanish."""
        import typing

        from backend.arch.schema import ModelLayer
        from backend.rendering._badges import _AUX_KIND_BADGE

        ann = ModelLayer.model_fields["aux_kind"].annotation
        lit = next(a for a in typing.get_args(ann) if a is not type(None))
        self.assertEqual(set(typing.get_args(lit)), set(_AUX_KIND_BADGE))

    def test_quant_badge_reads_spec_not_config(self):
        """The quant badge is spec.quant_context, even with no HF config."""
        from backend.rendering._badges import collect_badges
        _, _, r = _load("deepseek-ai__DeepSeek-V3.json")
        badges, _ = collect_badges(
            r,
            {"attention_info": {"attention_type": "MLA"},
             "weight_methods": [{"method": "fp8", "bytes": 1}]},
            None,
            fallback_attn_type="MLA",
        )
        self.assertIn("FP8", badges)

    def test_a_mixed_checkpoint_is_badged_by_its_largest_scheme(self):
        """DeepSeek-V4 lists FP8 attention first and holds ~96% of its weight
        bytes in MXFP4 experts (`mxfp4-dot-suffix` since 2d17bec2c): the badge
        is the scheme with the most bytes."""
        from backend.rendering._badges import collect_badges
        _, _, r = _load("deepseek-ai__DeepSeek-V4-Pro.json")
        badges, _ = collect_badges(
            r,
            {"weight_methods": [{"method": "mxfp4-dot-suffix", "bytes": 786},
                                {"method": "fp8_ue8m0", "bytes": 23}]},
            None,
            fallback_attn_type="MLA",
        )
        self.assertIn("MXFP4", badges)
        self.assertNotIn("FP8", badges)


    def test_a_block_with_no_attention_names_none(self):
        """Nemotron's MoE layers make no attention call: their cards carry no
        attention label, not the headline's (`GQA` in `_layers`)."""
        cards = []

        def collect(items):
            for card in items:
                cards.append(card)
                collect(card.get("cycle_blocks") or [])
        collect(self._layers("nvidia__NVIDIA-Nemotron-3-Nano-30B-A3B-BF16.json"))
        moe = [c for c in cards if c.get("template") == "nemotron_moe_block"]
        self.assertTrue(moe)
        for card in moe:
            self.assertNotIn("GQA", card.get("detail", ""))
            self.assertNotIn("Mamba2", card.get("detail", ""))

class TestModelConfig(unittest.TestCase):
    def test_public_layer_configs_cover_every_snapshot(self):
        """`layer_configs` (the one projection, since the engine twin
        retired) yields a non-empty, well-formed group list for every
        resolvable snapshot; the two MoE models it historically corrected
        stay corrected."""
        loader = shared_loader()
        corrected = {
            "Qwen__Qwen3-235B-A22B.json",
            "Qwen__Qwen3-30B-A3B.json",
        }
        for path in sorted(glob.glob(str(SNAPSHOTS_DIR / "*.json"))):
            with self.subTest(snapshot=Path(path).name):
                config = json.loads(Path(path).read_text())["config"]
                resolved = loader.match(config, model_id=Path(path).stem.replace("__", "/", 1))
                if resolved is None:
                    continue
                projected = layer_configs(resolved)
                self.assertTrue(projected)
                for item in projected:
                    self.assertIn("count", item)
                    self.assertIn("attn_display", item)
                if Path(path).name in corrected:
                    self.assertTrue(all(
                        item["ffn_type"] == "moe" for item in projected
                    ))

    def _model_config(self, snap_name: str) -> dict:
        _, _, r = _load(snap_name)
        # The analyzer states the expert width of every MoE model, and the
        # attention families of its walk.
        from backend.arch.eval.program import Program
        from backend.arch.eval.transforms import Deployment
        analysis = Deployment(Program.of(r)).analyze()
        summary = {"attention_info": {
                       "families": [family.as_dict()
                                    for family in analysis.attention_families]},
                   "ffn_info": {"ffn_type": "moe",
                                "expert_intermediate_size": None}}
        return build_model_config(summary, r)

    def test_dense_layers_are_the_declared_ones(self):
        # first_k_dense_replace = 3.
        mc = self._model_config("deepseek-ai__DeepSeek-V3.json")
        self.assertEqual(mc["ffn"]["dense_layers"], [0, 1, 2])
        # moe_layers_enum = 3, 4, …, 44.
        mc = self._model_config("stepfun-ai__Step-3.7-Flash.json")
        self.assertEqual(mc["ffn"]["dense_layers"], [0, 1, 2])
        # Nemotron has none: its attention and Mamba layers are not dense
        # FFN layers.
        mc = self._model_config("nvidia__NVIDIA-Nemotron-3-Nano-30B-A3B-BF16.json")
        self.assertEqual(mc["ffn"]["dense_layers"], [])

    def test_the_attention_card_names_each_familys_dims_by_config(self):
        # Each dim is named by the spec's binding of it, as authored — the
        # whole `??` chain, not the alternative a re-read of the config
        # would guess the resolution took.
        mc = self._model_config("deepseek-ai__DeepSeek-V3.json")
        (mla,) = mc["attention"]["families"]
        facts = {d["symbol"]: (d["config"], d["value"]) for d in mla["dims"]}
        self.assertEqual(mla["name"], "MLA")
        # The V width is a fact of MLA, shown whatever the other widths.
        self.assertEqual(facts["d_v_head"],
                         ("config.v_head_dim ?? d_qk_nope", 128))
        self.assertEqual(facts["d_kv_lora"], ("config.kv_lora_rank", 512))
        self.assertEqual(facts["d_qk_nope"],
                         ("config.qk_nope_head_dim", 128))
        self.assertNotIn("n_kv_heads", facts)
        # The config's num_key_value_heads, read by no MLA call.
        self.assertEqual(
            [(d["symbol"], d["config"], d["value"])
             for d in mc["attention"]["model_dims"]],
            [("n_kv_heads", "config.num_key_value_heads", 128)])

    def test_config_dims_are_the_authored_bindings_at_the_resolved_values(self):
        # Llama-2 publishes no `head_dim`: the row is the binding as written
        # and the value the one resolution bound through its fallback — a
        # row a re-read of the config's first path would have dropped.
        mc = self._model_config("meta-llama__Llama-2-7b-hf.json")
        rows = {row["symbol"]: (row["field"], row["value"])
                for row in mc["config_dims"]}
        self.assertEqual(rows["d_head"],
                         ('config.head_dim ?? "d_model // n_heads"', 128))
        self.assertEqual(rows["d_model"], ("config.hidden_size", 4096))

    def test_a_hybrids_kv_heads_are_its_linear_attentions(self):
        mc = self._model_config("inclusionAI__Ling-2.6-1T.json")
        families = {f["name"]: {d["symbol"]: (d["config"], d["value"])
                                for d in f["dims"]}
                    for f in mc["attention"]["families"]}
        self.assertEqual(families["GLA-Linear"]["n_kv_heads"],
                         ("config.num_kv_heads_for_linear_attn ?? n_heads", 64))
        self.assertNotIn("n_kv_heads", families["MLA"])
        # V width equal to the linear attention's head width, still shown.
        self.assertEqual(families["MLA"]["d_v_head"],
                         ("config.v_head_dim ?? d_qk_nope", 128))

    def test_block_types_name_only_what_a_block_runs(self):
        mc = self._model_config("nvidia__NVIDIA-Nemotron-3-Nano-30B-A3B-BF16.json")
        rows = {row["template"]: row for row in mc["block_types"]}
        self.assertEqual(
            {name: (row["attention_type"], row["ffn_type"])
             for name, row in rows.items()},
            {"nemotron_attention_block": ("GQA", None),
             "nemotron_mamba_block": ("Mamba2", None),
             "nemotron_moe_block": (None, "moe")})

    def test_mtp_layers_are_the_mtp_stack_and_not_the_backbone(self):
        mc = self._model_config("nvidia__NVIDIA-Nemotron-3-Ultra-550B-A55B-BF16.json")
        self.assertEqual(sum(row["count"] for row in mc["block_types"]), 108)
        self.assertEqual(mc["ffn"]["mtp_layers"], 2)

    def test_qwen3_precision_bf16(self):
        cfg, _patterns, r = _load("Qwen__Qwen3-8B.json")
        summary = {
            "attention_info": {}, "ffn_info": {"ffn_type": "dense"},
            "hidden_size": cfg["hidden_size"],
            "vocab_size": cfg["vocab_size"],
            "num_layers": cfg["num_hidden_layers"],
        }
        mc = build_model_config(summary, r)
        # precision falls back to torch_dtype when no patterns passed in
        self.assertIn("precision", mc)

    def test_precision_names_the_format_not_the_packed_container(self):
        # Kimi-K2.5 packs INT4 into I32, DeepSeek-V4-Pro MXFP4 (2d17bec2c)
        # into I8: the precision is the analysis' weight formats, not header
        # dtypes.
        from backend.analyzer import analyze_model
        from backend.rendering._quant import resolve_precision
        from tests.snapshots import resolve_snapshot
        for model, lead, scale in (
                ("moonshotai/Kimi-K2.5", "INT4", "bf16"),
                ("deepseek-ai/DeepSeek-V4-Pro", "MXFP4", "ue8m0"),
                ("deepseek-ai/DeepSeek-V3.1", "F8_E4M3", "fp32")):
            resolve_snapshot(model)
            summary = analyze_model(model_id=model)["summary"]
            self.assertEqual(summary["weight_dtype"], lead, model)
            p = resolve_precision(summary)
            self.assertTrue(p.startswith(f"{lead} "), (model, p))
            self.assertIn(f"(scale: {scale})", p)
            self.assertNotIn("I8", p)
            self.assertNotIn("I32", p)

    def test_precision_of_a_single_format_is_its_name(self):
        summary = {"weight_formats": [{"dtype": "BF16", "bytes": 10**9},
                                      {"dtype": "F32", "bytes": 10}]}
        from backend.rendering._quant import resolve_precision
        self.assertEqual(resolve_precision(summary), "BF16")
        self.assertIsNone(resolve_precision({}))

if __name__ == "__main__":
    unittest.main()
