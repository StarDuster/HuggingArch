"""The architecture view's topology: every read between model layers
(`Program.layer_reads`) is an edge between the cards that draw them, so the
frontend lays the overview out as the graph it is rather than a list.
"""
import json
import sys
import unittest

import pytest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

SNAPSHOTS_DIR = PROJECT_ROOT / "backend" / "arch" / "snapshots"

from tests.snapshots import shared_loader
from backend.rendering import build_architecture_view
from tools.corpus_pins.checks.arch_view_graph import ARCH_VIEW_GRAPH
from tools.corpus_pins.corpus import Corpus


def _format_params(n):
    return f"{n / 1e9:.1f}B" if n >= 1e9 else f"{n / 1e6:.0f}M"


def _load_resolved_and_config(snap_name: str):
    snap = json.loads((SNAPSHOTS_DIR / snap_name).read_text())
    cfg = snap["config"]
    model_id = Path(snap_name).stem.replace("__", "/", 1)
    return shared_loader().match(cfg, model_id=model_id), cfg


def _minimal_summary(model_type: str, hidden_size: int, vocab_size: int, num_layers: int):
    """`build_architecture_view` reads dims + attention/ffn descriptors
    off summary. The rendering paths under test don't depend on
    flops/params/dtype numbers — those flow into the dashboard, not the
    aux-section inputs propagation — so stub the minimum it touches."""
    return {
        "model_type": model_type,
        "hidden_size": hidden_size,
        "vocab_size": vocab_size,
        "num_layers": num_layers,
        "total_params": 0,
        "total_active_params": 0,
        "num_dense_ffn_layers": 0,
        "num_moe_layers": 0,
        "attention_info": {"attention_type": "MLA"},
        "ffn_info": {"ffn_type": "moe", "num_experts": 0, "top_k": 0},
        "architectures": [model_type],
    }


def _view(snap: str, model_type: str, hidden: int, vocab: int, layers: int):
    resolved, config = _load_resolved_and_config(snap)
    assert resolved is not None
    return build_architecture_view(
        _minimal_summary(model_type, hidden, vocab, layers),
        layer_groups=[], resolved_spec=resolved,
        format_params=_format_params, config=config)


def _named_edges(view):
    """`(from name, to name, kind)` per edge of `view`."""
    names = {m["id"]: m["name"] for m in view["model_inputs"]}
    names.update({card["id"]: card["name"] for card in view["layers"]})
    names.update({s["id"]: s["name"] for s in view["extra_sections"]})
    return {(names[e["from"]], names[e["to"]], e["kind"]): e
            for e in view["edges"]}


class TestViewTopology(unittest.TestCase):
    def test_dsv3_mtp_section_reads_the_final_norm(self):
        """DSv3's `mtp` group declares `inputs: [norm]` — the MTP head's
        backbone hidden_states source: an edge from the norm card into the
        MTP section, not a list position."""
        view = _view("deepseek-ai__DeepSeek-V3.json", "deepseek_v3", 7168, 129280, 61)
        edges = _named_edges(view)
        (mtp,) = [s["name"] for s in view["extra_sections"]
                  if s["name"].startswith("mtp")]
        self.assertIn(("norm", mtp, "data"), edges)

    def test_kimi_k25_embed_merge_reads_the_tower(self):
        """Kimi K2.5's `embed_merge` reads the text embedding and the vision
        tower: an edge from each, the tower's from its section."""
        view = _view("moonshotai__Kimi-K2.5.json", "kimi_k25", 7168, 163840, 61)
        edges = _named_edges(view)
        (merge,) = [c["name"] for c in view["layers"] if "embed_merge" in c["name"]]
        (vision,) = [s["name"] for s in view["extra_sections"]
                     if s["name"].startswith("vision")]
        self.assertIn(("language_model.model.embed_tokens", merge, "data"), edges)
        self.assertIn((vision, merge, "data"), edges)
        self.assertIn(("pixel_values", vision, "data"), edges)

    def test_dsv41_draws_kinds_streams_and_no_views(self):
        """DeepSeek-V4.1: a card per run of one block kind — layer 0, the
        Engram layer 1, CSA 2-13, the Engram layer 14, CSA 15-39 — the CSA
        layers that own the shared KV and those that borrow it variants of
        one card, the Engram wrapper a kind of its own; the hyper-connection
        views (`hc_expand`, `hc_flat`, …) draw no card; `input_ids` reaches
        only the Engram cards, whose templates read it; the seed, the final
        `hc_pre_mix`, a read of layer 14's compressed KV from the next card
        and the MTP's backbone taps are stream edges."""
        view = _view("deepseek-ai__DeepSeek-V4.1-Flash.json", "deepseek_v41",
                     5120, 129280, 40)
        names = [c["name"] for c in view["layers"]]
        self.assertFalse([n for n in names if "view_" in n or "broadcast" in n],
                         names)
        blocks = [c for c in view["layers"] if c["type"] == "moe_block"]
        self.assertEqual([[0], [1], list(range(2, 14)), [14], list(range(15, 40))],
                         [c["layers"] for c in blocks])
        swa, swa_engram, csa, csa_engram, csa_tail = blocks
        self.assertEqual("SWA + MoE FFN · layer 1", swa_engram["name"])
        self.assertIn("dsv41_hc_block_engram", swa_engram["template"])
        self.assertEqual("CSA + MoE FFN · layer 14", csa_engram["name"])
        self.assertIn("dsv41_hc_block_engram", csa_engram["template"])
        self.assertNotIn("engram", csa["name"])
        self.assertTrue(all("dsv41_hc_block_engram" not in variant["template"]
                            for variant in csa["variants"]))
        self.assertEqual(2, len(csa["variants"]))
        self.assertTrue(all(v["template"] for v in csa_tail["variants"]))
        edges = _named_edges(view)
        merge = "embed_merge (scatter_rows_replace)"
        self.assertIn((merge, swa["name"], "data"), edges)
        self.assertIn((swa["name"], swa_engram["name"], "data"), edges)
        readers = {to for (frm, to, _kind) in edges if frm == "input_ids"}
        self.assertIn(swa_engram["name"], readers)
        self.assertIn(csa_engram["name"], readers)
        self.assertNotIn(swa["name"], readers)
        seed = edges[("hc_seed (one_hot)", swa["name"], "stream")]
        self.assertEqual(["hc_pre_mix"], seed["streams"])
        self.assertIn("compress_kv",
                      edges[(csa_engram["name"], csa_tail["name"], "stream")]["streams"])
        exit_ = "hc_exit (hc_weighted_reduce)"
        self.assertIn((csa_tail["name"], exit_, "data"), edges)
        self.assertIn((csa_tail["name"], exit_, "stream"), edges)
        (mtp,) = [s["name"] for s in view["extra_sections"] if s["name"].startswith("mtp")]
        taps = edges[(csa_tail["name"], mtp, "stream")]
        self.assertEqual({"main_hidden_37", "main_hidden_38", "main_hidden_39"},
                         set(taps["streams"]))
        self.assertIn(("embed", mtp, "data"), edges)

    def test_dsv41_csa_card_maps_who_reads_whose_stream(self):
        """Inside a CSA card the layers read each other's state: the
        KV-owning layers 20 write `compress_kv` and each later layer reads
        it; the index-key owners 24..36 read `index_k` from layer 20, and
        the first layers read their `hc_pre_mix` from the Engram layer
        before the card. The card's `stream_map` carries that layer by
        layer."""
        view = _view("deepseek-ai__DeepSeek-V4.1-Flash.json", "deepseek_v41",
                     5120, 129280, 40)
        (tail,) = [c for c in view["layers"] if c.get("layers", [None])[0] == 15]
        rows = {row["name"]: {c["layer"]: c for c in row["cells"]}
                for row in tail["stream_map"]}
        kv = rows["compress_kv"]
        self.assertEqual([20], [layer for layer, c in kv.items() if c["writes"]])
        self.assertEqual(14, kv[15]["from"])
        self.assertEqual(20, kv[39]["from"])
        self.assertEqual({24: 20, 28: 20, 32: 20, 36: 20},
                         {layer: c["from"] for layer, c in rows["index_k"].items()
                          if not c["writes"]})
        self.assertEqual(set(range(15, 40)),
                         {l for v in tail["variants"] for l in v["layers"]})

    def test_a_variant_says_what_tells_it_from_the_others_on_its_card(self):
        """A variant row names the slots its card's variants bind
        differently, each with the component there and that component's
        own declared description: DSv4.1's layer 20 owns its compressed KV
        (`attn.compressed`) and is the candidate source (`attn.indexer`),
        where 24..36 borrow the KV and re-score shared keys."""
        view = _view("deepseek-ai__DeepSeek-V4.1-Flash.json", "deepseek_v41",
                     5120, 129280, 40)
        (tail,) = [c for c in view["layers"] if c.get("layers", [None])[0] == 15]
        by_layers = {tuple(v["layers"]): {s["slot"]: s for s in v["slots"]}
                     for v in tail["variants"]}
        source = by_layers[(20,)]
        self.assertEqual("csa2_compress_store", source["attn.compressed"]["component"])
        self.assertEqual("csa2_indexer_candidate_source",
                         source["attn.indexer"]["component"])
        self.assertTrue(all(s["description"] for s in source.values()))
        borrowers = by_layers[(24, 28, 32, 36)]
        self.assertEqual("csa2_compress_borrowed",
                         borrowers["attn.compressed"]["component"])
        # `ffn` binds one component on every variant: nothing to tell apart.
        self.assertNotIn("ffn", source)

    def test_an_input_no_layer_consumes_draws_no_edge(self):
        """DeepSeek-V4-Flash hands `input_ids` to every layer and to its MTP
        group; only its hash-routed layers (0-2: the SWA card and the cycle
        that starts at 2) consume it, not the score-routed layer 42 nor the
        MTP block, so neither of those is an `input_ids` reader."""
        view = _view("deepseek-ai__DeepSeek-V4-Flash.json", "deepseek_v4", 1, 1000, 1)
        readers = {to for (frm, to, _kind) in _named_edges(view) if frm == "input_ids"}
        (mtp,) = [s["name"] for s in view["extra_sections"] if s["name"].startswith("mtp")]
        self.assertNotIn(mtp, readers)
        self.assertEqual({"embed", "SWA + MoE FFN", "× 20 cycles"}, readers)

    def test_a_stream_no_layer_reads_from_another_draws_no_map(self):
        """GLM-5.1's layers each write `dsa_topk` for themselves and none
        reads another's: its cards relate no two layers, so they carry no
        stream map. GLM-5.2's every fourth layer writes it for the three
        after: its MoE card maps that."""
        flat = _view("zai-org__GLM-5.1.json", "glm", 1, 1000, 1)
        self.assertFalse([c["name"] for c in flat["layers"] if c.get("stream_map")])
        shared = _view("zai-org__GLM-5.2.json", "glm", 1, 1000, 1)
        (card,) = [c for c in shared["layers"] if c.get("stream_map")]
        ((row,),) = [card["stream_map"]]
        reads = {c["layer"]: c["from"] for c in row["cells"] if "from" in c}
        self.assertEqual({7: 6, 8: 6, 9: 6}, {k: reads[k] for k in (7, 8, 9)})

    def test_every_edge_joins_two_drawn_boxes(self):
        """No edge names a box the view does not draw."""
        view = _view("deepseek-ai__DeepSeek-V4.1-Flash.json", "deepseek_v41",
                     5120, 129280, 40)
        ids = ({m["id"] for m in view["model_inputs"]}
               | {c["id"] for c in view["layers"]}
               | {s["id"] for s in view["extra_sections"]})
        for edge in view["edges"]:
            self.assertIn(edge["from"], ids)
            self.assertIn(edge["to"], ids)


@pytest.mark.regression  # every snapshot
@pytest.mark.parametrize("model", Corpus.models(), ids=lambda m: m.stem)
def test_every_view_is_a_connected_graph(model):
    """On every snapshot (C7, `tools/corpus_pins/checks/arch_view_graph`):
    each edge joins two boxes the view draws, and every card and section is
    read into by something — a box with no incoming edge would be a layer
    the view lost the reads of."""
    assert ARCH_VIEW_GRAPH.violations(model) == []


if __name__ == "__main__":
    unittest.main()
