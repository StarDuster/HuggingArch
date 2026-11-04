"""C7: every model's architecture overview is a connected graph.

The Inspector lays the overview out from its `edges` (`build_architecture_view`):
an edge naming a box the view does not draw is a dangling arrow, and a card
or section no edge enters is a layer whose reads the view lost — both have
happened while the view's edges were being derived from `Program.layer_reads`.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List

from tools.corpus_pins.check import CheckKind, CorpusCheck
from tools.corpus_pins.corpus import CorpusModel

#: What the view reads off the summary; none of it decides an edge.
_SUMMARY = {
    "model_type": "x", "hidden_size": 1, "vocab_size": 1000, "num_layers": 1,
    "total_params": 0, "total_active_params": 0, "num_dense_ffn_layers": 0,
    "num_moe_layers": 0, "attention_info": {"attention_type": "MLA"},
    "ffn_info": {"ffn_type": "moe", "num_experts": 0, "top_k": 0},
    "architectures": ["x"],
}


def view_problems(resolved: Any, config: Any) -> List[str]:
    """The dangling edges and unread boxes of one model's overview."""
    from backend.rendering import build_architecture_view
    view = build_architecture_view(_SUMMARY, [], resolved, str, config)
    ids = ({m["id"] for m in view["model_inputs"]}
           | {c["id"] for c in view["layers"]}
           | {s["id"] for s in view["extra_sections"]})
    out = [f"dangling edge {e['from']} -> {e['to']}" for e in view["edges"]
           if e["from"] not in ids or e["to"] not in ids]
    read = {e["to"] for e in view["edges"]}
    out += [f"unread box {box['name']}"
            for box in (*view["layers"], *view["extra_sections"])
            if box["id"] not in read]
    return out


@dataclass(frozen=True)
class ArchViewGraphCheck(CorpusCheck):
    def violations(self, model: CorpusModel) -> List[str]:
        return view_problems(model.resolved(), model.config)


ARCH_VIEW_GRAPH = ArchViewGraphCheck(
    id="C7",
    title="the architecture overview is a connected graph",
    kind=CheckKind.ADMIT,
    owner_tests=(
        "tests/backend/arch/test_arch_view_aux_inputs.py::"
        "test_every_view_is_a_connected_graph",
    ),
)
