"""Leaf input tensor identities — the one place the activation-liveness edge
join is made.

`backend.inference.activation_peak` reads these instead of re-walking each
leaf's `inputs_obj` itself: eval is the producer of the edge → producer
resolution (traffic prices the same edges for bytes; this names them). The
resolution redirects a producer absorbed into a fused node to that node's
published port, and keeps an alias id for a ref that names an enclosing scope's
input.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from .launch import FusedNode
from .layer import Rewrite


def tensor_id(uid, port: str):
    """Stable identity for one node output tensor. Main output is the uid; a
    named port is `uid#port=` so liveness never rebuilds a path."""
    return uid if not port or port == "out" else f"{uid}#port={port}"


def leaf_input_ids(program, rewrite: Optional[Rewrite] = None) -> Dict[int, List]:
    """Per leaf uid → its input tensor identities.

    Each input's producer is resolved through fusion absorption (an absorbed
    producer redirects to the fused node's published port). A ref whose `src`
    is None names an enclosing scope's input and keeps its `external_sid`.
    """
    rewrite = rewrite or Rewrite()

    absorbed_to: Dict[int, Tuple[Any, str]] = {}
    for fused_uid in (rewrite.origins or {}):
        fused = program.by_uid(fused_uid)
        if fused is None:
            continue
        if not isinstance(fused, FusedNode):
            continue
        ports = fused.output_ports or {}
        for origin, port in ports.items():
            absorbed_to[origin] = (fused, port or "out")
        sink = fused.sink
        if sink is not None and sink not in absorbed_to:
            absorbed_to[sink] = (fused, "out")

    out: Dict[int, List] = {}
    for node in program.nodes():
        if node.children:
            continue
        inputs: List = []
        for ref in (node.inputs_obj or ()):
            if ref.src is None:
                inputs.append(ref.external_sid)
                continue
            producer, port = program.output_leaf(ref.src, ref.port or "out")
            published = absorbed_to.get(producer.uid)
            if published is not None:
                producer, port = published
            inputs.append(tensor_id(producer.uid, port))
        out[node.uid] = inputs
    return out
