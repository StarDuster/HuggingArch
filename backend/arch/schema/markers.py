"""Field markers (BINDING_ONLY / LOADER_DERIVED), retired-key rejection, OWN_NAME."""
from __future__ import annotations

from typing import (Any)





#: `json_schema_extra` marker: this field only BINDS the component a node
#: invokes — which op, which edge, which params, what the checkpoint calls
#: its nodes. It says nothing about the node itself, so a node that sets
#: only these is a pure delegation and the walk can inline it (see
#: `spec_tree.is_pure_delegation`). Marked at the field rather than listed
#: in the walker, so adding a field here without thinking about inlining
#: leaves it correctly disqualified.
BINDING_ONLY = {"binding_only": True}



#: `json_schema_extra` marker: the loader computes this field; no YAML may
#: declare it. Every other field on a model is part of the authoring schema, so
#: without the marker the two are indistinguishable at a glance and nothing
#: stops a derived field from quietly acquiring an authoring channel.
#: `test_loader.TestBlockGraphSourceSchema` pins the marked set equal to the
#: fields the loader's source-key gate refuses to read.
LOADER_DERIVED = {"loader_derived": True}



#: Checkpoint-naming fields that used to live on GraphNode. Retired rather
#: than aliased: `rename` / `hf_inner` rewrote structural ids to checkpoint
#: spellings, and `hf` mixed three states in an Optional[str]. The load
#: error has to name the replacement so a writer does not treat
#: extra="forbid" as "pick a nearby allowed field".
RETIRED_GRAPH_NODE_NAMING = {
    "rename": "`inner_modules:` (checkpoint names of the invoked component's "
              "nodes; structural ids stay)",
    "hf_inner": "`inner_modules:`",
    "hf": "`module:` (omit = own name, string = checkpoint name, "
          "null = no module)",
    "primary": "nothing — a one-node adapter is detected from the graph",
}



def reject_retired_graph_node_naming(data: Any) -> Any:
    """Raise on a retired checkpoint-naming key; otherwise return `data`."""
    if not isinstance(data, dict):
        return data
    retired = [key for key in RETIRED_GRAPH_NODE_NAMING if key in data]
    if not retired:
        return data
    raise ValueError("; ".join(
        f"`{key}:` is removed; use {RETIRED_GRAPH_NODE_NAMING[key]}"
        for key in retired
    ))



class _OwnName:
    """`GraphNode.module` absent: the checkpoint calls this position what the
    spec calls it. A distinct value rather than `None`, because `None` is the
    OTHER answer — no module here at all — and a field with three answers
    needs three values."""

    __slots__ = ()

    def __repr__(self) -> str:      # pragma: no cover - debugging aid
        return "OWN_NAME"

    @classmethod
    def __get_pydantic_core_schema__(cls, _source, _handler):
        # Serializes as the sentinel string so `model_dump(mode="json")` —
        # and everything hashed off it, like the calibration spec_hash —
        # keeps working. Refusing to serialize is how the spec hash went
        # structurally blind: the dump raised, a swallow turned the
        # fingerprint into None, and every mutation hashed the same.
        from pydantic_core import core_schema

        def _only_own_name(v):
            if isinstance(v, cls):
                return v
            raise ValueError("not OWN_NAME")

        return core_schema.no_info_plain_validator_function(
            _only_own_name,
            serialization=core_schema.plain_serializer_function_ser_schema(
                lambda _v: "OWN_NAME"),
        )



OWN_NAME = _OwnName()
