"""Hand-built node dicts for the view tests carry what the renderer stamps."""
from backend.arch.ops_base import OP_REGISTRY
from backend.rendering._node_build import declaration_card


def declared_roles(op: str) -> dict:
    """The declaration fields `NodeBuilder` stamps from the op's declaration."""
    spec = OP_REGISTRY.get(op)
    if not spec:
        return {}
    out = declaration_card(spec, op)
    return {"input_roles": list(spec.input_roles),
            "declaration": out} if spec.input_roles else {"declaration": out}
