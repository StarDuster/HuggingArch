"""Block synthesis — bind parameterized templates into resolved block graphs.

Extracted from loader.py, which mixed five concerns; this is the synthesizer
alone. Loading / matching stay in loader.py, which imports `_BlockSynthesis`
from here. One-directional: this layer depends only on schema (BlockGraph /
GraphNode), none of which imports it.
"""
from __future__ import annotations

import difflib
from typing import Any, Dict, List, Optional, Tuple

from backend.arch.ops import OP_REGISTRY
from backend.arch.schema import BlockGraph, GraphNode

_MAX_SYNTH_DEPTH = 16


def binding_of(spec: Any, called: str, component: str) -> bool:
    """Whether the component a call invokes IS `component` or a binding of
    it: a synthesized specialization whose `synth_template` chain roots at
    `component` (`csa2_attention__compressor-…` is `csa2_attention` with its
    slot params bound). A name outside the spec's namespace is no binding
    of anything in it."""
    blocks = {**(spec.components or {}), **(spec.blocks or {})}
    seen: set = set()
    while called and called not in seen:
        if called == component:
            return True
        seen.add(called)
        block = blocks.get(called)
        if block is None:
            return False
        called = block.synth_template
    return False


def binding_key(spec: Any, called: str,
                params: Dict[str, str]) -> Tuple[str, Tuple[Tuple[str, str], ...]]:
    """The binding a call runs, as (root template, full slot bindings):
    a synthesized specialization's chain accumulates each level's
    `synth_params`; a call written inline with literal `params:` carries
    them itself. Two calls with the same key run the same graph the same
    way, whichever syntax wrote them — an inline
    `attn: {op: csa2_attention, params: {…}}` and the per-layer map's
    synthesized `csa2_attention__…` of the same bindings are one
    parameterization — and a slot left at its declared default binds what
    naming the default binds (`pre_norm`'s `norm: rmsnorm`, written on the
    backbone's call and omitted on the MTP layer's)."""
    blocks = {**(spec.components or {}), **(spec.blocks or {})}
    merged = dict(params)
    seen: set = set()
    while called and called not in seen:
        seen.add(called)
        block = blocks.get(called)
        if block is None or not block.synth_template:
            break
        merged = {**{k: str(v)
                     for k, v in (block.synth_params or {}).items()},
                  **merged}
        called = block.synth_template
    root = blocks.get(called)
    if root is not None:
        merged = {**{name: slot.default for name, slot in root.params.items()
                     if slot.default is not None},
                  **merged}
    return called, tuple(sorted(merged.items()))


class _BlockSynthesis:
    """Bind parameterized templates into resolved block graphs.

    `lower` used to nest the synthesizer as closures over four maps that
    all mutate during assembly. Those maps are the assembly; these
    methods write them.
    """

    def __init__(
        self,
        component_lib: Dict[str, BlockGraph],
        inline_blocks: Dict[str, BlockGraph],
        components: Dict[str, BlockGraph],
        component_roles: Dict[str, str],
    ) -> None:
        self._component_lib = component_lib
        self.inline_blocks = inline_blocks
        self.components = components
        self.component_roles = component_roles
        self.synthesized: Dict[str, BlockGraph] = {}
        self._stack: List[str] = []

    def role_of(self, op: str) -> Optional[str]:
        comp = self._component_lib.get(op)
        declared = comp.role if comp is not None else None
        if not declared:
            return None
        # The schema validated `role:` against the vocabulary at load, so a
        # declared role is a registered one. The slot filter that lived here
        # (`_SLOT_ROLES`) dropped `router` — which is why eight specs'
        # `per_role.router` never matched a leaf — and `attention_swa`.
        return declared

    def unknown(self, name: str, where: str) -> ValueError:
        """The refusal for a template name that is neither a library
        component, an inline block of this spec, nor a registered op — most
        often a misspelling, so the nearest names are offered."""
        near = difflib.get_close_matches(
            name, [*self._component_lib, *self.inline_blocks, *OP_REGISTRY],
            n=3)
        return ValueError(
            f"{where}: template {name!r} is neither a component in the "
            f"library, an inline block of this spec, nor a registered op. "
            f"Possibly a misspelling; nearest names: {near or 'none'}.")

    def register(self, cname: str) -> None:
        if cname in self._component_lib and cname not in self.components:
            self.components[cname] = self._component_lib[cname]
            role = self.role_of(cname)
            if role is not None:
                self.component_roles.setdefault(cname, role)

    def resolve_param_value(self, val: Any) -> str:
        if isinstance(val, dict) and "template" in val:
            sub_params = {
                k: self.resolve_param_value(v)
                for k, v in (val.get("params") or {}).items()
            }
            return self.synthesize(val["template"], sub_params)
        return str(val)

    def bind_nested_params(self, graph: BlockGraph) -> BlockGraph:
        """`graph` with every nested template binding in its nodes' `params:`
        replaced by the synthesized block's name. Unchanged (the same object)
        when no node nests one."""
        nested = {name: node for name, node in graph.graph.items()
                  if any(isinstance(v, dict) for v in (node.params or {}).values())}
        if not nested:
            return graph
        nodes = dict(graph.graph)
        for name, node in nested.items():
            nodes[name] = node.model_copy(update={"params": {
                k: self.resolve_param_value(v) for k, v in node.params.items()}})
        return graph.model_copy(update={"graph": nodes})

    def synthesize(
        self,
        template: str,
        params: Dict[str, Any],
        as_name: Optional[str] = None,
    ) -> str:
        if template in self._stack:
            cycle = " -> ".join([*self._stack, template])
            raise ValueError(
                f"Cyclic compound-op params detected: {cycle}. A spec's "
                f"nested `params:` chain references {template!r} from "
                f"inside its own instantiation."
            )
        if len(self._stack) >= _MAX_SYNTH_DEPTH:
            chain = " -> ".join([*self._stack, template])
            raise ValueError(
                f"Compound-op params nesting exceeded depth "
                f"{_MAX_SYNTH_DEPTH}: {chain}. Likely a spec authoring "
                f"mistake — flatten or reuse a named component."
            )
        self._stack.append(template)
        try:
            resolved = {k: self.resolve_param_value(v) for k, v in params.items()}
        finally:
            self._stack.pop()
        param_str = "_".join(f"{k}-{v}" for k, v in sorted(resolved.items()))
        name = as_name or (f"{template}__{param_str}" if param_str else template)
        if name in self.synthesized:
            return name
        template_graph = self._component_lib.get(template) or self.inline_blocks.get(template)
        if template_graph is None:
            # A registered op is a template too: a one-node block running it.
            if template not in OP_REGISTRY:
                raise self.unknown(template, f"block {name!r}")
            node = GraphNode(op=template, inputs=["x"], params=dict(resolved))
            self.synthesized[name] = BlockGraph(
                inputs=["x"], graph={"layer": node}, outputs=["layer"],
            )
            return name

        full_bindings: Dict[str, str] = {}
        for pname, pspec in (template_graph.params or {}).items():
            if pspec.default is not None:
                full_bindings[pname] = pspec.default
        full_bindings.update(resolved)
        for pname, pspec in (template_graph.params or {}).items():
            bound = full_bindings.get(pname)
            if (pspec.kind == "component" and bound
                    and bound not in self._component_lib
                    and bound not in self.inline_blocks
                    and bound not in self.synthesized
                    and bound not in OP_REGISTRY):
                raise self.unknown(bound, f"block {name!r} param {pname!r}")

        new_nodes: Dict[str, GraphNode] = {}
        for nname, gn in template_graph.graph.items():
            op = gn.op
            if op.startswith("<") and op.endswith(">"):
                pn = op[1:-1]
                if pn not in full_bindings:
                    raise ValueError(
                        f"block {name!r}: component {template!r} node "
                        f"{nname!r} runs parameter <{pn}>, which the caller "
                        f"does not bind and the component declares no "
                        f"default for. Bind `{pn}:` in the call's `params:`.")
                op = full_bindings[pn]
            new_nodes[nname] = gn.model_copy(update={"op": op})

        # Everything the template declares is the bound graph's too: its
        # stream interface, formals and outputs, and its own declarations —
        # role, cache, axioms. Binding `<attn>` or `<rope>`
        # changes which op a node runs, not what the component is; an
        # attention bound to `mrope` still caches its K and V.
        self.synthesized[name] = template_graph.model_copy(update={
            "graph": new_nodes,
            "params": {},
            "synth_template": template,
            "synth_params": dict(resolved),
        })
        # A call site reads the role off `component_roles`, for a bound
        # component as for a library one: an attention bound to `mrope`
        # is attention.
        if template_graph.role:
            self.component_roles.setdefault(name, template_graph.role)
        return name

    def maybe_synth(
        self, template: Optional[str], params: Dict[str, str],
    ) -> Optional[str]:
        if not template:
            return template
        if template in self.inline_blocks:
            if not params:
                return template
            return self.synthesize(template, params)
        if template not in self._component_lib:
            # `block` is the `components:` slot map, synthesized under that
            # name once every reference is known; a registered op is
            # resolved where the layer runs it.
            if template == "block" or template in OP_REGISTRY:
                return template
            raise self.unknown(template, "model layer")
        self.register(template)
        if not params:
            return template
        return self.synthesize(template, params)
