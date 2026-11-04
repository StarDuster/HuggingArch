"""Single source of truth for spec_id + HF weight-name composition.

Two parallel namespaces share this module:

* **Structural id** (`compose_child_spec_id` → `SpecIRNode.display_path`
  → `aggregation_id`). Pure nesting: every node's id is
  ``<parent>.<local_name>``. Drives DAG topology, shape propagation,
  fusion / comm grafting, engine cost aggregation, rendering. Two
  IR nodes can NEVER share an id under this scheme — wrappers and
  the leaves they enclose stay distinguishable.

* **HF weight name** (`compose_weight_name` → `SpecIRNode.weight_name`
  → `compose_tensor_name`). Also pure nesting, over the nodes the
  CHECKPOINT has modules for: a node declaring `module: null`
  contributes no segment, so the linear inside an `mlp.gate` wrapper
  stores as ``mlp.gate.weight``, not ``mlp.gate.gate.weight``. Drives
  HF tensor matching, materialize, validator, quant overlay.

Keeping these separate is what lets one node be a level of the graph and
not a level of the checkpoint. Collapsing them is how a fused subgraph
came to show both wrapper and inner leaf under a single id, writing two
shape entries that aliased.

Aggregation id (template + display_path) stays unchanged — it's the
pipeline cost-aggregation key, derived live from the (now-pure-nested)
display_path. Component-scope suffix (`__<scope>`) still applies to
compute-only primitives via `compose_child_spec_id` to disambiguate
cross-component name collisions; weight-bearing leaves never receive
a scope (HF naming has no such namespace).
"""
from __future__ import annotations

from typing import Optional, Tuple


_AGGREGATION_ID_SEP = "|"


def compose_aggregation_id(template: str, display_path: str) -> str:
    """Build the pipeline cost-aggregation key: ``f"{template}|{display_path}"``
    for block-internal nodes, bare ``display_path`` at model root
    (``template == ""``).

    Single source of truth for structural nodes, reports and wire rows, so the
    format stays consistent if the separator ever changes. The ``|`` separator is
    reserved: ``display_path`` is ``.``-separated and never carries ``|``.
    """
    sep = _AGGREGATION_ID_SEP
    return f"{template}{sep}{display_path}" if (template and display_path) else display_path


def split_aggregation_id(agg: str) -> Tuple[str, str]:
    """Inverse of `compose_aggregation_id`: ``(template, display_path)``.
    Bare display_path inputs (top-level / ``template==""``) return
    ``("", agg)``. Idempotent — calling on an already-bare display_path
    is a no-op.
    """
    sep = _AGGREGATION_ID_SEP
    if sep in agg:
        tmpl, disp = agg.split(sep, 1)
        return tmpl, disp
    return "", agg


def compose_child_spec_id(
    parent_spec_id: str,
    child_local_name: str,
    component_scope: Optional[str] = None,
) -> str:
    """Compose a child's structural id by pure nested append:
    ``<parent_spec_id>.<child_local_name>``. At template root
    (``parent_spec_id == ""``) the result is just the bare local name.

    `component_scope`, when non-empty, is appended as ``__<scope>`` to
    disambiguate compute-only primitives that recur in multiple
    components (e.g. several attention variants each name their score
    op `softmax`). Weight-bearing leaves never carry a scope suffix
    (HF naming has no such namespace and the wrapper would never agree).
    """
    prefix = f"{parent_spec_id}." if parent_spec_id else ""
    base = prefix + child_local_name
    if component_scope:
        return f"{base}__{component_scope}"
    return base


def compose_weight_name(
    parent_weight_name: str,
    child_local_name: str,
) -> str:
    """Compose a weight-bearing leaf's HF tensor name by nested append:
    ``<parent>.<child_local_name>``, bare at root.

    One rule, because a checkpoint path is one rule. There used to be
    three: a `primary` child took the parent's name outright, and a
    sibling spelled ``<primary>.<rest>`` lifted to ``<parent>.<rest>``.
    Both said the same thing — that no module stands between the parent
    and this node — in a form keyed off an identity claim about a
    sibling, at most one per graph, that nothing could check. The claim
    is now `module: null` on the node itself, and it is checked
    against the checkpoint's own tensor names; a node the checkpoint has
    no module for is skipped here rather than composed away.
    """
    prefix = f"{parent_weight_name}." if parent_weight_name else ""
    return prefix + child_local_name
