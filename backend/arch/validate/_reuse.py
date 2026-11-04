"""Drafts that restate one another, or restate a library component.

A generated spec may bring drafts (`drafts:`). Two drafts whose graphs are
nearly the same are one structure written twice: the shared part belongs in
one component, and the difference in what its `params:` slots bind per call.
A draft nearly the same as a library component is that component with a
change the library may already express (a `when:` switch, a slot). Written
twice, the copies drift: DeepSeek-V4.1-Flash's generated spec carried four
CSA attentions about 90% node for node alike, and one of them — the only one
whose cache read law was edited separately — came out pricing a dense read
under a top-k score.

What "nearly the same" means — edges by op, containment, which pairs are
comparable at all — is `arch/component_reuse.py`, the same comparison the
writer can run on a candidate graph before authoring it
(`python -m backend.arch.component_reuse <draft.yaml>`). This module is
where that comparison meets a finished spec: the drafts it brought, against
each other and against the library.

A draft against a library component counts how much of the DRAFT the
component holds: a large draft containing a small library block whole is
composition, not a copy.

Warnings, never gating: how alike is "the same structure" is a judgement,
and the finding asks the writer to factor, not to fix a number.
"""
from __future__ import annotations

from itertools import combinations
from typing import Any, List

from backend.arch.component_reuse import (
    CONTAINED, MIN_NODES, comparable, edges, held_in, likenesses,
)

from ._finding import Finding


def run_reuse_check(resolved: Any) -> List[Finding]:
    loader = resolved.declared_by
    if loader is None:
        return []
    drafts = {name: resolved.spec.components[name]
              for name in loader.listed_drafts(resolved.spec.name)
              if name in resolved.spec.components}
    big = {name: graph for name, graph in drafts.items()
           if len(graph.graph or {}) >= MIN_NODES}
    shapes = {name: edges(graph) for name, graph in big.items()}
    out: List[Finding] = []
    for a, b in combinations(sorted(big), 2):
        if not comparable(a, big[a], b, big[b]):
            continue
        # A copy is usually a copy with parts deleted, so the question is how
        # much of the SMALLER graph the larger one holds.
        smaller, larger = sorted((shapes[a], shapes[b]),
                                 key=lambda shape: sum(shape.values()))
        alike = held_in(smaller, larger)
        if alike >= CONTAINED:
            out.append(Finding(
                kind="drafts_restate_each_other",
                template=a,
                detail=(
                    f"drafts `{a}` and `{b}`: {alike:.0%} of the smaller one's "
                    f"edges (by op) are the larger one's. Write the shared structure once — one "
                    f"component whose `params:` slots bind what differs per "
                    f"call — and bind it from each place that now uses a "
                    f"copy."),
                extra={"other": b, "likeness": round(alike, 3)}))
    library = {name: loader.component(name) for name in loader.component_names()}
    for name, graph in sorted(big.items()):
        for found in likenesses(graph, library, name=name):
            if found.share < CONTAINED:
                break
            out.append(Finding(
                kind="draft_restates_library",
                template=name,
                detail=(
                    f"{found.share:.0%} of draft `{name}`'s edges (by op) are "
                    f"library `{found.name}`'s. Bind the library component "
                    f"(a `when:` switch or a `params:` slot expresses "
                    f"most differences) rather than restating it. If it "
                    f"cannot express your structure — a part it fixes that "
                    f"your model must not carry, a slot it does not have — "
                    f"name that part in your report: `library/` is "
                    f"review-gated, so parameterising it is a maintainer's "
                    f"change, not yours."),
                extra={"library": found.name,
                       "likeness": round(found.share, 3)}))
    return out
