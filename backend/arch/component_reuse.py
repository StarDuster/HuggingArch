"""Which existing component already holds this graph.

One comparison, two moments. The writer asks it BEFORE authoring a draft
(`python -m backend.arch.component_reuse <draft.yaml>`); the validator asks
it again about the drafts a finished spec brought
(`validate/_reuse.run_reuse_check`). Asking late only names the copy after
the spec is built around it: a Step-5-Preview run wrote a draft holding 86%
of a library MoE's edges, read the warning, and shipped the copy because
refactoring then was the expensive path. Asking first makes the copy hard
to write by accident.

The comparison is structural: a graph is the multiset of its edges, each
`(op of the source, op of the reader, input port)`, names ignored. One
graph is compared to another by CONTAINMENT — how much of the first the
second holds — because a copy is usually a copy with parts deleted (a
cache-reading variant drops its compressor): DeepSeek-V4.1-Flash's reuse
attention had 31 nodes, 28 of them its source attention's.

Two graphs are compared only when a likeness between them would mean a copy
(`comparable`):

  * Same `role:`. A slot is filled by role, so a same-role component is the
    one the author could have bound instead — which is what the finding
    asks for. Across roles a high share is composition, not a copy, and the
    advice would be impossible: a `role: block` component CONTAINS an
    attention, so it holds nearly every edge of any `role: attention` graph,
    and no attention slot can bind a block. A component that declares no
    role claims no slot and is compared with the others that declare none.
  * Neither is the other's `decode_variant`: a component and its own decode
    form are alike by design.
  * Both at least `MIN_NODES` nodes — below that a graph is too small for
    likeness to mean a copy.

How alike is "the same structure" stays a judgement: the share is a number,
the remedy (bind it, parameterise it, or state why neither works) is the
author's call. Hence warnings, never a gate.
"""
from __future__ import annotations

import argparse
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Tuple

import yaml

from backend.arch.loader import SpecLoader, overlay_kwargs
from backend.arch.snapshots import default_overlay_dir
from backend.arch.spec_parsing import _parse_block_graph
from backend.arch.yaml_load import load_yaml

#: The share of a graph's edges found in another that reads as a copy.
#: Over the corpus no two components of one model reach 0.6; V4.1's four
#: CSA drafts hold one another at 0.89-0.94. Below `MIN_NODES` a graph is too
#: small for likeness to mean a copy.
CONTAINED = 0.85
MIN_NODES = 8

#: Rows the CLI prints — enough to see where the ranking falls off.
_SHOWN = 8


@dataclass(frozen=True)
class Likeness:
    """One existing component and the share of the candidate's edges it
    holds. `share >= CONTAINED` is what the validator reports."""

    name: str
    share: float


def edges(graph: Any) -> Counter:
    """The graph as the multiset of its edges, by op: what it computes
    from what, whatever its nodes are called."""
    nodes = graph.graph or {}

    def op_of(ref: str) -> str:
        if ref in nodes:
            return nodes[ref].op
        head = ref.rsplit(".", 1)[0]
        return nodes[head].op if head in nodes else "<input>"

    out: Counter = Counter()
    for node in nodes.values():
        for port, ref in enumerate(node.inputs or ()):
            out[(op_of(str(ref)), node.op, port)] += 1
    return out


def held_in(part: Counter, whole: Counter) -> float:
    """The share of `part`'s edges `whole` also has."""
    size = sum(part.values())
    return sum((part & whole).values()) / size if size else 0.0


def comparable(name_a: str, a: Any, name_b: str, b: Any) -> bool:
    """Whether a likeness between these two graphs would mean a copy — same
    role, not one another's decode form, and both big enough (see module
    docstring)."""
    if name_a == name_b:
        return False
    if len(a.graph or {}) < MIN_NODES or len(b.graph or {}) < MIN_NODES:
        return False
    if a.role != b.role:
        return False
    return not (a.decode_variant == name_b or b.decode_variant == name_a)


def likenesses(candidate: Any, existing: Mapping[str, Any], *,
               name: str = "") -> List[Likeness]:
    """Every component of `existing` comparable with `candidate`, ranked by
    the share of the candidate's edges it holds, highest first.

    Not thresholded: the validator reports the entries at `CONTAINED` and
    above, and the CLI prints the ranking so a writer sees how far the
    closest existing component actually is.
    """
    shape = edges(candidate)
    ranked = [
        Likeness(other, held_in(shape, edges(graph)))
        for other, graph in existing.items()
        if comparable(name, candidate, other, graph)
    ]
    ranked.sort(key=lambda found: (-found.share, found.name))
    return ranked


def read_candidate(source: str) -> Tuple[str, Any]:
    """Parse one component document — a draft file, or `-` for stdin — the
    way the loader parses one, so the graph compared is the graph that would
    load. The name is the file stem (stdin is nameless, and no existing
    component's name can match it)."""
    if source == "-":
        name, text = "<stdin>", sys.stdin.read()
    else:
        name, text = Path(source).stem, Path(source).read_text(encoding="utf-8")
    raw = load_yaml(text)
    if not isinstance(raw, dict):
        raise ValueError(f"{source}: a component document is a YAML mapping")
    return name, _parse_block_graph(raw, identity=name)


def reusable_components(loader: Any) -> Dict[str, Any]:
    """Every component a spec could reuse instead of a new draft: `library/`
    + `adapters/` by name, plus every draft the loader scanned — the merged
    `components/drafts/` tree and the run overlay the writer's own files sit
    in. A draft becomes reusable by being listed under a spec's `drafts:`,
    which is the writer's decision to make, so the set offered here does not
    wait for it. Library wins a name collision, as activation does."""
    components = dict(loader.draft_components())
    for name in loader.component_names():
        components[name] = loader.component(name)
    return components


#: This checkout's root — `backend/arch/component_reuse.py` is three down.
_REPO_ROOT = Path(__file__).resolve().parents[2]


def _as_written(path: Any) -> str:
    """A component's file as its reader can open it: under the working
    directory (a run's own drafts sit beside it) or repo-relative, the
    spelling the prompt uses and a run's reference tree mirrors."""
    if not path:
        return ""
    resolved = Path(path).resolve()
    for root in (Path.cwd(), _REPO_ROOT):
        if resolved.is_relative_to(root):
            return str(resolved.relative_to(root))
    return str(resolved)


def render(name: str, candidate: Any, ranked: List[Likeness],
           file_of: Any) -> str:
    """The CLI's answer: the ranking, then what the share at the top means.
    `file_of(component_name)` gives the yaml to read (the loader's
    `component_file`)."""
    role = candidate.role or "(none declared)"
    nodes = len(candidate.graph or {})
    out = [f"=== {name} — role={role}, {nodes} nodes, "
           f"{sum(edges(candidate).values())} edges (by op) ==="]
    if nodes < MIN_NODES:
        out.append(
            f"\nToo small to compare: {MIN_NODES} nodes is the floor below "
            f"which a likeness does not mean a copy. Pick by reading the "
            f"component manifest.")
        return "\n".join(out)
    if not ranked:
        out.append(
            f"\nNo existing component declares role {role} and is big enough "
            f"to compare. A new draft is the only way to write this graph.")
        return "\n".join(out)
    out.append(f"\nShare of this graph's edges held by each of the "
               f"{len(ranked)} existing components with role {role}:\n")
    out.extend(f"  {found.share:.2f}  {found.name:<44} "
               f"{_as_written(file_of(found.name))}"
               for found in ranked[:_SHOWN])
    if len(ranked) > _SHOWN:
        out.append(f"  … {len(ranked) - _SHOWN} more, all below "
                   f"{ranked[_SHOWN].share:.2f}")
    over = [found for found in ranked if found.share >= CONTAINED]
    if over:
        out.append(
            f"\n{len(over)} at or above {CONTAINED:.2f}: that reads as this "
            f"graph written twice, and the validator reports it. Read "
            f"{over[0].name} and bind it — a `when:` switch or a `params:` "
            f"slot expresses most differences. Keep the draft only if that "
            f"component cannot express your structure, and name the part it "
            f"cannot in your report: `library/` and `adapters/` are "
            f"review-gated, so parameterising one is a maintainer's change.")
    else:
        out.append(
            f"\nNothing at or above {CONTAINED:.2f}: no existing component "
            f"holds most of this graph, so a new draft is the honest way to "
            f"write it.")
    if candidate.role is None:
        out.append(
            "\nThis graph declares no `role:`, so it was compared only "
            "against components that declare none. Declare the role the "
            "component will carry and ask again.")
    return "\n".join(out)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Which existing components already hold this component "
                    "graph (edges by op), before you write a new draft")
    parser.add_argument("candidate",
                        help="Component YAML to compare (`-` reads stdin)")
    parser.add_argument("--overlay-dir", dest="overlay_dir", default=None,
                        help="Per-run overlay root (layers <dir>/models + "
                             "<dir>/components/drafts on the shared tree)")
    args = parser.parse_args()
    try:
        name, candidate = read_candidate(args.candidate)
    except (OSError, ValueError, yaml.YAMLError) as e:
        # The candidate is read the way the loader reads a component, so a
        # document that cannot be compared is a document that cannot load.
        print(f"{args.candidate} does not load as a component: {e}",
              file=sys.stderr)
        sys.exit(1)
    loader = SpecLoader(**overlay_kwargs(default_overlay_dir(args.overlay_dir)))
    ranked = likenesses(candidate, reusable_components(loader), name=name)
    print(render(name, candidate, ranked, loader.component_file))


if __name__ == "__main__":
    main()
