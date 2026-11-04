"""Which per-op rows belong to one declaration — the grouping, once.

Block enumeration splits a declaration into one invocation per visible
history length (`BlockMeta.visible`), so one declaration's rows arrive as
several derived groups. Two readers need to know which rows those are:

  * the measured fold, which states the declaration parent's measured time
    as the one formula's share of exactly those terms
    (`batch_sweep.apply_measured` → `auto_sweep.Folded.subtotal`);
  * the declarations view, which shows a parent carrying their sums
    (`views.declarations.per_op_rows`).

Both ask here, so a parent's measured number and the rows it is shown over
are the same set. Nothing is read out of a group's name: each row carries
its group as fields (`BlockMeta.derived_group`), and this groups on those.
"""
from __future__ import annotations

from typing import Any, List, Mapping, Optional, Sequence, Tuple

#: A declaration, hashable: its layer path, which assignment it is, and the
#: template it was written as.
Key = Tuple[Tuple[int, ...], Optional[int], str]


def declaration_key(group: Mapping[str, Any]) -> Key:
    """The declaration a `derived_group` record belongs to, hashable."""
    d = group["declaration"]
    return (tuple(d["layer_path"]), d["assignment_index"], d["base_template"])


def group_key(key: Key, path: str, op: str) -> str:
    """One parent's name as text: a declaration, a path in its block and an
    op. Text because it crosses a JSON boundary — the measured fold states
    each parent's share under this name and the view looks its own parent up
    by it, so the two cannot drift into different groupings."""
    layer_path, assignment, template = key
    return "|".join((
        "/".join(str(i) for i in layer_path),
        "" if assignment is None else str(assignment),
        template, path, op,
    ))


def declaration_terms(rows: Sequence[Mapping[str, Any]]
                      ) -> List[Tuple[str, List[int]]]:
    """`(parent name, its rows' indices)` for every op of every declaration
    that has derived groups, in the order of each parent's first row.

    A row of no derived group is in no parent: a model without derived
    groups groups into nothing and both readers pass it through.
    """
    joined: dict[str, List[int]] = {}
    for index, row in enumerate(rows):
        group = row.get("derived_group")
        if group is None:
            continue
        name = group_key(declaration_key(group), row["display_path"],
                         str(row.get("op") or ""))
        joined.setdefault(name, []).append(index)
    return list(joined.items())
