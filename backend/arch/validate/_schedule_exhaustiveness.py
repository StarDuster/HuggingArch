"""Per-layer schedule exhaustiveness.

The ONE grounded check kept from the old config-coverage anchor. It is NOT
name-matching / significance-guessing (that heuristic was removed — a spec's
faithfulness to the model is the inspector agent's job, grounded in the
`forward()` source, not a regex over config field names).

This check is different in kind: it fires ONLY when a spec has already OPTED IN
to modelling a per-layer schedule via `config_list` assignments, and then
verifies — against the config VALUE, the hard ground truth — that every distinct
value the list actually takes is handled. A `rest:` fallback does not count, so a
schedule value can't be silently absorbed into the wrong template.
"""
from __future__ import annotations

from backend.arch.dim_resolution import config_path_value

from ._finding import Finding

from typing import Dict, List


def run_layer_schedule_check(resolved) -> List[Finding]:
    """Every distinct value in a PRESENT per-layer config list must be handled
    by some `config_list`/`equals` (a `per_layer` map key). A `rest:` does NOT
    count — it is the fallback for when the FIELD is absent, not a licence to
    silently absorb unhandled values of a present list (exactly how a schedule
    change would go unmodelled). Field absent / scalar → no check. Returns a list
    of error strings (empty = clean)."""
    spec = resolved.spec
    config = resolved.config or {}
    handled: Dict[str, set] = {}   # field -> set of equals values
    optional: Dict[str, bool] = {}  # field -> every selector declared `?? absent`
    spans: Dict[str, set] = {}     # field -> layer indices its containers span
    # Every layer that declares assignments, at any depth (`layer_paths`).
    for _path, g in spec.layer_paths():
        assigns = g.assignments or []
        if not assigns:
            continue
        # A config_list schedule is exhaustive over the LAYERS ITS CONTAINER
        # SPANS, not the raw array length: a shared schedule array may carry
        # trailing entries for a sibling stack (DSv4's `compress_ratios` has
        # one extra 0 for the MTP layer, which the `mtp` group models on its
        # own). Clamp to `range(container.repeat)` so a value that only
        # occurs outside this stack is checked where it is actually built.
        rep = g.repeat
        total = resolved.get_dim(rep) if rep else None
        for assign in assigns:
            field = assign.config_list
            # A list of layer numbers (no `equals`) is not a schedule.
            if field is not None and assign.equals is not None:
                handled.setdefault(field, set()).add(str(assign.equals))
                optional[field] = (optional.get(field, True)
                                   and assign.config_list_optional)
                if total is not None:
                    spans.setdefault(field, set()).update(range(total))
    errors: List[str] = []
    for field, handled_vals in sorted(handled.items()):
        lst = config_path_value(field, config)
        if lst is None:
            if optional[field]:
                continue  # declared absent-able: the sibling selectors take the layers
            errors.append(Finding(
                kind="schedule_list_absent",
                detail=f"Per-layer config list '{field}' is absent from the config and "
                       f"no selector declares `?? absent` for it"))
            continue
        if not isinstance(lst, list):
            errors.append(Finding(
                kind="schedule_list_not_a_list",
                detail=f"Per-layer config list '{field}' is {type(lst).__name__}, not a list"))
            continue
        span = spans.get(field)
        if span:
            present_vals = {str(lst[i]) for i in span if 0 <= i < len(lst)}
        else:
            present_vals = {str(v) for v in lst}
        unhandled = present_vals - handled_vals
        if unhandled:
            errors.append(Finding(
                kind="schedule_value_unhandled",
                detail=f"Per-layer config list '{field}' has value(s) "
                       f"{sorted(unhandled)} that no assignment handles "
                       f"(handled: {sorted(handled_vals)}). Add a per_layer map entry "
                       f"for them — a `rest:` does not count, so a schedule value can't "
                       f"be silently absorbed into another template."))
    return errors
