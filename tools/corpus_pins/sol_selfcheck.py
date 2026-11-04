"""The SOL surface's scoped reconciliation against the checkpoint header.

`estimate_capacity` leaves every aux stack out of the model-level numbers
(`include_aux_weights=False`), so the `single` cell is the main model's
surface. The header side is scoped the same way: the header total less each
aux stack's bytes and elements, summed over the header patterns its stored
tensors are named (`Analysis.iter_checkpoint()`, quant siblings included). An
encoder stack's header bytes must also equal its TP=1 bytes per rank (pin
P3).

`eight` is not the header divided by eight (replicated nodes, uneven splits
and routing all move a rank's share), so its evidence is the main model's
per-node TP=8 bytes summed against `weight_bytes_per_rank`, and the per-token
KV divisor `single / eight`. Same-skeleton models (equal `single` parameters
and per-token KV) are listed with their differences, for reading only.

The result is evidence beside the P2 keys, not a gate. Where an aux stack
names a header pattern the main model also names (an MTP layer stored in the
backbone's index space) the header cannot be scoped and the check says it
skipped. Elsewhere a difference is reported with both numbers: the SOL
surface prices the deployment's weights (a precomputed buffer such as
`inv_freq`, or a checkpoint stored wider than it is served, makes the two
sides differ by design), so a person reads it.
"""
from __future__ import annotations

from typing import Any, Dict, List, Mapping, Set, Tuple

from tools.corpus_pins.corpus import CorpusModel
from tools.corpus_pins.pins.encoder_weight import weight_leaves

#: header dtypes that hold packed encodings, where elements are not
#: logical parameters (DeepSeek-V4 stores FP4 as I8)
PACKED_STORAGE = ("I8", "U8", "I16", "I32", "I64", "U32")


def header_patterns(model: CorpusModel) -> List[Tuple[Any, str, int, int]]:
    """(group, dtype, elements, bytes) per header tensor group."""
    from backend.arch.dtypes import dtype_bytes
    out = []
    for group in model.header:
        elements = group.elements * group.count
        out.append((group, group.dtype, elements,
                    int(elements * dtype_bytes(group.dtype))))
    return out


def stack_names(model: CorpusModel) -> Tuple[Dict[str, Set[str]], Set[str]]:
    """`({aux stack: checkpoint names}, main-model checkpoint names)`."""
    from backend.arch.eval.program import Program
    from backend.arch.eval.transforms import Deployment
    resolved = model.resolved()
    program = Program.of(resolved)
    aux = {stack.name for stack in program.stacks if stack.is_aux}
    analysis = Deployment(program).at(batch=1, input_tokens=8).checkpoint()
    names: Dict[str, Set[str]] = {}
    main: Set[str] = set()
    for row in analysis.iter_checkpoint():
        stack = program.stack_of(program.by_uid(row.uid))
        if stack in aux:
            names.setdefault(stack, set()).add(row.name)
        else:
            main.add(row.name)
    return dict(sorted(names.items())), main


def selfcheck(model: CorpusModel, cells: Mapping[str, Any],
              encoder_row: Mapping[str, Any], golden: Mapping[str, Any]) -> Dict[str, Any]:
    """The scoped evidence for one model's SOL cells."""
    single = cells[f"{model.filename}|single"]["model"]
    eight = cells[f"{model.filename}|eight"]["model"]
    patterns = header_patterns(model)
    total_bytes = sum(p[3] for p in patterns)
    total_elements = sum(p[2] for p in patterns)
    stacks, main_names = stack_names(model)
    out: Dict[str, Any] = {"header": {"bytes": total_bytes, "elements": total_elements}}
    aux: Dict[str, Any] = {}
    differences: List[str] = []
    for stack, names in stacks.items():
        shared = sorted(names & main_names)
        if shared:
            # an MTP layer stored in the backbone's own index space: the
            # header patterns it names are the main model's too
            out["skipped"] = (f"aux stack {stack!r} shares header patterns with "
                              f"the main model ({shared[0]!r}, …); the header "
                              f"cannot be scoped")
            out["differences"] = []
            return out
        from backend.arch.dtypes import dtype_bytes
        mine = [part for name in names
                for part in model.header.select(name).parts]
        aux[stack] = {"bytes": sum(int(g.elements * n * dtype_bytes(g.dtype))
                                   for g, n in mine),
                      "elements": sum(g.elements * n for g, n in mine)}
        tp1 = encoder_row.get("1", {}).get(stack)
        if tp1 is not None and tp1["bytes_per_rank"] != aux[stack]["bytes"]:
            differences.append(f"{stack}: header {aux[stack]['bytes']} B, TP=1 {tp1['bytes_per_rank']} B")
    out["aux"] = aux
    main_bytes = total_bytes - sum(a["bytes"] for a in aux.values())
    main_elements = total_elements - sum(a["elements"] for a in aux.values())
    out["single"] = {"header_bytes_less_aux": main_bytes,
                     "weight_bytes_per_rank": single["weight_bytes_per_rank"]}
    if int(single["weight_bytes_per_rank"]) != main_bytes:
        differences.append(f"single weight_bytes_per_rank {single['weight_bytes_per_rank']} != {main_bytes}")
    packed = sorted({p[0].display() for p in patterns if p[1] in PACKED_STORAGE})
    if packed:
        out["single"]["total_params"] = {"skipped": "packed storage", "families": packed}
    else:
        out["single"]["total_params"] = {"header_elements_less_aux": main_elements,
                                         "total_params": single["total_params"]}
        if int(single["total_params"]) != main_elements:
            differences.append(f"single total_params {single['total_params']} != {main_elements}")
    kv1, kv8 = single["per_token_kv_bytes_per_rank"], eight["per_token_kv_bytes_per_rank"]
    _, leaves = weight_leaves(model.resolved(), 8)
    main_leaves = {name: leaf for name, leaf in leaves.items() if leaf.stack not in stacks}
    node_sum = sum(int(leaf.resident_bytes_per_rank) for leaf in main_leaves.values())
    out["eight"] = {"per_token_kv_bytes_per_rank": kv8,
                    "kv_divisor": (kv1 / kv8) if kv8 else None,
                    "weight_bytes_per_rank": eight["weight_bytes_per_rank"],
                    "main_nodes": len(main_leaves),
                    "main_node_bytes_per_rank": node_sum}
    if node_sum != int(eight["weight_bytes_per_rank"]):
        differences.append(f"eight: main nodes hold {node_sum} B per rank, "
                        f"weight_bytes_per_rank {eight['weight_bytes_per_rank']}")
    out["same_skeleton"] = same_skeleton(model, cells, golden)
    out["differences"] = differences
    return out


def same_skeleton(model: CorpusModel, cells: Mapping[str, Any],
                  golden: Mapping[str, Any]) -> List[Dict[str, Any]]:
    """Corpus models whose `single` cell has the candidate's parameters and
    per-token KV, with every model-level field and decode op family's flops
    that differ."""
    mine = cells[f"{model.filename}|single"]
    out = []
    for key, cell in sorted(golden.items()):
        name, deployment = key.split("|")
        if deployment != "single" or name == model.filename:
            continue
        if (cell["model"]["total_params"] != mine["model"]["total_params"]
                or cell["model"]["per_token_kv_bytes_per_rank"]
                != mine["model"]["per_token_kv_bytes_per_rank"]):
            continue
        fields = {f: (cell["model"][f], mine["model"][f]) for f in mine["model"]
                  if cell["model"][f] != mine["model"][f]}
        theirs, ours = cell["ops"]["decode"]["by_op"], mine["ops"]["decode"]["by_op"]
        flops = {op: (theirs.get(op, {}).get("flops_per_rank", 0),
                      ours.get(op, {}).get("flops_per_rank", 0))
                 for op in sorted(set(theirs) | set(ours))
                 if theirs.get(op, {}).get("flops_per_rank", 0)
                 != ours.get(op, {}).get("flops_per_rank", 0)}
        out.append({"model": name, "model_fields": fields, "decode_flops_by_op": flops})
    return out
