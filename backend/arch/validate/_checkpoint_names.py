"""Every checkpoint name the spec declares is a name the checkpoint has.

The other direction of the one name join (`_matching.CheckpointJoin`): the
coverage gate asks which header tensors no stored tensor is named, this asks
which stored tensors — as `Analysis.iter_checkpoint()` names them, quant
siblings included — the header does not hold. The comparison is exact. Packed
storage needs no allowance: the emitted name already joins the op's suffix
with the quant scheme's storage suffix (`w1.weight_packed`, its
`weight_scale` sibling), the same names `tests/golden/tensor_names.txt` pins.
"""
from __future__ import annotations

from typing import Any, List

from ._finding import Finding
from ._matching import CheckpointJoin


def run_checkpoint_name_check(analysis: Any, header: Any) -> List[Finding]:
    """One finding per stored tensor the spec names that the header does
    not hold under that name."""
    findings: List[Finding] = []
    for row in CheckpointJoin.of(analysis, header).absent:
        node = analysis.program.by_uid(row.uid)
        path = node.display_path or row.name
        # The same module under another tensor name: the header holds
        # tensors inside this name's module, or a tensor named by a path
        # this name extends.
        segments = row.name.split(".")
        spellings = sorted(
            {group.display() for group in header.under(row.tensor.module)}
            | {group.display()
               for cut in range(1, len(segments))
               for group, _n in header.select(
                   ".".join(segments[:cut])).parts})
        if spellings:
            fix = (f"The header has {spellings} for this module: name the "
                   f"tensor the node stores after the one the checkpoint "
                   f"holds.")
        else:
            fix = ("No header tensor shares its module path either: check the "
                   "node's name and the `checkpoint:` address of the stack "
                   "it sits under against the safetensors index.")
        findings.append(Finding(
            kind="checkpoint_name_absent",
            node=path,
            op=node.op,
            detail=(f"{path} stores `{row.name}`, but the checkpoint has no "
                    f"tensor of that name. {fix}")))
    return findings
