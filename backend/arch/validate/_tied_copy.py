"""A tie and the checkpoint agree on whether the tied layer's tensor ships.

`tied_weight` makes a layer read an earlier layer's stored tensor. Most tied
checkpoints then omit the reader's own tensor — `save_pretrained` deduplicates
a tie — but some ship it anyway, and the serving loader skips it (HunyuanOCR's
`lm_head.weight`, dropped by vLLM when `tie_word_embeddings`). The spec states
which with `tied_weight.checkpoint_copy`, and this check holds that statement
to the file, per tied node:

* declared, and the checkpoint has no tensor at that name → the declaration
  adds bytes the file does not hold;
* not declared, and the checkpoint has a tensor at that name → the tie drops
  bytes the file does hold.

A tensor at the reader's name is a copy only when that name is not the one
the tie reads. A modality call (`call: vision`) ties each leaf of the written-
out layer to the called one's, and where the call does not rename the module
path (gemma-4's `vision_embedder`, Step-3.7's `vision_model`) both resolve to
one address: the tensor there is the stored weight itself.

Either way the byte anchor would also fail, but only with a total. This says
which layer, which tensor, and what to write.
"""
from __future__ import annotations

from typing import Any, List

from backend.arch.eval.storage import module_path

from ._finding import Finding


def _module_ships(module: str, header: Any) -> List[str]:
    """The header tensors stored under `module`, as their groups display."""
    return sorted({group.display() for group in header.under(module)})


def run_tied_copy_check(analysis: Any, header: Any) -> List[Finding]:
    """One finding per tied node whose `checkpoint_copy` declaration
    disagrees with the checkpoint's tensor names."""
    if not header:
        return []
    program = analysis.program
    copies = analysis.checkpoint_copies.values
    findings: List[Finding] = []
    for node in program.nodes():
        if node.tied_to is None:
            continue
        if node.stores_checkpoint_copy:
            stored, _reps = copies.get(node.uid, ((), ()))
            missing = [t.name for t in stored if not header.select(t.name)]
            if not stored or missing:
                findings.append(Finding(
                    kind="tied_checkpoint_copy_absent",
                    node=node.display_path,
                    op=node.op,
                    detail=(
                        f"{node.display_path} declares "
                        f"`tied_weight.checkpoint_copy`, but the checkpoint "
                        f"has no tensor named {missing or [node.display_path]}. "
                        f"This checkpoint omits the tied copy: delete "
                        f"`checkpoint_copy` and keep `tied_weight: {{ to: "
                        f"{node.tied_to} }}`.")))
            continue
        module = module_path(node, program.checkpoint_prefix(node))
        target = program.by_uid(node.tied_to_uid)
        if module == module_path(target, program.checkpoint_prefix(target)):
            continue
        shipped = _module_ships(module, header)
        if shipped:
            findings.append(Finding(
                kind="tied_checkpoint_copy_undeclared",
                node=node.display_path,
                op=node.op,
                detail=(
                    f"{node.display_path} is tied to {node.tied_to!r}, but "
                    f"the checkpoint also stores {shipped}. If the serving "
                    f"loader skips that copy, declare it on the tie: "
                    f"`tied_weight: {{ to: {node.tied_to}, ..., "
                    f"checkpoint_copy: {{ src: \"<file>:<line> where the "
                    f"loader skips it>\" }} }}`. If serving loads it as its "
                    f"own weight, the layer is untied: delete `tied_weight`.")))
    return findings
