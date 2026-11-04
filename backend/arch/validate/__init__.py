"""Spec validation — params + tensor naming + structural + shape consistency.

Public API:
  • `validate(model_id, spec_name=None) -> dict`
        Top-level entry. Used by the CLI (`python -m backend.arch.validate`)
        and by spec_worker prompt instructions.
  • `bind_leaves_to_hf(header, analysis) -> BindResult`
        Spec leaves bound to the header tensors their stored tensors'
        names select, consumed by `spec_vs_tensors_check`.
  • `spec_vs_tensors_check(resolved, tensor_patterns_meta) -> list`
        Estimate-path warnings from that same matcher.

Internal helper re-exported because the orchestrator constructs
`spec_blocks` from it:
  • `_flatten_block_nodes`

Internal layout:
  _shapes        cross-input geometry checks (axis_concat / attention_score / contract)
  _structural    spec block ↔ forward source diff (FYI warnings)
  _matching      spec-node ↔ HF-tensor pattern matching + per-tensor diff
  _orchestrator  top-level validate() that composes the above
  __main__       CLI: argparse + JSON output + diagnostic summary
"""
from ._matching import (
    BindResult,
    HFBinding,
    _flatten_block_nodes,
    bind_leaves_to_hf,
    spec_vs_tensors_check,
)
from ._orchestrator import validate
from ._type_contracts import (
    check_field_contracts,
    run_type_contract_checks,
)
from ._module_exec import run_module_exec_checks

__all__ = [
    "BindResult",
    "HFBinding",
    "_flatten_block_nodes",
    "bind_leaves_to_hf",
    "check_field_contracts",
    "run_module_exec_checks",
    "run_type_contract_checks",
    "spec_vs_tensors_check",
    "validate",
]
