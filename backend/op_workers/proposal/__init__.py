"""Op-proposal worker — agent CLI integration for primitive-op extension.

Sibling of `backend.spec_worker`. The spec worker writes a structured
gap statement (see the "Missing-primitive escape hatch" section of
`backend/spec_worker/_prompts/tasks.py`) when it cannot compose a model's
forward from existing primitives. This worker consumes that gap and
produces a candidate `OpSpec(...)` registration plus the supporting
artifacts a human / reviewer needs to accept the proposal:

  backend/arch/ops_proposals/<op_name>/
    gap.json              — copy of the gap statement (input record)
    patch.py              — structured: NEW_DIM_SYMBOLS / DIM_SECTION_COMMENT
                            / OP_REGISTRATIONS — consumed by promote tool
    dims.yaml             — new symbol entries for the dim registry
    tests.py              — pytest test methods (registration, shape, FLOPs)
    worked_example.yaml   — a minimal component yaml using the new
                            primitive to compose the blocking model's
                            kernel
    rationale.md          — agent's narrative: kernel source citation,
                            why no existing op composes, shape_kind
                            choice + justification, FLOPs derivation

Public API (consumed by api / scripts):
  • Runtime config: configure_runtime, get_agent_cli, get_agent_binary,
    get_agent_display_name, VALID_AGENT_CLIS (re-exported from
    spec_worker so the worker share a single CLI selection)
  • Concurrency: acquire, release, stop (per-op-name, not per-model)
  • Prompt building: prepare_propose_op_prompt
  • Execution: stream_agent (mirror of spec_worker.stream_agent)
"""
from .runtime import (
    acquire,
    release,
    stop,
    PROPOSALS_DIR,
    GAP_DIR,
    LOG_DIR,
)
from ._prompts.tasks import (
    build_propose_op_prompt,
    prepare_propose_op_prompt,
)
from .worker import stream_agent

# Reuse spec_worker's CLI selection so SPEC_AGENT_CLI applies to both.
from backend.spec_worker.runtime import (
    VALID_AGENT_CLIS,
    configure_runtime,
    get_agent_cli,
    get_agent_binary,
    get_agent_display_name,
)

__all__ = [
    "GAP_DIR",
    "LOG_DIR",
    "PROPOSALS_DIR",
    "VALID_AGENT_CLIS",
    "acquire",
    "build_propose_op_prompt",
    "configure_runtime",
    "get_agent_binary",
    "get_agent_cli",
    "get_agent_display_name",
    "prepare_propose_op_prompt",
    "release",
    "stop",
    "stream_agent",
]
