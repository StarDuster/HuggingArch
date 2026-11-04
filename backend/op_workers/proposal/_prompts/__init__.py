"""Prompt-construction subsystem for op-proposal worker.

Mirrors `backend.spec_worker._prompts` layout. Public entry points
(`build_propose_op_prompt`, `prepare_propose_op_prompt`) are re-exported
from the parent package.

Layout:
  components.py — read the live OP_REGISTRY + dim registry into
                  prompt-ready strings (so the agent sees exactly what
                  already exists before proposing something new).
  system.py     — reusable system prompt assembled from template + live
                  registry snapshots.
  tasks.py      — `build_propose_op_prompt(gap)` per-task body + entry
                  points.
"""
from .components import (
    build_existing_ops_doc,
    read_dim_registry,
)
from .system import build_system_prompt
from .tasks import (
    build_propose_op_prompt,
    prepare_propose_op_prompt,
)

__all__ = [
    "build_existing_ops_doc",
    "build_propose_op_prompt",
    "build_system_prompt",
    "prepare_propose_op_prompt",
    "read_dim_registry",
]
