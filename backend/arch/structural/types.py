"""Shared types for the structural-check package.

The two sides of the check (forward AST + spec block graph) emit the
same `StructOp` sequence so the comparator doesn't need to know where
they came from.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List


@dataclass
class StructOp:
    """Canonical description of one structural step in a block DAG.

    kind:
      - "call":      submodule invocation  (e.g. self.self_attn(x))
      - "add":       residual addition     (e.g. residual + x)
      - "snapshot":  variable snapshot     (e.g. residual = x)
      - "return":    return statement      — marks end
      - "branch_detected":  forward has if/for/try — heterogeneity hint
    name:    submodule name for "call", class name for "branch_detected", empty otherwise
    inputs:  variable names read (dataflow semantics)
    output:  variable name receiving the result
    line:    source line number (forward side only; 0 for spec side)
    """

    kind: str
    name: str = ""
    inputs: List[str] = field(default_factory=list)
    output: str = ""
    line: int = 0
