"""Compatibility shim — knobs live in `arch.layout.parallel_config`.

`Evaluator` and the layout rule already take a `ParallelConfig`; the
type belongs next to that walk, not in the roofline package that reads
its output.
"""
from backend.arch.layout.parallel_config import (
    CpScheme,
    ParallelConfig,
    ParallelStep,
)

__all__ = ["CpScheme", "ParallelConfig", "ParallelStep"]
