"""Kernel-Bench wrapper for the backend-owned prefill-chunk geometry.

The interval/tail/batch policy and launch-fold invariant are deliberately
owned by the inference backend.  Kernel Bench imports them here so campaign
planning and runtime estimates consume the same implementation.
"""
from __future__ import annotations

from backend.inference.chunk_terms import (ChunkLaunchTerm, chunk_term_count,
                                           merge_chunk_launch_terms)
from backend.inference.prefill_geometry import (
    DEFAULT_PREFILL_CHUNK_SIZE, PrefillChunkPoint, decompose_prefill,
    normalize_prefill_chunk_size, prefill_chunk_map)

__all__ = [
    "DEFAULT_PREFILL_CHUNK_SIZE",
    "ChunkLaunchTerm",
    "PrefillChunkPoint",
    "chunk_term_count",
    "decompose_prefill",
    "merge_chunk_launch_terms",
    "normalize_prefill_chunk_size",
    "prefill_chunk_map",
]
