"""
arch — model architecture: spec language, one evaluation walk, layout.

The YAML specs (models/, components/) are the source of truth. eval/
interprets a spec under a Binding. layout/ folds that walk onto a
deployment (ParallelConfig, mesh).

A node's op is a primitive or a component.
"""

from backend.arch.ops import OP_REGISTRY, OpSpec, get_op, is_primitive
from backend.arch.schema import ArchSpec, BlockGraph, GraphNode, ModelLayer
from backend.arch.loader import SpecLoader

__all__ = [
    "OP_REGISTRY",
    "ArchSpec",
    "BlockGraph",
    "GraphNode",
    "ModelLayer",
    "OpSpec",
    "SpecLoader",
    "get_op",
    "is_primitive",
]
