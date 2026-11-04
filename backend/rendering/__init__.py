"""Pure response-builder package.

Public API (consumed by api, inference.activation_peak, arch.validate):

    build_block_graph_data    — block DAG view JSON for the frontend
    build_architecture_view   — top-level architecture summary JSON
    build_model_config        — model-config panel JSON

Internal layout (not re-exported, but importable directly when useful):

    _consts        — style maps, dtype tables, label tables
    _quant         — the precision string, from the analysis' weight formats
    _periodic      — periodic-span detection on layer-template sequences
    builders       — orchestrator: builds the three frontend dicts
"""
from .builders import (
    build_block_graph_data,
    build_architecture_view,
    build_model_config,
    model_name,
)

__all__ = [
    "build_architecture_view",
    "build_block_graph_data",
    "build_model_config",
    "model_name",
]
