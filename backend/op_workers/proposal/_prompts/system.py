"""System prompt for op-proposal worker — template + live registry."""
from __future__ import annotations

import os
from functools import lru_cache

from .components import (
    build_existing_ops_doc,
    read_dim_registry,
    read_shape_kind_taxonomy,
)
from ...._agent_runner.source_paths import VLLM_SRC, SGLANG_SRC


_TEMPLATES_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "templates"
)


@lru_cache(maxsize=1)
def _load_template(name: str) -> str:
    with open(os.path.join(_TEMPLATES_DIR, name), encoding="utf-8") as f:
        return f.read().rstrip()


def build_system_prompt() -> str:
    """Assemble system prompt with live registry + glossary + taxonomy."""
    return _load_template("system.md").format(
        ops_doc=build_existing_ops_doc(),
        dim_registry=read_dim_registry(),
        shape_kind_taxonomy=read_shape_kind_taxonomy(),
        vllm_src=VLLM_SRC,
        sglang_src=SGLANG_SRC,
    )
