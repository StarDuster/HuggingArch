"""Local framework-source roots for the op-proposal / op-review agents.

Those agents derive FLOPs and params by reading real kernel source in
vLLM and sglang rather than trusting a paper's claimed cost. The two
checkouts are not vendored into this repo — point the agents at your
own local clones via the ``HA_VLLM_SRC`` / ``HA_SGLANG_SRC`` environment
variables. When unset they default to ``vendor/vllm`` and
``vendor/sglang`` under the project root, a conventional spot to clone
or symlink the sources into.

Keeping the resolution here means no host-specific path is baked into
any prompt template or task body.
"""
from __future__ import annotations

import os

_PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)


def _resolve(env_var: str, default_rel: str) -> str:
    raw = os.environ.get(env_var, "").strip()
    path = raw or os.path.join(_PROJECT_ROOT, default_rel)
    return os.path.abspath(os.path.expanduser(path))


# Absolute paths handed to the agents as "you MAY read source here".
VLLM_SRC = _resolve("HA_VLLM_SRC", "vendor/vllm")
SGLANG_SRC = _resolve("HA_SGLANG_SRC", "vendor/sglang")
