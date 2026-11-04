"""Coding-agent backends for spec generation / refinement.

Each backend wraps a local CLI tool (Codex, Claude Code, Kimi Code, ...)
behind a uniform interface so `spec_worker.stream_agent` doesn't need to
know about CLI-specific args or event schemas. Adding a new agent =
drop a new file in this package and register it below.
"""
from __future__ import annotations

from typing import Dict

from .base import AgentBackend
from .claude import ClaudeBackend
from .codex import CodexBackend
from .kimi import KimiBackend


_BACKENDS: Dict[str, AgentBackend] = {
    "codex": CodexBackend(),
    "claude": ClaudeBackend(),
    "kimi": KimiBackend(),
}


def available_clis() -> tuple[str, ...]:
    return tuple(_BACKENDS.keys())


def get_backend(agent_cli: str) -> AgentBackend:
    try:
        return _BACKENDS[agent_cli]
    except KeyError:
        raise ValueError(f"Unknown agent CLI {agent_cli!r}. Expected one of: {', '.join(_BACKENDS)}") from None
