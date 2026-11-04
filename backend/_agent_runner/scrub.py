"""Scrub deployment-local absolute paths from user-visible agent output.

The agent stream echoes whatever the CLI prints — assistant prose, `$ cd …`
commands, `[Read] <path>` lines, stderr excerpts. Those carry absolute paths:
the per-run overlay root (which embeds the deploy home AND, for signed-in
runs, the triggering user's UUID), the shared repo checkout, HF cache dirs.
None of that is the browser's business: events polled via /spec-status +
/api/spec/status and the owner-readable spec_generation_logs row should only
ever see workspace-relative paths.

The on-disk trace log keeps the raw text (operator debug artifact); only
frames leaving the process and the DB-bound upload are scrubbed.
"""
from __future__ import annotations

import os
import re

from backend.agents.base import PROJECT_ROOT


def _rules(target_root: str | None) -> list[tuple[str, str]]:
    """(prefix, replacement) pairs, longest prefix first.

    The run overlay and repo roots are STRIPPED, keeping the repo-relative
    tail the prompt templates already speak in (`<root>/backend/arch/x.yaml`
    → `backend/arch/x.yaml`). Home dirs collapse to `~`. Longest-first so a
    nested overlay root wins over the home dir that contains it — otherwise
    the user-UUID middle segment survives as `~/.huggingarch/spec_runs/…`.
    """
    rules: dict[str, str] = {}
    for root in (target_root, PROJECT_ROOT):
        if root and len(root) > 1:
            rules[root.rstrip("/")] = ""
    for home in (os.environ.get("HA_AGENT_HOME"), os.path.expanduser("~")):
        if home and len(home) > 1:
            rules[home.rstrip("/")] = "~"
    return sorted(rules.items(), key=lambda kv: len(kv[0]), reverse=True)


def scrub_paths(text: str, target_root: str | None = None) -> str:
    """Return `text` with known deployment roots replaced by display-safe
    stand-ins. `target_root` is the run's private overlay root when the run
    has one (service-layer per-user runs)."""
    if not text:
        return text
    for prefix, repl in _rules(target_root):
        if prefix not in text:
            continue
        esc = re.escape(prefix)
        if repl:
            # Path-boundary guard: the prefix must be followed by `/` or a
            # non-path character, so `/home/x` doesn't mangle `/home/xy`.
            text = re.sub(esc + r"(?=/|[^\w./-]|$)", repl, text)
        else:
            text = re.sub(esc + "/", "", text)
            text = re.sub(esc + r"(?![\w./-])", ".", text)
    return text
