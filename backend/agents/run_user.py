"""The OS user untrusted agent work runs as.

A leaf: the agent backends drop to this user at spawn, and the spec overlay
hands the same user ownership of the directories it creates. Both read it
here, so the overlay does not import the backend layer that itself resolves
overlay workspaces.
"""
from __future__ import annotations

import os
from typing import Optional


def untrusted_run_user() -> Optional[str]:
    """OS user to run untrusted code as (distinct from the root backend, so it
    can't read the parent's /proc/<pid>/environ). `HA_AGENT_USER` in the
    container; None in dev (no privilege drop)."""
    return os.environ.get("HA_AGENT_USER") or None
