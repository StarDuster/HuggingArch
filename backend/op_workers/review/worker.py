"""Agent execution for op-review worker.

Thin wrapper over `backend._agent_runner.stream_agent_core`. The
reviewer agent has no in-runner post-success step — verdict-reading +
rule-checking + `run_promote` live in `api/routes/op_reviews.py`. The
agent's only direct output is `review.json` in the artifact directory.
"""
from __future__ import annotations

import logging
import os
from datetime import datetime
from typing import Generator

from backend._agent_runner import stream_agent_core
from backend.spec_worker.runtime import get_agent_cli

from .runtime import LOG_DIR, _locker


logger = logging.getLogger(__name__)


def stream_agent(op_name: str, prep_result: dict) -> Generator[str, None, None]:
    """Run the configured agent CLI for an op review and stream SSE."""
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(LOG_DIR, f"{op_name}__{ts}.log")
    yield from stream_agent_core(
        key=op_name,
        prep_result=prep_result,
        agent_cli=get_agent_cli(),
        log_path=log_path,
        log_header=f"=== op-review {op_name} @ {ts} ===",
        launch_banner="Launching reviewer agent...",
        procs_dict=_locker.procs,
        locks_lock=_locker.locks_lock,
        release_fn=_locker.release,
    )
