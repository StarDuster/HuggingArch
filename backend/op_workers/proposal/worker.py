"""Agent execution for op-proposal worker.

Thin wrapper over `backend._agent_runner.stream_agent_core`. Op
proposals have no post-success side effects (no HF snapshot, no
Supabase cache publish — proposals are staged under
`backend/arch/ops_proposals/<op_name>/` and a human approves before
they hit `ops.py`), so the runner is invoked with `on_success=None`.

Per-run trace logs land in `logs/op_proposal_worker/<op_name>__<ts>.log`.
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
    """Run the configured agent CLI for an op proposal and stream SSE.

    `prep_result` is what `prepare_propose_op_prompt` returns — a dict
    with `prompt` and optional `prefetch_logs`. Mirror of the spec
    worker contract so the SSE wrapper in the API route can be the same
    shape.
    """
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(LOG_DIR, f"{op_name}__{ts}.log")
    yield from stream_agent_core(
        key=op_name,
        prep_result=prep_result,
        agent_cli=get_agent_cli(),
        log_path=log_path,
        log_header=f"=== op-proposal {op_name} @ {ts} ===",
        launch_banner="Launching coding agent...",
        procs_dict=_locker.procs,
        locks_lock=_locker.locks_lock,
        release_fn=_locker.release,
    )
