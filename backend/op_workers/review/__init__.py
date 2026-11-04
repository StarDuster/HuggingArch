"""Op-review worker — second-stage judge on op-proposal artifacts.

Sibling of `backend.op_workers.proposal`. The proposer writes an artifact
directory under `backend/arch/ops_proposals/<op_name>/`; this
worker inspects that directory, runs the agent CLI as a "reviewer", and
writes a structured `review.json` verdict:

  {"verdict": "auto_merge" | "needs_human" | "reject",
   "red_flags": [...], "summary": "markdown", "checked_at": ...}

After the agent done event, the route layer (`api/routes/op_reviews.py`)
runs a mechanical rule check (`_rules.check_auto_merge_eligibility`) and
calls `backend.arch.promote_op_proposal.run_promote` if BOTH
verdict == `auto_merge` AND the rule check passes. Rules are the gate;
the LLM is a necessary but not sufficient condition.

Public API:
  • Runtime config: re-exports from spec_worker (single CLI selection)
  • Concurrency: acquire, release, stop (per-op-name)
  • Prompt building: prepare_review_prompt
  • Execution: stream_agent
  • Rule check: check_auto_merge_eligibility (for the route to call
    post-agent)
"""
from .runtime import (
    acquire,
    release,
    stop,
    PROPOSALS_DIR,
    LOG_DIR,
)
from ._prompts.tasks import (
    build_review_prompt,
    prepare_review_prompt,
)
from ._rules import check_auto_merge_eligibility
from .worker import stream_agent

from backend.spec_worker.runtime import (
    VALID_AGENT_CLIS,
    configure_runtime,
    get_agent_cli,
    get_agent_binary,
    get_agent_display_name,
)

__all__ = [
    "LOG_DIR",
    "PROPOSALS_DIR",
    "VALID_AGENT_CLIS",
    "acquire",
    "build_review_prompt",
    "check_auto_merge_eligibility",
    "configure_runtime",
    "get_agent_binary",
    "get_agent_cli",
    "get_agent_display_name",
    "prepare_review_prompt",
    "release",
    "stop",
    "stream_agent",
]
