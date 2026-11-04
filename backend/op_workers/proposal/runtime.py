"""Op-proposal worker runtime — paths + per-op-name concurrency.

Path constants are consumed by the prompt templates + the API route +
the promote tool. Concurrency state is delegated to
`backend._agent_runner.InMemoryLocker`, sharing the spec_worker
semaphore so total concurrent agent CLI invocations stay capped at
SPEC_AGENT_MAX_CONCURRENT across all three workers.
"""
from __future__ import annotations

import os

from backend._agent_runner import InMemoryLocker
from backend.spec_worker.runtime import semaphore as _spec_semaphore


# --- Paths ----------------------------------------------------------------

_PROJECT_ROOT = os.path.dirname(
    os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    )
)

# Where the spec worker drops gap statements (and where the API reads
# them when starting a proposal).
GAP_DIR = os.path.join(_PROJECT_ROOT, "logs", "op_proposals")

# Per-run trace log directory (mirror of logs/spec_worker/).
LOG_DIR = os.path.join(_PROJECT_ROOT, "logs", "op_proposal_worker")

# Final artifact landing zone — staged under
# backend/arch/ops_proposals/<op_name>/, promoted into ops.py +
# dims.yaml + tests by scripts/promote_op_proposal.py.
PROPOSALS_DIR = os.path.join(
    _PROJECT_ROOT, "backend", "arch", "ops_proposals"
)

# Repo-relative paths for prompt templates (don't leak host home dirs).
GAP_DIR_REL = os.path.relpath(GAP_DIR, _PROJECT_ROOT)
PROPOSALS_DIR_REL = os.path.relpath(PROPOSALS_DIR, _PROJECT_ROOT)

os.makedirs(GAP_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(PROPOSALS_DIR, exist_ok=True)


# --- Concurrency ----------------------------------------------------------

_locker = InMemoryLocker(_spec_semaphore, key_label="op_name")

acquire = _locker.acquire
release = _locker.release
stop = _locker.stop
