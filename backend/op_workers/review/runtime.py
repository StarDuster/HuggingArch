"""Op-review worker runtime — paths + per-op-name concurrency.

Mirror of `backend.op_workers.proposal.runtime`. Concurrency state is
delegated to `backend._agent_runner.InMemoryLocker`, sharing the
spec_worker semaphore.
"""
from __future__ import annotations

import os

from backend._agent_runner import InMemoryLocker
from backend.spec_worker.runtime import semaphore as _spec_semaphore


_PROJECT_ROOT = os.path.dirname(
    os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    )
)

LOG_DIR = os.path.join(_PROJECT_ROOT, "logs", "op_review_worker")

# Same staging dir the proposer writes to — review reads from here and
# the promote tool reads from here.
PROPOSALS_DIR = os.path.join(
    _PROJECT_ROOT, "backend", "arch", "ops_proposals"
)

PROPOSALS_DIR_REL = os.path.relpath(PROPOSALS_DIR, _PROJECT_ROOT)

os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(PROPOSALS_DIR, exist_ok=True)


_locker = InMemoryLocker(_spec_semaphore, key_label="op_name")

acquire = _locker.acquire
release = _locker.release
stop = _locker.stop
