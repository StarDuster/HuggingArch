"""In-memory per-key locker shared by op_workers.proposal / op_workers.review.

Two-layer locking:
  • A *shared* semaphore (passed in; the spec_worker semaphore so total
    concurrent agent CLI invocations stay capped at
    SPEC_AGENT_MAX_CONCURRENT across all three workers).
  • An in-memory `{key: True}` set guarded by `locks_lock` to reject
    duplicate concurrent requests for the same key.

`acquire()` returns 'ok' / 'duplicate' / 'overloaded' — same contract as
`spec_worker.runtime.acquire`. spec_worker has its own Supabase-backed
variant and doesn't use this class; that's intentional, the cross-VM
mutex doesn't apply to op_*_worker (proposals/reviews are rare and
one-shot, not the hot path).

`procs` and `locks_lock` are public attributes because
`stream_agent_core` registers the subprocess into `procs[key]` from
within its own try-block.
"""
from __future__ import annotations

import logging
import threading


logger = logging.getLogger(__name__)


class InMemoryLocker:
    def __init__(
        self,
        semaphore: threading.BoundedSemaphore,
        *,
        key_label: str = "key",
    ) -> None:
        self.semaphore = semaphore
        self.procs: dict = {}
        self.locks_lock = threading.Lock()
        self._locks: dict = {}
        self._slot_held: dict = {}
        self._key_label = key_label

    def acquire(self, key: str, *, consume_slot: bool = True) -> str:
        """'ok' / 'duplicate' / 'overloaded'.

        `consume_slot=False` skips the shared agent-CLI semaphore: the
        caller already holds a spec-gen slot and is chaining an op
        writer into the same job. The per-key lock still rejects a
        duplicate in-flight propose/review for this op.
        """
        if consume_slot and not self.semaphore.acquire(blocking=False):
            return "overloaded"
        with self.locks_lock:
            if key in self._locks:
                if consume_slot:
                    self.semaphore.release()
                return "duplicate"
            self._locks[key] = True
            self._slot_held[key] = consume_slot
        return "ok"

    def release(self, key: str) -> None:
        """Release per-key lock + reap subprocess. Idempotent."""
        with self.locks_lock:
            proc = self.procs.pop(key, None)
            self._locks.pop(key, None)
            held = self._slot_held.pop(key, False)
        if proc is not None and proc.poll() is None:
            try:
                proc.kill()
            except OSError as e:
                logger.warning(
                    f"release: kill({self._key_label}={key}) failed: {e}"
                )
        if held:
            try:
                self.semaphore.release()
            except ValueError:
                # release() called without a prior successful acquire
                # (e.g. duplicate path) — semaphore at full capacity.
                pass

    def stop(self, key: str) -> bool:
        """Kill a running worker. Returns True if killed."""
        with self.locks_lock:
            proc = self.procs.get(key)
        if proc and proc.poll() is None:
            proc.kill()
            self.release(key)
            return True
        return False
