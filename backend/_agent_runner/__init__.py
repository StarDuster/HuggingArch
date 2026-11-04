"""Shared agent-CLI runner used by spec / op_proposal / op_review workers.

All three workers spawn the same coding-agent CLI (claude / codex / kimi),
drain stdout as JSON events, normalize them to SSE frames, and write a
per-run trace log. The mechanics are identical; the only per-worker
variation is:

  • The per-run log path (each worker has its own logs/<name>/ tree).
  • The log header + launch banner strings.
  • The (procs_dict, locks_lock, release_fn) tuple — each worker owns
    its own concurrency state because the lock *key* differs (model_id
    vs op_name).
  • An optional `on_success` hook — spec_worker writes the HF snapshot
    and publishes to Supabase after a successful run; the op workers
    have no such side effect.

This module owns the shared driver. The per-worker `worker.py` shrinks
to a thin façade that constructs paths + banners and calls in.
"""
from .locker import InMemoryLocker
from .runner import stream_agent_core, sse, format_usage_log
from .scrub import scrub_paths

__all__ = ["InMemoryLocker", "format_usage_log", "scrub_paths", "sse", "stream_agent_core"]
