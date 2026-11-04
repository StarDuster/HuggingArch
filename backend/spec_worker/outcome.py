"""The closed vocabulary of how a spec generation ends and where its record
stands.

One definition, read by every consumer: the SQL CHECK constraints in
`supabase/init.sql` (and any later migration) list the same
values (`tests/meta/test_generation_vocabulary.py` holds them equal), the
service routes whitelist filter parameters against them, and the frontend
keys its badges on the values `/api/spec/generation-vocabulary` returns.
Human text (a `done` message, a review report) is display only; nothing
branches on it.
"""
from __future__ import annotations

from enum import Enum

from backend.arch.validate._analysable import OPEN_KIND


class GenerationStatus(str, Enum):
    """Where a `spec_generations` record stands."""

    #: a valid spec, the user's working candidate
    GENERATED = "generated"
    #: the author opened a PR from it (frozen)
    CONTRIBUTED = "contributed"
    #: its PR merged (frozen)
    MERGED = "merged"
    #: an admin turned it down before a PR existed
    REJECTED = "rejected"
    #: its PR closed unmerged; still a valid spec
    CLOSED = "closed"
    #: every validate error is an analysis engine refusal: the spec states
    #: the model correctly and waits for the engine
    BLOCKED_BY_ENGINE = "blocked_by_engine"
    #: a re-check found errors the spec writer resolves
    NEEDS_REPAIR = "needs_repair"


#: Records that hold a valid spec: the Inspector serves the newest of them.
SERVABLE = frozenset({
    GenerationStatus.GENERATED, GenerationStatus.CONTRIBUTED,
    GenerationStatus.MERGED, GenerationStatus.CLOSED,
})
#: Records waiting on the engine or on a repair; at most one per
#: (user, model), beside the working candidate.
PENDING = frozenset({
    GenerationStatus.BLOCKED_BY_ENGINE, GenerationStatus.NEEDS_REPAIR,
})
#: The working candidate a successful run overwrites in place.
MUTABLE = frozenset({
    GenerationStatus.GENERATED, GenerationStatus.REJECTED,
    GenerationStatus.CLOSED,
})


class RunErrorKind(str, Enum):
    """Why a spec run (or a record action) ended without a valid spec, as a
    machine key. Rides on the `done` event and the run's log row."""

    SPEC_INVALID = "spec_invalid"
    SPEC_BUNDLE_INCOMPLETE = "spec_bundle_incomplete"
    SPEC_NEEDS_DECLARATION = "spec_needs_declaration"
    BUNDLE_INVALID = "bundle_invalid"
    #: every remaining validate error is an analysis engine refusal
    BLOCKED_BY_ENGINE = "blocked_by_engine"
    #: every remaining repairable error is a refusal whose raise site
    #: declared no owner; the writer kept repairing and could not resolve it
    ANALYSIS_REFUSAL_UNATTRIBUTED = OPEN_KIND
    # The run ended on its agent, not on its spec: the same words as the
    # invocation's own `backend.agents.base.AgentEnd` (`done.agent_end`), so
    # the record says what the agent said.
    #: no configured agent backend with credentials could serve the run
    #: (auth, quota, no model configured, 5xx, no connection)
    AGENT_UNAVAILABLE = "agent_unavailable"
    #: the run's overall deadline was spent; the draft stays resumable
    AGENT_TIMED_OUT = "timed_out"
    #: a signal the run's timers did not send ended the agent (a stop, the
    #: OOM killer, a restart)
    AGENT_KILLED = "killed"
    #: the agent's session outgrew the model's context window; the draft
    #: stays resumable, in a new session
    AGENT_CONTEXT_EXHAUSTED = "context_exhausted"


class RecheckOutcome(str, Enum):
    """What a re-check of a pending record found, and did."""

    #: validate passes: the record became the working candidate
    PASSED = "passed"
    #: still only engine refusals: the gap list is refreshed
    STILL_BLOCKED = "still_blocked"
    #: validate found errors the spec writer resolves
    NEEDS_REPAIR = "needs_repair"
    #: the saved bundle does not normalize or load
    UNLOADABLE = "unloadable"
    #: the record changed since it was read; nothing written
    CONFLICT = "conflict"
    #: validate or storage failed; nothing written, retry
    FAILED = "failed"


class RecordOutcome(str, Enum):
    """What became of a run's generation record (`done.record.outcome`)."""

    RECORDED = "recorded"
    #: a newer spec of this (user, model) stands; nothing written
    STALE = "stale"
    #: storage failed; the spec stays in the run's workspace
    FAILED = "failed"
    #: an anonymous or single-user run keeps no record
    NONE = "none"
    #: a pending record changed since it was read; nothing written
    CONFLICT = "conflict"
    #: promoting a pending record found a valid record from a later run;
    #: the pending record was dropped
    SUPERSEDED = "superseded"


def vocabulary() -> dict:
    """Every value of the two record-facing vocabularies, for the frontend's
    filters and badges."""
    return {
        "generation_status": [s.value for s in GenerationStatus],
        "pending": sorted(s.value for s in PENDING),
        "servable": sorted(s.value for s in SERVABLE),
        "run_error_kind": [k.value for k in RunErrorKind],
    }
