"""Per-user spec-generation records (Phase 1).

Wraps public.spec_generations. The three writes that decide on the rows
already there run as one locked transaction each, in SQL
(supabase/init.sql; added by migration 0002, since folded):
    record             — a successful run's spec becomes the working candidate
                         (`record_generation`)
    record_pending     — an engine-blocked run's spec becomes the pending record
                         (`record_pending_generation`)
    transition_pending — a re-check or a Fix moves the pending record on
                         (`transition_pending_generation`)
Statuses and their sets come from `backend.spec_worker.outcome`.
    (admin reads go through the get_spec_generations* RPCs, called from
     backend/service/routes/admin.py — same pattern as cached_specs)

Service-layer module: the OSS core (`backend.spec_worker.worker`) calls
`record` through a lazy import guarded by `supabase_env.configured()`, exactly
like `backend.service.shared_spec_cache`. That keeps the core↔service boundary
test (tests/meta/test_service_layout.py) happy — nothing here is imported at
core module top level.

The writes never raise: a records outage comes back as the `failed` record
outcome, which the run reports beside its verdict.
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Optional

import httpx

from backend import supabase_env, supabase_rest
from backend.service import github as _github
from backend.spec_worker.outcome import (
    PENDING,
    SERVABLE,
    GenerationStatus,
    RecordOutcome,
)


logger = logging.getLogger(__name__)


class SpecGenerationsUnavailable(Exception):
    """Supabase did not answer a spec_generations read.

    A miss is `None`. This is an outage: callers must not report it as
    "no such generation".
    """


#: The record fields every full read returns: the bundle, where the record
#: stands, and — for a pending record — why (0003 `pending_detail`).
_RECORD_COLUMNS = ("id,user_id,model_id,spec_yaml,snapshot_json,"
                   "new_component_yamls,agent_cli,status,pr_url,created_at,"
                   "revision,run_started_at,pending_detail,"
                   "blocked_engine_version")


def _project_pending(row: dict) -> dict:
    """Split `pending_detail` into the names callers already use. Storage
    is one jsonb; blocked vs repair is `status`. A payload that never went
    through this column (tests, pre-0003 rows) is left as it arrived."""
    if "pending_detail" not in row:
        return row
    detail = row.pop("pending_detail")
    status = row.get("status")
    if status == GenerationStatus.BLOCKED_BY_ENGINE.value:
        row["engine_gaps"] = detail or []
        row["repairable_errors"] = None
    elif status == GenerationStatus.NEEDS_REPAIR.value:
        row["repairable_errors"] = detail or []
        row["engine_gaps"] = None
    else:
        row["engine_gaps"] = None
        row["repairable_errors"] = None
    return row


def _write_rpc(fn: str, body: dict) -> dict:
    """Call one of the conditional write RPCs (migration 0002) as the
    service role. Each decides and writes inside one locked transaction and
    answers `{"outcome": ...}`; any storage failure comes back as the
    `failed` outcome with its reason, never as an exception."""
    url, key = supabase_env.service_role()
    try:
        with httpx.Client(timeout=10.0) as client:
            r = client.post(f"{url}/rest/v1/rpc/{fn}", json=body,
                            headers=supabase_rest.service_headers(key))
    except httpx.RequestError as e:
        logger.warning(f"{fn} errored: {e}")
        return {"outcome": RecordOutcome.FAILED.value, "reason": str(e)[:200]}
    if r.status_code >= 400:
        logger.warning(f"{fn} failed: {r.status_code} {r.text[:200]}")
        return {"outcome": RecordOutcome.FAILED.value,
                "reason": f"{r.status_code} {r.text[:200]}"}
    try:
        out = r.json()
    except ValueError:
        out = None
    if not isinstance(out, dict) or "outcome" not in out:
        logger.warning(f"{fn} answered without an outcome: {r.text[:200]}")
        return {"outcome": RecordOutcome.FAILED.value,
                "reason": f"{fn} answered without an outcome"}
    return out


def record(
    *,
    model_id: str,
    user_id: str,
    spec_yaml: str,
    snapshot_json: Optional[dict] = None,
    new_component_yamls: Optional[list[dict]] = None,
    agent_cli: Optional[str] = None,
    agent_model: Optional[str] = None,
    run_started_at: Optional[str] = None,
) -> dict:
    """Record a successful run's spec as the (user, model)'s working
    candidate (`record_generation`): the newest `MUTABLE` row is overwritten
    in place, else a `generated` row is inserted beside any frozen ones; a
    pending record from a run that began earlier is dropped, one from a run
    that began later stays. Returns the record outcome (`done.record`):
    `recorded` with the id, `failed` with the reason, or `none` when there
    is no user or no Supabase.

    Called from the spec-worker post-success hook with `user_id` = the
    triggering user's auth.users.id (JWT `sub`). Anonymous / single-user
    runs pass no user."""
    if not supabase_env.configured() or not user_id:
        return {"outcome": RecordOutcome.NONE.value}
    return _write_rpc("record_generation", {
        "p_user_id": user_id,
        "p_model_id": model_id,
        "p_spec_yaml": spec_yaml,
        "p_snapshot_json": snapshot_json or {},
        "p_new_component_yamls": new_component_yamls or [],
        "p_agent_cli": agent_cli,
        "p_agent_model": agent_model,
        "p_run_started_at": run_started_at or _now(),
    })


def record_pending(
    *,
    model_id: str,
    user_id: str,
    spec_yaml: str,
    engine_gaps: list,
    engine_version: str,
    snapshot_json: Optional[dict] = None,
    new_component_yamls: Optional[list[dict]] = None,
    agent_cli: Optional[str] = None,
    agent_model: Optional[str] = None,
    run_started_at: Optional[str] = None,
) -> dict:
    """Record an engine-blocked run's spec as the (user, model)'s pending
    record (`record_pending_generation`), beside — never over — its valid
    records. `stale` when a record of this (user, model) holds a bundle from
    a run that began later; else the pending record is inserted or
    overwritten. Same outcome shape as `record`."""
    if not supabase_env.configured() or not user_id:
        return {"outcome": RecordOutcome.NONE.value}
    return _write_rpc("record_pending_generation", {
        "p_user_id": user_id,
        "p_model_id": model_id,
        "p_spec_yaml": spec_yaml,
        "p_snapshot_json": snapshot_json or {},
        "p_new_component_yamls": new_component_yamls or [],
        "p_agent_cli": agent_cli,
        "p_agent_model": agent_model,
        "p_engine_gaps": engine_gaps,
        "p_engine_version": engine_version,
        "p_run_started_at": run_started_at or _now(),
    })


def transition_pending(
    record_id: str,
    *,
    expected_revision: int,
    to: GenerationStatus,
    engine_gaps: Optional[list] = None,
    repairable_errors: Optional[list] = None,
    engine_version: Optional[str] = None,
) -> dict:
    """Refresh a pending record in place as `to` (`blocked_by_engine` with
    its gaps and the engine version, or `needs_repair` with the errors the
    writer resolves), if it is still the record read at `expected_revision`;
    `conflict` otherwise (`transition_pending_generation`)."""
    return _transition(record_id, expected_revision, to.value, {
        "p_engine_gaps": engine_gaps,
        "p_repairable_errors": repairable_errors,
        "p_engine_version": engine_version,
    })


def promote_pending(
    record_id: str,
    *,
    expected_revision: int,
    spec_yaml: Optional[str] = None,
    new_component_yamls: Optional[list] = None,
    agent_cli: Optional[str] = None,
    agent_model: Optional[str] = None,
    run_started_at: Optional[str] = None,
) -> dict:
    """Make a pending record the working candidate, if it is still the
    record read at `expected_revision` (`conflict` otherwise) — with the
    bundle given here (a Fix's result, from the run that began at
    `run_started_at`), else its own. The same lock and the same overwrite
    as a successful generation: the newest `MUTABLE` record takes the bundle
    and the pending record is dropped; with none, the pending record itself
    becomes `generated`. When a valid record from a later run stands, the
    pending record is dropped instead (`superseded`)."""
    return _transition(record_id, expected_revision, PROMOTE, {
        "p_spec_yaml": spec_yaml,
        "p_new_component_yamls": new_component_yamls,
        "p_agent_cli": agent_cli,
        "p_agent_model": agent_model,
        "p_run_started_at": run_started_at,
    })


#: `transition_pending_generation`'s `p_to` for a promotion; every other
#: value it takes is a pending status.
PROMOTE = "promote"


def _transition(record_id: str, expected_revision: int, to: str,
                fields: dict) -> dict:
    if not supabase_env.configured():
        return {"outcome": RecordOutcome.NONE.value}
    return _write_rpc("transition_pending_generation", {
        "p_id": record_id, "p_expected_revision": expected_revision,
        "p_to": to, **fields})


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _lacks_agent_model(r: httpx.Response) -> bool:
    """PostgREST's answer to a write naming `agent_model` before migration
    0041 is deployed."""
    return r.status_code == 400 and "agent_model" in (r.text or "")


def _get(client, base, params, headers, select: str) -> httpx.Response:
    """GET `select` plus `agent_model`; the same read without it while
    migration 0041 is undeployed."""
    r = client.get(base, params={**params, "select": f"{select},agent_model"},
                   headers=headers)
    if _lacks_agent_model(r):
        r = client.get(base, params={**params, "select": select}, headers=headers)
    return r


#: PostgREST filters on a record's status against the pending set: the
#: plain writes below never touch a pending record (it moves on only through
#: `transition_pending` / `promote_pending`, or is discarded whole).
_NOT_PENDING = f"not.in.({','.join(sorted(s.value for s in PENDING))})"
_IS_PENDING = f"in.({','.join(sorted(s.value for s in PENDING))})"


def _svc() -> tuple[str, dict]:
    """(rest_base_url, service-role headers)."""
    url, key = supabase_env.service_role()
    return f"{url}/rest/v1/spec_generations", supabase_rest.service_headers(key)


def list_for_user(user_id: str) -> list[dict]:
    """Summary of a user's own generation records (newest first) for the
    account 'My spec generations' view. Omits the heavy spec_yaml payload."""
    if not supabase_env.configured():
        return []
    base, headers = _svc()
    try:
        with httpx.Client(timeout=10.0) as client:
            r = _get(client, base, {
                "user_id": f"eq.{user_id}",
                "order": "created_at.desc",
            }, headers, "id,model_id,agent_cli,status,pr_url,created_at,"
                        "pending_detail,blocked_engine_version")
        if r.status_code >= 400:
            logger.warning(f"spec_generations list failed: {r.status_code} {r.text[:200]}")
            return []
        return [_project_pending(row) for row in (r.json() or [])]
    except httpx.RequestError as e:
        logger.warning(f"spec_generations list errored: {e}")
        return []


def get(record_id: str, *, strict: bool = False) -> Optional[dict]:
    """Fetch one generation record (full payload incl. spec_yaml / components /
    snapshot) by id. Returns None on miss / error / unconfigured; with
    `strict`, any failed read (transport, 4xx such as 401/403, 5xx) raises
    SpecGenerationsUnavailable instead — PostgREST answers a missing row
    with 200 and no rows — so a caller can tell "gone" from "could not
    ask"."""
    if not supabase_env.configured():
        return None
    base, headers = _svc()
    try:
        with httpx.Client(timeout=10.0) as client:
            r = _get(client, base, {
                "id": f"eq.{record_id}",
            }, headers, _RECORD_COLUMNS)
        if r.status_code >= 400:
            logger.warning(f"spec_generations get failed: {r.status_code} {r.text[:200]}")
            if strict:
                raise SpecGenerationsUnavailable(f"spec_generations get {r.status_code}")
            return None
        rows = r.json() or []
        return _project_pending(rows[0]) if rows else None
    except httpx.RequestError as e:
        logger.warning(f"spec_generations get errored: {e}")
        if strict:
            raise SpecGenerationsUnavailable(str(e)) from e
        return None


def get_for_user(record_id: str, user_id: str) -> Optional[dict]:
    """Fetch one of the caller's OWN generation records (full payload incl.
    spec_yaml / components / snapshot), scoped by user_id. Returns None on miss,
    wrong owner, or unconfigured — so the account detail route answers a
    single 404 for both "no such record" and "not yours" (never leaks a record's
    existence to a non-owner). A Supabase timeout or 5xx raises
    SpecGenerationsUnavailable instead of pretending the row is gone.
    Mirrors get()'s columns with an owner filter."""
    if not supabase_env.configured():
        return None
    base, headers = _svc()
    try:
        with httpx.Client(timeout=10.0) as client:
            r = _get(client, base, {
                "id": f"eq.{record_id}",
                "user_id": f"eq.{user_id}",
            }, headers, _RECORD_COLUMNS)
        if r.status_code >= 500:
            logger.warning(f"spec_generations get_for_user failed: {r.status_code} {r.text[:200]}")
            raise SpecGenerationsUnavailable(
                f"spec_generations get_for_user {r.status_code}")
        if r.status_code >= 400:
            logger.warning(f"spec_generations get_for_user failed: {r.status_code} {r.text[:200]}")
            return None
        rows = r.json() or []
        return _project_pending(rows[0]) if rows else None
    except httpx.RequestError as e:
        logger.warning(f"spec_generations get_for_user errored: {e}")
        raise SpecGenerationsUnavailable(str(e)) from e


#: The records the Inspector analyses and a Refine starts from.
SERVED = SERVABLE | {GenerationStatus.BLOCKED_BY_ENGINE}


def serving_generation(user_id: str, model_id: str) -> Optional[dict]:
    """The user's record the Inspector answers `model_id` with (full
    payload), or None when the user has no record of the model at all.

    A valid record always wins: the newest `SERVABLE` one — `closed`
    included, since a PR closed unmerged says nothing about the spec. With
    no valid record, the `blocked_by_engine` record: its spec is right and
    analyses on every deployment the engine supports. With neither, the
    newest other record (`needs_repair`, `rejected`) — the caller does not
    analyse or seed from it (`SERVED`), but the user does have a spec of the
    model, so it is not a missing spec either: nothing generates one by
    itself.

    A read that fails (transport, any HTTP error) raises
    SpecGenerationsUnavailable: an outage is not "no record", which would
    start a paid generation."""
    if not supabase_env.configured():
        return None
    base, headers = _svc()
    try:
        with httpx.Client(timeout=10.0) as client:
            r = _get(client, base, {
                "user_id": f"eq.{user_id}",
                "model_id": f"eq.{model_id}",
                "order": "created_at.desc",
            }, headers, _RECORD_COLUMNS)
    except httpx.RequestError as e:
        logger.warning(f"spec_generations serving errored: {e}")
        raise SpecGenerationsUnavailable(str(e)) from e
    if r.status_code >= 400:
        logger.warning(f"spec_generations serving failed: {r.status_code} {r.text[:200]}")
        raise SpecGenerationsUnavailable(f"spec_generations serving {r.status_code}")
    rows = [_project_pending(row) for row in (r.json() or [])]
    for rank in (SERVABLE, {GenerationStatus.BLOCKED_BY_ENGINE}):
        row = next((row for row in rows if row.get("status") in rank), None)
        if row is not None:
            return row
    return rows[0] if rows else None


def set_status(
    record_id: str,
    status: str,
    *,
    user_id: Optional[str] = None,
    pr_url: Optional[str] = None,
    contributed_at: Optional[str] = None,
) -> bool:
    """Transition a record's status (service-role PATCH). When `user_id` is
    given the update is scoped to that owner (so a user can only move their own
    row) — used by the user-facing create-pr action; admin actions omit it.
    A pending record is never matched (`discard_pending` removes one): it
    leaves pending only by promotion, which the database enforces too."""
    if not supabase_env.configured():
        return False
    base, headers = _svc()
    params = {"id": f"eq.{record_id}", "status": _NOT_PENDING}
    if user_id:
        params["user_id"] = f"eq.{user_id}"
    patch: dict = {"status": status}
    if pr_url is not None:
        patch["pr_url"] = pr_url
    if contributed_at is not None:
        patch["contributed_at"] = contributed_at
    try:
        with httpx.Client(timeout=10.0) as client:
            r = client.patch(base, params=params,
                             headers={**headers, "Prefer": "return=representation"},
                             json=patch)
        if r.status_code >= 400:
            logger.warning(f"spec_generations patch failed: {r.status_code} {r.text[:200]}")
            return False
        # return=representation → the updated rows. Empty when the (owner-)scoped
        # filter matched nothing — a wrong owner or missing record — so callers
        # (create-pr / admin-reject / sync_pr_status) can tell a real transition
        # from a silent no-op instead of always reporting success on a 204.
        return bool(r.json())
    except httpx.RequestError as e:
        logger.warning(f"spec_generations patch errored: {e}")
        return False


def update_spec(
    record_id: str,
    *,
    spec_yaml: str,
    new_component_yamls: list,
    user_id: Optional[str] = None,
    snapshot_json: Optional[dict] = None,
) -> bool:
    """Replace a record's spec bundle IN PLACE (service-role PATCH). Used by the
    agent-fix flow: a scoped repair rewrites the record's own draft rather than
    appending a new generation row. `snapshot_json`, when given, replaces the
    record's snapshot with the one the fix ran against, so the checkpoint
    facts a PR ships are the ones the spec was last judged on. When `user_id` is given the update is scoped
    to that owner. `return=representation` so a wrong-owner / missing-record
    no-op is distinguishable from a real update (empty list vs non-empty).
    A pending record is never matched: its fix lands by `promote_pending`."""
    if not supabase_env.configured():
        return False
    base, headers = _svc()
    params = {"id": f"eq.{record_id}", "status": _NOT_PENDING}
    if user_id:
        params["user_id"] = f"eq.{user_id}"
    patch = {
        "spec_yaml": spec_yaml,
        "new_component_yamls": new_component_yamls or [],
    }
    if snapshot_json:
        patch["snapshot_json"] = snapshot_json
    try:
        with httpx.Client(timeout=10.0) as client:
            r = client.patch(base, params=params,
                             headers={**headers, "Prefer": "return=representation"},
                             json=patch)
        if r.status_code >= 400:
            logger.warning(f"spec_generations spec update failed: {r.status_code} {r.text[:200]}")
            return False
        return bool(r.json())
    except httpx.RequestError as e:
        logger.warning(f"spec_generations spec update errored: {e}")
        return False


def discard_pending(record_id: str) -> bool:
    """An admin's rejection of a pending record: the record is deleted (its
    reviews cascade), never turned `rejected` — a rejected row is a working
    candidate, and a pending record must not become a second one. True only
    when a pending row was deleted."""
    if not supabase_env.configured():
        return False
    base, headers = _svc()
    try:
        with httpx.Client(timeout=10.0) as client:
            r = client.delete(base, params={"id": f"eq.{record_id}", "status": _IS_PENDING},
                              headers={**headers, "Prefer": "return=representation"})
        if r.status_code >= 400:
            logger.warning(f"spec_generations discard failed: {r.status_code} {r.text[:200]}")
            return False
        return bool(r.json())
    except httpx.RequestError as e:
        logger.warning(f"spec_generations discard errored: {e}")
        return False


def delete_for_user(record_id: str, user_id: str) -> bool:
    """Delete one of the caller's OWN generation records (service-role DELETE
    scoped by user_id). `merged` records and `contributed` records with an open
    PR are NOT deletable — dropping one would orphan the PR reconciliation. A
    `contributed` record whose PR was never opened (`pr_url` null) is. Reviews of the record
    cascade (0030). Returns True only when a row was actually deleted — an
    empty result means wrong owner / protected status / missing record, so the
    route can answer 404 instead of claiming success on a no-op."""
    if not supabase_env.configured():
        return False
    base, headers = _svc()
    params = {
        "id": f"eq.{record_id}",
        "user_id": f"eq.{user_id}",
        "or": "(status.not.in.(contributed,merged),and(status.eq.contributed,pr_url.is.null))",
    }
    try:
        with httpx.Client(timeout=10.0) as client:
            r = client.delete(base, params=params,
                              headers={**headers, "Prefer": "return=representation"})
        if r.status_code >= 400:
            logger.warning(f"spec_generations delete failed: {r.status_code} {r.text[:200]}")
            return False
        return bool(r.json())
    except httpx.RequestError as e:
        logger.warning(f"spec_generations delete errored: {e}")
        return False


def sync_pr_status(record: dict) -> Optional[str]:
    """Reconcile a record's status against its PR, best-effort. Returns the new
    status if it changed (and was persisted), else None.

    There is no webhook infra: PR state is reconciled lazily when a record is
    viewed (poll-on-view). Only a record that carries a `pr_url` and has not
    already `merged` is checked; a still-open PR is left untouched. A PR that
    merged flips to `merged`; a PR closed without merging flips to `closed` —
    distinct from `rejected`, which is an admin's decision made *before* a PR
    ever existed. Guarding on `!= "merged"` keeps merged terminal even if a
    previously-closed PR is reopened and later lands. `rejected` is likewise
    terminal — an admin's decision the PR reconciler must never override.
    """
    pr_url = record.get("pr_url")
    if not pr_url or record.get("status") in ("merged", "rejected"):
        return None
    state = _github.pr_merged_state(pr_url)
    new = "merged" if state == "merged" else ("closed" if state == "closed" else None)
    if not new or new == record.get("status"):
        return None
    if set_status(record["id"], new):
        return new
    return None
