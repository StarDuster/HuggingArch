"""Per-user spec-generation run logs (public.spec_generation_logs, migration
0019).

Binds each agent run's log to the triggering user in the DB — success OR
failure — so logs survive cross-VM/redeploy and failed runs (the case you most
need the log for) are captured too. Called service-role from the spec-worker at
run completion via a lazy import guarded by `supabase_env.configured()`, exactly like
`shared_spec_cache` / `spec_generations`, so the core↔service boundary test stays happy.

`record` silent-fails (returns False, never raises) — a logging outage must not
affect the user-facing run.
"""
from __future__ import annotations

import logging
import threading
from typing import Optional

import httpx

from backend import supabase_env, supabase_rest


logger = logging.getLogger(__name__)

# Migration 0034 columns on spec_generation_logs. None = unprobed; False means
# PostgREST reported an unknown column — skip running-state persistence and
# keep the original transcript INSERT.
_durability_columns: Optional[bool] = None
_durability_guard = threading.Lock()
_durability_warned = False
_DURABILITY_FIELDS = ("state", "message", "host_id", "boot_id", "finished_at", "kind")
# Columns `record` writes beside the transcript: which credential and model
# ran (migration 0041), and the record the run wrote (migration 0005).
_RUN_FIELDS = ("agent_cli", "agent_model", "record_id")


def _reset_schema_compat_for_tests() -> None:
    global _durability_columns, _durability_warned
    _durability_columns = None
    _durability_warned = False


def _durability_on() -> bool:
    return _durability_columns is not False


def _note_unknown_column(status_code: int, text: str) -> bool:
    global _durability_columns, _durability_warned
    if not supabase_rest.is_unknown_column(status_code, text):
        return False
    with _durability_guard:
        _durability_columns = False
        if not _durability_warned:
            _durability_warned = True
            logger.warning(
                "spec_generation_logs missing the 0034 durability columns; "
                "not persisting running job state. The database schema is "
                "behind. In-memory job status and transcript INSERT still work."
            )
    return True

# Tail-keep cap: an agent transcript can run to hundreds of KB (a big prompt
# echo + many turns). Keep the END — the outcome + last actions are what matter
# for debugging — and mark the truncation.
_MAX_LOG_CHARS = 256 * 1024

# Insert used to time out at 5s on a concurrent 09-06 batch, so failure
# transcripts never landed. One retry after a transport/timeout error.
_LOG_HTTP_TIMEOUT = 30.0


def _client_call(client, method: str, *args, **kwargs):
    if method == "get":
        return client.get(*args, **kwargs)
    if method == "post":
        return client.post(*args, **kwargs)
    if method == "patch":
        return client.patch(*args, **kwargs)
    raise ValueError(f"unsupported spec_generation_logs method {method!r}")


def _log_request(method: str, *args, **kwargs):
    """POST/PATCH/GET with a single retry on timeout or transport error."""
    try:
        with httpx.Client(timeout=_LOG_HTTP_TIMEOUT) as client:
            return _client_call(client, method, *args, **kwargs)
    except httpx.RequestError as exc:
        logger.warning(f"spec_generation_logs {method} errored ({exc}); retrying once")
        with httpx.Client(timeout=_LOG_HTTP_TIMEOUT) as client:
            return _client_call(client, method, *args, **kwargs)


def _truncate(log: str) -> str:
    if len(log) <= _MAX_LOG_CHARS:
        return log
    kept = log[-_MAX_LOG_CHARS:]
    return f"[... truncated {len(log) - _MAX_LOG_CHARS} leading chars ...]\n{kept}"


def begin_run(*, run_key: str, model_id: str, host_id: str, boot_id: str,
              user_id: Optional[str] = None, kind: Optional[str] = None) -> Optional[str]:
    """Insert a `state=running` row at job start so a restart can find it.
    Silent-fail; returns the new row id, or None."""
    if not supabase_env.configured() or not run_key:
        return None
    if not _durability_on():
        return None
    # A leftover running row for this key (a previous finish that never
    # landed) would collide with the one-running unique index. Close it
    # first; acquire() already serializes a live run on the same key.
    finish_run(run_key=run_key, success=False,
               message="superseded by a new run", state="interrupted")
    if not _durability_on():
        return None
    url, key = supabase_env.service_role()
    payload = {
        "model_id": model_id,
        "run_key": run_key,
        "state": "running",
        "host_id": host_id,
        "boot_id": boot_id,
        "log": "",
    }
    if user_id:
        payload["user_id"] = user_id
    if kind:
        payload["kind"] = kind
    try:
        r = _log_request(
            "post",
            f"{url}/rest/v1/spec_generation_logs",
            json=payload,
            headers=supabase_rest.service_headers(
                key, prefer="return=representation"),
        )
        if r.status_code >= 400:
            if _note_unknown_column(r.status_code, r.text):
                return None
            logger.warning(
                f"spec_generation_logs begin failed: {r.status_code} {r.text[:200]}"
            )
            return None
        rows = r.json() or []
        if isinstance(rows, list) and rows:
            return rows[0].get("id")
        return None
    except httpx.RequestError as e:
        logger.warning(f"spec_generation_logs begin errored: {e}")
        return None


def finish_run(*, run_key: str, success: Optional[bool], message: str,
               state: str = "done", row_id: Optional[str] = None,
               error_kind: Optional[str] = None,
               engine_gaps: Optional[list] = None) -> bool:
    """Close one durability row. Prefers `row_id`; else the running row for
    `run_key`. `error_kind` / `engine_gaps` (migration 0002) are the
    verdict's machine fields; while those columns are undeployed the close
    retries without them. Silent-fail."""
    if not supabase_env.configured() or not (row_id or run_key):
        return False
    if not _durability_on():
        return False
    url, key = supabase_env.service_role()
    from datetime import datetime, timezone
    payload = {
        "state": state,
        "success": success,
        "message": message or "",
        "finished_at": datetime.now(timezone.utc).isoformat(),
    }
    # Migration 0002 columns: the run's error kind, a blocked run's gaps.
    terminal = {name: value for name, value in (
        ("error_kind", error_kind), ("engine_gaps", engine_gaps))
        if value is not None}
    payload.update(terminal)
    params = {"id": f"eq.{row_id}"} if row_id else {
        "run_key": f"eq.{run_key}", "state": "eq.running",
    }
    try:
        r = _log_request(
            "patch",
            f"{url}/rest/v1/spec_generation_logs",
            params=params,
            json=payload,
            headers=supabase_rest.service_headers(key, prefer="return=minimal"),
        )
        if _lacks_columns(r, terminal):
            r = _log_request(
                "patch",
                f"{url}/rest/v1/spec_generation_logs",
                params=params,
                json={k: v for k, v in payload.items() if k not in terminal},
                headers=supabase_rest.service_headers(key, prefer="return=minimal"),
            )
        if r.status_code >= 400:
            if _note_unknown_column(r.status_code, r.text):
                return False
            logger.warning(
                f"spec_generation_logs finish failed: {r.status_code} {r.text[:200]}"
            )
            return False
        return True
    except httpx.RequestError as e:
        logger.warning(f"spec_generation_logs finish errored: {e}")
        return False


def _lacks_columns(r: httpx.Response, names) -> bool:
    """PostgREST's answer to a request naming one of `names` before the
    migration that adds it is deployed."""
    return r.status_code == 400 and any(c in (r.text or "") for c in names)


def latest_by_run_key(run_key: str) -> Optional[dict]:
    """Newest durability row for a run_key (`state` in running/done/interrupted).
    Transcript-only rows without state are ignored. None on miss."""
    if not supabase_env.configured() or not run_key:
        return None
    if not _durability_on():
        return None
    url, key = supabase_env.service_role()
    select = ("id,model_id,user_id,run_key,kind,success,state,message,"
              "host_id,boot_id,created_at,finished_at")
    params = {
        "run_key": f"eq.{run_key}",
        "or": "(state.eq.running,state.eq.done,state.eq.interrupted)",
        "order": "created_at.desc",
        "limit": "1",
    }
    headers = supabase_rest.service_headers(key, json_body=False)
    try:
        with httpx.Client(timeout=10.0) as client:
            r = client.get(f"{url}/rest/v1/spec_generation_logs", params={
                **params, "select": f"{select},error_kind,engine_gaps",
            }, headers=headers)
            if _lacks_columns(r, ("error_kind", "engine_gaps")):
                r = client.get(f"{url}/rest/v1/spec_generation_logs",
                               params={**params, "select": select},
                               headers=headers)
        if r.status_code >= 400:
            if _note_unknown_column(r.status_code, r.text):
                return None
            logger.warning(
                f"spec_generation_logs latest failed: {r.status_code} {r.text[:200]}"
            )
            return None
        rows = r.json() or []
        return rows[0] if rows else None
    except httpx.RequestError as e:
        logger.warning(f"spec_generation_logs latest errored: {e}")
        return None


def list_running() -> list[dict]:
    """Running durability rows (service-role). Empty on miss / unconfigured."""
    if not supabase_env.configured():
        return []
    if not _durability_on():
        return []
    url, key = supabase_env.service_role()
    try:
        with httpx.Client(timeout=10.0) as client:
            r = client.get(f"{url}/rest/v1/spec_generation_logs", params={
                "state": "eq.running",
                "select": "id,model_id,run_key,host_id,boot_id,created_at",
            }, headers=supabase_rest.service_headers(key, json_body=False))
        if r.status_code >= 400:
            if _note_unknown_column(r.status_code, r.text):
                return []
            logger.warning(
                f"spec_generation_logs list_running failed: {r.status_code} {r.text[:200]}"
            )
            return []
        return r.json() or []
    except httpx.RequestError as e:
        logger.warning(f"spec_generation_logs list_running errored: {e}")
        return []


def interrupt_ids(ids: list[str], message: str) -> int:
    """Mark the given log row ids interrupted. Returns how many PATCH calls succeeded."""
    if not supabase_env.configured() or not ids:
        return 0
    if not _durability_on():
        return 0
    url, key = supabase_env.service_role()
    from datetime import datetime, timezone
    payload = {
        "state": "interrupted",
        "success": False,
        "message": message,
        "finished_at": datetime.now(timezone.utc).isoformat(),
    }
    n = 0
    try:
        with httpx.Client(timeout=10.0) as client:
            for row_id in ids:
                r = client.patch(
                    f"{url}/rest/v1/spec_generation_logs",
                    params={"id": f"eq.{row_id}"},
                    json=payload,
                    headers=supabase_rest.service_headers(key, prefer="return=minimal"),
                )
                if r.status_code < 400:
                    n += 1
                elif _note_unknown_column(r.status_code, r.text):
                    return n
                else:
                    logger.warning(
                        f"spec_generation_logs interrupt failed: {r.status_code} {r.text[:200]}"
                    )
    except httpx.RequestError as e:
        logger.warning(f"spec_generation_logs interrupt errored: {e}")
    return n


def _legacy_record_payload(*, user_id: str, model_id: str, log: str,
                           run_key: Optional[str], success: Optional[bool]) -> dict:
    return {
        "user_id": user_id,
        "model_id": model_id,
        "run_key": run_key,
        "success": success,
        "log": _truncate(log or ""),
    }


def _post_log(url: str, key: str, payload: dict) -> bool:
    try:
        r = _log_request(
            "post",
            f"{url}/rest/v1/spec_generation_logs",
            json=payload,
            headers=supabase_rest.service_headers(
                key, prefer="return=minimal"),
        )
        if r.status_code >= 400:
            if _note_unknown_column(r.status_code, r.text):
                stripped = {k: v for k, v in payload.items()
                            if k not in _DURABILITY_FIELDS}
                if stripped != payload:
                    return _post_log(url, key, stripped)
            logger.warning(
                f"spec_generation_logs insert failed: {r.status_code} {r.text[:200]}"
            )
            return False
        return True
    except httpx.RequestError as e:
        logger.warning(f"spec_generation_logs insert errored: {e}")
        return False


class _BoundRow(threading.local):
    """Per job thread: the durability row its run began. None elsewhere."""

    def __init__(self) -> None:
        self.row_id: Optional[str] = None


_bound = _BoundRow()


def bind_run_row(row_id: Optional[str]) -> None:
    """Name the durability row the calling job thread's run began; `record`
    writes that run's transcript onto it."""
    _bound.row_id = row_id


def bound_run_row() -> Optional[str]:
    """The durability row the calling job thread's run began, or None (no
    durability row, or a thread that runs no job)."""
    return _bound.row_id


def record(*, user_id: str, model_id: str, log: str,
           run_key: Optional[str] = None, success: Optional[bool] = None,
           agent_cli: Optional[str] = None,
           agent_model: Optional[str] = None,
           record_id: Optional[str] = None) -> bool:
    """Write the run transcript onto the durability row this run began
    (`begin_run`'s id, bound to the job thread by `bind_run_row`) — never
    onto whichever row of `run_key` is newest, which by then may be a later
    run's. Does not insert a parallel row and does not change
    `state` or `success` — an interrupted/done row keeps its terminal
    verdict. INSERTs a row with a terminal `state` only when the run began
    none. When 0034 columns are missing, INSERT the
    original transcript fields only. `agent_cli`/`agent_model` (migration
    0041) record which credential and model the run used; `record_id`
    (migration 0005) is the generation record the run wrote, the one link
    from a record to the run that produced it. While those columns are
    undeployed the write retries without them. Returns True on success,
    False on silent failure / no user / unconfigured. Only signed-in runs
    carry a user_id and reach here."""
    if not supabase_env.configured() or not user_id:
        return False
    if success is None:
        success = False
    url, key = supabase_env.service_role()
    legacy = _legacy_record_payload(
        user_id=user_id, model_id=model_id, log=log,
        run_key=run_key, success=success,
    )
    if not _durability_on():
        return _post_log(url, key, legacy)
    payload = {
        "user_id": user_id,
        "model_id": model_id,
        "log": _truncate(log or ""),
    }
    if agent_cli is not None:
        payload["agent_cli"] = agent_cli
    if agent_model is not None:
        payload["agent_model"] = agent_model
    if record_id is not None:
        payload["record_id"] = record_id
    row_id = _bound.row_id
    try:
        headers = supabase_rest.service_headers(key, prefer="return=minimal")
        if row_id:
            # Transcript only. Interrupted/done already carry the terminal
            # verdict; a late success=True must not overwrite it.
            r = _log_request(
                "patch",
                f"{url}/rest/v1/spec_generation_logs",
                params={"id": f"eq.{row_id}"},
                json=payload,
                headers=headers,
            )
        else:
            payload["run_key"] = run_key
            payload["state"] = "done"
            payload["success"] = success
            # begin_run already stamped kind; this INSERT path is the
            # fallback when no durability row exists. fix| / review| keys
            # still need kind so the account page can attach them.
            head = str(run_key or "").split("|", 1)[0]
            if head in ("fix", "review"):
                payload["kind"] = head
            r = _log_request(
                "post",
                f"{url}/rest/v1/spec_generation_logs",
                json=payload,
                headers=headers,
            )
        if r.status_code >= 400:
            if _lacks_columns(r, _RUN_FIELDS):
                # Migration 0041 / 0005 undeployed: retry without their columns.
                slim = {k: v for k, v in payload.items()
                        if k not in _RUN_FIELDS}
                if row_id:
                    r = _log_request(
                        "patch",
                        f"{url}/rest/v1/spec_generation_logs",
                        params={"id": f"eq.{row_id}"},
                        json=slim,
                        headers=headers,
                    )
                else:
                    r = _log_request(
                        "post",
                        f"{url}/rest/v1/spec_generation_logs",
                        json=slim,
                        headers=headers,
                    )
            if r.status_code >= 400:
                if _note_unknown_column(r.status_code, r.text):
                    return _post_log(url, key, legacy)
                logger.warning(
                    f"spec_generation_logs insert failed: {r.status_code} {r.text[:200]}"
                )
                return False
        return True
    except httpx.RequestError as e:
        logger.warning(f"spec_generation_logs insert errored: {e}")
        return False


def delete_for_user(log_id: str, user_id: str) -> bool:
    """Delete one of the caller's OWN run-log rows (service-role DELETE scoped
    by user_id) — the generations list manages orphan logs (failed runs with
    no spec) inline, so they need the same cleanup path as spec records.
    Returns True only when a row was actually deleted (empty result = wrong
    owner / missing), never raises."""
    if not supabase_env.configured() or not user_id:
        return False
    url, key = supabase_env.service_role()
    try:
        with httpx.Client(timeout=10.0) as client:
            r = client.delete(
                f"{url}/rest/v1/spec_generation_logs",
                params={"id": f"eq.{log_id}", "user_id": f"eq.{user_id}"},
                headers=supabase_rest.service_headers(
                    key, json_body=False, prefer="return=representation"),
            )
        if r.status_code >= 400:
            logger.warning(
                f"spec_generation_logs delete failed: {r.status_code} {r.text[:200]}"
            )
            return False
        return bool(r.json())
    except httpx.RequestError as e:
        logger.warning(f"spec_generation_logs delete errored: {e}")
        return False



def _get_summary(client, url: str, key: str, params: dict, select: str):
    """GET `select` plus agent_cli,agent_model (migration 0041),
    error_kind,engine_gaps (migration 0002) and record_id (migration 0005);
    the same read without them while a column is undeployed."""
    headers = supabase_rest.service_headers(key, json_body=False)
    extra = ("agent_cli", "agent_model", "error_kind", "engine_gaps", "record_id")
    r = client.get(f"{url}/rest/v1/spec_generation_logs",
                   params={**params, "select": f"{select},{','.join(extra)}"},
                   headers=headers)
    if _lacks_columns(r, extra):
        r = client.get(f"{url}/rest/v1/spec_generation_logs",
                       params={**params, "select": select}, headers=headers)
    return r


def get(record_id: str, user_id: str) -> Optional[dict]:
    """Full log row (incl. the `log` body) for the OWNER — scoped by user_id so
    a caller can never read another user's log. None on miss / not-owner."""
    if not supabase_env.configured() or not user_id:
        return None
    url, key = supabase_env.service_role()
    try:
        with httpx.Client(timeout=10.0) as client:
            r = _get_summary(client, url, key, {
                "id": f"eq.{record_id}",
                "user_id": f"eq.{user_id}",
            }, "id,model_id,run_key,kind,success,log,created_at")
        if r.status_code >= 400:
            logger.warning(f"spec_generation_logs get failed: {r.status_code} {r.text[:200]}")
            return None
        rows = r.json() or []
        return rows[0] if rows else None
    except httpx.RequestError as e:
        logger.warning(f"spec_generation_logs get errored: {e}")
        return None


def list_for_user(user_id: str, limit: int = 50) -> list[dict]:
    """A user's own run-log summaries (newest first), omitting the heavy `log`
    body — for a future 'my runs' view."""
    if not supabase_env.configured() or not user_id:
        return []
    url, key = supabase_env.service_role()
    try:
        with httpx.Client(timeout=10.0) as client:
            r = _get_summary(client, url, key, {
                "user_id": f"eq.{user_id}",
                "order": "created_at.desc",
                "limit": str(limit),
            }, "id,model_id,run_key,kind,success,created_at")
        if r.status_code >= 400:
            logger.warning(f"spec_generation_logs list failed: {r.status_code} {r.text[:200]}")
            return []
        return r.json() or []
    except httpx.RequestError as e:
        logger.warning(f"spec_generation_logs list errored: {e}")
        return []
