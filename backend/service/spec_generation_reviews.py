"""Per-reviewer agent review reports for a spec generation
(public.spec_generation_reviews, migration 0030).

An on-demand agent reads a generated spec against the model's real architecture
and returns a verdict (pass/no_pass, or error when it produced none) + a written
report. Both the generating user and admins can review; each reviewer stores
their own row. Written service-role from the spec-worker at review completion via
a lazy import guarded by `supabase_env.configured()`, exactly like
`spec_generation_logs` / `spec_generations`, so the core↔service boundary test
stays happy.

`record` silent-fails (returns None, never raises) — a storage outage must not
affect the user-facing review run.

A PASS row also carries its admission (migration 0038, ADMIT §4.2): the
`admission` JSONB column holds a job envelope `{"job": "running" | "done" |
"failed", "report": <AdmissionReport> | null, ...}`. The review INSERT never
names it, so a review row lands whether or not the migration is deployed;
`set_admission` PATCHes it afterwards and reads before the migration fall back
to rows without it, marked `admission_unavailable`.
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Optional

import httpx

from backend import supabase_env, supabase_rest
from backend.service import spec_generation_logs


logger = logging.getLogger(__name__)

# Tail-keep cap: a review report is small markdown, but the agent transcript it
# is derived from can be large; guard the row regardless. Keep the END (the
# verdict + findings are the outcome) and mark the truncation.
_MAX_REPORT_CHARS = 128 * 1024

#: newest first; `id` breaks a tie between two rows written in the same
#: instant, so "the latest review" is one row on every read (ADMIT I1)
_ORDER = "created_at.desc,id.desc"
_SELECT = ("id,model_id,reviewer_role,verdict,report,agent_cli,created_at,"
           "bundle_fingerprint,run_revision,engine_version")


def _truncate(report: str) -> str:
    if len(report) <= _MAX_REPORT_CHARS:
        return report
    kept = report[-_MAX_REPORT_CHARS:]
    return f"[... truncated {len(report) - _MAX_REPORT_CHARS} leading chars ...]\n{kept}"


def record(*, record_id: str, model_id: str, reviewer_id: str,
           reviewer_role: str, verdict: str, report: str,
           agent_cli: Optional[str] = None,
           bundle_fingerprint: Optional[str] = None,
           run_revision: Optional[str] = None,
           engine_version: Optional[str] = None) -> Optional[str]:
    """Store one review row. Service-role INSERT (bypasses RLS; no client
    writes). Returns the new row's id (the admission job PATCHes that row),
    or None on silent failure / no reviewer / unconfigured. Persisted for
    pass/no_pass/error alike. The row names the run log the review was
    written by (`log_id`, migration 0020): the durability row bound to the
    calling job thread (`spec_generation_logs.bind_run_row`). While the
    column is undeployed the insert retries without it."""
    if not supabase_env.configured() or not reviewer_id:
        return None
    url, key = supabase_env.service_role()
    payload = {
        "record_id": record_id,
        "model_id": model_id,
        "reviewer_id": reviewer_id,
        "reviewer_role": reviewer_role,
        "verdict": verdict,
        "report": _truncate(report or ""),
        "agent_cli": agent_cli,
        # What the verdict applies to (migration 0037): Open PR compares it
        # with the record's current bundle.
        "bundle_fingerprint": bundle_fingerprint,
        "run_revision": run_revision,
        "engine_version": engine_version,
    }
    log_id = spec_generation_logs.bound_run_row()
    if log_id:
        payload["log_id"] = log_id
    headers = {
        "apikey": key,
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
        "Prefer": "return=representation",
    }
    try:
        with httpx.Client(timeout=5.0) as client:
            r = client.post(f"{url}/rest/v1/spec_generation_reviews",
                            json=payload, headers=headers)
            if ("log_id" in payload and r.status_code >= 400
                    and supabase_rest.is_unknown_column(r.status_code, r.text)):
                del payload["log_id"]
                r = client.post(f"{url}/rest/v1/spec_generation_reviews",
                                json=payload, headers=headers)
        if r.status_code >= 400:
            logger.warning(
                f"spec_generation_reviews insert failed: {r.status_code} {r.text[:200]}"
            )
            return None
        rows = r.json() if r.content else []
        row = rows[0] if isinstance(rows, list) and rows else {}
        return str(row.get("id") or "") or None
    except (httpx.RequestError, ValueError) as e:
        logger.warning(f"spec_generation_reviews insert errored: {e}")
        return None


def set_admission(review_id: str, envelope: dict) -> bool:
    """Write a review row's admission envelope. False (logged, never
    raised) when the store is unreachable or the `admission` column is not
    deployed yet: the report then lives only in the job's log."""
    if not supabase_env.configured() or not review_id:
        return False
    url, key = supabase_env.service_role()
    try:
        with httpx.Client(timeout=10.0) as client:
            r = client.patch(
                f"{url}/rest/v1/spec_generation_reviews",
                params={"id": f"eq.{review_id}"},
                json={"admission": envelope},
                headers=supabase_rest.service_headers(key, prefer="return=minimal"),
            )
    except httpx.RequestError as e:
        logger.warning(f"spec_generation_reviews admission write errored: {e}")
        return False
    if r.status_code >= 400:
        if supabase_rest.is_unknown_column(r.status_code, r.text):
            logger.warning("spec_generation_reviews.admission column is not deployed "
                           "(migration 0038); the admission report stays in the job log")
        else:
            logger.warning(f"spec_generation_reviews admission write failed: "
                           f"{r.status_code} {r.text[:200]}")
        return False
    return True


def _select_with_admission(client: httpx.Client, url: str, headers: dict,
                           params: dict) -> tuple[Optional[httpx.Response], bool]:
    """GET with the `admission` column, or — before migration 0038 — the
    same read without it. Returns (response, whether the column exists)."""
    r = client.get(f"{url}/rest/v1/spec_generation_reviews",
                   params={**params, "select": f"{_SELECT},admission"}, headers=headers)
    if r.status_code < 400 or not supabase_rest.is_unknown_column(r.status_code, r.text):
        return r, True
    r = client.get(f"{url}/rest/v1/spec_generation_reviews",
                   params={**params, "select": _SELECT}, headers=headers)
    return r, False


def mark_admission_unavailable(rows: list[dict], deployed: bool) -> list[dict]:
    """Tag rows read without the `admission` column, so a reader tells "no
    report yet" from "this server cannot store one"."""
    if not deployed:
        for row in rows:
            row["admission_unavailable"] = True
    return rows


def _created(row: dict) -> datetime:
    try:
        return datetime.fromisoformat(str(row.get("created_at")).replace("Z", "+00:00"))
    except (ValueError, TypeError):
        return datetime(1970, 1, 1, tzinfo=timezone.utc)


def order_key(row: dict) -> tuple:
    return (_created(row), str(row.get("id") or ""))


def newer_than(rows: list[dict], review_id: str) -> list[dict]:
    """The rows written after review `review_id` — by (created_at, id), so a
    row stamped in the same instant still counts (ADMIT I1). All rows when
    `review_id` is not among them."""
    mine = next((row for row in rows if str(row.get("id")) == str(review_id)), None)
    if mine is None:
        return list(rows)
    return [row for row in rows if order_key(row) > order_key(mine)]


def list_for_record(record_id: str) -> list[dict]:
    """Every review of a record, newest first, unscoped. For the service's own
    jobs (an Open PR re-reading the latest verdict before it writes GitHub),
    never a response to a caller."""
    if not supabase_env.configured() or not record_id:
        return []
    url, key = supabase_env.service_role()
    headers = {"apikey": key, "Authorization": f"Bearer {key}"}
    try:
        with httpx.Client(timeout=10.0) as client:
            r, deployed = _select_with_admission(
                client, url, headers, {"record_id": f"eq.{record_id}", "order": _ORDER})
        if r is None or r.status_code >= 400:
            return []
        return mark_admission_unavailable(r.json() or [], deployed)
    except httpx.RequestError as e:
        logger.warning(f"spec_generation_reviews read errored: {e}")
        return []


def latest_pr_report(rows: list[dict]) -> Optional[dict]:
    """The newest admission report written by an Open PR / update-pr run
    (`mode: "pr"` with the PR's head commit): what the bot last pushed."""
    for row in sorted(rows, key=order_key, reverse=True):
        envelope = row.get("admission") or {}
        report = envelope.get("report") if isinstance(envelope, dict) else None
        if isinstance(report, dict) and report.get("mode") == "pr" \
                and (report.get("pr") or {}).get("head_sha"):
            return report
    return None


def admission_of(row: dict) -> Optional[dict[str, Any]]:
    envelope = row.get("admission")
    return envelope if isinstance(envelope, dict) else None


def latest_verdicts_for_records(record_ids: list[str]) -> dict[str, str]:
    """Latest review verdict per record, in one bulk query — for list rows.
    `record_ids` must already be owner-scoped by the caller (they come from the
    caller's own generation rows); this adds no ownership check of its own."""
    if not supabase_env.configured() or not record_ids:
        return {}
    url, key = supabase_env.service_role()
    headers = {"apikey": key, "Authorization": f"Bearer {key}"}
    try:
        with httpx.Client(timeout=10.0) as client:
            r = client.get(f"{url}/rest/v1/spec_generation_reviews", params={
                "record_id": f"in.({','.join(record_ids)})",
                "select": "record_id,verdict",
                "order": _ORDER,
            }, headers=headers)
        if r.status_code >= 400:
            logger.warning(f"spec_generation_reviews bulk list failed: {r.status_code} {r.text[:200]}")
            return {}
        out: dict[str, str] = {}
        for row in r.json() or []:
            out.setdefault(row["record_id"], str(row.get("verdict") or ""))
        return out
    except httpx.RequestError as e:
        logger.warning(f"spec_generation_reviews bulk list errored: {e}")
        return {}


def verdicts_by_log(record_ids: list[str]) -> dict[str, str]:
    """The recorded verdict of each review run, keyed by its run log
    (`log_id`, migration 0020), over the given records' reviews — so a list
    of review logs can say what each review concluded. Empty while the
    column is undeployed. `record_ids` must already be owner-scoped."""
    if not supabase_env.configured() or not record_ids:
        return {}
    url, key = supabase_env.service_role()
    headers = {"apikey": key, "Authorization": f"Bearer {key}"}
    try:
        with httpx.Client(timeout=10.0) as client:
            r = client.get(f"{url}/rest/v1/spec_generation_reviews", params={
                "record_id": f"in.({','.join(record_ids)})",
                "log_id": "not.is.null",
                "select": "log_id,verdict",
            }, headers=headers)
    except httpx.RequestError as e:
        logger.warning(f"spec_generation_reviews log verdicts errored: {e}")
        return {}
    if r.status_code >= 400:
        if not supabase_rest.is_unknown_column(r.status_code, r.text):
            logger.warning(f"spec_generation_reviews log verdicts failed: "
                           f"{r.status_code} {r.text[:200]}")
        return {}
    return {str(row["log_id"]): str(row.get("verdict") or "") for row in r.json() or []}


def list_for_record_owner(record_id: str, user_id: str) -> list[dict]:
    """Reviews of a generation the caller OWNS (their own + any admin's).

    Service-role bypasses RLS, so ownership is enforced here: confirm the record
    belongs to `user_id` first, else return [] (a caller can never read reviews
    of a record they don't own). `reviewer_id` is omitted from the response.
    """
    if not supabase_env.configured() or not user_id:
        return []
    url, key = supabase_env.service_role()
    headers = {"apikey": key, "Authorization": f"Bearer {key}"}
    try:
        with httpx.Client(timeout=10.0) as client:
            owns = client.get(f"{url}/rest/v1/spec_generations", params={
                "id": f"eq.{record_id}",
                "user_id": f"eq.{user_id}",
                "select": "id",
            }, headers=headers)
            if owns.status_code >= 400 or not (owns.json() or []):
                return []
            r, deployed = _select_with_admission(
                client, url, headers, {"record_id": f"eq.{record_id}", "order": _ORDER})
        if r.status_code >= 400:
            logger.warning(f"spec_generation_reviews list failed: {r.status_code} {r.text[:200]}")
            return []
        return mark_admission_unavailable(r.json() or [], deployed)
    except httpx.RequestError as e:
        logger.warning(f"spec_generation_reviews list errored: {e}")
        return []
