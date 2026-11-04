"""Requests made to Supabase AS the service role.

Every server-side Supabase call carries the same identity: the service-role
key twice, once as `apikey` (the gateway's project key) and once as the
`Authorization` bearer (the Postgres role PostgREST runs the query under).
`service_headers` spells that once; a caller adds only what its request
differs by (a JSON body, a `Prefer`).

`rpc` is the one POST to `/rest/v1/rpc/<fn>` the cache and lock modules make.
It never raises for a network error or an HTTP error status — both come back
as None and a warning — because every caller treats a failed RPC as a miss or
a refusal, not an outage. Whether an undecodable body is also a miss is the
caller's choice (`strict_json`): the caches say yes, the spec lock says no.

Lives beside `supabase_env` (not in `backend.service`) for the same reason:
the spec worker's lock RPC is core code and may not import the service layer.
"""
from __future__ import annotations

import logging
import os
import threading
from typing import Optional, Union

import httpx

from backend import supabase_env


def is_missing_rpc(status_code: int, text: str = "") -> bool:
    """PostgREST has no such function in the schema cache (404 / PGRST202)."""
    if status_code == 404:
        return True
    return "PGRST202" in (text or "")


def is_unknown_column(status_code: int, text: str = "") -> bool:
    """PostgREST / Postgres: the request named a column that is not there."""
    t = text or ""
    if "PGRST204" in t or "42703" in t:
        return True
    low = t.lower()
    return "column" in low and ("schema cache" in low or "does not exist" in low)


def service_headers(key: str, *, json_body: bool = True,
                    prefer: Optional[str] = None) -> dict[str, str]:
    """Headers for a request made as the service role. `json_body` adds the
    JSON `Content-Type` a request with a body sends; `prefer` sets
    PostgREST's `Prefer` (e.g. `return=minimal`)."""
    headers = {"apikey": key, "Authorization": f"Bearer {key}"}
    if json_body:
        headers["Content-Type"] = "application/json"
    if prefer is not None:
        headers["Prefer"] = prefer
    return headers


_clients: dict[int, httpx.Client] = {}
_clients_lock = threading.Lock()


def _client() -> httpx.Client:
    """This process's client. Its pool keeps connections open between
    calls, so an RPC pays the TCP and TLS handshake once per connection, not
    once per call: a fresh client per call spent 0.5 s (direct) to 4 s
    (through a local proxy) shaking hands before each request, past the
    2 s budget of a cache read. One per pid: a forked compute worker must not
    share its parent's sockets."""
    pid = os.getpid()
    with _clients_lock:
        client = _clients.get(pid)
        if client is None:
            client = _clients[pid] = httpx.Client()
        return client


def rpc(fn: str, body: Union[dict, bytes], *, label: str, logger: logging.Logger,
        timeout: float = 5.0, strict_json: bool = False) -> Optional[list]:
    """POST /rest/v1/rpc/<fn> with the service-role key. Parsed JSON on
    success; None when Supabase is not configured, the request fails, or it
    answers >= 400 (logged to `logger` as "<label> RPC <fn> ...").

    `body` is the arguments object, or that object already encoded as JSON
    bytes (a caller that had to measure it first does not encode it twice).

    With `strict_json` off, an empty body (a void function's 204) is `[]` and
    a body that is not JSON (a proxy's error page) is a logged None. With it
    on, the body is always decoded and a decode error raises."""
    if not supabase_env.configured():
        return None
    url, key = supabase_env.service_role()
    try:
        sent = ({"content": body, "headers": {**service_headers(key),
                                              "Content-Type": "application/json"}}
                if isinstance(body, bytes)
                else {"json": body, "headers": service_headers(key)})
        r = _client().post(f"{url}/rest/v1/rpc/{fn}", timeout=timeout, **sent)
        if r.status_code >= 400:
            logger.warning("%s RPC %s failed: %s %s", label, fn, r.status_code, r.text[:200])
            return None
        if strict_json:
            return r.json()
        if not r.content:
            return []
        return r.json()
    except httpx.RequestError as e:
        logger.warning("%s RPC %s errored: %s", label, fn, e)
        return None
    except ValueError as e:
        if strict_json:
            raise
        logger.warning("%s RPC %s errored: %s", label, fn, e)
        return None
