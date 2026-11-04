"""Agent CLI runtime — config + per-model concurrency control.

Holds the shared mutable state used across the spec worker subsystems:
  • `_max_concurrent` / `semaphore`            — per-VM global cap on
                                                   concurrent agent runs.
  • `_model_locks` / `_locks_lock`              — in-memory per-model
                                                   lock (dev fallback).
  • `_model_procs`                              — model_id → subprocess.Popen,
                                                   so `stop()` can kill the
                                                   running agent process.
  • Supabase `spec_generation_locks` table      — cross-process /
                                                   cross-VM mutex used
                                                   when SUPABASE_* env
                                                   is configured.

Lock layering:
  - The semaphore is per-VM. With Fly autoscale to N VMs, each gets
    its own SPEC_AGENT_MAX_CONCURRENT slots — that's intentional, it
    caps the *coding-agent CLI cost per machine*.
  - The Postgres-backed model lock is global. Two VMs both seeing
    /generate-spec for the same model_id → exactly one acquires the
    DB row, the other returns 'duplicate'.
  - When SUPABASE_* env vars aren't set (local dev / test), we fall
    back to the in-memory dict — single-VM correctness only.

`configure_runtime()` is the canonical mutation entry. `get_agent_cli()` /
`get_agent_binary()` / `get_agent_display_name()` read from environment
(SPEC_AGENT_CLI / SPEC_AGENT_MAX_CONCURRENT). Path constants used by
prompt builders also live here.
"""
from __future__ import annotations

import itertools
import json
import logging
import os
import socket
import threading
import time
from pathlib import Path
from typing import Optional

import httpx

from backend import supabase_env, supabase_rest

from backend.agents import available_clis, get_backend
from backend.arch.loader import invalidate_default_loader
from .limits import GENERATE_AGENT_SEC, LOCK_TTL_SEC


logger = logging.getLogger(__name__)

VALID_AGENT_CLIS = available_clis()

# Log directory — every agent invocation writes a per-model JSONL trace
# here for replay / debugging.
_LOG_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "logs", "spec_worker",
)
os.makedirs(_LOG_DIR, exist_ok=True)
_LOG_SIDECAR_SUFFIX = ".run.json"


def bind_job_log(run_key: str, log_path: str, *,
                 user_id: Optional[str] = None,
                 model_id: Optional[str] = None) -> None:
    """Remember the transcript path on the in-memory job and a local sidecar.

    Graceful shutdown reads the job dict. A killed process leaves the sidecar
    so boot recovery can patch the existing tail onto the durability row.
    """
    with _locks_lock:
        job = _jobs.get(run_key)
        if job is not None:
            job["log_path"] = log_path
            if user_id and not job.get("user_id"):
                job["user_id"] = user_id
            if model_id and not job.get("model_id"):
                job["model_id"] = model_id
            user_id = user_id or job.get("user_id")
            model_id = model_id or job.get("model_id")
    meta = {
        "run_key": run_key,
        "log_path": log_path,
        "user_id": user_id or "",
        "model_id": model_id or "",
    }
    try:
        Path(log_path + _LOG_SIDECAR_SUFFIX).write_text(
            json.dumps(meta), encoding="utf-8")
    except OSError:
        logger.warning("bind_job_log sidecar failed for %s", run_key, exc_info=True)


def clear_job_log_sidecar(log_path: str) -> None:
    try:
        Path(log_path + _LOG_SIDECAR_SUFFIX).unlink(missing_ok=True)
    except OSError:
        logger.debug("clear_job_log_sidecar failed for %s", log_path, exc_info=True)


def _upload_log_tail(log_path: str, *, user_id: Optional[str],
                     model_id: Optional[str], run_key: Optional[str]) -> bool:
    """Patch the durability row. Keep the sidecar until `record()` succeeds."""
    if not log_path:
        return False
    if not user_id or not os.path.isfile(log_path):
        # Nothing a later boot can upload from this sidecar.
        clear_job_log_sidecar(log_path)
        return False
    try:
        from backend._agent_runner import scrub_paths
        from backend.service import spec_generation_logs as _logs
        text = Path(log_path).read_text(encoding="utf-8", errors="replace")
        uploaded = bool(_logs.record(
            user_id=user_id, model_id=model_id or "", run_key=run_key,
            success=False, log=scrub_paths(text),
        ))
    except Exception:
        logger.warning("interrupted log tail upload failed for %s",
                       run_key, exc_info=True)
        return False
    if uploaded:
        clear_job_log_sidecar(log_path)
    return uploaded


def _flush_job_log(run_key: str) -> None:
    with _locks_lock:
        job = _jobs.get(run_key) or {}
        log_path = job.get("log_path")
        user_id = job.get("user_id")
        model_id = job.get("model_id")
    _upload_log_tail(
        log_path or "", user_id=user_id, model_id=model_id, run_key=run_key)


def flush_orphaned_log_sidecars() -> int:
    """On boot: upload tails for sidecars whose job is no longer running here."""
    n = 0
    if not os.path.isdir(_LOG_DIR):
        return 0
    for name in os.listdir(_LOG_DIR):
        if not name.endswith(_LOG_SIDECAR_SUFFIX):
            continue
        path = os.path.join(_LOG_DIR, name)
        try:
            meta = json.loads(Path(path).read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            logger.warning("skipping unreadable log sidecar %s: %s", path, exc)
            continue
        run_key = meta.get("run_key") or ""
        with _locks_lock:
            job = _jobs.get(run_key)
            if job is not None and job.get("state") == "running":
                continue
        if _upload_log_tail(
            meta.get("log_path") or "",
            user_id=meta.get("user_id") or None,
            model_id=meta.get("model_id") or None,
            run_key=run_key or None,
        ):
            n += 1
    return n


# --- Path constants (used by prompt builders + ground-truth assembly) ----
_ARCH_SPEC_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "arch",
)
_COMPONENTS_DIR = os.path.join(_ARCH_SPEC_DIR, "components")
_MODELS_DIR = os.path.join(_ARCH_SPEC_DIR, "models")
_PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
# Repo-relative paths for prompt templates — absolute host paths leak the
# developer's home directory into agent context and aren't portable.
_COMPONENTS_DIR_REL = os.path.relpath(_COMPONENTS_DIR, _PROJECT_ROOT)
_MODELS_DIR_REL = os.path.relpath(_MODELS_DIR, _PROJECT_ROOT)


# --- Concurrency state ---------------------------------------------------
_max_concurrent = int(os.environ.get("SPEC_AGENT_MAX_CONCURRENT", "10"))
_model_locks: dict = {}
semaphore = threading.BoundedSemaphore(_max_concurrent)
_locks_lock = threading.Lock()
_model_procs: dict = {}  # model_id → subprocess.Popen (for stop/kill)
# model_id → True while this run holds a semaphore slot. Makes release()
# idempotent per model_id: a run can be released more than once (stop() then
# the SSE finally; an explicit route release + the runner's release_fn) and the
# slot is returned exactly once. Without it, the 2nd BoundedSemaphore.release()
# over-releases (only raises ValueError when already at the bound), minting a
# phantom permit that belongs to another live run → the cap is silently defeated.
_slot_held: dict = {}
# model_id → the epoch of its current reservation (one per successful acquire),
# and the epoch a JOB thread was started under. A run whose reservation was
# reclaimed (stale past the TTL) can still wake up later; comparing its epoch
# with the key's current one tells a superseded run from the live owner, so
# its late release() / spawn cannot touch the run that replaced it.
_lock_epoch: dict = {}
_epoch_counter = itertools.count(1)
# model_id → a lock ordering that key's cross-VM lock RPCs. release() gives the
# key up locally and then tells the DB; without this, a new acquire landing
# between the two would take the DB row and the late release RPC would delete
# it under the new run. Taken before `_locks_lock`, never inside it.
_rpc_locks: dict = {}
# Restart drain (guarded by `_locks_lock`). While set, acquire() admits no new
# job; `in_flight()` lists what a restart would still interrupt. Checked in
# the same critical section that records a reservation, so every admission
# either lands before the drain (and is waited on) or is refused.
_draining = False
# run_key → True while start_job's thread has not finished (its finally
# persists the verdict after the reservation is already released).
_job_threads: dict = {}


class ServerDraining(RuntimeError):
    """acquire() refused: the server is draining for a restart."""

    MESSAGE = ("Server restarting: new spec jobs are paused for a moment. "
               "Retry in a minute.")

    def __init__(self) -> None:
        super().__init__(self.MESSAGE)


def _rpc_lock_for(model_id: str) -> threading.Lock:
    with _locks_lock:
        return _rpc_locks.setdefault(model_id, threading.Lock())


class _RunContext(threading.local):
    """Per-thread: the reservation epoch each key's job was started under.
    Empty on every thread that is not a job thread."""

    def __init__(self) -> None:
        self.epochs: dict = {}


_run_ctx = _RunContext()


# --- Agent CLI config ---------------------------------------------------

def get_agent_cli() -> str:
    """Return the configured agent CLI name."""
    cli = os.environ.get("SPEC_AGENT_CLI", "claude").strip().lower()
    if cli not in VALID_AGENT_CLIS:
        valid = ", ".join(VALID_AGENT_CLIS)
        raise ValueError(f"Invalid SPEC_AGENT_CLI={cli!r}. Expected one of: {valid}")
    return cli


def get_agent_binary() -> str:
    """Return the executable name for the configured agent CLI."""
    return get_agent_cli()


def get_agent_display_name() -> str:
    """Return a human-readable display name for the configured agent CLI."""
    return get_backend(get_agent_cli()).display_name


def configure_runtime(
    agent_cli: Optional[str] = None,
    max_concurrent: Optional[int] = None,
) -> None:
    """Apply explicit runtime configuration for the agent worker."""
    global _max_concurrent, semaphore

    if agent_cli is not None:
        cli = agent_cli.strip().lower()
        if cli not in VALID_AGENT_CLIS:
            valid = ", ".join(VALID_AGENT_CLIS)
            raise ValueError(f"Invalid agent_cli={cli!r}. Expected one of: {valid}")
        os.environ["SPEC_AGENT_CLI"] = cli

    if max_concurrent is not None:
        if max_concurrent <= 0:
            raise ValueError("max_concurrent must be > 0")
        os.environ["SPEC_AGENT_MAX_CONCURRENT"] = str(max_concurrent)
        _max_concurrent = max_concurrent
        semaphore = threading.BoundedSemaphore(_max_concurrent)


# --- Concurrency: per-model locks --------------------------------------
#
# Acquire is layered:
#   1. Local semaphore — caps concurrent agent runs *on this VM*.
#   2. Postgres lock   — global per-model mutex via
#                        public.spec_generation_locks. When the
#                        Supabase env isn't configured (dev / tests),
#                        we fall back to the in-memory _model_locks
#                        dict for single-VM correctness.
#
# Agent execution is capped separately from metadata/prompt preparation. The
# cap starts only after the subprocess is spawned (see stream_agent_core), so a
# large model's Hugging Face fetch time does not consume its reasoning budget.
# Defaults live in `limits` so generate / fix / review cannot drift.
_AGENT_HARD_CAP_SEC = GENERATE_AGENT_SEC
_LOCK_TTL_SECONDS = LOCK_TTL_SEC


def _supabase_lock_enabled() -> bool:
    return supabase_env.configured()


def _process_start_ns() -> int:
    try:
        return os.stat("/proc/self").st_ctime_ns
    except OSError:
        return time.time_ns()


# Process generation: pid + inode ctime of /proc/self. A restarted process
# (or a reused PID) gets a different value; uvicorn workers on one host each
# get their own.
_BOOT_ID = f"{os.getpid()}:{_process_start_ns()}"

INTERRUPTED_MESSAGE = "Interrupted by a server restart. You can retry."

# Migration 0034 adds v2 lock RPCs and keeps the 0009 signatures. Production
# may run this code before 0034 is applied: probe v2 once, cache the result,
# and fall back to the 0009 RPCs with hostname#pid identity.
_ACQUIRE_RPC = "try_acquire_spec_lock_v2"
_RELEASE_RPC = "release_spec_lock_v2"
_ACQUIRE_RPC_V1 = "try_acquire_spec_lock"
_RELEASE_RPC_V1 = "release_spec_lock"
# "v2" | "v1" | None (unprobed). Sticky for the process.
_lock_rpc_mode: Optional[str] = None
_lock_rpc_mode_guard = threading.Lock()
_lock_rpc_fallback_warned = False
# spec_generation_locks.boot_id present? None = unprobed.
_lock_select_boot: Optional[bool] = None
_lock_select_warned = False
_rpc_tls = threading.local()


def hostname() -> str:
    return socket.gethostname()


def boot_id() -> str:
    return _BOOT_ID


def machine_id() -> str:
    """Stable machine identity for lock/job rows.

    Container hostname is not a machine id: two containers on one host can
    share a hostname, and a PID namespace then looks local. Prefer an explicit
    id (HA_MACHINE_ID, Fly's FLY_MACHINE_ID, /etc/machine-id) and only then
    hostname.
    """
    for key in ("HA_MACHINE_ID", "FLY_MACHINE_ID"):
        v = (os.environ.get(key) or "").strip()
        if v:
            return v
    try:
        with open("/etc/machine-id", encoding="utf-8") as f:
            text = f.read().strip()
        if text:
            return text
    except OSError as exc:
        logger.debug("no /etc/machine-id (%s); lock rows use the hostname", exc)
    return hostname()


def _host_id() -> str:
    """Machine identity for lock rows. Process generation is `boot_id()`."""
    return machine_id()


def owner_process_alive(owner_boot_id: str, *, proc_root: str = "/proc") -> bool:
    """Whether `owner_boot_id` (`pid:start_ns`) still names a live process
    whose /proc inode ctime matches. Unknown / unparsable → not alive."""
    pid_s, sep, start_s = (owner_boot_id or "").partition(":")
    if not (sep and pid_s.isdigit() and start_s.isdigit()):
        return False
    path = os.path.join(proc_root, pid_s)
    try:
        st = os.stat(path)
    except OSError as exc:
        logger.debug("lock owner %s is not alive: %s", owner_boot_id, exc)
        return False
    return st.st_ctime_ns == int(start_s)


def lock_holder_is_local_and_dead(
    row: dict, *, host: Optional[str] = None, live_boot: Optional[str] = None,
    proc_root: str = "/proc",
) -> bool:
    """Same machine, holder process gone. Cross-machine rows stay until TTL.
    An unparseable identity is not proof of death — wait for TTL."""
    host = host if host is not None else _host_id()
    live_boot = live_boot if live_boot is not None else boot_id()
    held_host = row.get("host_id") or row.get("held_by") or ""
    held_boot = row.get("boot_id") or ""
    local = held_host == host or (host and held_host.startswith(host + "#"))
    if not local:
        hn = hostname()
        local = bool(hn) and held_host.startswith(hn + "#")
    if not local:
        return False
    if held_boot and held_boot == live_boot:
        return False
    if held_boot:
        return not owner_process_alive(held_boot, proc_root=proc_root)
    # Pre-0034 rows stored hostname#pid in host_id and have empty boot_id.
    if "#" in held_host:
        pid = held_host.rsplit("#", 1)[-1]
        return not os.path.exists(os.path.join(proc_root, pid))
    # Same machine, no boot id, no #pid: cannot prove the holder is dead.
    return False


def _legacy_lock_host_id() -> str:
    """0009 lock identity: hostname#pid. Dead-pid reclaim still works on
    the old 2-arg RPCs; machine-id + boot_id need v2."""
    return f"{hostname()}#{os.getpid()}"


def _reset_schema_compat_for_tests() -> None:
    global _lock_rpc_mode, _lock_rpc_fallback_warned
    global _lock_select_boot, _lock_select_warned
    _lock_rpc_mode = None
    _lock_rpc_fallback_warned = False
    _lock_select_boot = None
    _lock_select_warned = False
    _rpc_tls.missing = False


def _remember_lock_mode(mode: str) -> None:
    global _lock_rpc_mode, _lock_rpc_fallback_warned
    with _lock_rpc_mode_guard:
        if _lock_rpc_mode is not None:
            return
        _lock_rpc_mode = mode
        if mode == "v1" and not _lock_rpc_fallback_warned:
            _lock_rpc_fallback_warned = True
            logger.warning(
                "spec lock v2 RPC missing (404/PGRST202); using "
                "try_acquire_spec_lock / release_spec_lock (hostname#pid). "
                "The database schema is behind (0034 durability) — "
                "machine-id + boot_id locks need it."
            )


def _supabase_rpc(fn: str, body: dict) -> Optional[list]:
    """A spec-lock RPC: parsed JSON on success, None on network/HTTP error.
    Lock-acquire failures must NOT raise — instead we treat as "cannot
    acquire", letting the caller fall back to the in-memory path or surface
    'overloaded'. A body that is not JSON does raise (`strict_json`).

    404 / PGRST202 sets this thread's missing-rpc flag so the first
    acquire can fall back to the 0009 signatures and cache that choice.
    """
    _rpc_tls.missing = False
    if not supabase_env.configured():
        return None
    url, key = supabase_env.service_role()
    try:
        with httpx.Client(timeout=5.0) as client:
            r = client.post(
                f"{url}/rest/v1/rpc/{fn}",
                json=body,
                headers=supabase_rest.service_headers(key),
            )
        if r.status_code >= 400:
            logger.warning(
                "spec lock RPC %s failed: %s %s", fn, r.status_code, r.text[:200],
            )
            _rpc_tls.missing = supabase_rest.is_missing_rpc(r.status_code, r.text)
            return None
        return r.json()
    except httpx.RequestError as e:
        logger.warning("spec lock RPC %s errored: %s", fn, e)
        return None


def _invoke_lock_rpc(fn: str, body: dict) -> tuple[Optional[list], bool]:
    """Call `_supabase_rpc` and report whether that call was a missing RPC.
    A mock of `_supabase_rpc` never sets the flag (treated as a real error)."""
    _rpc_tls.missing = False
    rows = _supabase_rpc(fn, body)
    return rows, _rpc_tls.missing


def _first_row(rows: Optional[list]) -> Optional[dict]:
    if rows is None:
        return None
    return rows[0] if rows else {}


def _acquire_payload(model_id: str, *, v2: bool) -> tuple[str, dict]:
    if v2:
        return _ACQUIRE_RPC, {
            "p_model_id": model_id,
            "p_host_id": _host_id(),
            "p_boot_id": boot_id(),
            "p_ttl_seconds": _LOCK_TTL_SECONDS,
        }
    return _ACQUIRE_RPC_V1, {
        "p_model_id": model_id,
        "p_host_id": _legacy_lock_host_id(),
        "p_ttl_seconds": _LOCK_TTL_SECONDS,
    }


def _release_holder_payload(model_id: str, row: dict, *, v2: bool) -> tuple[str, dict]:
    held = row.get("held_by") or row.get("host_id") or (
        _host_id() if v2 else _legacy_lock_host_id()
    )
    if v2:
        return _RELEASE_RPC, {
            "p_model_id": model_id,
            "p_host_id": held,
            "p_boot_id": row.get("boot_id") or "",
        }
    return _RELEASE_RPC_V1, {
        "p_model_id": model_id,
        "p_host_id": held,
    }


def _acquire_now(model_id: str, *, v2: bool) -> Optional[dict]:
    rows = _supabase_rpc(*_acquire_payload(model_id, v2=v2))
    row = _first_row(rows)
    if row is not None and not row.get("acquired") and lock_holder_is_local_and_dead(row):
        _supabase_rpc(*_release_holder_payload(model_id, row, v2=v2))
        rows = _supabase_rpc(*_acquire_payload(model_id, v2=v2))
        row = _first_row(rows)
    return row


def _cross_vm_acquire(model_id: str) -> Optional[dict]:
    with _lock_rpc_mode_guard:
        mode = _lock_rpc_mode
    if mode == "v1":
        return _acquire_now(model_id, v2=False)
    if mode == "v2":
        return _acquire_now(model_id, v2=True)
    rows, missing = _invoke_lock_rpc(*_acquire_payload(model_id, v2=True))
    if missing:
        _remember_lock_mode("v1")
        return _acquire_now(model_id, v2=False)
    if rows is None:
        return None
    _remember_lock_mode("v2")
    row = _first_row(rows)
    if row is not None and not row.get("acquired") and lock_holder_is_local_and_dead(row):
        _supabase_rpc(*_release_holder_payload(model_id, row, v2=True))
        rows = _supabase_rpc(*_acquire_payload(model_id, v2=True))
        row = _first_row(rows)
    return row


def _cross_vm_release(model_id: str) -> None:
    with _lock_rpc_mode_guard:
        mode = _lock_rpc_mode
    if mode == "v1":
        _supabase_rpc(_RELEASE_RPC_V1, {
            "p_model_id": model_id,
            "p_host_id": _legacy_lock_host_id(),
        })
        return
    if mode == "v2":
        _supabase_rpc(_RELEASE_RPC, {
            "p_model_id": model_id,
            "p_host_id": _host_id(),
            "p_boot_id": boot_id(),
        })
        return
    rows, missing = _invoke_lock_rpc(_RELEASE_RPC, {
        "p_model_id": model_id,
        "p_host_id": _host_id(),
        "p_boot_id": boot_id(),
    })
    if missing:
        _remember_lock_mode("v1")
        _supabase_rpc(_RELEASE_RPC_V1, {
            "p_model_id": model_id,
            "p_host_id": _legacy_lock_host_id(),
        })
    elif rows is not None:
        _remember_lock_mode("v2")


def acquire(model_id: str) -> str:
    """Acquire per-model lock. Returns:
        'ok'         — caller holds the lock
        'duplicate'  — same model_id already being generated elsewhere
        'overloaded' — local VM at concurrency cap

    Two layers: semaphore (per-VM cap) then mutex (cross-VM model id).
    The semaphore is checked first because it's cheap and returns the
    most actionable error; the DB hit only happens when we know we
    have a slot.
    """
    if not semaphore.acquire(blocking=False):
        return "overloaded"

    # Local reservation first, atomically, in BOTH modes. The cross-VM lock
    # reports `acquired` only when hostname AND boot_id match, so a second
    # caller on this process is invisible to it. Without the local reservation,
    # a double-clicked review/generate on one VM starts two agents on one
    # key; they share one overlay and one job slot, and the first to finish
    # calls release(), which kills the other mid-run.
    #
    # The reservation records when it was taken. One older than the lock TTL
    # (the agent's hard cap plus slack) belongs to a run that cannot still be
    # going — a caller that never reached release(). Reclaim it the way the
    # DB lock self-heals, rather than refusing the key until a restart.
    #
    # Check, reclaim and re-reserve in ONE critical section: two callers that
    # both saw the same stale entry must not both reclaim it, or the second
    # would free the reservation the first had just taken (and kill its run).
    reclaimed = None
    with _locks_lock:
        if _draining:
            semaphore.release()
            raise ServerDraining()
        held_since = _model_locks.get(model_id)
        if held_since is not None and time.monotonic() - held_since > _LOCK_TTL_SECONDS:
            del _model_locks[model_id]
            reclaimed = (_model_procs.pop(model_id, None), _slot_held.pop(model_id, False))
        if model_id in _model_locks:
            semaphore.release()
            return "duplicate"
        _model_locks[model_id] = time.monotonic()
        _lock_epoch[model_id] = next(_epoch_counter)
    if reclaimed is not None:
        # Outside the lock: reap what the abandoned run left behind. Its DB lock
        # row is this host's, so the cross-VM acquire below re-takes it.
        proc, held_slot = reclaimed
        logger.warning(f"acquire: reclaimed a stale local reservation for {model_id}")
        if proc is not None and proc.poll() is None:
            try:
                proc.kill()
            except OSError as e:
                logger.warning(f"acquire: kill({model_id}) failed: {e}")
        if held_slot:
            try:
                semaphore.release()
            except ValueError:
                logger.warning(f"acquire: slot for {model_id} was already returned")

    # Cross-VM lock (production). Skipped when SUPABASE isn't configured
    # (dev / tests) — the local reservation above is then the whole lock.
    if _supabase_lock_enabled():
        def _undo() -> None:
            with _locks_lock:
                _model_locks.pop(model_id, None)
                _lock_epoch.pop(model_id, None)
            semaphore.release()

        try:
            with _rpc_lock_for(model_id):
                row = _cross_vm_acquire(model_id)
        except BaseException:
            # _supabase_rpc turns network errors into None; anything else (a
            # body that is not JSON, a misconfigured key) raises. Give back the
            # reservation and the slot before it propagates, or both leak and
            # even the TTL reclaim cannot return the slot.
            _undo()
            raise
        if row is None or not row.get("acquired"):
            _undo()
            # RPC failure → let the caller retry; held elsewhere → duplicate.
            return "overloaded" if row is None else "duplicate"

    with _locks_lock:
        _slot_held[model_id] = True
    return "ok"


def owns(model_id: str) -> bool:
    """Whether the calling thread may act for `model_id`'s reservation. A job
    thread started under an epoch that has since been reclaimed is a superseded
    run: it must not spawn or release on behalf of the key's new owner. Any
    other caller (a request thread, a test) is not tied to an epoch and owns
    nothing it could clobber, so it answers True."""
    mine = _run_ctx.epochs.get(model_id)
    return mine is None or mine == _lock_epoch.get(model_id)


def release(model_id: str) -> None:
    """Release per-model lock + reap any still-running agent subprocess.
    Idempotent — safe to call from finally handlers even if acquire
    failed.

    Why kill here: the SSE generator's finally block calls release()
    when the client disconnects mid-stream. Without the kill, the
    spawned coding-agent CLI keeps running — blocked on its own API
    socket if the upstream is slow — and accumulates one zombie per
    aborted request. After a few stuck runs the machine has multiple
    leaked claude/codex procs each holding network state, which is
    what the V4-Pro hang ultimately surfaced (cf. runtime.py header).
    On a normal completion the proc has already exited, `poll()` is
    not None, and the kill is skipped.
    """
    # Ownership is decided in the same critical section that frees the key, so
    # a reclaim cannot slip in between; the DB release follows under the key's
    # RPC lock, so a new owner's cross-VM acquire cannot run before it.
    with _rpc_lock_for(model_id):
        with _locks_lock:
            if not owns(model_id):
                # This job thread's reservation is gone: either it already
                # released (a repeat release — nothing left to free) or it was
                # reclaimed and a new run holds the key. Either way what
                # release() would free now is not this run's; its own process
                # and slot were reaped already.
                if _lock_epoch.get(model_id) is not None:
                    logger.warning(f"release: superseded run of {model_id} released late; ignored")
                return
            proc = _model_procs.pop(model_id, None)
            _model_locks.pop(model_id, None)
            _lock_epoch.pop(model_id, None)
            held = _slot_held.pop(model_id, False)
        # Everything local is settled before the DB is told, so an RPC that
        # raises (a body that is not JSON) cannot leave the agent running or
        # the slot taken.
        if proc is not None and proc.poll() is None:
            try:
                proc.kill()
            except OSError as e:
                logger.warning(f"release: kill({model_id}) failed: {e}")
        # Return the slot only if THIS model_id actually holds one, and only
        # once (the pop above clears it) — so a second release() for the same
        # run, or a release after a non-'ok' acquire, is a true no-op instead
        # of over-releasing.
        if held:
            try:
                semaphore.release()
            except ValueError:
                pass
        if _supabase_lock_enabled():
            _cross_vm_release(model_id)


def stop(model_id: str) -> bool:
    """Kill a running spec worker for a run_key. Returns True if a live
    subprocess was killed or a running job was marked interrupted
    (prep can sit with no proc yet)."""
    with _locks_lock:
        proc = _model_procs.get(model_id)
        job = _jobs.get(model_id)
        running = job is not None and job.get("state") == "running"
    killed = False
    if proc and proc.poll() is None:
        proc.kill()
        killed = True
    if running:
        _mark_interrupted(model_id)
        killed = True
    if killed:
        try:
            release(model_id)
        except Exception:
            logger.warning("stop(%s) release failed", model_id, exc_info=True)
    return killed


def _list_lock_rows() -> list:
    if not _supabase_lock_enabled():
        return []
    url, key = supabase_env.service_role()
    global _lock_select_boot, _lock_select_warned
    with _lock_rpc_mode_guard:
        mode = _lock_rpc_mode
    want_boot = _lock_select_boot is not False and mode != "v1"
    select = "model_id,host_id,boot_id" if want_boot else "model_id,host_id"
    try:
        with httpx.Client(timeout=5.0) as client:
            r = client.get(
                f"{url}/rest/v1/spec_generation_locks",
                params={"select": select},
                headers=supabase_rest.service_headers(key, json_body=False),
            )
        if r.status_code >= 400:
            if want_boot and supabase_rest.is_unknown_column(r.status_code, r.text):
                _lock_select_boot = False
                if not _lock_select_warned:
                    _lock_select_warned = True
                    logger.warning(
                        "spec_generation_locks.boot_id missing (schema behind "
                        "0034 durability); listing host_id only."
                    )
                return _list_lock_rows()
            logger.warning("list spec locks failed: %s %s", r.status_code, r.text[:200])
            return []
        if want_boot:
            _lock_select_boot = True
        return r.json() or []
    except Exception as e:
        logger.warning("list spec locks errored: %s", e)
        return []


def _delete_lock_row(model_id: str, host_id: str, owner_boot: str) -> bool:
    with _lock_rpc_mode_guard:
        mode = _lock_rpc_mode
    if mode == "v1":
        rows = _supabase_rpc(
            _RELEASE_RPC_V1, {"p_model_id": model_id, "p_host_id": host_id},
        )
        return bool(rows)
    if mode == "v2":
        rows = _supabase_rpc(
            _RELEASE_RPC,
            {"p_model_id": model_id, "p_host_id": host_id, "p_boot_id": owner_boot},
        )
        return bool(rows)
    rows, missing = _invoke_lock_rpc(
        _RELEASE_RPC,
        {"p_model_id": model_id, "p_host_id": host_id, "p_boot_id": owner_boot},
    )
    if missing:
        _remember_lock_mode("v1")
        rows = _supabase_rpc(
            _RELEASE_RPC_V1, {"p_model_id": model_id, "p_host_id": host_id},
        )
    elif rows is not None:
        _remember_lock_mode("v2")
    return bool(rows)


def reclaim_dead_owner_locks(*, proc_root: str = "/proc") -> int:
    """Drop same-host lock rows whose holder process is gone. Cross-host
    rows stay until `expires_at`. Returns how many rows were released."""
    n = 0
    for row in _list_lock_rows():
        if not lock_holder_is_local_and_dead(row, proc_root=proc_root):
            continue
        if _delete_lock_row(row.get("model_id") or "",
                            row.get("host_id") or "",
                            row.get("boot_id") or ""):
            n += 1
            logger.warning(
                "reclaimed dead-owner spec lock %s host=%s boot=%s",
                row.get("model_id"), row.get("host_id"), row.get("boot_id"),
            )
    return n


def _persist_begin(run_key: str, *, model_id: str,
                   user_id: Optional[str] = None,
                   kind: str = "generate") -> Optional[str]:
    try:
        from backend.service import spec_generation_logs as _logs
        return _logs.begin_run(
            run_key=run_key, model_id=model_id, user_id=user_id, kind=kind,
            host_id=_host_id(), boot_id=boot_id(),
        )
    except Exception:
        logger.warning("job persist begin failed for %s", run_key, exc_info=True)
        return None


def _persist_finish(run_key: str, *, success: Optional[bool], message: str,
                    state: str = "done", row_id: Optional[str] = None,
                    error_kind: Optional[str] = None,
                    engine_gaps: Optional[list] = None) -> None:
    try:
        from backend.service import spec_generation_logs as _logs
        _logs.finish_run(
            run_key=run_key, success=success, message=message, state=state,
            row_id=row_id, error_kind=error_kind, engine_gaps=engine_gaps,
        )
    except Exception:
        logger.warning("job persist finish failed for %s", run_key, exc_info=True)


def _persist_lookup(run_key: str) -> Optional[dict]:
    try:
        from backend.service import spec_generation_logs as _logs
        return _logs.latest_by_run_key(run_key)
    except Exception:
        logger.warning("job persist lookup failed for %s", run_key, exc_info=True)
        return None


def interrupt_orphaned_runs(*, proc_root: str = "/proc") -> int:
    """Mark DB `running` rows whose holder is this host and gone as interrupted."""
    try:
        from backend.service import spec_generation_logs as _logs
        rows = _logs.list_running()
    except Exception:
        logger.warning("interrupt_orphaned_runs list failed", exc_info=True)
        return 0
    ids = []
    for row in rows:
        fake_lock = {
            "host_id": row.get("host_id") or "",
            "boot_id": row.get("boot_id") or "",
        }
        if not lock_holder_is_local_and_dead(fake_lock, proc_root=proc_root):
            continue
        ids.append(row["id"])
        run_key = row.get("run_key")
        if run_key:
            with _locks_lock:
                _job_store[run_key] = {
                    "state": "interrupted",
                    "success": False,
                    "message": INTERRUPTED_MESSAGE,
                }
    if not ids:
        return 0
    try:
        from backend.service import spec_generation_logs as _logs
        return _logs.interrupt_ids(ids, INTERRUPTED_MESSAGE)
    except Exception:
        logger.warning("interrupt_orphaned_runs patch failed", exc_info=True)
        return 0


def recover_on_boot() -> None:
    """Startup: reclaim same-host dead locks and close orphaned running jobs."""
    if os.environ.get("PYTEST_CURRENT_TEST"):
        return
    try:
        n = reclaim_dead_owner_locks()
        if n:
            logger.warning("reclaimed %s dead-owner spec lock(s)", n)
        m = interrupt_orphaned_runs()
        if m:
            logger.warning("marked %s spec job(s) interrupted", m)
        tails = flush_orphaned_log_sidecars()
        if tails:
            logger.warning("uploaded %s interrupted run log tail(s)", tails)
    except Exception:
        logger.exception("recover_on_boot failed")


def interrupt_on_shutdown() -> None:
    """SIGTERM / lifespan: persist interrupted, release locks, kill agents."""
    with _locks_lock:
        keys = [k for k, j in _jobs.items() if j.get("state") == "running"]
    for key in keys:
        _mark_interrupted(key)
        try:
            _flush_job_log(key)
        except Exception:
            logger.warning("interrupt_on_shutdown log flush(%s) failed", key, exc_info=True)
        try:
            release(key)
        except Exception:
            logger.warning("interrupt_on_shutdown release(%s) failed", key, exc_info=True)


def running_job_count() -> int:
    with _locks_lock:
        return sum(1 for j in _jobs.values() if j.get("state") == "running")


def set_draining(on: bool) -> None:
    """Start (or end) a restart drain: while on, acquire() raises
    ServerDraining instead of admitting a job."""
    global _draining
    with _locks_lock:
        _draining = on


def in_flight() -> list[dict]:
    """What a restart of this process would interrupt: every admitted
    reservation (including one still preparing, before its job starts) and
    every job thread that has not finished."""
    now = time.time()
    with _locks_lock:
        keys = sorted(set(_model_locks) | set(_job_threads)
                      | {k for k, j in _jobs.items() if j.get("state") == "running"})
        out = []
        for key in keys:
            job = _jobs.get(key) or {}
            started = job.get("started_at")
            out.append({
                "run_key": key,
                "model_id": job.get("model_id", key),
                "kind": job.get("kind", "admitting"),
                "running_s": round(now - started) if started else 0,
            })
        return out


def drain_state() -> dict:
    with _locks_lock:
        draining = _draining
    return {"draining": draining, "in_flight": in_flight()}


def _mark_interrupted(run_key: str) -> None:
    row_id = None
    with _locks_lock:
        job = _jobs.get(run_key)
        if job is not None and job.get("state") == "running":
            job["state"] = "interrupted"
            job["success"] = False
            job["message"] = INTERRUPTED_MESSAGE
            job["finished_at"] = time.time()
            job["events"].append({
                "type": "done", "success": False, "message": INTERRUPTED_MESSAGE,
            })
            row_id = job.get("log_id")
        _job_store[run_key] = {
            "state": "interrupted",
            "success": False,
            "message": INTERRUPTED_MESSAGE,
        }
    _persist_finish(
        run_key, success=False, message=INTERRUPTED_MESSAGE, state="interrupted",
        row_id=row_id,
    )


# --- Background job registry ---------------------------------------------
# Spec generation/refinement runs the agent for many minutes. We consume the
# stream_agent generator server-side in a daemon thread and buffer its events
# here, so the HTTP request that kicks a run off returns immediately and the
# browser polls job_status() — decoupling the run from any single connection.
# The cloudflared tunnel in front of the backend caps proxied streams at
# ~10 min (cutting long runs mid-flight even while bytes flow, which no
# heartbeat can defeat); buffering here sidesteps that entirely. A client
# disconnect no longer reaches release()/proc.kill(), so a survivable tunnel
# cut no longer destroys an almost-finished run.
_jobs: dict = {}  # model_id → job dict (guarded by _locks_lock)


def _terminal(event: dict) -> dict:
    """What a run's log row keeps of its terminal `done` beside the verdict
    and message: the failure's machine key and an engine-blocked run's
    refusals (Admin filters runs on them)."""
    return {"error_kind": event.get("error_kind"),
            "engine_gaps": event.get("engine_gaps")}
# Survives `_jobs` being dropped for a key (a client asking after this
# process restarted still hits `_persist_lookup`; in-process tests hit here).
_job_store: dict = {}

def _agent_model_id(cli: Optional[str], env: Optional[dict] = None) -> Optional[str]:
    """Model id this run will spawn: BYOK env first, else the host env var."""
    from backend.agents.base import AGENT_ENV_VARS
    names = AGENT_ENV_VARS.get(cli or "")
    var = names.model if names else None
    if not var:
        return None
    return (env or {}).get(var) or os.environ.get(var)


def start_job(run_key: str, run_fn, *, model_id: Optional[str] = None,
              user_id: Optional[str] = None, kind: str = "generate",
              agent_cli: Optional[str] = None,
              agent_env: Optional[dict] = None) -> None:
    """Consume `run_fn()` (a generator of event dicts) in a daemon thread,
    buffering events into `_jobs[run_key]` for `job_status()` to serve.

    Caller must already hold a successful `acquire(run_key)`; the thread's
    `stream_agent_core` finally releases it. The subprocess execution cap is
    enforced by the runner after spawn, not here, so metadata/prompt
    preparation does not reduce the agent's available runtime.

    `model_id` / `user_id` / `kind` are the domain identity written on the
    durability row. `run_key` is only the concurrency key.
    """
    domain_model = model_id or run_key
    cli = (agent_cli or "").strip() or get_agent_cli()
    job = {
        "state": "running",
        "events": [],
        "success": None,
        "message": "",
        "usage": None,
        "started_at": time.time(),
        "finished_at": None,
        "model_id": domain_model,
        "user_id": user_id,
        "kind": kind,
        "agent_cli": cli,
        "agent_model": _agent_model_id(cli, agent_env),
    }
    with _locks_lock:
        _jobs[run_key] = job
        _job_threads[run_key] = True
        _job_store[run_key] = {"state": "running", "success": None, "message": ""}
        epoch = _lock_epoch.get(run_key)
    job["log_id"] = _persist_begin(
        run_key, model_id=domain_model, user_id=user_id, kind=kind,
    )

    def _run():
        # Tie this job thread to the reservation it was started under (see
        # owns()): if that reservation is later reclaimed, this thread's
        # spawns and releases stop acting for the key.
        _run_ctx.epochs = {run_key: epoch}
        # The run's transcript goes onto the row this run began, not onto
        # whichever row of the key is newest when the upload lands.
        from backend.service import spec_generation_logs as _logs
        _logs.bind_run_row(job.get("log_id"))
        # Buffered events are served verbatim to the browser poller. Frames
        # coming through stream_agent_core arrive pre-scrubbed (the runner
        # knows the run's overlay root); the pre-prep lines and the crash
        # `done` below don't pass the runner, so apply the generic roots
        # (repo / homes) here. Re-scrubbing a runner frame is idempotent.
        from backend._agent_runner import scrub_paths

        def _scrub(evt: dict) -> dict:
            for field in ("text", "message"):
                value = evt.get(field)
                if isinstance(value, str) and value:
                    evt[field] = scrub_paths(value)
            return evt

        try:
            for evt in run_fn():
                etype = evt.get("type")
                with _locks_lock:
                    job["events"].append(_scrub(evt))
                    if etype == "usage":
                        job["usage"] = evt
                    # A late transcript after interrupt must not flip the
                    # terminal verdict: interrupted is always success=false.
                    elif etype == "done" and job.get("state") != "interrupted":
                        job["success"] = bool(evt.get("success"))
                        job["message"] = evt.get("message", "")
                        # The verdict's machine fields reach the log row.
                        job["terminal"] = _terminal(evt)
                        if evt.get("usage"):
                            job["usage"] = evt["usage"]
        except Exception as e:
            logger.warning(f"start_job({run_key}) crashed: {e}")
            with _locks_lock:
                job["events"].append(_scrub(
                    {"type": "done", "success": False, "message": f"Job crashed: {e}"}
                ))
                if job.get("state") != "interrupted" and job["success"] is None:
                    job["success"] = False
                    job["message"] = f"Job crashed: {scrub_paths(str(e))}"
        finally:
            with _locks_lock:
                if job["state"] == "interrupted":
                    job["success"] = False
                    job["message"] = job["message"] or INTERRUPTED_MESSAGE
                else:
                    if job["success"] is None:
                        # Generator ended without a terminal `done` — treat as
                        # failure so the poller resolves instead of hanging.
                        job["success"] = False
                        job["message"] = job["message"] or "Ended without result"
                    job["state"] = "done"
                    job["finished_at"] = time.time()
                _job_store[run_key] = {
                    "state": job["state"],
                    "success": job["success"],
                    "message": job["message"],
                    **job.get("terminal", {}),
                }
            if job["state"] != "interrupted":
                _persist_finish(
                    run_key, success=job["success"], message=job["message"] or "",
                    state="done", row_id=job.get("log_id"),
                    **job.get("terminal", {}),
                )
            # The agent just (re)wrote the model/component yaml on disk; drop
            # the backend's cached SpecLoader so the next /analyze rebuilds
            # from the fresh yaml instead of serving the pre-edit spec (which
            # would re-trigger refine in a loop). Moved here from the old SSE
            # route's finally.
            try:
                invalidate_default_loader()
            finally:
                with _locks_lock:
                    _job_threads.pop(run_key, None)

    threading.Thread(target=_run, daemon=True).start()


def _cache_terminal(run_key: str, *, state: str, success, message: str,
                    **terminal) -> dict:
    stored = {"state": state, "success": success, "message": message,
              **terminal}
    with _locks_lock:
        _job_store[run_key] = stored
    return stored


def _interpret_persisted_running(row: dict) -> str:
    """How a persisted `running` row looks on *this* process.

    Same machine + this boot: `_jobs` already missed, so we are not running
    it → interrupted. Same machine + holder dead → interrupted. Same machine
    + another live process, or another machine → running. Never invent
    interrupted for a live remote owner.
    """
    held_host = row.get("host_id") or ""
    held_boot = row.get("boot_id") or ""
    if held_host == _host_id() and held_boot == boot_id():
        return "interrupted"
    fake = {"host_id": held_host, "boot_id": held_boot}
    if lock_holder_is_local_and_dead(fake):
        return "interrupted"
    return "running"


def _persist_interrupt_row(row: dict) -> None:
    rid = row.get("id")
    if not rid:
        return
    try:
        from backend.service import spec_generation_logs as _logs
        _logs.interrupt_ids([rid], INTERRUPTED_MESSAGE)
    except Exception:
        logger.warning("job persist interrupt failed for %s", rid, exc_info=True)


def job_status(run_key: str, cursor: int = 0) -> dict:
    """Return job state + events after `cursor`. `{state: 'idle'}` on miss.

    The cursor lets the frontend fetch only newly-buffered events per poll;
    `next_cursor` is the cursor to pass on the following request. After a
    restart the in-memory registry is empty; a persisted interrupted/done
    row is returned instead of idle. A persisted `running` row is read with
    host_id/boot_id: only a same-machine dead owner becomes interrupted.
    Live remote (or live local other-process) owners stay `running`, and
    that guess is not written into `_job_store`.
    """
    with _locks_lock:
        job = _jobs.get(run_key)
        stored = None if job is not None else _job_store.get(run_key)
    if job is not None:
        events = job["events"]
        new = events[cursor:] if 0 <= cursor < len(events) else []
        return {
            "state": job["state"],
            "events": list(new),
            "next_cursor": len(events),
            "success": job["success"],
            "message": job["message"],
            "usage": job["usage"],
            # Wall-clock start, so a client that attaches mid-run (drawer
            # reopened, page reloaded) shows true elapsed time, not "0s".
            "started_at": job.get("started_at"),
            # How long a client should keep polling a running job: the lock
            # TTL, which covers every round of the run (`limits`).
            "poll_budget_sec": _LOCK_TTL_SECONDS,
            "agent_cli": job.get("agent_cli"),
            "agent_model": job.get("agent_model"),
        }

    persist = _persist_lookup(run_key)
    if persist and persist.get("state") in ("interrupted", "done"):
        interrupted = persist.get("state") == "interrupted"
        stored = _cache_terminal(
            run_key,
            state=persist.get("state") or "interrupted",
            success=False if interrupted else persist.get("success"),
            message=persist.get("message") or (
                INTERRUPTED_MESSAGE if interrupted else ""
            ),
            **_terminal(persist),
        )
    elif persist and persist.get("state") == "running":
        interpreted = _interpret_persisted_running(persist)
        if interpreted == "interrupted":
            _persist_interrupt_row(persist)
            stored = _cache_terminal(
                run_key, state="interrupted", success=False,
                message=persist.get("message") or INTERRUPTED_MESSAGE,
            )
        else:
            return {
                "state": "running",
                "events": [],
                "next_cursor": cursor,
                "success": persist.get("success"),
                "message": persist.get("message") or "",
                "poll_budget_sec": _LOCK_TTL_SECONDS,
            }
    elif persist:
        persist = None

    if stored is None:
        with _locks_lock:
            stored = _job_store.get(run_key)
    if stored is None:
        return {"state": "idle"}
    state = stored.get("state") or "idle"
    if state == "running":
        # In-memory running with no live `_jobs` entry and no persist row
        # that said otherwise: this process started it and it is gone.
        state = "interrupted"
    return {
        "state": state,
        "events": [],
        "next_cursor": cursor,
        "success": False if state == "interrupted" else stored.get("success"),
        "message": stored.get("message") or (
            INTERRUPTED_MESSAGE if state == "interrupted" else ""
        ),
        **(_terminal(stored) if state == "done" else {}),
    }
