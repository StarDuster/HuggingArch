"""One closed vocabulary for where a generation record stands and why a spec
run failed: `backend/spec_worker/outcome.py` defines it, and the SQL schema
and the frontend's badge tables must list exactly its values.

The schema is read the way a fresh database is built — `supabase/init.sql`,
then every migration in number order — and the last definition of each
CHECK wins. Every status list the schema's functions branch on must be
one of the sets `outcome.py` names, so a set cannot drift inside SQL
either.
"""
from __future__ import annotations

import re
from pathlib import Path

from backend.service.spec_generations import PROMOTE
from backend.spec_worker.outcome import (
    MUTABLE,
    PENDING,
    SERVABLE,
    GenerationStatus,
    RunErrorKind,
)

ROOT = Path(__file__).resolve().parents[2]
SUPABASE = ROOT / "supabase"
INIT = SUPABASE / "init.sql"
STATUSES = {s.value for s in GenerationStatus}
ERROR_KINDS = {k.value for k in RunErrorKind}


def _schema_in_order() -> list[str]:
    files = [INIT, *sorted((SUPABASE / "migrations").glob("*.sql"))]
    return [f.read_text(encoding="utf-8") for f in files]


def _values(array_body: str) -> set[str]:
    return set(re.findall(r"'([^']*)'", array_body))


def _last_check(constraint: str) -> set[str]:
    pattern = re.compile(
        re.escape(constraint) + r"\s+CHECK\s*\(.*?ARRAY\[(.*?)\]", re.S)
    found = None
    for sql in _schema_in_order():
        for match in pattern.finditer(sql):
            found = _values(match.group(1))
    assert found is not None, f"no CHECK named {constraint}"
    return found


def test_status_check_is_the_generation_status_vocabulary():
    assert _last_check("spec_generations_status_check") == STATUSES


def test_log_error_kind_check_is_the_run_error_kind_vocabulary():
    assert _last_check("spec_generation_logs_error_kind_check") == ERROR_KINDS


def test_every_status_set_the_schema_names_is_one_outcome_names():
    named = {frozenset(STATUSES), frozenset(s.value for s in SERVABLE),
             frozenset(s.value for s in PENDING), frozenset(s.value for s in MUTABLE)}
    transitions = frozenset({PROMOTE, *(s.value for s in PENDING)})
    arrays = []
    # The error-kind lists of the baseline and of each migration that lists
    # them. A migration that widens the vocabulary re-states the CHECK and
    # the readers that whitelist it, so only the newest such file must list
    # all of it, everywhere it lists it; an older list is superseded and may
    # only lack what was added since.
    error_kind_lists: list[list[set[str]]] = []
    for sql in _schema_in_order():
        listed = [_values(body) for body in re.findall(r"ARRAY\[(.*?)\]", sql, re.S)]
        kinds = [v for v in listed if v <= ERROR_KINDS and not v <= STATUSES]
        if kinds:
            error_kind_lists.append(kinds)
        # A status list names a status (or the promote transition); an array
        # naming none (a function's filter-dimension list) is not one.
        arrays.extend(v for v in listed if v not in kinds and v & (STATUSES | {PROMOTE}))
    assert arrays and error_kind_lists
    for values in error_kind_lists[-1]:
        assert values == ERROR_KINDS
    for older in error_kind_lists[:-1]:
        for values in older:
            assert values < ERROR_KINDS
    for values in arrays:
        if PROMOTE in values:
            assert frozenset(values) == transitions
        else:
            assert values <= STATUSES, values
            assert frozenset(values) in named, sorted(values)


def _badge_keys(table: str) -> tuple[set[str], set[str]]:
    """The keys of a `Record<string, …>` table in specStatus.ts, and the
    keys whose entry says `pending: true`."""
    src = (ROOT / "web" / "src" / "specStatus.ts").read_text(encoding="utf-8")
    start = src.index(f"export const {table}")
    body = src[src.index("{", src.index("=", start)) + 1:src.index("\n};", start)]
    keys, pending, depth, key = set(), set(), 0, None
    for line in body.splitlines():
        if depth == 0:
            match = re.match(r"\s*(\w+):", line)
            if match:
                key = match.group(1)
                keys.add(key)
        if "pending: true" in line and key:
            pending.add(key)
        depth += line.count("{") - line.count("}")
    return keys, pending


def test_frontend_badges_cover_both_vocabularies():
    status_keys, pending = _badge_keys("GENERATION_STATUS_BADGE")
    assert status_keys == STATUSES
    assert pending == {s.value for s in PENDING}
    error_keys, _ = _badge_keys("RUN_ERROR_KIND_BADGE")
    assert error_keys == ERROR_KINDS


def _function_grantees(sql: str, name: str) -> set[str]:
    return set(re.findall(
        rf"^GRANT ALL ON FUNCTION public\.{name}\([^;]*\) TO (\w+);", sql, re.M))


def test_the_generation_functions_are_granted_to_who_may_call_them():
    """Admin readers to signed-in users (they check is_admin themselves) and
    the service role; the record writes to the service role only; nothing
    to anon or PUBLIC. Read off the baseline's ACL (pg_dump form)."""
    sql = INIT.read_text(encoding="utf-8")
    readers = ("get_spec_generations", "get_spec_generations_count",
               "get_spec_generation_detail", "get_spec_generation_runs",
               "get_spec_generation_runs_count")
    writes = ("record_generation", "record_pending_generation",
              "transition_pending_generation")
    for name in (*readers, *writes, "spec_generations_bump_revision"):
        assert re.search(rf"^CREATE FUNCTION public\.{name}\(", sql, re.M), name
        assert re.search(rf"^REVOKE ALL ON FUNCTION public\.{name}\([^;]*\) FROM PUBLIC;",
                         sql, re.M), name
    for name in readers:
        assert _function_grantees(sql, name) == {"authenticated", "service_role"}, name
    for name in writes:
        assert _function_grantees(sql, name) == {"service_role"}, name
    assert _function_grantees(sql, "spec_generations_bump_revision") == set()
