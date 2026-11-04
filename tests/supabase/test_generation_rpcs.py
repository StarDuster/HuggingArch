"""Behaviour of the spec-generation write and read functions (migration 0002,
folded into the baseline) on a real Postgres.

The functions decide on the rows already there inside one locked
transaction (stale / CAS / promotion / the pending invariant), which no
mock can exercise. This suite builds a disposable cluster, loads the schema
the way a fresh project gets it — `supabase/init.sql` with
minimal Supabase stand-ins (roles, `auth.users`, `auth.uid()`), then every
migration in order — and drives the functions directly.

It needs Postgres server binaries (`initdb`, `pg_ctl`, `psql`) and runs when
it finds them: `HA_TEST_PG_BIN=<dir with the binaries>`, else on `PATH`.
Without them it skips. It never connects to an existing database. Run:

    HA_TEST_PG_BIN=/usr/lib/postgresql/16/bin \
        python -m pytest tests/supabase/test_generation_rpcs.py

Postgres 15+ is enough; the dump's PG17-only bits (`transaction_timeout`,
the MAINTAIN privilege) are dropped from the loaded copy.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
import threading
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

pg = pytest.importorskip("pg8000.native")

ROOT = Path(__file__).resolve().parents[2]
SUPABASE = ROOT / "supabase"

# What a Supabase project provides before init.sql runs.
_STUBS = """
DO $$ BEGIN
  IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname='anon') THEN CREATE ROLE anon NOLOGIN; END IF;
  IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname='authenticated') THEN CREATE ROLE authenticated NOLOGIN; END IF;
  IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname='service_role') THEN CREATE ROLE service_role NOLOGIN BYPASSRLS; END IF;
  IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname='supabase_admin') THEN CREATE ROLE supabase_admin NOLOGIN; END IF;
END $$;
CREATE SCHEMA IF NOT EXISTS auth;
CREATE TABLE IF NOT EXISTS auth.users (id uuid PRIMARY KEY, email text);
CREATE OR REPLACE FUNCTION auth.uid() RETURNS uuid LANGUAGE sql STABLE AS $$
  SELECT nullif(current_setting('request.jwt.claim.sub', true), '')::uuid $$;
CREATE OR REPLACE FUNCTION auth.role() RETURNS text LANGUAGE sql STABLE AS $$ SELECT 'service_role'::text $$;
CREATE OR REPLACE FUNCTION auth.jwt() RETURNS jsonb LANGUAGE sql STABLE AS $$ SELECT '{}'::jsonb $$;
GRANT USAGE ON SCHEMA auth TO anon, authenticated, service_role;
GRANT USAGE ON SCHEMA public TO anon, authenticated, service_role;
"""


def _pg_bin() -> Path | None:
    named = os.environ.get("HA_TEST_PG_BIN")
    if named:
        return Path(named)
    found = shutil.which("pg_ctl")
    return Path(found).parent if found else None


def _baseline() -> str:
    """init.sql minus the dump's PG17-only lines."""
    text = (SUPABASE / "init.sql").read_text(encoding="utf-8")
    text = "\n".join(line for line in text.split("\n") if "transaction_timeout" not in line)
    return text.replace(",MAINTAIN", "")


class _Cluster:
    def __init__(self, bindir: Path):
        self.bin = bindir
        self.root = Path(tempfile.mkdtemp(prefix="hapg"))
        self.data = self.root / "data"
        self.sock = self.root / "s"
        self.sock.mkdir()

    def _run(self, *args, **kw):
        return subprocess.run([str(self.bin / args[0]), *args[1:]], capture_output=True,
                              text=True, check=False, timeout=120, **kw)

    def start(self) -> None:
        p = self._run("initdb", "-D", str(self.data), "-U", "postgres", "--auth=trust", "-E", "UTF8")
        assert p.returncode == 0, p.stderr
        p = self._run("pg_ctl", "-D", str(self.data), "-w", "-l", str(self.root / "log"),
                      "-o", f"-k {self.sock} -c listen_addresses=''", "start")
        assert p.returncode == 0, p.stderr + (self.root / "log").read_text(errors="replace")

    def load(self, sql: str, name: str) -> None:
        path = self.root / name
        path.write_text(sql, encoding="utf-8")
        p = self._run("psql", "-h", str(self.sock), "-U", "postgres", "-d", "postgres",
                      "-v", "ON_ERROR_STOP=1", "-q", "-f", str(path))
        assert p.returncode == 0, f"{name}: {p.stderr[-2000:]}"

    def stop(self) -> None:
        self._run("pg_ctl", "-D", str(self.data), "-m", "immediate", "stop")
        shutil.rmtree(self.root, ignore_errors=True)

    def connect(self):
        sock = next(p for p in self.sock.iterdir() if p.name.startswith(".s.PGSQL.")
                    and not p.name.endswith(".lock"))
        return pg.Connection("postgres", unix_sock=str(sock), database="postgres")


@pytest.fixture(scope="module")
def server():
    bindir = _pg_bin()
    if bindir is None or not (bindir / "initdb").exists():
        pytest.skip("no Postgres server binaries (set HA_TEST_PG_BIN)")
    cluster = _Cluster(bindir)
    cluster.start()
    try:
        cluster.load(_STUBS, "stubs.sql")
        cluster.load(_baseline(), "init.sql")
        for migration in sorted((SUPABASE / "migrations").glob("*.sql")):
            cluster.load(migration.read_text(encoding="utf-8"), migration.name)
        yield cluster
    finally:
        cluster.stop()


@pytest.fixture
def c(server):
    conn = server.connect()
    conn.run("TRUNCATE public.spec_generations CASCADE")
    conn.run("TRUNCATE public.spec_generation_logs")
    conn.run("DELETE FROM public.admins")
    conn.run("DELETE FROM auth.users")
    conn.run("INSERT INTO auth.users (id, email) VALUES (CAST(:u AS uuid), 'x@y')", u=U)
    yield conn
    conn.close()


T0 = datetime(2026, 10, 1, tzinfo=timezone.utc)
def t(m): return T0 + timedelta(minutes=m)
U = str(uuid.uuid4())
M = "org/model"


def rec_gen(c, at, yaml="ok", user=U):
    return json.loads(json.dumps(c.run(
        "SELECT public.record_generation(CAST(:u AS uuid), :m, :y, '{}'::jsonb, '[]'::jsonb, 'claude', NULL, :at)",
        u=user, m=M, y=yaml, at=at)[0][0]))


def rec_pending(c, at, yaml="blocked", gaps=None, user=U):
    return c.run(
        "SELECT public.record_pending_generation(CAST(:u AS uuid), :m, :y, '{}'::jsonb, '[]'::jsonb, 'claude', NULL, CAST(:g AS jsonb), 'ev1', :at)",
        u=user, m=M, y=yaml, g=json.dumps(gaps or [{"capability": "cap"}]), at=at)[0][0]


def transition(c, rid, rev, to, **kw):
    cols = ", ".join(f"{k} => :{k}" for k in kw)
    q = f"SELECT public.transition_pending_generation(p_id => CAST(:i AS uuid), p_expected_revision => :r, p_to => :to{', ' + cols if cols else ''})"
    return c.run(q, i=rid, r=rev, to=to, **kw)[0][0]


def rows(c):
    out = []
    for r in c.run(
        "SELECT id::text, status, spec_yaml, run_started_at, pending_detail, "
        "blocked_engine_version, revision, pr_url "
        "FROM public.spec_generations ORDER BY created_at"
    ):
        d = dict(zip(
            ("id", "status", "spec_yaml", "run_started_at", "pending_detail",
             "blocked_engine_version", "revision", "pr_url"), r, strict=True))
        detail = d.pop("pending_detail")
        if d["status"] == "blocked_by_engine":
            d["engine_gaps"], d["repairable_errors"] = detail, None
        elif d["status"] == "needs_repair":
            d["repairable_errors"], d["engine_gaps"] = detail, None
        else:
            d["engine_gaps"] = d["repairable_errors"] = None
        out.append(d)
    return out


def insert(c, status, at, yaml="v", created=None, pr_url=None):
    return c.run("INSERT INTO public.spec_generations (user_id, model_id, spec_yaml, status, run_started_at, created_at, pr_url) VALUES (CAST(:u AS uuid), :m, :y, :s, :at, COALESCE(:cr, now()), :pr) RETURNING id::text",
                 u=U, m=M, y=yaml, s=status, at=at, cr=created, pr=pr_url)[0][0]


# --- pending beside valid records ---------------------------------------
@pytest.mark.parametrize("status", ["generated", "closed", "contributed", "merged"])
def test_blocked_never_touches_a_valid_record(c, status):
    vid = insert(c, status, t(0), yaml="valid")
    out = rec_pending(c, t(5))
    assert out["outcome"] == "recorded"
    by = {r["id"]: r for r in rows(c)}
    assert by[vid]["spec_yaml"] == "valid" and by[vid]["status"] == status
    assert sorted(r["status"] for r in rows(c)) == sorted([status, "blocked_by_engine"])


def test_blocked_run_older_than_valid_record_is_stale(c):
    insert(c, "generated", t(10))
    assert rec_pending(c, t(5))["outcome"] == "stale"
    assert [r["status"] for r in rows(c)] == ["generated"]


def test_legacy_row_without_run_start_reads_created_at(c):
    insert(c, "generated", None, created=t(10))
    assert rec_pending(c, t(5))["outcome"] == "stale"
    assert rec_pending(c, t(11))["outcome"] == "recorded"


# --- pending CAS -------------------------------------------------------
def test_earlier_started_late_pending_cannot_overwrite_newer_pending(c):
    first = rec_pending(c, t(10), yaml="newer")
    assert first["outcome"] == "recorded"
    late = rec_pending(c, t(5), yaml="older")
    assert late["outcome"] == "stale"
    assert [r["spec_yaml"] for r in rows(c)] == ["newer"]


def test_later_pending_overwrites_in_place(c):
    a = rec_pending(c, t(1), yaml="a")
    b = rec_pending(c, t(2), yaml="b", gaps=[{"capability": "x"}])
    assert b["outcome"] == "recorded" and b["id"] == a["id"]
    (r,) = rows(c)
    assert r["spec_yaml"] == "b" and r["engine_gaps"] == [{"capability": "x"}]


def test_later_pending_overwrites_needs_repair_and_clears_its_errors(c):
    a = rec_pending(c, t(1))
    rev = rows(c)[0]["revision"]
    assert transition(c, a["id"], rev, "needs_repair", p_repairable_errors=json.dumps(["e"]))["outcome"] == "recorded"
    b = rec_pending(c, t(2))
    assert b["id"] == a["id"]
    (r,) = rows(c)
    assert r["status"] == "blocked_by_engine" and r["repairable_errors"] is None


def test_late_write_after_pending_deleted_records_new(c):
    a = rec_pending(c, t(1))
    c.run("DELETE FROM public.spec_generations WHERE id = CAST(:i AS uuid)", i=a["id"])
    b = rec_pending(c, t(0))   # began before the deleted one: nothing newer stands
    assert b["outcome"] == "recorded" and b["id"] != a["id"]


def test_late_write_behind_a_newer_rejected_candidate_is_stale(c):
    insert(c, "rejected", t(10))
    assert rec_pending(c, t(5))["outcome"] == "stale"
    assert rec_pending(c, t(11))["outcome"] == "recorded"


def test_concurrent_pending_writes_leave_one(c, server):
    errs = []
    def go(m):
        conn = server.connect()
        try:
            rec_pending(conn, t(m), yaml=f"y{m}")
        except Exception as e:  # pragma: no cover
            errs.append(e)
        finally:
            conn.close()
    ts = [threading.Thread(target=go, args=(m,)) for m in range(12)]
    [x.start() for x in ts]; [x.join() for x in ts]
    assert not errs
    rs = rows(c)
    assert len(rs) == 1 and rs[0]["spec_yaml"] == "y11"


# --- success ------------------------------------------------------------
def test_success_drops_earlier_pending_keeps_later(c):
    rec_pending(c, t(1), yaml="early")
    out = rec_gen(c, t(5))
    assert out["outcome"] == "recorded"
    assert [r["status"] for r in rows(c)] == ["generated"]
    rec_pending(c, t(9), yaml="later")
    rec_gen(c, t(7))
    assert sorted(r["status"] for r in rows(c)) == ["blocked_by_engine", "generated"]


def test_success_overwrites_mutable_and_inserts_beside_frozen(c):
    gid = insert(c, "rejected", t(0))
    out = rec_gen(c, t(1), yaml="new")
    assert out["id"] == gid
    (r,) = rows(c)
    assert r["status"] == "generated" and r["spec_yaml"] == "new"
    c.run("UPDATE public.spec_generations SET status='contributed', pr_url='p'")
    out2 = rec_gen(c, t(2), yaml="newer")
    assert out2["id"] != gid
    assert sorted(r["status"] for r in rows(c)) == ["contributed", "generated"]


def test_needs_user_and_run_start(c):
    with pytest.raises(pg.DatabaseError):
        c.run("SELECT public.record_generation(NULL, 'm', 'y', NULL, NULL, NULL, NULL, now())")
    with pytest.raises(pg.DatabaseError):
        c.run("SELECT public.record_pending_generation(CAST(:u AS uuid), 'm', 'y', NULL, NULL, NULL, NULL, NULL, NULL, NULL)", u=U)


# --- transition ---------------------------------------------------------
def test_revision_advances_on_any_update(c):
    rec_pending(c, t(1))
    r0 = rows(c)[0]["revision"]
    c.run("UPDATE public.spec_generations SET pr_url = NULL")
    assert rows(c)[0]["revision"] == r0 + 1


def test_stale_revision_conflicts(c):
    a = rec_pending(c, t(1))
    rev = rows(c)[0]["revision"]
    rec_pending(c, t(2))  # someone wrote since
    assert transition(c, a["id"], rev, "blocked_by_engine")["outcome"] == "conflict"


def test_transition_after_discard_conflicts(c):
    a = rec_pending(c, t(1))
    rev = rows(c)[0]["revision"]
    c.run("DELETE FROM public.spec_generations WHERE id = CAST(:i AS uuid)", i=a["id"])
    assert transition(c, a["id"], rev, "promote")["outcome"] == "conflict"
    assert transition(c, a["id"], rev, "needs_repair")["outcome"] == "conflict"


def test_transition_of_missing_record_conflicts(c):
    assert transition(c, str(uuid.uuid4()), 0, "promote")["outcome"] == "conflict"


def test_unknown_transition_raises(c):
    a = rec_pending(c, t(1))
    with pytest.raises(pg.DatabaseError):
        transition(c, a["id"], 0, "generated")


def test_refresh_blocked_and_needs_repair(c):
    a = rec_pending(c, t(1))
    rev = rows(c)[0]["revision"]
    out = transition(c, a["id"], rev, "needs_repair", p_repairable_errors=json.dumps(["fix me"]))
    assert out["outcome"] == "recorded" and out["status"] == "needs_repair"
    (r,) = rows(c)
    assert r["repairable_errors"] == ["fix me"] and r["engine_gaps"] is None and r["blocked_engine_version"] is None
    out = transition(c, a["id"], r["revision"], "blocked_by_engine", p_engine_gaps=json.dumps([{"capability": "z"}]), p_engine_version="ev2")
    (r,) = rows(c)
    assert r["status"] == "blocked_by_engine" and r["engine_gaps"] == [{"capability": "z"}]
    assert r["repairable_errors"] is None and r["blocked_engine_version"] == "ev2"


def test_promote_in_place_without_mutable(c):
    a = rec_pending(c, t(1), yaml="blk")
    rev = rows(c)[0]["revision"]
    out = transition(c, a["id"], rev, "promote")
    assert out == {"outcome": "recorded", "id": a["id"]}
    (r,) = rows(c)
    assert r["status"] == "generated" and r["spec_yaml"] == "blk"
    assert r["engine_gaps"] is None and r["repairable_errors"] is None and r["blocked_engine_version"] is None
    assert r["run_started_at"] == t(1)


def test_promote_beside_frozen_converts_in_place(c):
    insert(c, "merged", t(0), yaml="old", pr_url="p")
    a = rec_pending(c, t(1))
    rev = next(r["revision"] for r in rows(c) if r["status"] == "blocked_by_engine")
    out = transition(c, a["id"], rev, "promote", p_spec_yaml="fixed", p_run_started_at=t(3))
    assert out["id"] == a["id"]
    assert sorted((r["status"], r["spec_yaml"]) for r in rows(c)) == [("generated", "fixed"), ("merged", "old")]


def test_promote_overwrites_mutable_and_drops_pending(c):
    mid = insert(c, "closed", t(0), yaml="old", pr_url="p")
    a = rec_pending(c, t(1), yaml="blk")
    rev = next(r for r in rows(c) if r["id"] == a["id"])["revision"]
    out = transition(c, a["id"], rev, "promote", p_spec_yaml="fixed", p_run_started_at=t(4))
    assert out == {"outcome": "recorded", "id": mid}
    (r,) = rows(c)
    assert r["id"] == mid and r["status"] == "generated" and r["spec_yaml"] == "fixed"
    assert r["pr_url"] is None and r["run_started_at"] == t(4)
    assert r["engine_gaps"] is None and r["repairable_errors"] is None and r["blocked_engine_version"] is None


def test_promote_superseded_by_later_valid_record(c):
    a = rec_pending(c, t(1))
    insert(c, "contributed", t(5), pr_url="p")
    rev = next(r for r in rows(c) if r["id"] == a["id"])["revision"]
    assert transition(c, a["id"], rev, "promote")["outcome"] == "superseded"
    assert [r["status"] for r in rows(c)] == ["contributed"]


def test_promote_and_success_serialize(c, server):
    """Many promotes racing successes: never two pending rows, never a
    pending row older than the newest success, always one mutable."""
    for i in range(8):
        c.run("TRUNCATE public.spec_generations CASCADE")
        a = rec_pending(c, t(10))
        rev = rows(c)[0]["revision"]
        res = {}
        def promote():
            conn = server.connect(); res["p"] = transition(conn, a["id"], rev, "promote"); conn.close()
        def success():
            conn = server.connect(); res["s"] = rec_gen(conn, t(20), yaml="succ"); conn.close()
        th = [threading.Thread(target=promote), threading.Thread(target=success)]
        if i % 2: th.reverse()
        [x.start() for x in th]; [x.join() for x in th]
        rs = rows(c)
        # Either order ends in one working candidate holding the later run's
        # spec: promote-then-success overwrites it; success-then-promote finds
        # the pending record gone (conflict).
        assert [(r["status"], r["spec_yaml"]) for r in rs] == [("generated", "succ")], (rs, res)
        assert res["p"]["outcome"] in ("recorded", "conflict")


# --- admin reads ----------------------------------------------------------
def as_admin(c):
    admin = str(uuid.uuid4())
    c.run("INSERT INTO auth.users (id) VALUES (CAST(:a AS uuid))", a=admin)
    c.run("INSERT INTO public.admins (user_id) VALUES (CAST(:a AS uuid))", a=admin)
    c.run("SELECT set_config('request.jwt.claim.sub', :a, false)", a=admin)


def test_admin_filter_and_count(c):
    insert(c, "generated", t(0))
    rec_pending(c, t(1))
    as_admin(c)
    assert c.run("SELECT count(*) FROM public.get_spec_generations(50, 0, 'recent', 'blocked_by_engine')")[0][0] == 1
    assert c.run("SELECT public.get_spec_generations_count('blocked_by_engine')")[0][0] == 1
    assert c.run("SELECT public.get_spec_generations_count(NULL)")[0][0] == 2
    assert c.run("SELECT public.get_spec_generations_count()")[0][0] == 2
    assert c.run("SELECT count(*) FROM public.get_spec_generations(50, 0, 'recent')")[0][0] == 2
    (gaps,) = c.run("SELECT engine_gaps FROM public.get_spec_generations(p_status => 'blocked_by_engine')")[0]
    assert gaps == [{"capability": "cap"}]
    with pytest.raises(pg.DatabaseError):
        c.run("SELECT public.get_spec_generations_count('bogus')")


def test_admin_runs_filter(c):
    c.run("INSERT INTO public.spec_generation_logs (model_id, error_kind, state) VALUES ('m', 'blocked_by_engine', 'done'), ('m', 'analysis_refusal_unattributed', 'done'), ('m', NULL, 'done')")
    as_admin(c)
    assert c.run("SELECT count(*) FROM public.get_spec_generation_runs(50, 0, 'analysis_refusal_unattributed')")[0][0] == 1
    assert c.run("SELECT public.get_spec_generation_runs_count('blocked_by_engine')")[0][0] == 1
    assert c.run("SELECT public.get_spec_generation_runs_count()")[0][0] == 3
    with pytest.raises(pg.DatabaseError):
        c.run("INSERT INTO public.spec_generation_logs (model_id, error_kind) VALUES ('m', 'nope')")


def test_detail_carries_pending_fields(c):
    a = rec_pending(c, t(1))
    as_admin(c)
    r = c.run("SELECT status, engine_gaps, blocked_engine_version, revision FROM public.get_spec_generation_detail(CAST(:i AS uuid))", i=a["id"])[0]
    assert r[0] == "blocked_by_engine" and r[1] == [{"capability": "cap"}] and r[2] == "ev1" and r[3] == 0


def test_non_admin_reads_refused(c):
    c.run("SELECT set_config('request.jwt.claim.sub', :a, false)", a=str(uuid.uuid4()))
    with pytest.raises(pg.DatabaseError):
        c.run("SELECT public.get_spec_generation_runs_count()")


def test_writes_are_service_role_only(c):
    for role in ("authenticated", "anon"):
        for fn in ("record_generation(uuid, text, text, jsonb, jsonb, text, text, timestamp with time zone)",
                   "record_pending_generation(uuid, text, text, jsonb, jsonb, text, text, jsonb, text, timestamp with time zone)",
                   "transition_pending_generation(uuid, bigint, text, jsonb, jsonb, text, text, jsonb, text, text, timestamp with time zone)"):
            assert c.run(f"SELECT has_function_privilege('{role}', 'public.{fn}', 'EXECUTE')")[0][0] is False
        assert c.run(f"SELECT has_function_privilege('{role}', 'public.get_spec_generation_runs(integer, integer, text)', 'EXECUTE')")[0][0] is (role == "authenticated")
    assert c.run("SELECT has_function_privilege('service_role', 'public.record_generation(uuid, text, text, jsonb, jsonb, text, text, timestamp with time zone)', 'EXECUTE')")[0][0] is True


def test_old_backend_writes_still_work(c):
    """The backend running before deploy: plain INSERT/PATCH with old columns."""
    rid = c.run("INSERT INTO public.spec_generations (user_id, model_id, spec_yaml) VALUES (CAST(:u AS uuid), 'm', 'y') RETURNING id::text", u=U)[0][0]
    c.run("UPDATE public.spec_generations SET status='rejected' WHERE id = CAST(:i AS uuid)", i=rid)
    assert rows(c)[0]["revision"] == 1


def test_a_pending_record_cannot_be_rejected_in_place(c):
    """Regression (review 86ca0218): an unlocked reject of a pending record
    used to turn it into a second MUTABLE row, and the next success then
    left two `generated` rows. Any UPDATE moving a record out of PENDING
    other than promotion is refused; the admin discards it instead."""
    gid = insert(c, "generated", t(0), yaml="orig")
    a = rec_pending(c, t(10), yaml="blk")
    for status in ("rejected", "generated", "closed", "contributed"):
        with pytest.raises(pg.DatabaseError):
            c.run("UPDATE public.spec_generations SET status=:s WHERE id = CAST(:i AS uuid)",
                  s=status, i=a["id"])
    c.run("DELETE FROM public.spec_generations WHERE id = CAST(:i AS uuid) "
          "AND status IN ('blocked_by_engine','needs_repair')", i=a["id"])
    out = rec_gen(c, t(20), yaml="succ")
    assert out["id"] == gid
    assert [(r["status"], r["spec_yaml"]) for r in rows(c)] == [("generated", "succ")]


def test_promotion_is_the_one_way_out_and_does_not_leak_its_mark(c):
    a = rec_pending(c, t(1))
    rev = rows(c)[0]["revision"]
    assert transition(c, a["id"], rev, "promote")["outcome"] == "recorded"
    b = rec_pending(c, t(5))
    # the promotion's mark is gone: a plain reject of the new pending record
    # in the same session is still refused
    with pytest.raises(pg.DatabaseError):
        c.run("UPDATE public.spec_generations SET status='rejected' WHERE id = CAST(:i AS uuid)",
              i=b["id"])
    assert sorted(r["status"] for r in rows(c)) == ["blocked_by_engine", "generated"]


def test_old_get_latest_neq_rejected_would_serve_blocked(c):
    """The backend still running after 0002 uses status <> rejected.
    Once a blocked row exists, that filter would pick it over an older
    generated row. The 0002-first window is safe only because the old
    backend never writes blocked rows."""
    insert(c, "generated", t(0), yaml="valid")
    rec_pending(c, t(5), yaml="blk")
    latest = c.run(
        "SELECT status, spec_yaml FROM public.spec_generations "
        "WHERE user_id = CAST(:u AS uuid) AND model_id = :m AND status <> 'rejected' "
        "ORDER BY created_at DESC LIMIT 1",
        u=U, m=M)[0]
    assert list(latest) == ["blocked_by_engine", "blk"]

