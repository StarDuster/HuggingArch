"""A throwaway database on the local Postgres 17 (socket /tmp, port 55432,
user postgres) for tests that need a Postgres catalog; they skip when it is
unreachable."""
from __future__ import annotations

import contextlib
from typing import Iterator

import pytest

SOCKET = "/tmp/.s.PGSQL.55432"


def _admin():
    pg8000 = pytest.importorskip("pg8000.native")
    try:
        return pg8000.Connection("postgres", unix_sock=SOCKET, database="postgres", timeout=5)
    except Exception as e:     # any failure to connect means "no local Postgres"
        pytest.skip(f"local Postgres unreachable: {e}")


@contextlib.contextmanager
def scratch_database(name: str) -> Iterator[str]:
    """Create database `name` (dropping a leftover), yield its libpq dsn,
    drop it afterwards."""
    admin = _admin()
    try:
        admin.run(f"DROP DATABASE IF EXISTS {name} WITH (FORCE)")
        admin.run(f"CREATE DATABASE {name}")
        yield f"host=/tmp port=55432 user=postgres dbname={name}"
    finally:
        admin.run(f"DROP DATABASE IF EXISTS {name} WITH (FORCE)")
        admin.close()


def advisory_locks(dbname: str) -> list:
    """The advisory locks held or awaited in `dbname`: [objid, granted]."""
    import pg8000.native
    con = pg8000.native.Connection("postgres", unix_sock=SOCKET, database=dbname)
    try:
        return con.run("SELECT objid, granted FROM pg_locks WHERE locktype = 'advisory' "
                       "ORDER BY granted DESC")
    finally:
        con.close()
