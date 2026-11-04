"""Read the bench lake's rows off-line; delete them through the service.

A maintenance tool that has to look at every row of the lake (an audit, a
purge) used to page `GET /api/bench/rows` upload by upload, which makes the
service hold each page in its own DuckDB — and over the whole lake that ran
the service out of memory. Reading needs no service at all: the catalog is a
DuckLake file, so a fresh COPY of it (the `.ducklake` and its `.wal`, taken
together) attached READ_ONLY over the live data files gives the committed
rows at that instant without a lock on the service's catalog or a byte of
its memory. Only the deletes go through the service
(`POST /api/bench/rows/delete`, 500 ids a call), which deletes only the rows
its owner holds — so `rows` reports the owner of each row, and the caller
keeps to its own.

    from tools.lake_rows import rows, delete
    for row in rows("m.outcome = 'error'"):   # dicts, every column + owner
        ...
    delete(ids)                                 # in chunks of 500
"""
from __future__ import annotations

import contextlib
import json
import os
import shutil
import tempfile
import urllib.request
from pathlib import Path
from typing import Any, Dict, Iterator, List, Sequence

REPO = Path(__file__).resolve().parents[1]
#: The lake the local service serves (`backend/service/bench/lake.py`):
#: this checkout's, unless `HA_BENCH_LAKE` names another (a worktree has no
#: lake of its own).
LAKE = Path(os.environ.get("HA_BENCH_LAKE", REPO / ".cache" / "bench_lake"))
SERVICE = "http://127.0.0.1:8100"
#: The uploader the local service authenticates the maintenance calls as.
OWNER = os.environ.get("HA_BENCH_USER", "07a6e2a7-54b4-4e52-ba0d-21ce44f06b09")
CHUNK = 500


@contextlib.contextmanager
def attached(lake: Path = LAKE) -> Iterator[Any]:
    """A DuckDB connection with the lake attached READ_ONLY as `lake`, over
    a fresh copy of its catalog (the `.ducklake` and its `.wal`, taken
    together) and the live data files: the committed rows at that instant,
    with no lock on the service's catalog. The copy is removed on exit."""
    import duckdb
    if not (lake / "catalog.ducklake").exists():
        raise FileNotFoundError(
            f"no DuckLake catalog at {lake}; set HA_BENCH_LAKE or pass "
            f"lake= (a worktree has no lake of its own)")
    copy = Path(tempfile.mkdtemp(prefix="lakecopy-"))
    try:
        for name in ("catalog.ducklake", "catalog.ducklake.wal"):
            if (lake / name).exists():
                shutil.copy2(lake / name, copy / name)
        con = duckdb.connect()
        try:
            con.execute(
                f"ATTACH 'ducklake:{copy / 'catalog.ducklake'}' AS lake "
                f"(DATA_PATH '{lake.resolve() / 'data'}/', READ_ONLY)")
            yield con
        finally:
            con.close()
    finally:
        shutil.rmtree(copy, ignore_errors=True)


def query(sql: str, params: Sequence[Any] = (), *,
          lake: Path = LAKE) -> List[Dict[str, Any]]:
    """The rows of one SQL statement over the attached lake (`attached`),
    as dicts — for an aggregate a maintenance tool reads, where `rows`
    would stream every measurement to Python."""
    with attached(lake) as con:
        cur = con.execute(sql, list(params))
        cols = [d[0] for d in cur.description]
        return [dict(zip(cols, values, strict=True))
                for values in cur.fetchall()]


def rows(where: str = "TRUE", params: Sequence[Any] = (), *,
         lake: Path = LAKE, batch: int = 20000) -> Iterator[Dict[str, Any]]:
    """Every measurement row matching `where` (SQL over `m`, the
    measurements table), as a dict of its columns plus `owner` (the batch's
    owner, text). Values are as stored: the JSON columns are TEXT, a
    missing value is None.

    Read from the read-only attach (`attached`), streamed `batch` rows at a
    time."""
    with attached(lake) as con:
        cur = con.execute(
            "SELECT m.*, b.owner::VARCHAR AS owner FROM lake.measurements m "
            f"JOIN lake.batches b USING (batch_id) WHERE {where}",
            list(params))
        cols = [d[0] for d in cur.description]
        while True:
            got = cur.fetchmany(batch)
            if not got:
                break
            for values in got:
                row = dict(zip(cols, values, strict=True))
                row["id"] = str(row["id"])
                yield row


def call(method: str, path: str, body: Any = None) -> Dict[str, Any]:
    """One call to the local bench service, as the uploader authenticates."""
    secret = Path("/tmp/ha_bench/bench_secret").read_text().strip()
    request = urllib.request.Request(
        SERVICE + path, method=method,
        data=None if body is None else json.dumps(body).encode(),
        headers={"X-HA-Internal": secret, "X-HA-User": OWNER,
                 "Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=600) as response:
        return json.loads(response.read().decode())


def delete(ids: Sequence[str]) -> int:
    """Delete `ids` through the service, `CHUNK` at a time; the number it
    reports deleted (only rows `OWNER` holds are)."""
    deleted = 0
    for start in range(0, len(ids), CHUNK):
        deleted += int(call("POST", "/api/bench/rows/delete",
                            {"ids": list(ids[start:start + CHUNK])}
                            ).get("deleted") or 0)
    return deleted
