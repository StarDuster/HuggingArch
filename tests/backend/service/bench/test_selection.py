"""The lookup's memory-budgeted selection table (`lookup.Selection`,
docs/measured_calibration.md §7.2): one table, for the newest snapshot
asked; an older pin or an over-budget lake takes the pinned query, with the
same answer; a table is never dropped under a lookup reading it."""
from __future__ import annotations

import threading
import uuid

import pytest

from backend.service.bench import lookup as LK
from backend.service.bench import store


def _upload(lake, annotate, owner, rows, *, status="private"):
    store.upload(lake, owner=owner, upload_id=str(uuid.uuid4()), run_id=None, label=None,
                 rows=rows, annotate=annotate, status=status,
                 decided_by=owner if status == "shared" else None)
    return lake.snapshot_id()


def _key(annotate, row) -> dict:
    ann = annotate(row)
    return {"signature": ann["signature"], "physical_shape": ann["physical_shape"],
            "dtype_env_json": row["dtype_env_json"], "gpu": row["gpu"], "phase": row["phase"]}


def _tables(lake) -> set:
    return {r["name"] for r in lake.read(
        "SELECT table_name AS name FROM duckdb_tables() WHERE table_name LIKE 'sel_%'")}


def test_the_held_table_answers_as_the_pinned_query(lake, annotate, make_rows, new_owner):
    alice, admin = new_owner(), new_owner()
    rows = make_rows(3)
    s1 = _upload(lake, annotate, admin, rows[:2], status="shared")
    s2 = _upload(lake, annotate, alice, rows[1:])
    keys = [_key(annotate, r) for r in rows]
    held, pinned = LK.Selection(), LK.Selection()
    pinned.BUDGET_BYTES = 0
    for owner in (None, alice):
        for snap in (s2, s1):
            a = LK.lookup(lake, keys, owner=owner, snapshot=snap, verdicts=LK.Verdicts(),
                          selection=held)
            b = LK.lookup(lake, keys, owner=owner, snapshot=snap, verdicts=LK.Verdicts(),
                          selection=pinned)
            assert a == b and a["rows"] != [[], [], []]
    assert len(_tables(lake)) == 1 and pinned._current[lake][1] is None


def test_a_table_is_never_dropped_under_its_reader(lake, annotate, make_rows, new_owner):
    """A lookup holding snapshot 2's table while two newer snapshots' tables
    are built reads its table to the end; it is dropped once released."""
    owner = new_owner()
    rows = make_rows(3)
    snaps = [_upload(lake, annotate, owner, [r], status="shared") for r in rows]
    sel = LK.Selection()
    entered, release = threading.Event(), threading.Event()
    got: dict = {}

    def reader():
        with sel.use(lake, snaps[0]) as table:
            got["table"] = table
            entered.set()
            release.wait(10)
            got["rows"] = lake.read(f"SELECT count(*) AS n FROM {table}")[0]["n"]

    t = threading.Thread(target=reader)
    t.start()
    assert entered.wait(10)
    for snap in snaps[1:]:                      # two newer snapshots replace it
        with sel.use(lake, snap) as table:
            assert table is not None
    assert got["table"] in _tables(lake)
    release.set()
    t.join(10)
    assert got["rows"] == 1
    assert got["table"] not in _tables(lake) and len(_tables(lake)) == 1


def test_an_older_pin_takes_the_pinned_query(lake, annotate, make_rows, new_owner):
    owner = new_owner()
    rows = make_rows(2)
    s1 = _upload(lake, annotate, owner, rows[:1], status="shared")
    s2 = _upload(lake, annotate, owner, rows[1:], status="shared")
    sel = LK.Selection()
    with sel.use(lake, s2) as table:
        assert table is not None
    with sel.use(lake, s1) as table:
        assert table is None


@pytest.mark.parametrize("budget", [0, LK.Selection.BUDGET_BYTES])
def test_concurrent_lookups_at_moving_snapshots_never_fail(lake, annotate, make_rows,
                                                           new_owner, budget):
    owner = new_owner()
    rows = make_rows(6)
    snaps = [_upload(lake, annotate, owner, [r], status="shared") for r in rows]
    keys = [_key(annotate, r) for r in rows]
    sel = LK.Selection()
    sel.BUDGET_BYTES = budget
    errors: list = []

    def run(snap):
        try:
            for _ in range(5):
                out = LK.lookup(lake, keys, owner=None, snapshot=snap,
                                verdicts=LK.Verdicts(), selection=sel)
                assert sum(map(len, out["rows"])) == snaps.index(snap) + 1
        except Exception as e:
            errors.append(e)
    threads = [threading.Thread(target=run, args=(s,)) for s in snaps]
    for t in threads:
        t.start()
    for t in threads:
        t.join(30)
    assert errors == []
    assert len(_tables(lake)) <= 1
