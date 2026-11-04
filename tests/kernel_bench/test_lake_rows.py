"""The lake is read off-line and only deleted through the service.

`tools.unserved_rows --lake` (and the purge scripts) used to page
`GET /api/bench/rows` over every upload, and over the whole lake that ran
the service's DuckDB out of memory. `tools.lake_rows` reads a fresh copy of
the catalog attached READ_ONLY, and the audit asks the service only to
delete.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

duckdb = pytest.importorskip("duckdb")

OWNER = "07a6e2a7-54b4-4e52-ba0d-21ce44f06b09"
OTHER = "11111111-2222-3333-4444-555555555555"

_SINK_DIMS = {"B": 1, "d_head": 64, "kv_block_size": 16, "n_attn_sinks": 8,
              "n_heads": 8, "n_kv_heads": 1, "s_kv": 128, "s_query": 1,
              "s_window": 128}
_PLAIN_DIMS = {k: v for k, v in _SINK_DIMS.items() if k != "n_attn_sinks"}
_ENV = json.dumps({"activation": "bf16", "kv": "bf16", "output": "bf16"})


def _lake(tmp_path):
    """A small DuckLake with the service's two tables: a refused row and a
    served row of ours, and a refused row of another owner's."""
    lake = tmp_path / "bench_lake"
    (lake / "data").mkdir(parents=True)
    con = duckdb.connect()
    con.execute(f"ATTACH 'ducklake:{lake / 'catalog.ducklake'}' AS lake "
                f"(DATA_PATH '{lake / 'data'}/')")
    con.execute("CREATE TABLE lake.batches (batch_id BIGINT, owner UUID)")
    con.execute("CREATE TABLE lake.measurements (id UUID, batch_id BIGINT, "
                "framework VARCHAR, kernel_impl VARCHAR, op VARCHAR, "
                "phase VARCHAR, gpu VARCHAR, outcome VARCHAR, "
                "shape_json VARCHAR, dtype_env_json VARCHAR)")
    con.execute("INSERT INTO lake.batches VALUES (1, ?), (2, ?)", [OWNER, OTHER])
    rows = [("00000000-0000-0000-0000-000000000001", 1, _SINK_DIMS),
            ("00000000-0000-0000-0000-000000000002", 1, _PLAIN_DIMS),
            ("00000000-0000-0000-0000-000000000003", 2, _SINK_DIMS)]
    for rid, batch, dims in rows:
        con.execute(
            "INSERT INTO lake.measurements VALUES (?, ?, 'flashinfer', "
            "'flashinfer_paged_decode@_b16_tc', 'flash_attention', 'decode', "
            "'H100-SXM', 'measured', ?, ?)", [rid, batch, json.dumps(dims), _ENV])
    con.close()
    return lake


def test_rows_are_read_off_a_copy_of_the_catalog(tmp_path):
    from tools import lake_rows
    lake = _lake(tmp_path)
    before = sorted(p.name for p in lake.iterdir())
    got = list(lake_rows.rows("m.gpu = ?", ["H100-SXM"], lake=lake))
    assert {r["id"][-1] for r in got} == {"1", "2", "3"}
    assert {r["owner"] for r in got} == {OWNER, OTHER}
    assert sorted(p.name for p in lake.iterdir()) == before


def test_the_lake_audit_never_pages_the_service(tmp_path, monkeypatch):
    import urllib.request
    from kernel_bench.providers import all_providers
    from tools import lake_rows, unserved_rows

    def no_network(*a, **k):
        raise AssertionError("the audit reached the live bench service")
    # Hermetic whatever the code under test does: nothing here may reach the
    # local service (the old audit paged it).
    monkeypatch.setattr(urllib.request, "urlopen", no_network)
    lake = _lake(tmp_path)
    real_rows = lake_rows.rows
    monkeypatch.setattr(lake_rows, "rows",
                        lambda *a, **k: real_rows(*a, **{**k, "lake": lake}))
    monkeypatch.setattr(lake_rows, "OWNER", OWNER)
    asked = []

    def call(method, path, body=None):
        asked.append((method, path))
        assert path == "/api/bench/rows/delete", path
        return {"deleted": len(body["ids"])}
    monkeypatch.setattr(lake_rows, "call", call)
    found, deleted = unserved_rows._purge_lake(
        list(all_providers()), True, because="variant_dims:n_attn_sinks",
        record=str(tmp_path / "record.csv"))
    # Ours and refused: deleted. Served, or another owner's: left.
    assert (found, deleted) == (1, 1)
    assert asked == [("POST", "/api/bench/rows/delete")]
    assert "00000000-0000-0000-0000-000000000001" in (
        tmp_path / "record.csv").read_text()


def test_a_checkout_without_a_lake_says_so(tmp_path):
    # A worktree has no `.cache/bench_lake`; attaching a copy of nothing
    # failed inside DuckDB with "database does not exist".
    from tools import lake_rows
    with pytest.raises(FileNotFoundError, match="HA_BENCH_LAKE"):
        list(lake_rows.rows(lake=tmp_path / "nowhere"))


def _results(tmp_path, name, dims):
    """One results tree holding a single sink cell of the paged decode."""
    import csv
    path = tmp_path / name / "m" / "flashinfer.csv"
    path.parent.mkdir(parents=True)
    fields = ["framework", "kernel_impl", "op", "phase", "gpu", "outcome",
              "shape_json", "dtype_env_json"]
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerow({"framework": "flashinfer",
                         "kernel_impl": "flashinfer_paged_decode@_b16_tc",
                         "op": "flash_attention", "phase": "decode",
                         "gpu": "H100-SXM", "outcome": "measured",
                         "shape_json": json.dumps(dims), "dtype_env_json": _ENV})
    return path


def _age(path, seconds):
    import os
    import time
    then = time.time() - seconds
    os.utime(path, (then, then))


def test_a_file_still_being_written_stops_every_rewrite(tmp_path):
    # Two stores in one run: the quiet one must not be rewritten while the
    # other is still being appended to by the harness.
    from tools import unserved_rows
    quiet = _results(tmp_path, "quiet", _SINK_DIMS)
    busy = _results(tmp_path, "busy", _SINK_DIMS)
    _age(quiet, 2 * unserved_rows.FRESH_SECONDS)
    before = quiet.read_text(), busy.read_text()
    code = unserved_rows.main([
        "--dir", str(tmp_path / "quiet"), "--dir", str(tmp_path / "busy"),
        "--because", "variant_dims:n_attn_sinks", "--apply",
        "--record", str(tmp_path / "record.csv")])
    assert code == 3
    assert (quiet.read_text(), busy.read_text()) == before
    assert not (tmp_path / "record.csv").exists()


def test_quiet_files_are_rewritten(tmp_path):
    from tools import unserved_rows
    quiet = _results(tmp_path, "quiet", _SINK_DIMS)
    _age(quiet, 2 * unserved_rows.FRESH_SECONDS)
    code = unserved_rows.main([
        "--dir", str(tmp_path / "quiet"),
        "--because", "variant_dims:n_attn_sinks", "--apply",
        "--record", str(tmp_path / "record.csv")])
    assert code == 0
    assert "n_attn_sinks" not in quiet.read_text()


def test_a_lake_delete_short_of_the_listing_fails(tmp_path, monkeypatch):
    import urllib.request
    from tools import lake_rows, unserved_rows

    def no_network(*a, **k):
        raise AssertionError("the audit reached the live bench service")
    monkeypatch.setattr(urllib.request, "urlopen", no_network)
    lake = _lake(tmp_path)
    real_rows = lake_rows.rows
    monkeypatch.setattr(lake_rows, "rows",
                        lambda *a, **k: real_rows(*a, **{**k, "lake": lake}))
    monkeypatch.setattr(lake_rows, "OWNER", OWNER)
    monkeypatch.setattr(lake_rows, "call",
                        lambda method, path, body=None: {"deleted": 0})
    code = unserved_rows.main([
        "--lake", "--because", "variant_dims:n_attn_sinks", "--apply",
        "--record", str(tmp_path / "record.csv")])
    assert code == 5
