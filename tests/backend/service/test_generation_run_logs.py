"""Spec ↔ run-log pairing on GET /api/account/generations.

A log names its record: a generate/refine run stamps `record_id` on its log
row when it writes a record, and fix/review transcripts carry
`fix|{user}|{record_id}` / `review|{user}|{record_id}`. A log that names no
record — a failed run, a run logged before the column existed — pairs with
nothing, whatever its model and time."""
from __future__ import annotations

from backend.service import spec_generations as _gens
from backend.service import spec_generation_logs as _genlogs
from backend.service import spec_generation_reviews as _reviews
from backend.service.routes import account as _account
from backend.service.routes.account import _attach_run_logs


def _g(id_, model, ts, **kw):
    return {"id": id_, "model_id": model, "created_at": ts, **kw}


def _l(id_, model, ts, success=True, **kw):
    return {"id": id_, "model_id": model, "created_at": ts, "success": success, **kw}


class TestAttachRunLogs:
    def test_basic_pair(self):
        items = [_g("g1", "m", "2026-09-13T22:00:00+00:00")]
        logs = [_l("l1", "m", "2026-09-13T21:59:00+00:00", record_id="g1")]
        _attach_run_logs(items, logs)
        assert items[0]["log_id"] == "l1" and items[0]["log_success"] is True

    def test_a_later_failed_run_is_not_the_records_log(self):
        """A failed run of the same model after the record wrote no record:
        its log is its own, never the record's (deleting the record must not
        take it)."""
        items = [_g("g1", "m", "2026-09-13T22:00:00+00:00")]
        logs = [_l("l-ok", "m", "2026-09-13T21:59:00+00:00", record_id="g1"),
                _l("l-fail", "m", "2026-09-13T23:00:00+00:00", success=False)]
        _attach_run_logs(items, logs)
        assert items[0]["log_id"] == "l-ok"

    def test_a_log_naming_no_record_pairs_with_none(self):
        items = [_g("g1", "m", "2026-09-13T22:00:00+00:00")]
        _attach_run_logs(items, [_l("l1", "m", "2026-09-13T22:00:05+00:00")])
        assert "log_id" not in items[0]

    def test_two_generations_each_pair_their_own(self):
        items = [_g("g2", "m", "2026-09-13T23:00:00+00:00"),
                 _g("g1", "m", "2026-09-13T22:00:00+00:00")]
        logs = [_l("l2", "m", "2026-09-13T21:00:00+00:00", record_id="g2"),
                _l("l1", "m", "2026-09-13T23:30:00+00:00", record_id="g1")]
        _attach_run_logs(items, logs)
        by_id = {g["id"]: g for g in items}
        assert by_id["g1"]["log_id"] == "l1"
        assert by_id["g2"]["log_id"] == "l2"

    def test_a_regenerated_record_pairs_its_newest_run(self):
        items = [_g("g1", "m", "2026-09-13T23:00:00+00:00")]
        logs = [_l("l-old", "m", "2026-09-13T21:00:00+00:00", record_id="g1"),
                _l("l-new", "m", "2026-09-13T22:59:00+00:00", record_id="g1")]
        _attach_run_logs(items, logs)
        assert items[0]["log_id"] == "l-new"

    def test_no_logs_no_keys(self):
        items = [_g("g1", "m", "2026-09-13T22:00:00+00:00")]
        _attach_run_logs(items, [])
        assert "log_id" not in items[0]

    def test_fix_log_does_not_steal_produce_pairing(self):
        items = [_g("g1", "m", "2026-09-13T22:00:00+00:00")]
        logs = [
            _l("l-gen", "m", "2026-09-13T22:00:05+00:00", record_id="g1"),
            _l("l-fix", "m", "2026-09-13T23:00:00+00:00",
               run_key="fix|u|g1", kind="fix"),
        ]
        _attach_run_logs(items, logs)
        assert items[0]["log_id"] == "l-gen"
        assert items[0]["fix_logs"] == [{
            "id": "l-fix", "success": True,
            "created_at": "2026-09-13T23:00:00+00:00",
        }]
        assert "review_logs" not in items[0]

    def test_fix_log_attaches_from_run_key_without_kind(self):
        items = [_g("g1", "m", "2026-09-13T22:00:00+00:00")]
        logs = [_l("l-fix", "m", "2026-09-13T23:00:00+00:00",
                   run_key="fix|u|g1")]
        _attach_run_logs(items, logs)
        assert "log_id" not in items[0]
        assert items[0]["fix_logs"][0]["id"] == "l-fix"

    def test_review_log_attaches_newest_first(self):
        items = [_g("g1", "m", "2026-09-13T22:00:00+00:00")]
        logs = [
            _l("r1", "m", "2026-09-13T22:10:00+00:00",
               run_key="review|u|g1", kind="review"),
            _l("r2", "m", "2026-09-13T23:00:00+00:00",
               run_key="review|u|g1", kind="review"),
        ]
        _attach_run_logs(items, logs)
        assert [x["id"] for x in items[0]["review_logs"]] == ["r2", "r1"]

    def test_a_fix_that_promoted_names_the_record_it_wrote(self):
        """A fix of a pending record that promotion folded into the working
        record names that record on its log row; its key names the dropped
        pending one."""
        items = [_g("g1", "m", "2026-09-13T22:00:00+00:00")]
        logs = [_l("l-fix", "m", "2026-09-13T23:00:00+00:00",
                   run_key="fix|u|pending", kind="fix", record_id="g1")]
        _attach_run_logs(items, logs)
        assert [x["id"] for x in items[0]["fix_logs"]] == ["l-fix"]

    def test_fix_log_unknown_record_stays_unattached(self):
        items = [_g("g1", "m", "2026-09-13T22:00:00+00:00")]
        logs = [_l("l-fix", "m", "2026-09-13T23:00:00+00:00",
                   run_key="fix|u|other")]
        _attach_run_logs(items, logs)
        assert "log_id" not in items[0]
        assert "fix_logs" not in items[0]


class TestListRoute:
    def test_items_carry_log_id(self, client, user_token, monkeypatch):
        monkeypatch.setattr(_account, "_release_dates_for",
                            lambda ids: {m: "2025-03-01" for m in ids if m})
        monkeypatch.setattr(_gens, "list_for_user",
                            lambda uid: [_g("g1", "m", "2026-09-13T22:00:00+00:00")])
        seen = {}
        def fake_logs(uid, limit=50):
            seen["limit"] = limit
            return [_l("l1", "m", "2026-09-13T22:00:05+00:00", record_id="g1")]
        monkeypatch.setattr(_genlogs, "list_for_user", fake_logs)
        r = client.get("/api/account/generations",
                       headers={"Authorization": f"Bearer {user_token}"})
        assert r.status_code == 200, r.text
        item = r.json()["items"][0]
        assert item["log_id"] == "l1" and item["log_success"] is True
        assert item["release_date"] == "2025-03-01"
        # The paired log is not also reported as an orphan.
        assert r.json()["orphan_logs"] == []
        # Pairing needs more than the 50-row panel default — older generations
        # must still find their logs.
        assert seen["limit"] >= 1000

    def test_orphan_logs_returned_for_inline_management(self, client, user_token, monkeypatch):
        """A failed run produces no spec record; its log comes back under
        `orphan_logs` so the page lists (and deletes) it alongside specs."""
        monkeypatch.setattr(_account, "_release_dates_for",
                            lambda ids: {m: None for m in ids})
        monkeypatch.setattr(_gens, "list_for_user",
                            lambda uid: [_g("g1", "m", "2026-09-13T22:00:00+00:00")])
        monkeypatch.setattr(_genlogs, "list_for_user", lambda uid, limit=50: [
            _l("l-ok", "m", "2026-09-13T22:00:05+00:00", record_id="g1"),
            _l("l-fail", "m", "2026-09-13T23:00:00+00:00", success=False),
        ])
        r = client.get("/api/account/generations",
                       headers={"Authorization": f"Bearer {user_token}"})
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["items"][0]["log_id"] == "l-ok"
        assert [l["id"] for l in body["orphan_logs"]] == ["l-fail"]

    def test_fix_log_on_item_not_orphan(self, client, user_token, monkeypatch):
        monkeypatch.setattr(_account, "_release_dates_for",
                            lambda ids: {m: None for m in ids})
        monkeypatch.setattr(_gens, "list_for_user",
                            lambda uid: [_g("g1", "m", "2026-09-13T22:00:00+00:00")])
        monkeypatch.setattr(_genlogs, "list_for_user", lambda uid, limit=50: [
            _l("l-ok", "m", "2026-09-13T22:00:05+00:00", record_id="g1"),
            _l("l-fix", "m", "2026-09-13T23:00:00+00:00",
               run_key="fix|u|g1", kind="fix"),
            _l("l-rev", "m", "2026-09-13T23:05:00+00:00",
               run_key="review|u|g1", kind="review"),
        ])
        monkeypatch.setattr(_reviews, "verdicts_by_log", lambda ids: {"l-rev": "error"})
        r = client.get("/api/account/generations",
                       headers={"Authorization": f"Bearer {user_token}"})
        assert r.status_code == 200, r.text
        item = r.json()["items"][0]
        assert item["log_id"] == "l-ok"
        assert [x["id"] for x in item["fix_logs"]] == ["l-fix"]
        # the review log names the verdict recorded for it, not the agent's
        assert [(x["id"], x["verdict"]) for x in item["review_logs"]] == [("l-rev", "error")]
        assert r.json()["orphan_logs"] == []

    def test_anonymous_401(self, client):
        assert client.get("/api/account/generations").status_code == 401
