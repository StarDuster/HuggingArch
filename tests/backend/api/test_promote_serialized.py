"""Promote (graft code, run pytest, archive) runs once per op at a time: the
review lock is gone by then, so a second review of the same op must not start
a second promote."""
from __future__ import annotations

import asyncio
import json


def test_a_second_promote_of_one_op_does_not_start(monkeypatch, tmp_path):
    from backend.api.routes import op_reviews as R
    (tmp_path / "op_x").mkdir()
    (tmp_path / "op_x" / "review.json").write_text(json.dumps({"verdict": "auto_merge"}))
    monkeypatch.setattr(R, "PROPOSALS_DIR", str(tmp_path))
    monkeypatch.setattr(R.op_review_worker, "check_auto_merge_eligibility",
                        lambda path: (True, []))
    started = []
    monkeypatch.setattr(R, "run_promote", started.append)

    async def run():
        return [c async for c in R._post_agent_pipeline("op_x", True)]

    lock = R._promote_lock("op_x")
    assert lock.acquire(blocking=False)          # a promote is in flight
    try:
        frames = asyncio.run(run())
    finally:
        lock.release()
    assert started == []
    assert "promote_in_progress" in frames[-1]


def test_the_lock_lives_as_long_as_the_promote(monkeypatch):
    """Held in the promoting thread: a second promote is refused for as long
    as the first runs, whatever happens to the stream that started it."""
    import threading
    from backend.api.routes import op_reviews as R
    started, finish = threading.Event(), threading.Event()

    def slow_promote(op):
        started.set()
        finish.wait(5)
        return "done"

    monkeypatch.setattr(R, "run_promote", slow_promote)
    first = threading.Thread(target=R._promote_exclusive, args=("op_y",))
    first.start()
    assert started.wait(5)
    assert R._promote_exclusive("op_y") is None      # refused while it runs
    finish.set()
    first.join(5)
    monkeypatch.setattr(R, "run_promote", lambda op: "again")
    assert R._promote_exclusive("op_y") == "again"
