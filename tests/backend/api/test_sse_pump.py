"""The op routes stream a blocking agent generator through one thread pump.

Route-level cases drive `/propose-op`'s and `/review-op`'s stream generators
with a fake worker and pin the exact frames the client sees; the pump cases
pin the heartbeat and pass-through contract on their own.
"""
from __future__ import annotations

import asyncio
import json
import time


def _frame(d: dict) -> str:
    return f"data: {json.dumps(d)}\n\n"


async def _collect(agen) -> list:
    return [x async for x in agen]


class TestPump:
    def test_passes_items_through_in_order(self):
        from backend.api import _sse
        out = asyncio.run(_collect(_sse.pump(iter(["a", "b", "c"]))))
        assert out == ["a", "b", "c"]

    def test_a_slow_pull_gets_heartbeats(self):
        from backend.api import _sse

        def slow():
            time.sleep(0.25)
            yield "x"

        out = asyncio.run(_collect(_sse.pump(slow(), heartbeat_s=0.05)))
        assert out[-1] == "x"
        assert out.count(": heartbeat\n\n") >= 2
        assert set(out[:-1]) == {": heartbeat\n\n"}

    def test_payload(self):
        from backend.api import _sse
        assert _sse.payload(_frame({"type": "done"})) == {"type": "done"}
        assert _sse.payload(": heartbeat\n\n") is None
        assert _sse.payload("data: {not json\n\n") is None

    def test_headers(self):
        from backend.api import _sse
        assert _sse.headers() == {"Cache-Control": "no-cache", "X-Accel-Buffering": "no"}


class TestProposeStream:
    def _run(self, monkeypatch, frames, *, chain_review):
        from backend.api.routes import op_proposals as opp
        released = []
        monkeypatch.setattr(opp.op_proposal_worker, "stream_agent",
                            lambda op, prep: iter(frames))
        monkeypatch.setattr(opp.op_proposal_worker, "release", released.append)
        out = asyncio.run(_collect(
            opp.stream_with_prep("op", dict, chain_review=chain_review)))
        return out, released

    def test_frames_pass_through_and_the_lock_is_released(self, monkeypatch):
        frames = [_frame({"type": "log", "text": "hi"}),
                  _frame({"type": "done", "success": True})]
        out, released = self._run(monkeypatch, frames, chain_review=False)
        assert out == [_frame({"type": "log", "text": "Starting op proposal..."}), *frames]
        assert released == ["op"]

    def test_a_failed_proposal_ends_the_chained_stream(self, monkeypatch):
        failed = _frame({"type": "done", "success": False})
        out, released = self._run(
            monkeypatch, [failed, _frame({"type": "log", "text": "after"})],
            chain_review=True)
        assert out == [_frame({"type": "log", "text": "Starting op proposal..."}), failed]
        assert released == ["op"]


class TestReviewStream:
    def test_frames_come_with_their_payload(self, monkeypatch):
        from backend.api.routes import op_reviews as opr
        frames = [_frame({"type": "log", "text": "r"}), "not a frame"]
        monkeypatch.setattr(opr.op_review_worker, "prepare_review_prompt", lambda op: {})
        monkeypatch.setattr(opr.op_review_worker, "stream_agent",
                            lambda op, prep: iter(frames))
        out = asyncio.run(_collect(opr._stream_review_agent("op")))
        assert out == [(frames[0], {"type": "log", "text": "r"}), ("not a frame", None)]
