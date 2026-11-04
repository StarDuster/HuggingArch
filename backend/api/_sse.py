"""Streaming a blocking agent generator to an SSE client.

The op-proposal and op-review workers yield SSE frames from a synchronous
generator (a subprocess read loop). A route streams them without blocking the
event loop by pulling each frame in a worker thread, and keeps the connection
alive with a comment frame whenever a pull takes longer than the heartbeat
interval — proxies close an idle stream.
"""
from __future__ import annotations

import asyncio
import json
from typing import AsyncIterator, Iterator, Optional

HEARTBEAT = ": heartbeat\n\n"

_STOP = object()


def headers() -> dict:
    """Response headers for an SSE stream: no caching, no proxy buffering."""
    return {"Cache-Control": "no-cache", "X-Accel-Buffering": "no"}


def payload(chunk: str) -> Optional[dict]:
    """The JSON dict of a `data: ...\\n\\n` frame. Workers emit only
    single-line `data:` frames; a heartbeat or anything malformed is None."""
    if not chunk.startswith("data: "):
        return None
    try:
        return json.loads(chunk[6:].strip())
    except json.JSONDecodeError:
        return None


def _next_or_stop(it: Iterator):
    try:
        return next(it)
    except StopIteration:
        return _STOP


async def pump(frames: Iterator, *, heartbeat_s: float = 15) -> AsyncIterator:
    """Yield each item of the blocking iterator `frames`, pulled in a thread,
    with a `HEARTBEAT` frame for every `heartbeat_s` a pull is still waiting.
    A pull still in flight when the consumer stops is cancelled."""
    while True:
        next_task = asyncio.create_task(asyncio.to_thread(_next_or_stop, frames))
        try:
            while True:
                try:
                    chunk = await asyncio.wait_for(asyncio.shield(next_task),
                                                   timeout=heartbeat_s)
                    break
                except asyncio.TimeoutError:
                    yield HEARTBEAT
        finally:
            if not next_task.done():
                next_task.cancel()
        if chunk is _STOP:
            return
        yield chunk
