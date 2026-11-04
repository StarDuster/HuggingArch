"""A chained op writer must not consume a second agent-CLI slot."""
from __future__ import annotations

import threading

from backend._agent_runner.locker import InMemoryLocker


def test_nested_acquire_skips_semaphore():
    sem = threading.BoundedSemaphore(1)
    locker = InMemoryLocker(sem)
    assert locker.acquire("spec") == "ok"
    nested = InMemoryLocker(sem)
    assert nested.acquire("op", consume_slot=False) == "ok"
    # The one slot is still held by spec; a third consumer is overloaded.
    other = InMemoryLocker(sem)
    assert other.acquire("other") == "overloaded"
    nested.release("op")
    locker.release("spec")
    assert other.acquire("other") == "ok"
    other.release("other")


def test_nested_release_does_not_over_release_semaphore():
    sem = threading.BoundedSemaphore(1)
    locker = InMemoryLocker(sem)
    locker.acquire("spec")
    nested = InMemoryLocker(sem)
    nested.acquire("op", consume_slot=False)
    nested.release("op")
    nested.release("op")  # idempotent
    # spec still holds the only slot
    assert InMemoryLocker(sem).acquire("x") == "overloaded"
    locker.release("spec")
