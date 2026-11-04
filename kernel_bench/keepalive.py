"""Hold the card while the queue has nothing to measure.

The cluster reclaims a GPU left idle for ten minutes, so a box between
models — or a box whose queue has crashed — loses the card it is holding
for the corpus. This runs a short matmul burst once the card has been
idle long enough, and yields the moment real work appears.

Separate from the queue on purpose: the queue dying is exactly when the
card needs covering, and a keepalive inside it would die with it.

It is not a benchmark and nothing it does is recorded. The only thing it
must never do is hold the card against real work, which is why it yields
on any compute process it did not start and runs at a duty cycle rather
than flat out.
"""
from __future__ import annotations

import subprocess
import time
from typing import Callable, FrozenSet, Optional

#: Seconds of idleness before the card is at risk. The cluster's limit is
#: ten minutes; half of it leaves a poll and a whole burst of room.
IDLE_LIMIT = 300.0
BURST = 90.0
#: Fraction of the burst the card is actually computing. Enough to count
#: as "in use", little enough to leave the machine responsive.
DUTY = 0.6


def compute_pids(run: Optional[Callable[..., object]] = None) -> FrozenSet[str]:
    """PIDs with a compute context on the card, per `nvidia-smi`.

    Host-namespace PIDs: in a container our own process cannot be matched
    by `os.getpid()`, so a caller tells its own burst apart by snapshot
    rather than by identity.
    """
    runner = run or subprocess.run
    try:
        out = runner(["nvidia-smi", "--query-compute-apps=pid",
                      "--format=csv,noheader"],
                     capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.SubprocessError):
        return frozenset()
    return frozenset(p.strip() for p in (out.stdout or "").split() if p.strip())


def should_burst(idle_seconds: float, busy: bool, *,
                 idle_limit: float = IDLE_LIMIT) -> bool:
    """Whether the card needs holding right now.

    A pure decision so the policy is testable without a GPU: the burst
    itself is the only part that needs one.
    """
    return not busy and idle_seconds >= idle_limit


def burst(seconds: float, *, duty: float = DUTY,
          yield_to: FrozenSet[str] = frozenset()) -> None:
    """Keep the card busy for `seconds`, yielding to anyone who arrives."""
    import torch

    from .gpulock import gpu_uuid, held

    uuid = gpu_uuid(0)
    a = torch.randn(4096, 4096, device="cuda", dtype=torch.bfloat16)
    b = torch.randn(4096, 4096, device="cuda", dtype=torch.bfloat16)
    torch.cuda.synchronize()
    # After init, so our own context is in the snapshot a caller yields by.
    baseline = yield_to or compute_pids()
    end = time.time() + seconds
    checked = 0.0
    while time.time() < end:
        started = time.time()
        for _ in range(50):
            a @ b
        torch.cuda.synchronize()
        computing = time.time() - started
        time.sleep(computing * (1 - duty) / duty)
        if time.time() - checked > 3:
            checked = time.time()
            # A run takes the card's lock BEFORE it creates a context and
            # waits for us to leave (`gpulock.FOREIGN_GRACE_S`), so the lock
            # is the signal; a new compute pid covers anything that
            # measures without it.
            if (uuid and held(uuid)) or compute_pids() - baseline:
                return


def hold(seconds: float, *, duty: float = DUTY,
         target: Callable[..., None] = burst) -> None:
    """Run one burst in a child process and wait for it.

    The child is the point: a CUDA context outlives the tensors that made
    it, so a burst run in this process left the keepalive holding the card
    (712 MiB) between bursts. The sweep refuses a card another process has a
    context on, so every model after the first burst was refused and the
    queue emptied itself with the card idle — the opposite of the job. A
    child's context ends with the child."""
    import multiprocessing

    child = multiprocessing.get_context("spawn").Process(
        target=target, args=(seconds,), kwargs={"duty": duty})
    child.start()
    child.join(seconds + 120)
    if child.is_alive():
        child.terminate()
        child.join()


#: How often the longest unheld stretch so far is written down, whether or
#: not a burst was needed. Without it the only record of this watch is the
#: bursts it DID run, which cannot answer the question anyone actually asks
#: of it — how long was the card unheld — and a 5-minute sampler reading the
#: card from outside cannot answer it either: it sees instants, so a run of
#: "nothing on the card" samples is equally a 25-minute hole and a series of
#: 3-minute gaps the sampler happened to land in. That ambiguity was read as
#: a hole in the invariant (2026-10-23) and it was not one.
REPORT_EVERY = 600.0


def serve(*, poll: float = 60.0, idle_limit: float = IDLE_LIMIT,
          burst_seconds: float = BURST, duty: float = DUTY,
          report_every: float = REPORT_EVERY,
          log: Optional[Callable[[str], None]] = None) -> int:
    """Watch the card and hold it when it goes quiet. Runs until killed.

    Every line it writes is stamped, and it writes one per `report_every`
    even when nothing happened: the invariant this process exists for is
    "the card is held by a measurement or by us", and a log that only
    records bursts says nothing about the stretches where neither was
    true. What it reports is the LONGEST unheld stretch of the window,
    which is the number the cluster's ten-minute reclaim is measured
    against — a mean would hide exactly the run that loses the card.
    """
    say = log or (lambda line: print(
        f"{time.strftime('%Y-%m-%d %H:%M:%S')} {line}", flush=True))
    idle = 0.0
    worst = 0.0
    bursts = 0
    reported = time.time()
    while True:
        pids = compute_pids()
        idle = 0.0 if pids else idle + poll
        worst = max(worst, idle)
        if should_burst(idle, bool(pids), idle_limit=idle_limit):
            say(f"idle {idle:.0f}s -> keepalive burst {burst_seconds:.0f}s")
            hold(burst_seconds, duty=duty)
            bursts += 1
            idle = 0.0
        if time.time() - reported >= report_every:
            say(f"window {(time.time() - reported) / 60:.0f}m: longest "
                f"unheld {worst:.0f}s, {bursts} burst(s)"
                + (" — the card is held by a measurement" if not worst else ""))
            reported, worst, bursts = time.time(), idle, 0
        time.sleep(poll)
