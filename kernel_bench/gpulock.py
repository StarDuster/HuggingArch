"""Exclusive lock on the GPU a measurement run is about to use.

Measured latency IS the product. Two processes sharing a device slow each
other down, and nothing in the emitted row records that it happened — so a
contended sweep is indistinguishable from a clean one after the fact and the
whole batch has to be thrown away. This has cost real corpus: five sweeps once
ran concurrently on one H100 for 22 hours, and 16k landed rows had to be
quarantined.

`--skip-existing` cannot prevent it: that asks what the CORPUS already holds,
not what another process is measuring right now.

The lock is per PHYSICAL DEVICE, keyed on the CUDA UUID rather than the device
index — `CUDA_VISIBLE_DEVICES` remaps indices, so index 0 in two shells can be
two different cards, or the same one. A multi-GPU box therefore runs one sweep
per card concurrently, which is correct and wanted; only a second sweep on the
SAME card is refused.

Default is fail-fast, not wait: the caller is a long GPU sweep, and silently
queueing for hours behind another run is rarely what anyone meant. The refusal
names the holder (pid + command) so the operator can decide.
"""
from __future__ import annotations

import errno
import os
import subprocess
import time
from dataclasses import dataclass
from typing import List, Optional, Tuple


class GpuBusy(RuntimeError):
    """Raised when another process already holds this GPU's lock."""


@dataclass(frozen=True)
class ComputeProcess:
    """One process already resident on the physical GPU."""

    pid: int
    command: str
    used_memory_mib: Optional[int]


def _query_compute_apps(uuid: str) -> Tuple[List[ComputeProcess], Optional[str]]:
    """Return compute processes already resident on ``uuid``.

    The warning is non-empty when the box cannot provide a trustworthy
    ``nvidia-smi`` answer. The flock remains useful in that case, but the
    caller must not mistake it for proof that an older uncoordinated process
    is absent.
    """
    cmd = [
        "nvidia-smi", f"--id={uuid}",
        "--query-compute-apps=pid,used_gpu_memory,process_name",
        "--format=csv,noheader,nounits",
    ]
    try:
        proc = subprocess.run(
            cmd, capture_output=True, text=True, timeout=5, check=False)
    except FileNotFoundError:
        return [], "nvidia-smi not found; external GPU occupancy was not checked"
    except (OSError, subprocess.TimeoutExpired) as exc:
        return [], f"nvidia-smi occupancy probe failed: {exc}"
    if proc.returncode != 0:
        detail = (proc.stderr or proc.stdout).strip().replace("\n", " ")
        return [], (f"nvidia-smi occupancy probe exited {proc.returncode}"
                    + (f": {detail}" if detail else ""))

    rows: List[ComputeProcess] = []
    for line in proc.stdout.splitlines():
        if not line.strip():
            continue
        fields = [field.strip() for field in line.split(",", 2)]
        if len(fields) != 3:
            return [], f"unparseable nvidia-smi compute-apps row: {line!r}"
        try:
            pid = int(fields[0])
        except ValueError:
            return [], f"unparseable nvidia-smi compute-apps pid: {fields[0]!r}"
        try:
            used_memory_mib = int(fields[1])
        except ValueError:
            used_memory_mib = None
        rows.append(ComputeProcess(
            pid=pid, command=fields[2], used_memory_mib=used_memory_mib))
    return rows, None


def gpu_uuid(device: int = 0) -> Optional[str]:
    """Stable identity of the physical device CUDA index `device` maps to.

    None when there is no CUDA device (CPU-only env — nothing to lock)."""
    try:
        import torch
        if not torch.cuda.is_available():
            return None
        uid = getattr(torch.cuda.get_device_properties(device), "uuid", None)
        if not uid:
            return None
        value = str(uid)
        return value if value.startswith("GPU-") else f"GPU-{value}"
    except Exception:
        return None


def _lock_dir() -> str:
    # /var/lock is the conventional home and is tmpfs on modern systems (so a
    # stale file cannot outlive a reboot); fall back to /tmp when it is not
    # writable, e.g. an unprivileged container.
    for d in ("/var/lock", "/tmp"):
        if os.path.isdir(d) and os.access(d, os.W_OK):
            return d
    return "/tmp"


def lock_path_for(uuid: str) -> str:
    return os.path.join(_lock_dir(), f"kernel_bench.gpu.{uuid}.lock")


#: How long a run that holds the lock waits for a process already on the
#: card to leave before refusing. The keepalive's burst is such a process:
#: it starts in the gaps between a sweep's envs (minutes of `uv` resolving,
#: no CUDA context), and it yields to a held lock (`held`) within seconds.
#: Without the wait, every env that started inside a burst was refused.
FOREIGN_GRACE_S = 30.0


def held(uuid: str) -> bool:
    """Whether a run holds `uuid`'s lock right now — probed without taking
    it for longer than the probe."""
    import fcntl
    try:
        fh = open(lock_path_for(uuid), "a+", encoding="utf-8")  # noqa: SIM115
    except OSError:
        return False
    with fh:
        try:
            fcntl.flock(fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            return True
        fcntl.flock(fh.fileno(), fcntl.LOCK_UN)
        return False


def _holder_of(path: str) -> str:
    """Best-effort description of the process holding `path`."""
    try:
        with open(path, encoding="utf-8") as f:
            info = f.read().strip()
        return info or "(holder wrote no info)"
    except OSError:
        return "(holder unknown)"


class GpuLock:
    """flock-based exclusive lock on one physical GPU.

    A no-op when there is no CUDA device, so CPU-only test runs and dry runs
    are unaffected. `flock` is released by the kernel when the process exits —
    including SIGKILL — so a crashed sweep never leaves the card locked.
    """

    def __init__(self, device: int = 0, *, wait: bool = False,
                 enabled: bool = True):
        self._wait = wait
        self._enabled = enabled
        self._uuid = gpu_uuid(device) if enabled else None
        self._path = lock_path_for(self._uuid) if self._uuid else None
        self._fh = None
        self._preflight_warning = None

    @property
    def path(self) -> Optional[str]:
        return self._path

    @property
    def active(self) -> bool:
        return self._fh is not None

    @property
    def preflight_warning(self) -> Optional[str]:
        return self._preflight_warning

    def acquire(self) -> "GpuLock":
        if not self._path:
            return self                       # no CUDA device → nothing to lock
        import fcntl
        # The flock lives as long as this handle: release() closes it.
        fh = open(self._path, "a+", encoding="utf-8")  # noqa: SIM115
        flags = fcntl.LOCK_EX if self._wait else (fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            fcntl.flock(fh.fileno(), flags)
        except OSError as e:
            if e.errno in (errno.EACCES, errno.EAGAIN):
                holder = _holder_of(self._path)
                fh.close()
                raise GpuBusy(
                    f"GPU {self._uuid} is already being measured by: {holder}\n"
                    f"  A second run on the same card makes BOTH sets of "
                    f"latencies wrong, and nothing in the emitted rows would "
                    f"record that it happened.\n"
                    f"  Wait for it, pick another card "
                    f"(CUDA_VISIBLE_DEVICES=…), or pass --wait-for-gpu to "
                    f"queue behind it."
                ) from e
            fh.close()
            raise
        # Holding the lock is the signal a yielding process (the keepalive)
        # watches for, so wait out its exit before calling the card busy.
        waited = 0.0
        while True:
            processes, warning = _query_compute_apps(self._uuid)
            foreign = [proc for proc in processes if proc.pid != os.getpid()]
            if not foreign or waited >= FOREIGN_GRACE_S:
                break
            time.sleep(2.0)
            waited += 2.0
        self._preflight_warning = warning
        if foreign:
            fcntl.flock(fh.fileno(), fcntl.LOCK_UN)
            fh.close()
            details = "; ".join(
                f"pid={proc.pid} cmd={proc.command}"
                + (f" memory={proc.used_memory_mib} MiB"
                   if proc.used_memory_mib is not None else "")
                for proc in foreign)
            raise GpuBusy(
                f"GPU {self._uuid} already has compute process(es): {details}\n"
                f"  They did not acquire kernel_bench's lock, but they can "
                f"still contaminate every measured latency. Stop them or "
                f"select another card before running the sweep."
            )
        # Record who holds it, for the next process's error message.
        fh.seek(0)
        fh.truncate()
        try:
            with open("/proc/self/cmdline", "rb") as c:
                cmd = c.read().replace(b"\0", b" ").decode(errors="replace").strip()
        except OSError:
            cmd = "?"
        fh.write(f"pid={os.getpid()} cmd={cmd}\n")
        fh.flush()
        self._fh = fh
        return self

    def release(self) -> None:
        if self._fh is None:
            return
        import fcntl
        try:
            self._fh.seek(0)
            self._fh.truncate()
            self._fh.flush()
            fcntl.flock(self._fh.fileno(), fcntl.LOCK_UN)
        except OSError:
            pass
        finally:
            self._fh.close()
            self._fh = None

    def __enter__(self) -> "GpuLock":
        return self.acquire()

    def __exit__(self, *exc) -> None:
        self.release()
