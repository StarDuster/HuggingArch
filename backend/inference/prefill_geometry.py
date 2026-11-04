"""Backend-owned geometry for scheduler prefill chunks.

This is the one implementation of the interval, tail and launch-batch policy.
Kernel Bench imports it for campaign planning; runtime estimates import it
directly.  The endpoint is any immutable point-like object with the four
``PlanPoint`` workload counts, so the backend module depends on neither a
model snapshot nor Kernel Bench.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Protocol, Tuple


@dataclass(frozen=True)
class PrefillEndpointGeometry:
    """Backend endpoint counts for callers that do not own ``PlanPoint``."""

    seq_len: int
    s_prefix_cached: int
    s_verify: int = 1
    batch: int = 1


class PrefillEndpoint(Protocol):
    """The endpoint fields the decomposition and fold provenance read."""

    seq_len: int
    s_prefix_cached: int
    s_verify: int
    batch: int


#: A deployment/campaign knob, not a framework physical constant.
DEFAULT_PREFILL_CHUNK_SIZE = 8192


def normalize_prefill_chunk_size(value: Optional[int]) -> Optional[int]:
    """Normalize ``0``/``None`` to unchunked and validate the campaign knob."""
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(
            "prefill_chunk_size must be a positive integer, 0, or None; "
            f"got {value!r}")
    if value < 0:
        raise ValueError(
            "prefill_chunk_size must be a positive integer, 0, or None; "
            f"got {value!r}")
    return value or None


@dataclass(frozen=True)
class PrefillChunkPoint:
    """One scheduler launch inside a prefill endpoint.

    ``endpoint`` retains the user-visible workload.  ``point`` is the launch
    at ``Lq=end`` and ``Sq=end-start``.  A multi-chunk endpoint launches
    batch 1; endpoint repetition is applied by the estimator's fold.
    """

    endpoint: PrefillEndpoint
    index: int
    start: int
    end: int
    launch_batch: int

    def __post_init__(self) -> None:
        if self.index < 0 or self.launch_batch < 1:
            raise ValueError(
                "a chunk point needs index >= 0 and launch_batch >= 1 "
                f"(index={self.index!r}, launch_batch={self.launch_batch!r})")
        if (self.start < self.endpoint.s_prefix_cached
                or self.end <= self.start
                or self.end > self.endpoint.seq_len):
            raise ValueError(
                "chunk bounds must satisfy endpoint.s_prefix_cached <= "
                "start < end <= endpoint.seq_len "
                f"(start={self.start!r}, end={self.end!r}, "
                f"endpoint={self.endpoint!r})")

    @property
    def width(self) -> int:
        return self.end - self.start

    @property
    def point(self) -> PrefillEndpoint:
        return type(self.endpoint)(
            self.end, self.start, 1, self.launch_batch)


@dataclass(frozen=True)
class PrefillChunkSchedule:
    """One endpoint's materialized scheduler launches.

    Runtime estimates first ask this question at batch one (the batch that
    discovers the endpoint's admitted batch).  ``with_endpoint_batch`` only
    changes endpoint repetition; it deliberately does not run the decomposition
    again, so ceilings, SOL and the measured fold continue to read one interval
    list.  The serialized map is derived from that list, never decomposed a
    second time.
    """

    endpoint: PrefillEndpointGeometry
    chunks: Tuple[PrefillChunkPoint, ...]
    prefill_chunk_size: Optional[int]
    endpoint_batch: int = 1

    @classmethod
    def materialize(
        cls, endpoint: PrefillEndpointGeometry, *,
        prefill_chunk_size: Optional[int],
    ) -> "PrefillChunkSchedule":
        chunks = decompose_prefill(
            endpoint, prefill_chunk_size=prefill_chunk_size)
        return cls(
            endpoint=endpoint,
            chunks=chunks,
            prefill_chunk_size=prefill_chunk_size,
            endpoint_batch=endpoint.batch,
        )

    def __post_init__(self) -> None:
        if self.endpoint_batch < 1:
            raise ValueError(
                "endpoint_batch must be positive "
                f"(got {self.endpoint_batch!r})")
        if not self.chunks:
            raise ValueError("a prefill schedule needs at least one launch")

    @property
    def launch_width(self) -> int:
        """The widest per-forward query, which is the activation footprint."""
        return max(chunk.width for chunk in self.chunks)

    def with_endpoint_batch(self, endpoint_batch: int) -> "PrefillChunkSchedule":
        """Read the same intervals as this many endpoint repetitions."""
        endpoint_batch = int(endpoint_batch)
        if endpoint_batch == self.endpoint_batch:
            return self
        endpoint = PrefillEndpointGeometry(
            self.endpoint.seq_len,
            self.endpoint.s_prefix_cached,
            self.endpoint.s_verify,
            endpoint_batch,
        )
        chunks = tuple(
            PrefillChunkPoint(
                endpoint, chunk.index, chunk.start, chunk.end,
                chunk.launch_batch)
            for chunk in self.chunks)
        return PrefillChunkSchedule(
            endpoint=endpoint,
            chunks=chunks,
            prefill_chunk_size=self.prefill_chunk_size,
            endpoint_batch=endpoint_batch,
        )

    def chunk_map(self, *, deployment: Optional[str] = None,
                  endpoint_index: int = 0) -> Optional[Dict[str, Any]]:
        """Serialize this materialized list in the campaign map schema."""
        if len(self.chunks) <= 1:
            return None
        return {
            **({"deployment": deployment} if deployment is not None else {}),
            "phase": "prefill",
            "endpoint_index": endpoint_index,
            "endpoint": {
                "seq_len": self.endpoint.seq_len,
                "s_prefix_cached": self.endpoint.s_prefix_cached,
                "s_verify": self.endpoint.s_verify,
                "batch": self.endpoint_batch,
            },
            "chunks": [{
                "index": chunk.index,
                "start": chunk.start,
                "end": chunk.end,
                "width": chunk.width,
                "launch_batch": chunk.launch_batch,
            } for chunk in self.chunks],
        }


def decompose_prefill(
    endpoint: PrefillEndpoint,
    *,
    prefill_chunk_size: Optional[int],
) -> Tuple[PrefillChunkPoint, ...]:
    """Decompose one endpoint using the single interval/tail/batch policy.

    A positive width emits full chunks and a nonzero tail.  ``0``/``None``
    and a query no wider than the chunk emit the endpoint unchanged.
    """
    chunk_size = normalize_prefill_chunk_size(prefill_chunk_size)
    total = endpoint.seq_len
    prefix = endpoint.s_prefix_cached
    if isinstance(total, bool) or not isinstance(total, int):
        raise TypeError(f"endpoint.seq_len must be an integer, got {total!r}")
    if isinstance(prefix, bool) or not isinstance(prefix, int):
        raise TypeError(
            f"endpoint.s_prefix_cached must be an integer, got {prefix!r}")
    if prefix < 0 or prefix >= total:
        raise ValueError(
            "a prefill endpoint needs 0 <= s_prefix_cached < seq_len "
            f"(seq_len={total!r}, s_prefix_cached={prefix!r})")
    if endpoint.batch < 1:
        raise ValueError(
            f"endpoint.batch must be positive, got {endpoint.batch!r}")

    query = total - prefix
    launch_batch = (
        endpoint.batch
        if chunk_size is None or query <= chunk_size else 1
    )
    if chunk_size is None or query <= chunk_size:
        return (PrefillChunkPoint(
            endpoint, 0, prefix, total, endpoint.batch),)

    full = query // chunk_size
    out = [PrefillChunkPoint(
        endpoint=endpoint,
        index=index,
        start=prefix + index * chunk_size,
        end=prefix + (index + 1) * chunk_size,
        launch_batch=launch_batch,
    ) for index in range(full)]
    remainder = query - full * chunk_size
    if remainder:
        out.append(PrefillChunkPoint(
            endpoint=endpoint,
            index=full,
            start=prefix + full * chunk_size,
            end=total,
            launch_batch=launch_batch,
        ))
    return tuple(out)


def prefill_chunk_map(
        endpoint: PrefillEndpoint,
        *,
        prefill_chunk_size: Optional[int],
        deployment: Optional[str] = None,
        endpoint_index: int = 0,
) -> Optional[Dict[str, Any]]:
    """Serialize the explicit endpoint-to-launch map used by plan and fold.

    ``None`` is the shared one-chunk policy: those points keep the existing
    endpoint measure.  Multi-chunk records are identical in campaign stats
    and runtime estimate provenance.
    """
    chunks = decompose_prefill(endpoint, prefill_chunk_size=prefill_chunk_size)
    if len(chunks) <= 1:
        return None
    return {
        **({"deployment": deployment} if deployment is not None else {}),
        "phase": "prefill",
        "endpoint_index": endpoint_index,
        "endpoint": {
            "seq_len": endpoint.seq_len,
            "s_prefix_cached": endpoint.s_prefix_cached,
            "s_verify": endpoint.s_verify,
            "batch": endpoint.batch,
        },
        "chunks": [{
            "index": chunk.index,
            "start": chunk.start,
            "end": chunk.end,
            "width": chunk.width,
            "launch_batch": chunk.launch_batch,
        } for chunk in chunks],
    }
