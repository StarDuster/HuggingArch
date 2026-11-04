"""The neutral identity shared by every SOL walk producer and consumer.

The pool stores walk facts, not a page payload. This module is the one place
that names those facts: a walk belongs to a model/engine version, a layout, a
card class, a precision profile, and the workload anchor its rows were taken
at. Gallery, Scaling, and Compare may project such a record; none of them owns
its spelling.
"""
from __future__ import annotations

from typing import Any, Mapping

from backend.arch.eval.precision import AS_SHIPPED
from backend.inference.card import Card
from backend.inference.defaults import GPU_MEMORY_FRACTION_DEFAULT
from backend.inference.sol_pool import defaultPool, recordId
from backend.inference.sweep_grid import (
    GALLERY_OUTPUT_TOKENS,
    ROUTINE_SEQ_LENS,
)
from backend.measure.scope import ResultIdentity, of_walk


#: The schema of a canonical law record and of the bundle manifest that lists
#: it — one name for both. v2: a record names the card it was walked on
#: (`walked_on`), and a bundle is one walk class's (`servingWalkIdentity`).
walkRecordSchema = "serving-law-v2"
servingContexts = tuple(ROUTINE_SEQ_LENS)
defaultPrefillChunkSize = 8192

# The capacity row is not part of the public decode ladder. It binds both
# phases at each card's own ceiling, which is what a deployment-scale sweep's
# default point reads.
capacityBatchPoint = {
    "key": "capacity",
    "prefill_batch_size": -1,
    "decode_batch_size": None,
}


def producerWalkFacts() -> dict[str, Any]:
    """The workload facts the canonical serving walk is taken at.

    Floors, selection mode, measurement scope, and a card's own HBM are
    deliberately absent: they are read when a record is projected, not when it
    is walked.
    """
    return {
        "outputTokens": GALLERY_OUTPUT_TOKENS,
        "gpuMemoryFraction": GPU_MEMORY_FRACTION_DEFAULT,
        "precision": AS_SHIPPED.key,
        "prefixCachedTokens": 0,
        "verifyWidth": 1,
        "acceptanceRate": 1.0,
        "speculativeDraft": None,
        "includeAuxWeights": False,
        "hostOffload": True,
        "framework": "",
        "frameworkOptions": {},
        "kvBlockSize": None,
        "prefillChunkSize": defaultPrefillChunkSize,
        "expertLoadFactor": 1.0,
    }


def walkFacts(environment: Mapping[str, Any]) -> dict[str, Any]:
    """The same facts, read from an estimate environment or a layout payload.

    Point-owned fields (card, GPU count, layout, context, and batch) and every
    measurement-only field are intentionally not named here.
    """
    precision = environment.get("precision")
    return {
        "outputTokens": int(environment["avg_output_tokens"]),
        "gpuMemoryFraction": float(environment["gpu_memory_fraction"]),
        "precision": getattr(precision, "key", precision),
        "prefixCachedTokens": int(environment.get("s_prefix_cached") or 0),
        "verifyWidth": int(environment.get("s_verify") or 1),
        "acceptanceRate": float(environment.get("acceptance_rate") or 1.0),
        "speculativeDraft": environment.get("speculative_draft"),
        "includeAuxWeights": bool(environment.get("include_aux_weights", False)),
        "hostOffload": bool(environment.get("host_offload", True)),
        "framework": environment.get("framework") or "",
        "frameworkOptions": dict(environment.get("framework_options") or {}),
        "kvBlockSize": environment.get("kv_block_size"),
        "prefillChunkSize": environment.get("prefill_chunk_size"),
        "expertLoadFactor": float(environment.get("ep_load_factor") or 1.0),
    }


def servingWalkIdentity(version: str, *, model: str, precision: str,
                        card: Card) -> ResultIdentity:
    """The identity of one model's canonical serving-walk bundle for the walk
    class of `card` (`Card.klass`).

    A law is over card SYMBOLS — peaks, bandwidth, fabric — but its terms are
    the rows of one walk, and which rows exist is a fact of the card's class:
    a kernel a framework gates on compute capability, the format a routed
    activation is cast to (`Card.refuse_unless_one_class`). So the class keys
    the bundle: cards of one class share it, and a card of another class
    names a different bundle — reading another class's law is not
    representable, rather than refused after it is read. The bundle's
    capacity facts are its own class's cards'."""
    return of_walk(
        version,
        model=model,
        layout=None,
        card_class=card.klass.key(),
        precision=precision,
        pp_stage=None,
        anchor={
            "kind": "serving-law",
            "schema": walkRecordSchema,
            **producerWalkFacts(),
        })


def layoutRecordName(layout: Any) -> str:
    """The one record name for a layout, however the caller holds it."""
    encoded = layout.to_dict() if hasattr(layout, "to_dict") else layout
    return recordId("layout", encoded)


def walkManifestIsValid(manifest: Mapping[str, Any] | None) -> bool:
    """Whether a manifest really is the current canonical walk record."""
    if manifest is None:
        return False
    metadata = manifest.get("metadata") or {}
    return (
        metadata.get("recordSchema") == walkRecordSchema
        and metadata.get("facts") == producerWalkFacts()
        and bool(metadata.get("layoutsByCard"))
    )


# One process reads and writes one pool. Distinct wrappers around the same
# directory would still be correct on disk, but they would split the hot record
# cache and make the data layer harder to reason about.
sharedSolPool = defaultPool()
