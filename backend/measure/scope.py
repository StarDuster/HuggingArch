"""What an estimate's measured numbers are read under, and what identifies
an estimate's result (docs/measured_calibration.md §7.1, §7.5).

`MeasurementScope` — the owner whose own rows join the shared ones (or
none), the lake, the snapshot a lookup is pinned at, the snapshot the shared
rows last changed at, and the revision of the service's annotation rules;
`NONE` for an estimate that reads no measurements (no service configured,
or a mode that reads none: `theoretical_only`).

`ResultIdentity` — what an estimate's result is a function of:

  * `content`  — the request content that reaches the result: the request
                 body as the client sent it (the server-set scope excluded),
                 its precision profile by key, without the fields a sweep's
                 points set for themselves; for the gallery, its render
                 config. A sweep's theory half (`theory_of_request`) drops
                 the measurement-only fields too;
  * `version`  — the engine part: the model's `result_version`, or
                 `engine_version()` for what spans every model, or the spec
                 generation for a join of work in flight;
  * `scope`    — the `MeasurementScope` its measured track was read under,
                 or `NONE` where the result reads no measurements — theory
                 keys carry no scope element and stay byte-stable whatever
                 Kernel Bench holds;
  * `posted_rows` — whether the request carried rows of its own
                 (`calibration_rows`).

From it come the cache key (`key`, through `make_key`, the one key hash),
whether the value may enter a tier other users read (`remote_ok`: no owner,
no posted rows) and the scope's digest every sweep and estimate reports what
its measured track was read under with (`scope_digest`, the response's
`measurement_scope`; informational — nothing keys on it). Every cache of an
estimate's result takes its key and its remote eligibility from here and
builds none of its own: the offload single-flight
(`backend/api/offload.call_key`, where an estimate request stands for its
identity), the Scaling sweeps' memory / disk / remote tiers
(`backend/api/sweep_runtime`: the theory half and the SOL answer only — a
measured sweep is made live and never stored), the gallery's facts and decode
frontier (`backend/api/gallery_overview`, `backend.api.gallery_frontier`), and
the warm jobs (the same code). The browser's in-tab sweep cache
(`web/src/sweepCache.ts`) holds SOL answers only, keyed on their body; a
measured sweep or estimate is always asked of the server.
`tests/meta/test_result_identity.py` fails when one of them builds a key
another way.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, FrozenSet, Optional


def make_key(*parts: Any) -> str:
    """Canonical cache key from arbitrary JSON-able parts (sorted-key JSON →
    sha256), so callers stop re-implementing the same hashing dance."""
    raw = json.dumps(list(parts), sort_keys=True, default=str)
    return hashlib.sha256(raw.encode()).hexdigest()


@dataclass(frozen=True)
class MeasurementScope:
    """What an estimate's measured numbers are a function of (module
    docstring). `key` is the identity a cache stores a measured result
    under: the shared rows' snapshot for a shared estimate (a private upload
    elsewhere does not move it), the snapshot itself for an owner's."""
    owner: Optional[str]
    lake_id: str
    snapshot: Optional[int]
    shared_snapshot: Optional[int]
    rules_rev: str
    #: The identity of the lake's content at `snapshot` and at
    #: `shared_snapshot` (`Lake.content_id`: a hash of the snapshot and the
    #: catalog rows live there — uniquely named data and delete files, inlined
    #: rows by value; equal ids are equal row and table content).
    #: A lake restored or copied keeps its id and its history's snapshot
    #: ids, and once the copies diverge one id (even one commit time) names
    #: two different contents — the content identity tells them apart by
    #: construction, so a cache another machine fills (the shared remote
    #: tier) never serves one copy's numbers for the other's.
    snapshot_content: str = ""
    shared_snapshot_content: str = ""

    @property
    def reads(self) -> bool:
        """Whether an estimate under this scope looks measurements up."""
        return self.lake_id != "none"

    @property
    def remote_ok(self) -> bool:
        """Whether a result under this scope may enter a cache tier other
        users read: the shared rows alone."""
        return self.owner is None

    @property
    def key(self) -> str:
        """The cache identity (`"none"` for the constant scope)."""
        if not self.reads:
            return "none"
        if self.owner is None:
            return (f"{self.lake_id}@{self.shared_snapshot}:{self.shared_snapshot_content}"
                    f"/{self.rules_rev}")
        return (f"{self.owner}:{self.lake_id}@{self.snapshot}:{self.snapshot_content}"
                f"/{self.rules_rev}")

    @property
    def digest(self) -> str:
        """`key` hashed: what a response reports its measured track was read
        under (`measurement_scope`)."""
        return hashlib.sha256(self.key.encode()).hexdigest()[:16]

    def to_json(self) -> dict:
        return {"owner": self.owner, "lake_id": self.lake_id, "snapshot": self.snapshot,
                "shared_snapshot": self.shared_snapshot, "rules_rev": self.rules_rev,
                "snapshot_content": self.snapshot_content,
                "shared_snapshot_content": self.shared_snapshot_content}

    @classmethod
    def from_version(cls, version: dict, owner: Optional[str]) -> "MeasurementScope":
        """The scope of `GET /internal/bench/version`'s answer for `owner`."""
        return cls(owner=owner, lake_id=str(version["lake_id"]),
                   snapshot=int(version["snapshot"]),
                   shared_snapshot=int(version["shared_snapshot"]),
                   rules_rev=str(version["rules_rev"]),
                   snapshot_content=str(version["snapshot_content"]),
                   shared_snapshot_content=str(version["shared_snapshot_content"]))


#: No measurements: no service configured, or a mode that reads none.
NONE = MeasurementScope(owner=None, lake_id="none", snapshot=None,
                        shared_snapshot=None, rules_rev="")


@dataclass(frozen=True)
class ResultIdentity:
    """What an estimate's result is a function of (module docstring)."""
    kind: str
    version: Any
    content: Any
    scope: MeasurementScope
    #: Whether the result reads measurements (its mode is not theory).
    reads: bool
    posted_rows: bool = False
    #: A process-local artifact that must never enter a shared cache tier,
    #: even under the constant no-measurement scope (`sweep-step`).
    local_only: bool = False

    @property
    def key(self) -> str:
        scope = [self.scope.key] if self.reads else []
        return make_key(self.kind, self.version, self.content, *scope)

    @property
    def remote_ok(self) -> bool:
        """May the value enter a tier every user and machine reads: the
        shared rows alone, no rows of the request's own."""
        return not (self.posted_rows or self.local_only) and (
            not self.reads or self.scope.remote_ok)

    @property
    def scope_digest(self) -> str:
        """The digest of the scope the result was read under, reported in
        the response (`MeasurementScope.digest`; the constant scope's for
        theory). Informational: the browser caches theory only and keys on
        no digest."""
        return (self.scope if self.reads else NONE).digest

    @property
    def measurements(self) -> str:
        """What the measured track was read from, for the response:
        `kernel_bench`, or `none` (SOL, and it says so)."""
        return "kernel_bench" if self.reads and self.scope.reads else "none"


def request_content(req: Any, *, drop: FrozenSet[str] = frozenset()) -> dict:
    """An estimate request's content: its body (`measurement_scope` is not
    in it), the precision profile by its key — two spellings of one profile
    are one result — without `drop`."""
    body = req.model_dump(mode="json")
    body["precision"] = req.precision_profile().key
    return {k: v for k, v in body.items() if k not in drop}


def of_request(kind: str, req: Any, version: Any, *,
               drop: FrozenSet[str] = frozenset()) -> ResultIdentity:
    """The identity of an estimate request's result (the API's
    `InferenceEstimateRequest` and its sweeps; `req.scope()` the scope the
    server set)."""
    reads = req.reads_measurements()
    return ResultIdentity(kind=kind, version=version, content=request_content(req, drop=drop),
                          scope=req.scope() if reads else NONE, reads=reads,
                          posted_rows=bool(req.calibration_rows))


def measurement_fields(req: Any) -> FrozenSet[str]:
    """The request's fields marked measurement-only on its model
    (`json_schema_extra={"measurement_only": True}`): they decide how a
    measured track is read, never the theory."""
    return frozenset(name for name, f in type(req).model_fields.items()
                     if (f.json_schema_extra or {}).get("measurement_only"))


def theory_of_request(kind: str, req: Any, version: Any, *,
                      drop: FrozenSet[str] = frozenset()) -> ResultIdentity:
    """The identity of a request's theory half — what no measurement and no
    selection moves: its content without `drop` and without its
    measurement-only fields (`measurement_fields`), under no scope. One theory
    entry serves every selection mode and every Kernel Bench state; it may
    enter the shared tier (no owner's rows reach it)."""
    return ResultIdentity(kind=f"{kind}-theory", version=version,
                          content=request_content(req, drop=drop | measurement_fields(req)),
                          scope=NONE, reads=False)


def of_render(kind: str, version: Any, content: Any, scope: MeasurementScope) -> ResultIdentity:
    """The identity of a server-composed estimate (the gallery's: shared rows,
    `latest`, nothing posted)."""
    return ResultIdentity(kind=kind, version=version, content=content, scope=scope,
                          reads=scope.reads)


def of_sweep_step(theory_key: str, cell: int, index: int) -> ResultIdentity:
    """The identity of one locally kept sweep step.

    This is not an estimate result and never enters a remote tier: it names a
    partial progress entry under the sweep's theory identity, one cell and one
    step index, so cancellation and resume read exactly what that run computed.
    Giving the local artifact the identity-module key removes the one API-side
    hand-assembled cache key while keeping `MeasurementScope` out of it.
    """
    return ResultIdentity(kind="sweep-step", version=theory_key,
                          content=(cell, index), scope=NONE, reads=False,
                          local_only=True)


def of_walk(version: Any, *, model: str, layout: Any, card_class: str,
            precision: str, pp_stage: Any, anchor: Any) -> ResultIdentity:
    """The identity of a WALK — the expensive half of every number this
    site publishes, named by the facts it is a function of.

    A walk is the graph walked once for one deployment at one point, with
    every row carrying its law in the workload's counts and the card rather
    than a number. Every other point of that deployment — another context,
    another batch, another card of the same class, another K, another floor,
    another routing skew (`ep_load_factor`) — is read off it by substitution.
    So the walk is worth keying, and the
    key is the enforcement: a caller asks for the walk OF something rather
    than being handed a walk object to move onto a card itself, which is
    how one card's rows came twice to be published as another card's
    (`inference.card.Card.refuse_unless_one_class` names both).

    The six components, each a fact a walk's ROWS depend on:

      * `model`     — the model and the engine, as `fingerprint.
                      result_version` states it: code, shared data (the GPU
                      sheet included) and this model's footprint, the
                      framework-binding entries it can reach among them. So
                      a plan edit, a binding edit or a sheet edit lands on
                      a fresh key with no invalidation pass;
      * `layout`    — the `ParallelConfig`. A walk is layout-specific: the
                      per-rank projection is a rule ON the walk;
      * `card_class` — `inference.card.Card.klass`, by its `key()`. The
                      structural half of a card and the only half that may
                      be keyed: what a framework's dispatch reads of it and
                      which peak columns the sheet lists. Two cards it
                      answers alike walk alike, so they share this entry;
                      two it answers differently cannot, which makes the
                      cross-class read unrepresentable rather than merely
                      refused;
      * `precision` — `PrecisionProfile.key`. Deployment precision has one
                      entry (CLAUDE.md §2.4);
      * `pp_stage`  — the pipeline stage the card's HBM binds
                      (`capacity.CapacityCeilings.pp_stage`), or None
                      without pipelining. Class alone is too coarse here:
                      two cards of one class with different HBM can run out
                      on different stages, and then they are genuinely
                      different walks;
      * `anchor`    — the point the walk was TAKEN at, and the walk-time
                      inputs that change which rows exist at all: the
                      context, the chunk schedule, the verify width's class
                      (a width of 1 is the number, every wider one the
                      symbol), `host_offload` (it removes rows before
                      anything is priced), and the framework, its options
                      and the paged-KV page the rows are written against.

    NOT in the key, deliberately: the SKU, HBM, the HBM fraction, the two
    time floors, the batch, K and the acceptance rate beyond the anchor's
    width class, the routing skew `ep_load_factor`, the measurement scope and
    the selection mode. Each is
    either priced or substituted into an already-priced form
    (`row_roofline.FloorForm`, `CardForm`), so keying on it would split one
    walk into entries that hold identical rows.

    Under no scope and reading no measurements: a walk is theory, and the
    measured track is folded over its points afterwards. So one walk entry
    serves every selection mode and every Kernel Bench state, and it may
    enter the shared tier.
    """
    return ResultIdentity(
        kind="walk", version=version,
        content={"model": model, "layout": layout, "card_class": card_class,
                 "precision": precision, "pp_stage": pp_stage,
                 "anchor": anchor},
        scope=NONE, reads=False)
