"""The Kernel Bench page's views over the lake, computed per request.

`browse(lake, q)` answers what `GET /api/bench/browse` returns before the
route's display fields (`_as_row`, `_coverage`): the same JSON shape the
old Postgres `bench_browse` (migration 0022, dropped by 0026; in git history)
returned, without `pending`: nothing is ever in flight.

Views: envelope, rows, workload, curves, coverage; every view but curves
also returns `facets` (each filter's values under the other filters).
Scopes (`Query.source`, `run`, `upload`): the shared corpus, the caller's rows ("mine"), shared
∪ mine (a row counted once), or one of the caller's sets (run / upload).
Sorts and filters as in 0022; bounds kept as product limits (workload candidates 50,000; curves 50,000 envelopes per op /
4,000 points). The bounds 0022 had only because of how it read stored
aggregates — shared ∪ mine by gain or candidates up to 20,000 of the
caller's op shapes, rows sorted by text up to 50,000 — are gone: nothing is
merged from stored parts any more, every view reads the rows.

An envelope is one physical op — (signature, physical shape) at one
(dtype env, gpu) and phase: one kernel at one shape on one card, whichever
deployment (`parallel`, kept on the row) cut the shape — over the scope's
rows of annotated inputs: its winner is the fastest row (ties: the smaller id), its gain the
runner-up's latency over the winner's, less one; its candidates the 20
fastest; its kernels and models every distinct one among its rows. The
envelope's model, framework and roofline check are its winner's; the kernel
filter matches an envelope any of whose rows ran that kernel.

Query shape (docs/measured_calibration.md §5.4): project only the columns a
view reads; envelope groups by (phys, env, phase) with the per-group
aggregates the sort and the page need, then builds candidate lists, kernels
and models for the page's groups only; rows find the page's ids over the
narrow columns, then load those rows.
"""
from __future__ import annotations

import json
import math
import statistics
import uuid
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Any, Optional

from backend.arch.dims import call_token_axes, symbols as dim_symbols
from backend.inference.benchmarks import OUTCOME_VOCABULARY
from backend.inference.op_signatures import all_signatures
from backend.service.bench import store
from backend.service.bench.lake import Lake


@dataclass(frozen=True)
class Query:
    """One browse request, validated by the route."""
    view: str = "envelope"            # envelope | rows | workload | curves | coverage
    caller: Optional[str] = None       # the signed-in user, if any
    source: str = ""                   # "" (shared ∪ mine) | "shared" | "mine"
    run: str = ""                      # one of the caller's runs …
    upload: str = ""                   # … or uploads (the route resolves an admin's reads)
    set_owner: Optional[str] = None    # owner of run/upload (the caller, or anyone for an admin)
    filters: dict = field(default_factory=dict)   # model, signature, gpu, framework, dtype, phase, kernel, outcome, check, key, batch, context, call_tokens (`call_tokens()`)
    sort: str = ""
    desc: Optional[bool] = None        # None: the view's own order (workload descending, others ascending)
    page: int = 0
    page_size: int = 50


class Refused(ValueError):
    """A request beyond a view's bounds; the message names the bound (400)."""


def browse(lake: Lake, q: Query) -> dict:
    return _Browse(lake, q).answer()


# The joined row: a measurement (m), its batch (b) and its input's
# annotations (a: the in-memory `annot`, store.py — none for a placeholder).
_FROM = ("FROM lake.measurements m JOIN lake.batches b ON b.batch_id = m.batch_id "
         "LEFT JOIN annot a ON a.input_id = m.input_id")
# Rows of annotated inputs — the members of envelopes (a placeholder has none).
_PHYSICAL = "a.signature IS NOT NULL AND a.physical_shape IS NOT NULL"


@dataclass(frozen=True)
class _Field:
    """One quantity of the browse surface, declared once: its SQL over the
    joined row (aliases m, b, a), and the parts of the page that read it —
    a row's struct (`shown`), a facet the page filters by (`filter`, the
    page's name for it), a sort key (`sort`, the page's name; the rows view
    takes every sort key, `views` the other views that take it), or a
    roofline check (`check`). The page's names are not always the column's:
    the `kernel` filter is the catalog kernel, the `kernel` sort the
    implementation. An envelope aggregate has no row SQL (the envelope
    view computes it)."""
    name: str
    sql: Optional[str]
    shown: bool = False
    filter: Optional[str] = None
    sort: Optional[str] = None
    views: tuple = ()
    check: bool = False


def _positive(sql: str) -> str:
    return f"(CASE WHEN {sql} > 0 THEN {sql} END)"


_DTYPE = "json_extract_string(m.dtype_env_json, '$.activation')"   # see facets
#: The views that read a latency: only a measurement has one.
_LATENCY_VIEWS = ("envelope", "workload", "curves")
_WORKLOAD, _ENVELOPE = ("workload",), ("workload", "envelope")

# The surface. Filters are listed in facet order. A row's rates are shown
# only when positive; MFU / MBU and the check are the store's
# (`measured_utilization`'s rule, parity-tested there). The row is 0021b's
# `_bench_row_json`; `origin` is added per request.
_FIELDS = (
    _Field("model", "m.model", shown=True, filter="model", sort="model"),
    _Field("signature", "a.signature", shown=True, filter="signature", sort="op", views=_WORKLOAD),
    _Field("gpu", "m.gpu", shown=True, filter="gpu", sort="gpu"),
    _Field("framework", "m.framework", shown=True, filter="framework", sort="framework",
           views=_WORKLOAD),
    _Field("dtype", _DTYPE, filter="dtype", sort="dtype"),
    _Field("phase", "m.phase", shown=True, filter="phase", sort="phase"),
    _Field("catalog_kernel", "a.catalog_kernel", shown=True, filter="kernel"),
    # What happened (`benchmarks.OUTCOMES`). Filtered in `scope`, not with the
    # other filters: it decides which rows are in view at all.
    _Field("outcome", store.OUTCOME, shown=True, filter="outcome"),
    _Field("outcome_detail", "m.outcome_detail", shown=True),
    # Which clock timed it (`benchmarks.TIMING_METHODS`); NULL on a row timed
    # before the clock recorded itself, which the page marks.
    _Field("timing_method", "m.timing_method", shown=True),
    _Field("id", "m.id", shown=True),
    _Field("op", "m.op", shown=True),
    _Field("framework_version", "m.framework_version", shown=True, sort="version"),
    _Field("kernel_impl", "m.kernel_impl", shown=True, sort="kernel", views=_WORKLOAD),
    _Field("parallel", "m.parallel", shown=True),
    _Field("shape_json", "m.shape_json", shown=True),
    _Field("physical_shape", "a.physical_shape", shown=True, sort="shape", views=_WORKLOAD),
    _Field("dtype_env_json", "m.dtype_env_json", shown=True),
    _Field("node_spec_id", "m.node_spec_id", shown=True, sort="layer", views=_WORKLOAD),
    _Field("latency_us", "m.latency_us", shown=True, sort="latency", views=_ENVELOPE),
    _Field("gain", None, sort="gain", views=("envelope",)),
    _Field("n_candidates", None, sort="candidates", views=("envelope",)),
    _Field("achieved_tflops", "m.achieved_tflops", shown=True),
    _Field("achieved_bw_gbps", "m.achieved_bw_gbps", shown=True),
    _Field("peak_tflops_at_test", "m.peak_tflops_at_test", shown=True),
    _Field("peak_bw_at_test", "m.peak_bw_at_test", shown=True),
    _Field("samples", "m.samples", shown=True, sort="samples", views=_WORKLOAD),
    _Field("date", "m.date", shown=True, sort="date"),
    _Field("upload_id", "b.upload_id", shown=True),
    _Field("run_id", "b.run_id", shown=True),
    _Field("workload_batch", "a.workload_batch", shown=True),
    _Field("workload_context", "a.workload_context", shown=True),
    _Field("tflops", _positive("m.achieved_tflops"), sort="tflops", views=_ENVELOPE),
    _Field("gbps", _positive("m.achieved_bw_gbps"), sort="gbps", views=_ENVELOPE),
    _Field("mfu", _positive(store.MFU), sort="mfu", views=_ENVELOPE),
    _Field("mbu", _positive(store.MBU), sort="mbu", views=_ENVELOPE),
    _Field("ai", "(CASE WHEN m.achieved_tflops > 0 AND m.achieved_bw_gbps > 0"
                 " THEN 1000 * m.achieved_tflops / m.achieved_bw_gbps END)", sort="ai"),
    _Field("over", f"(NOT {store.WITHIN_PEAK})", check=True),
    _Field("ok", f"(({store.MFU} IS NOT NULL OR {store.MBU} IS NOT NULL) AND {store.WITHIN_PEAK})",
           check=True),
)


class _Surface:
    """The lookups the views read, each derived from `_FIELDS`."""

    def __init__(self, fields: tuple) -> None:
        self.sql = {f.name: f.sql for f in fields if f.sql is not None}
        self.row = {f.name: f.sql for f in fields if f.shown}
        self.dims = {f.filter: f.sql for f in fields if f.filter}
        self.checks = {f.name: f.sql for f in fields if f.check}
        sortable = [f for f in fields if f.sort]
        self.sorts = {
            "rows": {f.sort: f.sql for f in sortable if f.sql is not None},
            "workload": {f.sort: f.sql for f in sortable if "workload" in f.views},
            # the envelope view sorts its grouped CTE, by column name
            "envelope": {f.sort: f.name for f in sortable if "envelope" in f.views},
        }
        #: what an envelope's winner can carry beyond a row column
        self.derived = {f.name for f in fields if f.sql is not None and not f.shown}


_SURFACE = _Surface(_FIELDS)

_WORKLOAD_CANDIDATES = 50_000
_CURVE_ENVELOPES = 50_000
_CURVE_POINTS = 4_000
_CANDIDATES = 20


class _Sql:
    """A statement being built: text and its positional parameters, in
    order. Names come only from the whitelists above; values go in params."""

    def __init__(self) -> None:
        self.parts: list[str] = []
        self.params: list[Any] = []

    def add(self, text: str, *params: Any) -> "_Sql":
        self.parts.append(text)
        self.params.extend(params)
        return self

    def text(self) -> str:
        return " ".join(self.parts)


class _Browse:
    """One request: its scope, filters, page and the views over them."""

    def __init__(self, lake: Lake, q: Query) -> None:
        self.lake, self.q = lake, q
        self.view = q.view or "envelope"
        self.size = min(max(q.page_size, 1), 200)
        self.off = max(q.page, 0) * self.size
        self.f = {k: v for k, v in q.filters.items() if v is not None and v != ""}
        # Measurements unless the request names another outcome: every view
        # without the filter reads what it read before outcomes were more
        # than `measured` / `oom`, and the facets list the rest.
        self.f.setdefault("outcome", "measured")
        # `outcome=all` (rows view only, `check_request`): measurements and
        # failures together, so a batch's page can show what failed beside
        # what measured. Everywhere the filter would read it (scope, facets)
        # it is absent instead.
        self.all_outcomes = self.f["outcome"] == "all"
        if self.all_outcomes:
            del self.f["outcome"]
        self.sort = q.sort or "latency"
        in_set = bool(q.run or q.upload)
        # Whose rows are "mine" on the page: the set's owner when an admin
        # reads someone's set, else the caller.
        self.reader = (q.set_owner or q.caller) if in_set else q.caller
        self._scope: Optional[tuple[str, list]] = None

    # ---------------------------------------------------------------- scope

    def scope(self, s: _Sql, outcome: bool = True) -> _Sql:
        """The scope's row predicate over m and b — and the outcome filter.

        One outcome is in view at a time, decided here, once, for every view
        this class builds: measurements unless the request names another
        (`outcome`). A row recording an outcome has no latency, and left in
        a view that reads one it inflated the candidate count or became a
        winner the page drew as `0.0us` — so the views that read latencies
        (`_LATENCY_VIEWS`) refuse any other outcome, and the rows and
        coverage views show and count the one asked for. The rows view alone
        takes `all` (`self.all_outcomes`, `check_request`): measurements and
        failures together. `outcome=False` leaves it out, for the facets,
        which list each filter's values under the OTHER filters.
        """
        if self._scope is None:
            self._scope = self._resolve_scope()
        text, params = self._scope
        if outcome and not self.all_outcomes:
            if self.f["outcome"] == "measured":
                s.add(f"{store.IS_MEASUREMENT} AND")
            else:
                s.add(f"{store.OUTCOME} = ? AND", self.f["outcome"])
        return s.add("(").add(text, *params).add(")")

    def _resolve_scope(self) -> tuple[str, list]:
        q = self.q
        if q.run or q.upload:
            owner = q.set_owner or q.caller
            if owner is None:
                return "FALSE", []
            ref = (store.SetRef(owner, "run", q.run.strip()) if q.run
                   else store.SetRef(owner, "upload", q.upload))
            where, params = ref.where()
            return f"m.batch_id IN (SELECT batch_id FROM lake.batches WHERE {where})", params
        parts, params = [], []
        if q.source in ("", "shared"):
            parts.append(f"({store.SHARED_ROW})")
        if q.caller is not None and q.source in ("", "mine"):
            parts.append("b.owner = ?::UUID")
            params.append(q.caller)
        return (f"({' OR '.join(parts)})" if parts else "FALSE"), params

    def origin(self, s: _Sql) -> _Sql:
        if self.reader is None:
            return s.add("'shared'")
        return s.add("CASE WHEN b.owner = ?::UUID THEN 'mine' ELSE 'shared' END", self.reader)

    def row_struct(self, s: _Sql) -> _Sql:
        s.add("struct_pack(" + ", ".join(f"{k} := {v}" for k, v in _SURFACE.row.items())
              + ", origin := ")
        return self.origin(s).add(")")

    # -------------------------------------------------------------- filters

    def filters(self, s: _Sql, dims=tuple(_SURFACE.dims), check: bool = True) -> _Sql:
        """`AND`-ed row filters on `dims` (and the roofline check)."""
        for d in dims:
            if d in self.f and d != "outcome":          # `scope` filters it
                s.add(f"AND {_SURFACE.dims[d]} = ?", self.f[d])
        if check and self.check():
            s.add("AND " + _SURFACE.checks[self.check()])
        return s

    def check(self) -> Optional[str]:
        """The roofline check the request asks for, if it names one."""
        c = self.f.get("check")
        return c if c in _SURFACE.checks else None

    def one_envelope(self, s: _Sql) -> _Sql:
        """Narrow to the one envelope `key` names, as the envelope view
        returns it (the rows and curves views ask for its members)."""
        key = self.f.get("key")
        if isinstance(key, dict):
            s.add("AND a.signature = ? AND a.physical_shape = ? AND json(m.dtype_env_json) = json(?)"
                  " AND m.gpu = ? AND m.phase = ?",
                  key.get("signature"), key.get("physical_shape"),
                  json.dumps(key.get("dtype_env_json")), key.get("gpu"), key.get("phase"))
        return s

    def desc(self) -> bool:
        if self.q.desc is None:
            return self.view == "workload"
        return self.q.desc

    def order(self, key: str, tie: str) -> str:
        d = "DESC" if self.desc() else "ASC"
        return f"{key} {d} NULLS LAST, {tie} {d}"

    # --------------------------------------------------------------- answer

    def answer(self) -> dict:
        views = {"envelope": self.envelope, "rows": self.rows, "workload": self.workload,
                 "curves": self.curves, "coverage": self.coverage}
        if self.view not in views:
            raise Refused(f"unknown view {self.view}")
        self.check_request()
        out: dict = {}
        if self.view != "curves":
            out["facets"] = self.facets()
        out.update(views[self.view]())
        return out

    def check_request(self) -> None:
        """Refuse what the view cannot answer before reading anything."""
        sorts = _SURFACE.sorts
        if self.all_outcomes:
            if self.view != "rows":
                raise Refused("outcome=all lists measurements and failures together; "
                              "only the rows view shows both")
            outcome = "measured"   # the latency-view guard below reads it
        else:
            outcome = self.f["outcome"]
            if outcome not in OUTCOME_VOCABULARY:
                raise Refused(f"outcome is one of {', '.join(sorted(OUTCOME_VOCABULARY))}, "
                              f"not {outcome}")
        if self.view in _LATENCY_VIEWS and outcome != "measured":
            raise Refused(f"the {self.view} view reads latencies, and only a "
                          f"measurement has one; see outcome={outcome} in the rows view")
        if self.view == "envelope" and self.sort not in sorts["envelope"]:
            *head, last = sorts["envelope"]
            raise Refused(f"envelopes sort by {', '.join(head)} or {last}, not {self.sort}")
        if self.view == "rows" and self.sort not in sorts["rows"]:
            raise Refused(f"rows do not sort by {self.sort}")
        if self.view == "workload":
            if "model" not in self.f or "phase" not in self.f:
                raise Refused("pick a model and a phase to see its workloads")
            if self.sort not in sorts["workload"]:
                raise Refused(f"workload rows do not sort by {self.sort}")
        if self.view == "curves":
            tokens = self.f.get("call_tokens")
            if not isinstance(tokens, dict) or not isinstance(tokens.get("total"), (list, tuple)) \
                    or not isinstance(tokens.get("factor"), (list, tuple)) \
                    or not isinstance(tokens.get("runs"), dict):
                raise Refused("the curves view needs call_tokens {total, factor, runs}")

    def read(self, s: _Sql) -> list[dict]:
        return self.lake.read(s.text(), s.params)

    # --------------------------------------------------------------- facets

    def facets(self) -> dict:
        """Each filter's values over the scope under every other filter.

        The scope's rows are grouped first, by the filter columns (dtype by
        its whole env JSON, read once per group) and the roofline check as a
        flag: a check in the WHERE made DuckDB group 2M filtered rows ~8×
        slower than grouping them all (2.2M-row scale copy: 1.1 s vs 0.14 s)."""
        check = self.check()
        dims = _SURFACE.dims
        s = _Sql().add("WITH g AS (SELECT " + ", ".join(
            "m.dtype_env_json AS dtype_env_json" if d == "dtype" else f"{e} AS {d}"
            for d, e in dims.items()))
        s.add(f", bool_or({_SURFACE.checks[check]}) AS passes" if check else ", true AS passes")
        s.add(_FROM + " WHERE")
        self.scope(s, outcome=False)
        s.add("GROUP BY " + ", ".join(str(n + 1) for n in range(len(dims))) + ")")
        s.add(", d AS (SELECT * EXCLUDE (dtype_env_json),"
              " json_extract_string(dtype_env_json, '$.activation') AS dtype FROM g WHERE passes)")
        cols = []
        for d in dims:
            col = _Sql().add(f"list_sort(list(DISTINCT {d}) FILTER (WHERE {d} IS NOT NULL")
            for o in dims:
                if o != d and o in self.f:
                    col.add(f"AND {o} = ?", self.f[o])
            col.add(f")) AS {d}")
            cols.append(col)
        s.add("SELECT")
        for n, col in enumerate(cols):
            s.add(("," if n else "") + col.text(), *col.params)
        s.add("FROM d")
        row = self.read(s)[0]
        return {d: _plain(row[d] or []) for d in dims}

    # ------------------------------------------------------------- envelope

    def envelopes(self, s: _Sql, winner: tuple[str, ...], extra_where: str = "",
                  listed: bool = False) -> _Sql:
        """CTEs `g` (one row per envelope in scope: count, winner struct
        `win`, two fastest latencies) and `e` (filtered; winner id, latency,
        gain and the `winner` fields — row columns or display rates — the
        caller reads). Filters on key columns (op, gpu, phase, dtype — equal
        on every member) select rows before grouping; a `key` selects the
        one envelope it names there too. The model, framework
        and check filters are the winner's, the kernel filter any member's.
        The winner struct carries only what is read: every field rides
        through `min_by` for every row in scope.

        With `listed`, `r` (materialized) also keeps the annotated rows
        without a physical shape and flags `listed`, a row passing every
        filter — the ops the curves view lists, from the same pass."""
        kernel = self.f.get("kernel")
        check = self.check()
        names = ("id", "latency_us", *(d for d in ("model", "framework") if d in self.f), *winner)
        exprs = {c: _SURFACE.sql[c] for c in dict.fromkeys(names)}
        if check:
            exprs["passes_check"] = _SURFACE.checks[check]
        s.add("WITH r AS MATERIALIZED (" if listed else "WITH r AS (")
        s.add("SELECT a.phys, m.env, m.phase,")
        s.add(", ".join(f"{e} AS {c}" for c, e in exprs.items()))
        if kernel:
            s.add(", a.catalog_kernel AS member_kernel")
        if listed:
            s.add(f", a.signature AS listed_signature, {_PHYSICAL} AS physical, (TRUE")
            self.filters(s, dims=("model", "framework", "kernel"))
            s.add(f") AS listed {_FROM} WHERE a.signature IS NOT NULL AND")
        else:
            s.add(f"{_FROM} WHERE {_PHYSICAL} AND")
        self.scope(s)
        self.filters(s, dims=("signature", "gpu", "phase", "dtype"), check=False)
        self.one_envelope(s)
        s.add("), g AS (SELECT phys, env, phase, count(*) AS n_candidates,")
        s.add("min_by(struct_pack(" + ", ".join(exprs) + "), (latency_us, id)) AS win,")
        s.add("min_by(latency_us, latency_us, 2) AS fastest")  # the two smallest latencies
        if kernel:
            s.add(", bool_or(member_kernel = ?) AS has_kernel", kernel)
        s.add(("FROM r WHERE physical" if listed else "FROM r") + " GROUP BY phys, env, phase)")
        s.add(", e AS MATERIALIZED (SELECT phys, env, phase, n_candidates, win, win.id AS winner_id,"
              " win.latency_us AS latency_us, fastest[2] AS second_latency_us,"
              " fastest[2] / NULLIF(win.latency_us, 0) - 1 AS gain"
              + "".join(f", win.{c} AS {c}" for c in winner) + " FROM g WHERE TRUE")
        if kernel:
            s.add("AND has_kernel")
        for d in ("model", "framework"):
            if d in self.f:
                s.add(f"AND win.{d} = ?", self.f[d])
        if check:
            s.add("AND win.passes_check")
        return s.add(extra_where + ")")

    def envelope(self) -> dict:
        sort_col = _SURFACE.sorts["envelope"][self.sort]
        order = self.order(sort_col, "winner_id")
        s = self.envelopes(_Sql(), winner=(sort_col,) if sort_col in _SURFACE.derived else ())
        s.add(f", p AS MATERIALIZED (SELECT * FROM e ORDER BY {order} LIMIT ? OFFSET ?)", self.size, self.off)
        # The page's members, read once: the winner's row, candidate lists,
        # kernels, models — by the page's inputs first (`pa`, materialized:
        # otherwise DuckDB joins the measurements to the page on env alone,
        # 40 s at 2.2M rows).
        s.add(", pa AS MATERIALIZED (SELECT p.winner_id, p.env, p.phase, a.* EXCLUDE (phys)"
              " FROM p JOIN annot a ON a.phys = p.phys)")
        s.add(", c AS (SELECT a.winner_id, min_by(")
        self.row_struct(s)
        s.add(", (m.latency_us, m.id)) AS winner,"
              " min_by(struct_pack(id := m.id, latency_us := m.latency_us, framework := m.framework,"
              " kernel_impl := m.kernel_impl, catalog_kernel := a.catalog_kernel, model := m.model),"
              f" (m.latency_us, m.id), {_CANDIDATES}) AS candidates,"
              " list_sort(list(DISTINCT a.catalog_kernel) FILTER (WHERE a.catalog_kernel IS NOT NULL)) AS kernels,"
              " list_sort(list(DISTINCT m.model) FILTER (WHERE m.model IS NOT NULL)) AS models"
              " FROM pa a JOIN lake.measurements m ON m.input_id = a.input_id AND m.env = a.env"
              " AND m.phase = a.phase"
              f" JOIN lake.batches b ON b.batch_id = m.batch_id WHERE {_PHYSICAL} AND")
        self.scope(s)
        s.add("GROUP BY a.winner_id)")
        s.add(f"SELECT t.total, x.* FROM (SELECT count(*) AS total FROM e) t LEFT JOIN ("
              f" SELECT p.{sort_col} AS sort_key, p.winner_id, p.n_candidates, p.gain,"
              " c.winner, c.candidates, c.kernels, c.models FROM p JOIN c ON c.winner_id = p.winner_id"
              f") x ON true ORDER BY {self.order('x.sort_key', 'x.winner_id')}")
        got = self.read(s)
        total = int(got[0]["total"])
        items = []
        for r in got:
            if r["winner_id"] is None:
                continue
            w = _row(r["winner"])
            items.append({
                "winner": w, "n_candidates": int(r["n_candidates"]), "gain": _plain(r["gain"]),
                "candidates": _plain(r["candidates"]), "models": list(r["models"] or []),
                "kernels": list(r["kernels"] or []),
                "key": {k: w[k] for k in ("signature", "physical_shape", "dtype_env_json",
                                          "gpu", "phase")}})
        return {"total": total, "items": items}

    # ----------------------------------------------------------------- rows

    def page_of_rows(self, s: _Sql, source: str, totals: str) -> list[dict]:
        """Finish a statement whose CTE `source` has (id, batch_id,
        input_id, k): `totals` over it, then the page's rows in order."""
        s.add(f", p AS (SELECT * FROM {source} ORDER BY {self.order('k', 'id')} LIMIT ? OFFSET ?)",
              self.size, self.off)
        s.add(f"SELECT t.*, x.* FROM (SELECT {totals} FROM {source}) t LEFT JOIN (SELECT p.k AS sort_key, p.id AS page_id,")
        self.row_struct(s)
        s.add("AS row FROM p JOIN lake.measurements m ON m.id = p.id AND m.batch_id = p.batch_id"
              " AND m.input_id = p.input_id"
              " JOIN lake.batches b ON b.batch_id = m.batch_id LEFT JOIN annot a ON a.input_id = m.input_id"
              f") x ON true ORDER BY {self.order('x.sort_key', 'x.page_id')}")
        return self.read(s)

    def rows(self) -> dict:
        """One page of the scope's rows under the filters (the envelope
        view's scope and filters), or of one envelope's (`key` as the
        envelope view returns it — its rows from every deployment)."""
        s = _Sql().add(f"WITH f AS (SELECT m.id, m.batch_id, m.input_id, {_SURFACE.sorts['rows'][self.sort]} AS k"
                       f" {_FROM} WHERE")
        self.scope(s)
        self.one_envelope(s)
        self.filters(s)
        s.add(")")
        got = self.page_of_rows(s, "f", "count(*) AS total")
        total = int(got[0]["total"])
        return {"total": total,
                "items": [_row(r["row"]) for r in got if r["page_id"] is not None]}

    # ------------------------------------------------------------- workload

    def workload(self) -> dict:
        model, phase = self.f["model"], self.f["phase"]
        batch, context = self.f.get("batch"), self.f.get("context")
        s = _Sql().add("SELECT list_sort(list(DISTINCT a.workload_batch)"
                       " FILTER (WHERE a.workload_batch IS NOT NULL)) AS batches,"
                       " count(*) FILTER (WHERE a.workload_batch IS NULL) AS unplaced,"
                       " list(DISTINCT a.workload_context) FILTER (WHERE a.workload_batch = ?) AS contexts,"
                       " count(*) FILTER (WHERE a.workload_batch = ? AND a.workload_context IN (?, 0)) AS candidates",
                       batch, batch, context)
        s.add(f"{_FROM} WHERE m.model = ? AND m.phase = ? AND", model, phase)
        self.scope(s)
        st = self.read(s)[0]
        out: dict = {"batches": list(st["batches"] or []), "unplaced": int(st["unplaced"])}
        if batch is None:
            return {**out, "contexts": []}
        # Real contexts ascending, then 0 (shapes without a context dimension).
        out["contexts"] = sorted(st["contexts"] or [], key=lambda c: (c is None, c == 0, c or 0))
        if context is None:
            return out
        out["candidates"] = n = int(st["candidates"])
        if n > _WORKLOAD_CANDIDATES:
            raise Refused(f"this workload has {n} measurements; the view reads up to 50,000 — pick a source")
        # The candidates — context c and context 0 (they belong to every
        # context) — and per (layer, op) the fastest of them.
        s = _Sql().add("WITH c AS (SELECT m.id, m.batch_id, m.input_id, m.node_spec_id, m.op, m.latency_us,"
                       f" {_SURFACE.sorts['workload'][self.sort]} AS k {_FROM}"
                       " WHERE m.model = ? AND m.phase = ? AND a.workload_batch = ?"
                       " AND a.workload_context IN (?, 0) AND", model, phase, batch, context)
        self.scope(s)
        self.filters(s, dims=[d for d in _SURFACE.dims if d not in ("model", "phase")])
        s.add("), w AS (SELECT min_by(struct_pack(id, batch_id, input_id, latency_us, k), (latency_us, id)) AS best"
              " FROM c GROUP BY node_spec_id, op),"
              " pick AS (SELECT best.id AS id, best.batch_id AS batch_id, best.input_id AS input_id,"
              " best.latency_us AS latency_us, best.k AS k FROM w)")
        got = self.page_of_rows(s, "pick", "count(*) AS total, COALESCE(sum(latency_us), 0) AS total_latency_us")
        return {**out, "total": int(got[0]["total"]),
                "total_latency_us": _plain(got[0]["total_latency_us"]),
                "items": [_row(r["row"]) for r in got if r["page_id"] is not None]}

    # --------------------------------------------------------------- curves

    def curves(self) -> dict:
        """Per op, its envelopes along the tokens one kernel call runs over,
        as `call_tokens()` derives them (passed by the route as {total,
        runs}): a shape with one `total` axis runs along it, the factors it
        also has moving with it (a grouped expert GEMM's routed `n_tokens`,
        its per-expert `s_expert_rows`); otherwise along the first run of
        factors
        its signature's contract lays out that the shape has whole
        (`call_tokens()`): batch × context where the contract has a context
        axis (attention along `B×s_kv`, one curve per `s_query`; not a
        sparse attention, whose keys are capped by its top-k), the product
        where the contract folds factors into one axis (`B×s_query`, in
        decode the batch), else one factor with the others kept in the
        curve. A signature with no contract is unplaced. An op
        shape with those axes taken out — a GEMM's (n, k) — at one gpu and
        dtype is one curve, across phases (each point says its phase). A
        point needs an MFU or an MBU: a memory-only op records no FLOP rate.
        The longest curves first, each whole, up to 4,000 points per op (the
        first curve always); envelopes with no token axis are `unplaced`; an
        op with more than 50,000 envelopes in view answers `too_many`."""
        tokens = self.f["call_tokens"]
        # One pass over the scope's rows (`listed`): the envelopes and the ops
        # listed come from the same materialized rows, narrow; the winners'
        # text by one join on their ids. (Measured both ways on the live lake
        # and the 2.2M-row copy: text carried through the grouping is slower
        # at scale, a separate pass for the op list costs a scan of every file.)
        s = self.envelopes(_Sql(), winner=("tflops", "gbps", "mfu", "mbu"),
                           extra_where=" AND (mfu IS NOT NULL OR mbu IS NOT NULL)", listed=True)
        s.add(", w AS (SELECT e.phase, e.tflops, e.gbps, e.mfu, e.mbu, "
              + ", ".join(f"{_SURFACE.row[c]} AS {c}" for c in (
                  "signature", "physical_shape", "gpu", "dtype_env_json", "framework",
                  "model", "latency_us"))
              + ", coalesce(a.catalog_kernel, m.kernel_impl) AS kernel"
              " FROM e JOIN lake.measurements m ON m.id = e.winner_id"
              " JOIN annot a ON a.input_id = m.input_id)")
        s.add(", s AS (SELECT *, count(*) OVER (PARTITION BY signature) AS n_op FROM w)"
              " SELECT false AS op, * FROM s"
              f" QUALIFY n_op <= {_CURVE_ENVELOPES} OR row_number() OVER (PARTITION BY signature) = 1"
              " UNION ALL BY NAME SELECT DISTINCT true AS op, listed_signature AS signature"
              " FROM r WHERE listed")
        signatures: list = []
        per_op: dict = {}
        for r in self.read(s):
            if r["op"]:
                signatures.append(r["signature"])
            else:
                per_op.setdefault(r["signature"], []).append(r)
        signatures.sort()
        place = _Placement(tokens["runs"], list(tokens["total"]),
                           list(tokens["factor"]))
        return {"ops": [place.curves_of(sig, per_op.get(sig, [])) for sig in signatures]}

    # ------------------------------------------------------------- coverage

    def coverage(self) -> dict:
        """Rows per catalog kernel under the gpu / dtype / phase / check
        filters, and the rows no catalog kernel claims, per framework — of
        the outcome in view (measurements unless one is named) — and
        `outcomes`, the rows of every outcome under the same filters, so the
        page can say how many of a kernel's attempts failed and how."""
        check = self.check()
        # The check counted per group rather than filtered in the WHERE (see
        # `facets`: same answer, several times faster at scale).
        n = f"count(*) FILTER (WHERE {_SURFACE.checks[check]})" if check else "count(*)"
        s = _Sql().add(f"SELECT a.catalog_kernel AS kernel, m.framework AS framework,"
                       f" {store.OUTCOME} AS outcome, {n} AS n {_FROM} WHERE")
        self.scope(s, outcome=False)
        self.filters(s, dims=("gpu", "dtype", "phase"), check=False)
        s.add(f"GROUP BY ALL HAVING {n} > 0")
        kernels: dict = {}
        unclassified: dict = {}
        outcomes: dict = {}
        for r in self.read(s):
            outcomes[r["outcome"]] = outcomes.get(r["outcome"], 0) + int(r["n"])
            if r["outcome"] != self.f["outcome"]:
                continue
            if r["kernel"] is not None:
                kernels[r["kernel"]] = kernels.get(r["kernel"], 0) + int(r["n"])
            else:
                f = r["framework"] or ""
                unclassified[f] = unclassified.get(f, 0) + int(r["n"])
        return {"kernels": kernels, "unclassified": unclassified, "outcomes": outcomes}


class _Placement:
    """Envelopes onto curves. A shape's token axes, x and the rest of it
    are read once per distinct shape text, and points are built only for
    the curves the answer keeps."""

    def __init__(self, runs: dict, totals: list[str], factors: list[str]) -> None:
        self.runs = runs
        self.totals = totals
        self.factors = factors
        self.shapes: dict = {}
        self.dtypes: dict = {}

    def place(self, signature: str, physical_shape: str):
        """(axis, x, rest, rest's key, x's factors) of a shape, or None:
        unplaced — no token axis, more than one total, or no contract for
        the signature."""
        key = (signature, physical_shape)
        if key not in self.shapes:
            shape = json.loads(physical_shape)

            def has(a: str) -> bool:
                v = shape.get(a)
                return isinstance(v, int) and not isinstance(v, bool) and v > 0
            runs = [list(r) for r in self.runs.get(signature, ())]
            totals = [a for a in self.totals if has(a)]
            if signature not in self.runs:
                # No contract: no declaration says what its axes are.
                x_axes = taken = []
            elif totals:
                # A total is the count itself; every factor present moves with
                # it — the declared factors (dims.yaml `call_tokens: factor`),
                # not only the ones the contract's operands lay out: a
                # combine writes its token side, so its B and s_query are on
                # no operand and still grow with the routed rows.
                x_axes = totals if len(totals) == 1 else []
                taken = totals + [a for a in self.factors if has(a)] + [
                    a for r in runs for a in r if has(a)]
            else:
                x_axes = next((r for r in runs if all(has(a) for a in r)), [])
                taken = x_axes
            if not x_axes:
                self.shapes[key] = None
            else:
                rest = {k: v for k, v in shape.items() if k not in taken}
                self.shapes[key] = ("×".join(x_axes), math.prod(shape[a] for a in x_axes),
                                    rest, json.dumps(rest, sort_keys=True),
                                    tuple(shape[a] for a in x_axes))
        return self.shapes[key]

    def dtype(self, text: str) -> Any:
        if text not in self.dtypes:
            self.dtypes[text] = json.loads(text)
        return self.dtypes[text]

    def curves_of(self, signature: str, envelopes: list[dict]) -> dict:
        if envelopes and envelopes[0]["n_op"] > _CURVE_ENVELOPES:
            return {"signature": signature, "too_many": True}
        members: dict = {}
        unplaced = 0
        for e in envelopes:
            at = self.place(signature, e["physical_shape"])
            if at is None:
                unplaced += 1
                continue
            key = (e["gpu"], e["dtype_env_json"], at[0], at[3])
            members.setdefault(key, []).append((at[1], at[4], e))
        kept, upto = [], 0
        for key in sorted(members, key=lambda k: (-len(members[k]), k)):
            upto += len(members[key])
            if upto > _CURVE_POINTS and kept:
                break
            gpu, dtype, axis, _ = key
            placed = sorted(members[key], key=lambda m: (m[0], m[2]["physical_shape"], m[2]["phase"],
                                                         m[2]["latency_us"]))
            pts = [{"x": x, "shape": e["physical_shape"], "phase": e["phase"], "framework": e["framework"],
                    "kernel": e["kernel"], "model": e["model"], "latency_us": e["latency_us"],
                    "tflops": e["tflops"], "gbps": e["gbps"], "mfu": e["mfu"], "mbu": e["mbu"]}
                   for x, _, e in placed]
            kept.append({"gpu": gpu,
                         "dtype_env_json": dict(self.dtype(dtype)), "axis": axis,
                         "shape": dict(self.place(signature, members[key][0][2]["physical_shape"])[2]),
                         "saturation": _saturation(pts), "points": pts,
                         "lines": _lines([(p["phase"], f) for p, (_, f, _) in zip(pts, placed, strict=True)])})
        return {"signature": signature, "points": sum(len(m) for m in members.values()),
                "curves_total": len(members), "unplaced": unplaced, "curves": kept}


def _lines(points: list[tuple]) -> list[list[int]]:
    """The lines through a curve's points (each a (phase, x's factors) in
    x order), as point indices: one factor of x moving, the phase and every
    other factor held. A product x is a grid of launches — `B=4 s_query=1`
    and `B=1 s_query=4` share x and are different calls — so a line through
    every point would join launches of different series; a point on no line
    is a line of its own."""
    lines: list[list[int]] = []
    width = len(points[0][1]) if points else 0
    for moving in range(width):
        groups: dict = {}
        for i, (phase, f) in enumerate(points):
            groups.setdefault((phase, f[:moving] + f[moving + 1:]), []).append(i)
        lines.extend(g for g in groups.values() if len(g) > 1)
    on = {i for g in lines for i in g}
    return lines + [[i] for i in range(len(points)) if i not in on]


def _saturation(points: list[dict]) -> dict:
    """Where a curve turns compute-bound: the token count at which its
    arithmetic intensity reaches the card's ridge (peak FLOP/s over peak
    bytes/s). A point's intensity over the ridge is its MFU over its MBU —
    both divide the same work by the same latency, against the same card's
    two peaks — so the crossing is where MFU/MBU reaches 1, interpolated on
    log tokens between the last point below and the first at or above (the
    median ratio of the points at one x). `crossed` with `tokens`;
    `compute` — at or above from the first point (`tokens`: that point's);
    `memory` — below at every point (`tokens`: the last point's);
    `no_flops` — no point has both ratios (an op that records no FLOP rate)."""
    by_x: dict = {}
    for p in points:
        f, m = p["mfu"], p["mbu"]
        if f is not None and m is not None and f > 0 and m > 0:
            by_x.setdefault(p["x"], []).append(math.log(float(f) / float(m)))
    if not by_x:
        return {"state": "no_flops"}
    xs = sorted(by_x)
    r = [statistics.median(by_x[x]) for x in xs]
    if r[0] >= 0:
        return {"state": "compute", "tokens": xs[0]}
    for x0, r0, x1, r1 in zip(xs[:-1], r[:-1], xs[1:], r[1:], strict=True):
        if r0 < 0 <= r1:
            t = -r0 / (r1 - r0)
            return {"state": "crossed",
                    "tokens": math.exp(math.log(x0) + t * (math.log(x1) - math.log(x0)))}
    return {"state": "memory", "tokens": xs[-1]}


def call_tokens() -> dict:
    """What the curves view places shapes by (`Query.filters["call_tokens"]`),
    from the declarations: dims.yaml's `total` and `factor` token axes, and per signature
    the runs its contract lays the `factor` axes out in
    (`OpSignature.token_runs`) — folded rows multiply, separate axes do not,
    and a `context` axis runs with the batch (dims.yaml `kind: batch`)."""
    roles = call_token_axes()
    batch = [n for n, sym in sorted(dim_symbols().items()) if sym.kind == "batch"]
    return {"total": list(roles["total"]),
            "factor": list(roles["factor"]),
            "runs": {s.name: [list(r) for r in s.token_runs(roles["factor"], roles["context"], batch)]
                     for s in all_signatures()}}


def _plain(v: Any) -> Any:
    """A lake value as JSON: ids as text, decimals as numbers."""
    if isinstance(v, uuid.UUID):
        return str(v)
    if isinstance(v, Decimal):
        return float(v)
    if isinstance(v, dict):
        return {k: _plain(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_plain(x) for x in v]
    return v


def _row(r: dict) -> dict:
    out = _plain(r)
    for k in store.JSON_COLUMNS:
        if isinstance(out.get(k), str):
            out[k] = json.loads(out[k])
    return out
