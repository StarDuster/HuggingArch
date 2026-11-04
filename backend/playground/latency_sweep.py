"""Module bench, prefill / decode time across context lengths.

The bench prices each entry's two phases at ONE length (the contract's
`input_tokens`). This sweeps that length: for every compare entry and phase,
the same per-op roofline the bench's Summary row reads (`_timing_view`), at
each requested length, at the bench's batch and GPU. Nothing is re-derived —
a point is the bench's timing at that length, trimmed to its totals.

Prefill at length L is a prompt of L tokens per request; decode at L is one
new token per request against L cached positions — each point carries the
`phase_contract` binding it ran under, so the page never restates the rule.

The work is cut per (entry, phase) — `LatencySweep.cells()` — so the server
spreads one sweep across its compute workers (`map_off_loop`); `cell` is the
top-level function those workers run.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from backend.arch.eval.transforms import DECODE, PREFILL, isolate_resolved
from backend.arch.loader import get_default_loader
from backend.inference.pipeline import NoPeakForDtype
from backend.playground.module_bench import (
    CONTRACT_ERRORS,
    EntryRefused,
    _match_model,
    _timing_view,
    inject_component,
    phase_contract,
    resolve_entry,
)

PHASES = (PREFILL, DECODE)


@dataclass(frozen=True)
class LatencySweep:
    """One sweep request: the bench's entries and shared contract (batch,
    dims, precision, GPU), and the lengths to time each phase at."""

    entries: Tuple[Dict[str, Any], ...]
    batch: int
    lens: Tuple[int, ...]
    gpu_sku: str
    precision: Any
    base_model_id: Optional[str] = None
    dim_overrides: Tuple[Tuple[str, Any], ...] = ()

    def cells(self) -> List[Tuple["LatencySweep", int, str]]:
        return [(self, i, phase)
                for i in range(len(self.entries)) for phase in PHASES]

    def assemble(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """The cells' answers as one row (or error) per entry, in order."""
        by_entry: Dict[int, List[Dict[str, Any]]] = {}
        for (_, i, _phase), res in zip(self.cells(), results, strict=True):
            by_entry.setdefault(i, []).append(res)
        rows: List[Dict[str, Any]] = []
        errors: List[Dict[str, Any]] = []
        for i in range(len(self.entries)):
            parts = by_entry[i]
            failed = next((p["error"] for p in parts if "error" in p), None)
            if failed is not None:
                errors.append(failed)
                continue
            rows.append({**{k: parts[0][k] for k in ("entry", "component", "model_id")},
                         "phases": {p["phase"]: p["points"] for p in parts}})
        return {
            "contract": {"batch": self.batch, "gpu_sku": self.gpu_sku,
                         "lens": list(self.lens),
                         "unit": "seconds · per layer (one call of the module) · whole batch"},
            "results": rows,
            "errors": errors,
        }


def cell(item: Tuple[LatencySweep, int, str]) -> Dict[str, Any]:
    """One entry's one phase, timed at every length of the sweep."""
    sweep, i, phase = item
    entry = sweep.entries[i]
    loader = get_default_loader()
    host_id = (sweep.base_model_id or "").strip() or None
    base = _match_model(host_id, loader) if host_id else None
    try:
        ready = resolve_entry(entry, base=base, host_id=host_id, loader=loader)
    except EntryRefused as exc:
        return {"error": exc.error}

    def refused(message: str) -> Dict[str, Any]:
        return {"error": {"component": ready.name, "entry": ready.key,
                          "model_id": ready.model_id, "error": message,
                          "missing_dims": []}}

    if ready.site.encoder is not None:
        return refused("an encoder call has no prefill or decode to time")
    host = base if base is not None else ready.owner
    dims = {**dict(sweep.dim_overrides), **ready.own}
    try:
        injected = inject_component(host, ready.name, loader)
        isolated = isolate_resolved(injected, ready.name, dims, ready.site)
        points = []
        for n in sweep.lens:
            timing = _timing_view(
                isolated, ready.name, batch=sweep.batch, input_tokens=n,
                site=ready.site, precision=sweep.precision,
                phase=phase, gpu_sku=sweep.gpu_sku)
            points.append({"len": n, **phase_contract(sweep.batch, n)[phase],
                           # the timing's totals; the per-op rows stay the bench's
                           **timing["totals"]})
    except NoPeakForDtype as exc:
        return refused(str(exc))
    except CONTRACT_ERRORS as exc:
        return refused(str(exc))
    return {"entry": ready.key, "component": ready.name,
            "model_id": ready.model_id, "phase": phase, "points": points}
