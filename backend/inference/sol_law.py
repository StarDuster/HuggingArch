"""Canonical SOL laws: one walked formula, every serving point substituted.

This layer is deliberately smaller than an estimate: it carries a phase's
SOL account and time as SymPy laws over workload and card symbols, with the
common subexpressions named once. Per-op rows, module tables, shapes and
display formulas belong to an explicit breakdown request and are not stored
here.
"""
from __future__ import annotations

import json
import math
from functools import lru_cache
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

import sympy as sp

from backend.arch.law_text import read_walk_law
from backend.measure.auto_sweep import compile_law

from .capacity import (
    CapacityKnobs,
    capacity_ceilings,
    _restamp_gpu,
    _capacity_pending,
    _capacity_preparation,
)
from .card import Card
from .comm_cost import CollectiveRung, collective_ladder
from .gpu_specs import peak_flops_for_dtype, peak_hbm_bw
from .phases import Speculation, phase_binding, phase_throughput, step_tokens
from .parallel_config import ParallelConfig
from .prefill_geometry import PrefillChunkSchedule, PrefillEndpointGeometry
from .row_roofline import StepSum, law_symbol, rung_symbols
from .sol_data import walkRecordSchema as servingLawSchema


_lawOutputs = {
    "t_sol_s": "t_op",
}

# One draft law serves every speculative K: the request says which draft stack
# exists, while this symbol says how many forwards the point runs.
draftForwardsLaw = sp.Symbol("s_verify") - 1


def _csePreservingExtrema(values: Sequence[sp.Expr]):
    """CSE without asking SymPy to compare the arms of a `Max`/`Min`.

    A roofline extremum is written as-written: its arms can hold `Cases`, so
    SymPy cannot order them until every axis is numeric. CSE rebuilds function
    applications by calling their constructors again, and `Max(...)`
    constructor tries that comparison. An inert function stands in for every
    extremum while CSE runs; the restored graph still prints the law language's
    as-written `Max` / `Min`.
    """
    replacements: dict[sp.Expr, sp.Symbol] = {}
    extrema: dict[sp.Symbol, sp.Expr] = {}

    def mask(expr: sp.Expr) -> sp.Expr:
        for current in sp.preorder_traversal(expr):
            if not isinstance(current, (sp.Max, sp.Min)) or current in replacements:
                continue
            symbol = sp.Symbol(f"law_inert_extremum_{len(replacements)}")
            extrema[symbol] = current
            replacements[current] = symbol
        return expr.xreplace(replacements)

    masked = [mask(value) for value in values]
    definitions, reduced = sp.cse(
        masked, symbols=sp.numbered_symbols("u"), order="canonical", list=True)

    def restore(expr: sp.Expr) -> sp.Expr:
        return expr.xreplace(extrema)

    return ([(symbol, restore(value)) for symbol, value in definitions],
            [restore(value) for value in reduced])


def _withoutITE(expr: sp.Expr) -> sp.Expr:
    """Rewrite SymPy's printed ITE as the law language's Piecewise."""
    if not isinstance(expr, sp.Basic) or not expr.args:
        return expr
    old_args = expr.args
    args = [_withoutITE(arg) for arg in old_args]
    if expr.func == sp.ITE:
        return sp.Piecewise((args[1], args[0]), (args[2], sp.true))
    if any(new is not old for new, old in zip(args, old_args, strict=True)):
        return expr.func(*args)
    return expr


@lru_cache(maxsize=4096)
def _read(text: str, local: frozenset[str]) -> sp.Expr:
    def symbol(name: str) -> sp.Symbol:
        if name in local:
            return sp.Symbol(name)
        return law_symbol(name)

    return read_walk_law(text, symbol=symbol)


@lru_cache(maxsize=4096)
def _compiled(expr: sp.Expr, names: tuple[str, ...]):
    return compile_law((expr,), names)


@dataclass(frozen=True)
class LawGraph:
    """A CSE graph: external inputs, local definitions, named outputs."""

    inputs: tuple[str, ...]
    definitions: tuple[tuple[str, str], ...]
    outputs: dict[str, str]

    @classmethod
    def of(cls, exprs: Mapping[str, sp.Expr]) -> "LawGraph":
        values = [_withoutITE(exprs[name]) for name in _lawOutputs]
        replacements, reduced = _csePreservingExtrema(values)
        outputs = {
            name: sp.sstr(value)
            for name, value in zip(_lawOutputs, reduced, strict=True)
        }
        inputs = tuple(sorted(
            {symbol.name for expr in values for symbol in expr.free_symbols}))
        return cls(
            inputs,
            tuple((symbol.name, sp.sstr(expr))
                  for symbol, expr in replacements),
            outputs,
        )

    def to_json(self) -> dict[str, Any]:
        return {"inputs": list(self.inputs),
                "definitions": [[name, text] for name, text in self.definitions],
                "outputs": self.outputs}

    @classmethod
    def from_json(cls, value: Mapping[str, Any]) -> "LawGraph":
        return cls(tuple(value["inputs"]),
                   tuple((name, text) for name, text in value["definitions"]),
                   dict(value["outputs"]))

    def evaluate(self, binding: Mapping[str, Any]) -> dict[str, Any]:
        values = dict(binding)
        local: set[str] = set()
        for name, text in self.definitions:
            expr = _read(text, frozenset(local))
            values[name] = self._at(expr, values)
            local.add(name)
        return {
            name: self._at(_read(text, frozenset(local)), values)
            for name, text in self.outputs.items()
        }

    @staticmethod
    def _at(expr: sp.Expr, values: Mapping[str, Any]) -> Any:
        names = tuple(sorted(symbol.name for symbol in expr.free_symbols))
        return _compiled(expr, names)(
            *(values[name] for name in names))[0]


def _collectives(rows: Iterable[Mapping[str, Any]]) -> tuple[str, ...]:
    out: list[str] = []
    seen: set[str] = set()
    for row in rows:
        if not row.get("is_comm") or int(row.get("comm_world") or 1) <= 1:
            continue
        signature = json.dumps({
            "op": row["comm_op"], "world": int(row["comm_world"]),
            "count": int(row["count_per_step"]),
            "stride": int(row["comm_base_stride"]),
            "rungs": [dict(rung) for rung in row.get("comm_rungs") or ()],
        }, sort_keys=True)
        if signature not in seen:
            seen.add(signature)
            out.append(signature)
    return tuple(out)


def _phase(rows: list[Mapping[str, Any]],
           draft_rows: Iterable[Mapping[str, Any]] = (),
           draft_forwards: int = 0) -> dict[str, Any]:
    step = StepSum.of(rows, draft_rows=draft_rows,
                      draft_forwards=draft_forwards)
    graph = LawGraph.of({
        wire: step.law(name) for wire, name in _lawOutputs.items()
    })
    return {"law": graph.to_json(), "collectives": list(_collectives(rows)),
            "terms": len(step.terms)}


def servingLawRecord(environment: Mapping[str, Any], *, gpu_sku: str,
                     anchorContext: int, outputTokens: int,
                     parallel: Any,
                     capacityCards: Sequence[str] = (),
                     capacityContexts: Sequence[int] = (),
                     draft_forwards: Any = None) -> dict[str, Any]:
    """Walk one layout once and keep its SOL laws, no expanded points."""
    knobs = CapacityKnobs.from_kwargs(environment, envelope=frozenset({
        "config", "gpu_sku", "avg_output_tokens", "prefill_batch_size",
        "decode_batch_size", "prefill_chunk_size"}))
    preparation = _capacity_preparation(
        config=environment["config"], gpu_sku=gpu_sku, knobs=knobs)
    pending = _capacity_pending(
        preparation,
        avg_input_tokens=anchorContext,
        avg_output_tokens=outputTokens,
        prefill_batch_override=1,
        decode_batch_override=1,
        breakdown=False,
        fold_prefill_chunks=False,
        prefill_capacity=False,
    )
    prefillRows = pending.walk_of_rows["per_op_timing"].analysis.per_op_timing
    decodeRows = (
        pending.walk_of_rows["per_op_timing_decode"].analysis.per_op_timing)
    draftRows = (
        (pending.walk_of_rows.get("per_op_timing_draft")
         and pending.walk_of_rows["per_op_timing_draft"].analysis.per_op_timing)
        or ())

    draftForwards = (0 if preparation.draft is None
                     else preparation.draft.forwards)
    if draft_forwards is not None:
        draftForwards = draft_forwards
    # A record is one walk class's: its capacity facts are for the cards
    # the walk may be re-priced on, and `_restamp_gpu` refuses any other
    # (`Card.refuse_unless_one_class`). Another class walks its own record.
    def preparationOn(cardSku: str) -> Any:
        return _restamp_gpu(preparation, Card.of(cardSku))

    capacity = []
    for cardSku in capacityCards or (gpu_sku,):
        onCard = preparationOn(cardSku)
        for context in capacityContexts or (anchorContext,):
            ceilings = capacity_ceilings(
                onCard, avg_input_tokens=int(context),
                avg_output_tokens=int(outputTokens),
                prefill_chunk_size=environment.get("prefill_chunk_size"),
                prefill_capacity=False)
            capacity.append({
                "gpu_sku": cardSku, "context": int(context),
                "max_decode_batch": int(ceilings.max_batch),
                "max_prefill_batch": int(ceilings.max_prefill_batch),
                "beyond_max_context": bool(ceilings.beyond_max_context),
                "infeasibility_reason": ceilings.infeasibility_reason,
            })
    return {
        "schema": servingLawSchema,
        # The card the phase laws' rows were walked on: every binding of a
        # card into them asks whether that card walks alike (`cardBinding`).
        "walked_on": gpu_sku,
        "parallel": parallel.to_dict(),
        "speculative": preparation.draft is not None,
        "max_context_tokens": preparation.max_context,
        "parallel_warnings": preparation.wb.get("parallel_warnings", []),
        "phases": {
            "prefill": _phase(prefillRows),
            "decode": _phase(decodeRows, draft_rows=draftRows,
                             draft_forwards=draftForwards),
        },
        "capacity": capacity,
    }


def servingLawCapacity(environment: Mapping[str, Any], *, gpu_sku: str,
                       parallel: Any,
                       capacityCards: Sequence[str] = (),
                       capacityContexts: Sequence[int] = ()) -> list[Any]:
    """The plain (non-speculative) capacity facts without a phase law.

    A speculative record's decode law already covers K=0 through its
    `s_verify` symbol, but its preparation also loads the draft stack. Capacity
    keeps both answers: this plain one and the drafted one
    (`servingLawRecord.capacity`)."""
    knobs = CapacityKnobs.from_kwargs(environment, envelope=frozenset({
        "config", "gpu_sku", "avg_output_tokens", "prefill_batch_size",
        "decode_batch_size", "prefill_chunk_size"}))
    preparation = _capacity_preparation(
        config=environment["config"], gpu_sku=gpu_sku, knobs=knobs)
    # A record is one walk class's: its capacity facts are for the cards
    # the walk may be re-priced on, and `_restamp_gpu` refuses any other
    # (`Card.refuse_unless_one_class`). Another class walks its own record.
    def preparationOn(cardSku: str) -> Any:
        return _restamp_gpu(preparation, Card.of(cardSku))

    out = []
    for cardSku in capacityCards or (gpu_sku,):
        onCard = preparationOn(cardSku)
        for context in capacityContexts or (1024,):
            ceilings = capacity_ceilings(
                onCard, avg_input_tokens=int(context),
                avg_output_tokens=int(environment["avg_output_tokens"]),
                prefill_chunk_size=environment.get("prefill_chunk_size"),
                prefill_capacity=False)
            out.append({
                "gpu_sku": cardSku, "context": int(context),
                "max_decode_batch": int(ceilings.max_batch),
                "max_prefill_batch": int(ceilings.max_prefill_batch),
                "beyond_max_context": bool(ceilings.beyond_max_context),
                "infeasibility_reason": ceilings.infeasibility_reason,
            })
    return out


def _capacityAt(record: Mapping[str, Any], card: Card, context: int):
    for item in record.get("capacity") or ():
        if item["gpu_sku"] == card.sku and item["context"] == context:
            return item
    raise KeyError(
        f"the serving law has no capacity fact for {card.sku} at {context}")


def servingLawPoint(record: Mapping[str, Any], *, card: Card,
                    context: int, outputTokens: int,
                    decodeBatch: int | None = None,
                    prefillBatch: int | None = 1,
                    verifyWidth: int = 1,
                    draftTokens: int = 0,
                    acceptanceRate: float = 1.0,
                    comm_latency_us: float, launch_floor_us: float) -> dict:
    """One served point on `card`, derived from the law and the card's
    capacity facts. `card` must walk as the record's walk card does
    (`cardBinding`, the one place a card enters a law)."""
    capacityRecord = (
        {"capacity": record["draft_capacity"]} if draftTokens
        and "draft_capacity" in record else record)
    cardValues = cardBinding(
        record, card, comm_latency_us=comm_latency_us,
        launch_floor_us=launch_floor_us)
    facts = _capacityAt(capacityRecord, card, context)
    decodeB = (int(decodeBatch) if decodeBatch is not None
               else int(facts["max_decode_batch"]))
    prefillB = (int(prefillBatch) if prefillBatch is not None
                else int(facts["max_prefill_batch"]))
    phases = {}
    parallel = ParallelConfig.from_dict(record["parallel"])
    groups = parallel.dp_attention * parallel.dp_replicas
    num_gpus = parallel.cluster_size
    for name, batch in (("prefill", prefillB), ("decode", decodeB)):
        graph = LawGraph.from_json(record["phases"][name]["law"])
        point = {
            **cardValues, "B": max(1, batch), "s_input": context,
            "s_output": outputTokens, "s_prefix": 0,
            "s_verify": verifyWidth,
            "rho": 1,
        }
        if name == "prefill":
            schedule = PrefillChunkSchedule.materialize(
                PrefillEndpointGeometry(int(context), 0, 1, max(1, batch)),
                prefill_chunk_size=8192)
            t = 0.0
            for chunk in schedule.chunks:
                repeat = (schedule.endpoint_batch // chunk.launch_batch
                          if len(schedule.chunks) > 1 else 1)
                chunkTime = float(graph.evaluate({
                    **point, "B": chunk.launch_batch,
                    "s_input": chunk.end, "s_prefix": chunk.start,
                })["t_sol_s"])
                t += chunkTime * repeat
        else:
            t = float(graph.evaluate(point)["t_sol_s"])
        phaseBinding = phase_binding(
            phase=name, avg_input_tokens=context,
            avg_output_tokens=outputTokens, s_prefix_cached=0,
            s_verify=verifyWidth)
        tokensPerStep = float(phaseBinding["tokens_per_step"])
        if name == "decode" and draftTokens:
            effectiveTokens, effectiveFormula = step_tokens(
                "decode", batch, tokensPerStep,
                Speculation(draftTokens, acceptanceRate))
        else:
            effectiveTokens = tokensPerStep * batch
            effectiveFormula = {
                "symbolic": f"B × {phaseBinding['s_query']:g}",
                "substituted": f"{batch} × {tokensPerStep:g}",
            }
        binding = {
            **phaseBinding,
            "effective_tokens_per_step": effectiveTokens,
            "effective_tokens_formula": effectiveFormula,
        }
        phase = {
            "phase": name,
            "batch": batch,
            "binding": binding,
            "t_sol_s": t,
            "measured_rows": 0,
            "total_rows": int(record["phases"][name].get("terms") or 0),
            "coverage": {
                "measured": 0,
                "total": int(record["phases"][name].get("terms") or 0),
            },
        }
        phase["throughput"] = phase_throughput(
            phase, unique_tokens_multiplier=groups, num_gpus=num_gpus)
        phases[name] = phase
    return {
        "avg_input_tokens": context,
        "avg_output_tokens": outputTokens,
        "max_batch_per_gpu": facts["max_decode_batch"],
        "max_decode_batch_per_gpu": facts["max_decode_batch"],
        "max_prefill_batch_per_gpu": facts["max_prefill_batch"],
        "beyond_max_context": facts["beyond_max_context"],
        "infeasibility_reason": facts["infeasibility_reason"],
        "max_context_tokens": record.get("max_context_tokens"),
        "parallel_warnings": list(record.get("parallel_warnings") or []),
        "phases": phases,
    }


def cardBinding(record: Mapping[str, Any], card: Card, *,
                comm_latency_us: float, launch_floor_us: float,
                dtypes: Sequence[str] = ()) -> dict[str, float]:
    """The card half of a law graph's inputs — `card`'s peaks, bandwidth and
    fabric — for a card of the class the record's rows were walked on
    (`walked_on`), refused for any other (`Card.refuse_unless_one_class`).

    The one door a card enters a law through, so the rule is asked here and
    not by each projection: a law's terms are one walk's rows, and a card of
    another class runs other rows (another kernel, another activation
    format). Its symbols would bind and the number would look right."""
    Card.of(record["walked_on"]).refuse_unless_one_class(card)
    gpu = card.spec
    fabric = card.fabric(comm_latency_us * 1e-6)
    out: dict[str, float] = {
        "PEAK_BW": float(peak_hbm_bw(gpu) or math.inf),
        "COMM_S_PER_BYTE": 0.0,
        "COMM_LATENCY_S": 0.0,
        "LAUNCH_S": launch_floor_us * 1e-6,
    }
    wanted = set(dtypes)
    for phase in record["phases"].values():
        graph = LawGraph.from_json(phase["law"])
        for name in graph.inputs:
            if name == "PEAK_FLOPS":
                out[name] = float(peak_flops_for_dtype(gpu, "") or math.inf)
            elif name.startswith("PEAK_FLOPS_"):
                dtype = name[len("PEAK_FLOPS_"):]
                wanted.add(dtype)
                out[name] = float(
                    peak_flops_for_dtype(gpu, dtype) or math.inf)

    def collectiveValues(signature: str) -> None:
        facts = json.loads(signature)
        rungs = tuple(CollectiveRung.from_dict(rung)
                      for rung in facts.get("rungs") or ())
        ladder = collective_ladder(
            op=facts["op"], world=int(facts["world"]),
            count_per_step=int(facts["count"]),
            base_stride=int(facts["stride"]), fabric=fabric, rungs=rungs)
        base = (f"_{facts['op']}_{int(facts['world'])}_"
                f"{int(facts['count'])}_{int(facts['stride'])}")
        names = (f"COMM_S_PER_BYTE{base}", f"COMM_LATENCY_S{base}")
        values = (ladder[-1].s_per_byte, ladder[-1].latency_s)
        for symbol, value in zip(names, values, strict=True):
            out[symbol] = float(value)
        for index, segment in enumerate(ladder[:-1]):
            names = tuple(
                f"{symbol.name[:-len(f'_r{index}')] if symbol.name.endswith(f'_r{index}') else symbol.name}{base}_r{index}"
                for symbol in rung_symbols(index))
            values = (segment.max_bytes, segment.s_per_byte,
                      segment.latency_s)
            for symbol, value in zip(names, values, strict=True):
                out[symbol] = float(value)

    seen: set[str] = set()
    for phase in record["phases"].values():
        for signature in phase["collectives"]:
            if signature not in seen:
                seen.add(signature)
                collectiveValues(signature)
    return out
