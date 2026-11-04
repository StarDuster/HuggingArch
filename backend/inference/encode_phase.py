"""The encode phase of an estimate: each encoder over the media a request
carries, priced on its own and never added to the language model's step.

An encoder (`Stack.is_encoder`) runs once per media item, before the language
model's prefill reads what it hands over (vLLM `_execute_mm_encoder` ahead of
the forward, `gpu_model_runner.py:3072, 4335`). Its work is its own phase
(`Phase(kind="encode", stack=...)`): the walk binds the batch and the media
axes (`Workload.media`) and none of the language model's step, so nothing
here reads `s_ctx` and nothing here reaches the prefill / decode totals,
TTFT or TPOT.

Each encoder stack is one step, priced the way a language-model step is —
the same `pipeline.analyze` rows, the same calibration overlay, the same row
sums — with only that stack's leaves timed. What the walk reports about its
media (`Analysis.media_rows`) and the prompt positions one request's media
fill (`Deployment.media_tokens`) ride beside the numbers; the language model
reads neither.

The encoder's own capacity is its account too, off the same walks: the
weights each encoder holds on one rank (the encode walk's resident weight
ledger, whatever the media), the encode step's activation peak (the same
liveness the language model's prefill peak runs, over the encoder's blocks),
and the encoder output cache a serving framework keeps for the language
model to read (`EncoderOutputCache`). None of it is in the language model's
capacity.
"""
from __future__ import annotations

from typing import Any, Callable, Dict, List, Mapping, Optional, Tuple

from dataclasses import dataclass

from backend.arch.eval.dim import value_of
from backend.arch.eval.program import Program
from backend.arch.eval.transforms import Deployment
from backend.arch.layout.persistent import physical_artifacts, resident_weights

from .activation_peak import activation_peak_at
from .calibration_overlay import overlay_walk
from .phase_formulas import enrich_module_breakdown
from .row_roofline import RowRoofline
from .step_eval import evaluate_encode


def _sums(rows: List[Mapping[str, Any]]) -> Dict[str, Any]:
    """The step's figures, each the sum of its rows' — the language model's
    phase sums (`phases.estimate_phase`), and the measured track beside
    them."""
    def total(field: str) -> float:
        return sum(float(r.get(field) or 0.0) for r in rows)
    return {
        "flops_per_rank": total("flops_per_rank"),
        "bytes_per_rank": total("access_bytes_per_rank"),
        "t_compute": total("t_compute_s"),
        "t_memory": total("t_memory_s"),
        "t_comm": total("t_comm_s"),
        "t_sol": total("t_op_s"),
        "t_sol_by_bound": RowRoofline.by_bound(rows),
        # The measured fold states it (`batch_sweep.apply_measured`).
        "t_measured": None,
    }


@dataclass(frozen=True)
class EncoderOutputCache:
    """The encoder outputs a serving framework holds for the language model
    to read, sized as vLLM sizes it: `compute_mm_encoder_budget` caches
    `max(encoder_cache_size, max tokens of one item)` tokens, the scheduler's
    `encoder_cache_size` starting at `max_num_batched_tokens`
    (vLLM v0.30.0 v1/core/encoder_cache_manager.py:293-340,
    config/scheduler.py:285) — the estimate's `prefill_chunk_size`. One cached token holds
    every tensor an encoder hands over at that prompt position (a domain's
    defining edge and the edges adding onto its positions, Qwen3-VL's
    DeepStack features included: vLLM caches the tower's output with the
    DeepStack levels concatenated on the width, qwen3_vl.py:596, 887,
    split back at 2936), read off the encode walks
    (`Deployment.handed_over`), each at the activation format its producing
    leaf has under the deployment's precision; the cache is shared by the
    model's media inputs, so a token is priced at the widest of them.

    `scheduler_tokens` None (un-chunked prefill) states no budget, and the
    cache is not sized."""

    scheduler_tokens: Optional[int]
    max_tokens_per_item: int
    bytes_per_token: float

    @classmethod
    def of(cls, deployment: Deployment, *, scheduler_tokens: Optional[int]
           ) -> "EncoderOutputCache":
        program = deployment.program
        # The widths are the walk's at any media; one item's rows are this
        # request's media divided by its items.
        widths = deployment.handed_over({})
        per_token: Dict[str, float] = {}
        for handoff in program.handoffs:
            per_token[handoff.domain] = per_token.get(handoff.domain, 0) + sum(
                int(value_of(t.width)) * t.value_bytes
                for t in widths[handoff.name])
        media = deployment.media
        per_item = []
        for domain, tokens in deployment.media_tokens().items():
            entry = program.model_inputs[domain]
            items = int(value_of(
                entry.media_values(media)[entry.axes[0]]))
            if items:
                per_item.append(tokens // items)
        return cls(
            scheduler_tokens=(int(scheduler_tokens) if scheduler_tokens
                              else None),
            max_tokens_per_item=max(per_item, default=0),
            bytes_per_token=float(max(per_token.values(), default=0)))

    def payload(self) -> Dict[str, Any]:
        tokens = (None if self.scheduler_tokens is None
                  else max(self.scheduler_tokens, self.max_tokens_per_item))
        return {
            "scheduler_tokens": self.scheduler_tokens,
            "max_tokens_per_item": self.max_tokens_per_item,
            "tokens": tokens,
            "bytes_per_token": self.bytes_per_token,
            "bytes_per_rank": (None if tokens is None
                               else tokens * self.bytes_per_token),
        }


def _weight_bytes_per_rank(analysis: Any, stack: str) -> int:
    """The encoder's resident weights on one rank: its encode walk's
    persistent-artifact ledger (parameters, quant metadata, buffers)."""
    # A stack that runs another's weights (`Stack.reruns`) holds none.
    return int(resident_weights(physical_artifacts(analysis)).by_stack.get(
        stack, 0))


def _per_item(entry: Dict[str, Any]) -> None:
    """Time per media item and items per second, off the step's SOL."""
    items, t_sol = entry["items"], entry["t_sol"]
    entry["t_sol_per_item"] = t_sol / items if items else None
    entry["items_per_s"] = items / t_sol if items and t_sol else None


@dataclass(frozen=True)
class EncodePlan:
    """The encode phase up to its overlays' walks (`plan_encode`): one walk
    per encoder that runs items, and `finish()` — the phase's answer over
    them, no measurement read. `points[i]` are stack `i`'s `PointRow`s once
    finished: what the estimate's measured fold keys its rows by."""
    walks: List[Any]
    finish: Callable[[], Dict[str, Any]]
    #: (index into the payload's `stacks`, walk) per encoder that runs items.
    indexed: List[Tuple[int, Any]]
    points: Dict[int, list]

    def timed(self) -> List[Tuple[int, Any]]:
        return self.indexed


def plan_encode(*, resolved: Any, parallel: Any, batch: int,
                media: Mapping[str, Mapping[str, int]],
                precision: Any, overlay_kwargs: Mapping[str, Any],
                scheduler_tokens: Optional[int] = None,
                ) -> Optional[EncodePlan]:
    """The encode phase, None for a model with no encoder; its `finish`
    answers `{batch, media, media_tokens, stacks: [...], subtotal,
    output_cache}`
    — one entry per encoder stack, None for a model with no encoder.
    Each entry carries the encoder's own capacity: `weight_bytes_per_rank`
    (whatever the media) and `activation_peak` (the encode step's, None when
    it runs no item). `output_cache` is `EncoderOutputCache` at
    `scheduler_tokens`.

    `overlay_kwargs` is the language-model estimate's own calibration
    context (`capacity._capacity_finish`): the same card, framework, dtypes
    and lookup. An encoder none of whose inputs `media` names runs no item:
    its entry says so and carries no rows. `subtotal` sums the encoders when
    there are several — the encoders' total alone, no part of any
    language-model figure."""
    program = Program.of(resolved)
    encoders = [stack for stack in program.stacks if stack.is_encoder]
    if not encoders:
        return None
    batch = max(1, int(batch))
    deployment = Deployment(program, config=parallel, precision=precision).at(
        batch=batch, media=dict(media))
    media = deployment.media  # held to the spec's declarations
    tokens = deployment.media_tokens()
    stacks: List[Dict[str, Any]] = []
    timed: List[tuple] = []         # (entry, walk) of each encoder that runs items
    for stack in encoders:
        own = {name: entry for name, entry in program.model_inputs.items()
               if entry.stack == stack.name}
        given = {name: entry for name, entry in own.items()
                 if name in media and entry.unmodeled is None}
        entry: Dict[str, Any] = {
            "stack": stack.name,
            "aux_kind": stack.aux_kind,
            # The stack whose weights this one runs (a model's videos
            # through its image tower); its own weight figure is then 0.
            "reruns_weights_of": stack.reruns,
            "inputs": {name: {axis: int(value_of(value)) for axis, value
                              in e.media_values(media).items()}
                       for name, e in given.items()},
            "handoff_tokens_per_request": {
                h.domain: tokens[h.domain] for h in program.handoffs
                if h.stack == stack.name and h.defines
                and h.domain in tokens},
        }
        if not given:
            # No item: the walk still holds the encoder's weights.
            idle = Deployment(program, config=parallel,
                              precision=precision).at(
                batch=batch).during("encode", stack=stack.name).analyze()
            entry.update(items=0, media_rows={}, per_op_timing=[],
                         flops_by_module=[], vs_naive_delta=None,
                         weight_bytes_per_rank=_weight_bytes_per_rank(
                             idle, stack.name),
                         activation_peak=None,
                         **{key: None for key in _sums([])})
            _per_item(entry)
            stacks.append(entry)
            continue
        step = evaluate_encode(resolved, parallel, stack=stack.name,
                               batch=batch, media=media,
                               precision=precision)
        walk = overlay_walk(**{
            **overlay_kwargs,
            "phase": "encode",
            "max_batch": batch,
            "step_eval": step,
            "timed_stacks": frozenset({stack.name}),
            # The encoder runs whole on one pipeline stage
            # (`LayoutPropagation.encoder_view`): its rows are that
            # stage's, every one of them.
            "stage": None,
            "pipeline_stages": 1,
        })
        media_rows = step.analysis.media_rows()
        peak = activation_peak_at(
            step, framework_binding=overlay_kwargs["framework_binding"],
            overhead_bytes=0)
        entry.update(
            weight_bytes_per_rank=_weight_bytes_per_rank(
                step.analysis, stack.name),
            activation_peak={
                "bytes_per_rank": peak["peak_activation_bytes_per_rank"],
                "worst_block": peak.get("worst_block"),
                "by_block": peak["peak_by_block"],
            },
            # A media item is one of an input's own items (an image, a clip,
            # a video), over the step's requests — not a segment (a video's
            # temporal patches) and not the items of another input's items
            # (Step-3.7's crops).
            items=batch * sum(entry["inputs"][name][e.axes[0]]
                              for name, e in given.items()
                              if e.per_item_of is None),
            # This encoder's own inputs (one that runs another's weights
            # walks beside it, at no item of that one's).
            media_rows={name: {"segments": r.segments,
                               "rows_per_segment": r.rows,
                               "valid_rows_per_segment": r.valid_rows}
                        for name, r in media_rows.items() if name in given},
        )
        timed.append((len(stacks), entry, walk))
        stacks.append(entry)

    points: Dict[int, list] = {}

    def finish() -> Dict[str, Any]:
        for index, entry, walk in timed:
            rows, fusion, _zone, module_rows, _pipeline, points[index] = walk.finish()
            entry.update(per_op_timing=rows, flops_by_module=module_rows,
                         vs_naive_delta=fusion, **_sums(rows))
            # The encoder's table is the same fold the language model's
            # phases draw: subtree rollups, parent popovers and its Total row.
            enrich_module_breakdown(entry)
            _per_item(entry)
        return _encode_payload(batch, media, tokens, stacks, deployment, scheduler_tokens)

    return EncodePlan(walks=[walk for _i, _entry, walk in timed], finish=finish,
                      indexed=[(index, walk) for index, _entry, walk in timed],
                      points=points)


def _encode_payload(batch: int, media: Any, tokens: Any, stacks: List[Dict[str, Any]],
                    deployment: Any, scheduler_tokens: Optional[int]) -> Dict[str, Any]:
    """The encode phase's answer over its finished stacks."""
    out: Dict[str, Any] = {
        "batch": batch, "media": media, "media_tokens": tokens,
        "stacks": stacks,
        "output_cache": EncoderOutputCache.of(
            deployment, scheduler_tokens=scheduler_tokens).payload(),
    }
    if len(stacks) > 1:
        ran = [s for s in stacks if s["items"]]
        out["subtotal"] = {
            key: (sum(s[key] for s in ran) if ran else None)
            for key in ("flops_per_rank", "bytes_per_rank", "t_compute",
                        "t_memory", "t_comm", "t_sol")}
        out["subtotal"]["weight_bytes_per_rank"] = sum(
            s["weight_bytes_per_rank"] for s in stacks)
    return out
