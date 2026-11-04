"""The analysability checks that need no deployment, and the one shape every
analysis refusal takes in `errors[]`.

Validate evaluates the spec on one rank, where nothing is distributed and the
layout rule is never asked anything. Every multi-rank analysis — SOL, the
Inference page, a parallelism sweep — walks the same graph with
`LayoutPropagation` and then joins the per-rank weights. Two of its refusals
need no deployment to find, so they run here, on the one-rank walk validate
already has:

1. Layout dispatch. The walk sends each primitive to a transition handler
   (`layout.propagation.layout_handler`) and refuses an op it has none for.
   This asks that same dispatch for every primitive the walk reaches in both
   phases — decode rebinds components to their declared decode forms, so its
   graph can hold ops prefill never reaches — and names the node, the op and
   what to write instead.
2. Weight identity. The per-rank weight join (`physical_weight_leaves`)
   refuses two live weights with one id at any degree, one GPU included.

The model card's sizes (`Program.headline`) are read off the model-level
dims, and every analysis page shows them; a spec that declares no `d_model`
or `n_vocab` is refused here too, naming the dim to bind.

The replay of the analysis on the product's deployments lives with the
analysis (`backend.inference.analysable`), which arch does not import; it
reaches the orchestrator as an injected gate and reports its refusals through
`Refusal` / `grouped_findings` below, so every analysis refusal reads
the same.
"""
from __future__ import annotations

import os
import traceback
from dataclasses import asdict, dataclass, fields
from typing import Any, Dict, List, Optional, Sequence

from backend.arch.eval.op_ref import OpRef
from backend.arch.eval.program import Program
from backend.arch.eval.refusal import (SpecRefusal, engine_gap, refusal_node,
                                       spec_refusal)
from backend.arch.eval.transforms import Deployment
from backend.arch.layout.persistent import physical_weight_leaves
from backend.arch.layout.propagation import layout_gap

from ._finding import Finding

#: The spec writer resolves it: the finding names the node, the declaration
#: and the fix.
SPEC_KIND = "analysis_refusal"
#: The engine resolves it: the raise site declared the capability it lacks
#: (`EngineGap`, or Python's `NotImplementedError`). The finding names the
#: node, the capability, the raise site and the deployment; the spec-gen
#: worker leaves it out of the repair loop, and a run whose every error is
#: one ends `blocked_by_engine`.
ENGINE_KIND = "analysis_engine_refusal"
#: The raise site declared no owner. The finding names the node, the raise
#: site and the deployment, and goes back to the spec writer with the one
#: question that settles it: does the refusing node compute what the source
#: does?
OPEN_KIND = "analysis_refusal_unattributed"


@dataclass(frozen=True)
class Refusal:
    """One refusal of one deployment, as plain data: it crosses process
    boundaries and is memoised. `kind` is SPEC_KIND, ENGINE_KIND or
    OPEN_KIND; a spec refusal also names the declaration and the fix, an
    engine one the capability it lacks."""
    kind: str
    node: Optional[str]
    site: str
    function: str
    exception: str
    message: str
    deployment: str
    stage: str
    declaration: Optional[str] = None
    fix: Optional[str] = None
    capability: Optional[str] = None

    @classmethod
    def of(cls, exc: BaseException, *, deployment: str,
           stage: str) -> "Refusal":
        frames = traceback.extract_tb(exc.__traceback__)
        inner = frames[-1] if frames else None
        spec = spec_refusal(exc)
        gap = engine_gap(exc) if spec is None else None
        # `NotImplementedError` is Python's own statement that a capability
        # is missing: its message names it.
        capability = (gap.capability if gap is not None
                      else str(exc) if spec is None
                      and isinstance(exc, NotImplementedError) else None)
        return cls(
            kind=(SPEC_KIND if spec is not None
                  else ENGINE_KIND if capability is not None else OPEN_KIND),
            node=refusal_node(exc),
            site=(f"{os.path.basename(inner.filename)}:{inner.lineno}"
                  if inner is not None else ""),
            function=inner.name if inner is not None else "",
            exception=type(exc).__name__,
            message=str(exc),
            deployment=deployment,
            stage=stage,
            declaration=spec.declaration if spec is not None else None,
            fix=spec.fix if spec is not None else None,
            capability=capability,
        )

    @property
    def cause(self) -> tuple:
        """What makes two refusals one: kind, node, raise site, exception
        type — never the message, whose numbers vary by deployment."""
        return (self.kind, self.node, self.site, self.exception)


def refusal_finding(refusals: Sequence[Refusal]) -> Finding:
    """One `errors[]` entry for refusals with one cause, seen at several
    deployments."""
    first = refusals[0]
    deployments = sorted({r.deployment for r in refusals})
    stages = sorted({r.stage for r in refusals})
    where = f"node `{first.node}`" if first.node else "the model"
    at = f"{', '.join(deployments)} ({', '.join(stages)})"
    if first.kind == SPEC_KIND:
        detail = (f"analysis refuses {where} at {at}: {first.message}. "
                  f"Fix `{first.declaration}`: {first.fix}.")
    elif first.kind == ENGINE_KIND:
        detail = (f"analysis engine refusal at {where}, raised at "
                  f"{first.site} ({first.function}) on {at}: "
                  f"{first.message}. The engine does not yet support "
                  f"{first.capability}; the spec states it correctly, and "
                  "is blocked until the engine does.")
    else:
        detail = (f"analysis refuses {where}, raised at {first.site} "
                  f"({first.function}) on {at}: {first.message}. Check that "
                  "the refusing node computes what the forward source does "
                  "— its op, its operands and their declared dims — and "
                  "write it so that it does.")
    extra = {"deployments": deployments, "stages": stages,
             "site": first.site, "exception": first.exception}
    if first.kind == SPEC_KIND:
        extra.update(declaration=first.declaration, fix=first.fix)
    elif first.kind == ENGINE_KIND:
        extra.update(capability=first.capability)
    return Finding(kind=first.kind, detail=detail, node=first.node,
                   extra=extra)


@dataclass(frozen=True)
class EngineGapEntry:
    """One engine refusal as a blocked spec stores and shows it: what the
    engine lacks, the node and raise site, and the deployments and stages
    validate met it on. Read off validate's own finding (`refusal_finding`),
    never off its text."""
    capability: str
    node: Optional[str]
    site: str
    exception: str
    deployments: List[str]
    stages: List[str]
    detail: str

    @classmethod
    def of(cls, finding: Dict[str, Any]) -> "EngineGapEntry":
        entry = cls(**{f.name: finding.get(f.name) for f in fields(cls)})
        # An engine finding always names these (`refusal_finding`).
        if not (entry.capability and entry.deployments and entry.stages):
            raise ValueError(f"engine refusal finding is incomplete: {finding}")
        return entry


def engine_gap_entries(result: Dict[str, Any]) -> List[Dict[str, Any]]:
    """The engine refusals of one validate result, one entry per cause
    (`EngineGapEntry`), from `analysis_refusals` — the finding list validate
    itself judged `blocked_by_engine` on."""
    return [asdict(EngineGapEntry.of(finding))
            for finding in result.get("analysis_refusals") or []
            if finding.get("kind") == ENGINE_KIND]


def only_unattributed(result: Dict[str, Any]) -> bool:
    """Whether every error the spec writer was left with is a refusal whose
    raise site declared no owner: the errors, less the engine refusals, are
    all `OPEN_KIND` findings (and there is at least one)."""
    details = {kind: {f.get("detail") for f in result.get("analysis_refusals")
                      or [] if f.get("kind") == kind}
               for kind in (ENGINE_KIND, OPEN_KIND)}
    left = [e for e in result.get("errors") or []
            if e not in details[ENGINE_KIND]]
    return bool(left) and all(e in details[OPEN_KIND] for e in left)


def grouped_findings(refusals: Sequence[Refusal]) -> List[Finding]:
    """One finding per cause (`Refusal.cause`)."""
    groups: Dict[tuple, List[Refusal]] = {}
    for refusal in refusals:
        groups.setdefault(refusal.cause, []).append(refusal)
    return [refusal_finding(group) for group in groups.values()]


def _dispatch_gaps(analysis: Any, phase: str) -> List[Refusal]:
    program = analysis.program
    out: List[Refusal] = []
    for uid in analysis.shapes.values:
        node = program.by_uid(uid)
        if node.is_component or node.children:
            continue
        op = OpRef.resolve(node.op or "", program, axis=node.axis)
        gap = layout_gap(node.op, None if op is None else op.spec)
        if gap is None:
            continue
        out.append(Refusal(
            kind=SPEC_KIND, node=node.display_path,
            site="layout dispatch", function="layout_handler",
            exception="SpecRefusal",
            message=f"op {node.op!r} has no layout transition rule",
            deployment="every multi-GPU deployment", stage=phase,
            declaration="op", fix=gap,
        ))
    return out


def run_analysable_check(resolved: Any, prefill_analysis: Any, *,
                         ledger_built: bool = True) -> List[Finding]:
    """The layout dispatch of every reached op in both phases, and the
    weight identity join at one GPU. The join reads the per-rank weight
    ledger, so it runs only when the ledger built — a ledger that did not
    is already an error naming its cause."""
    refusals = _dispatch_gaps(prefill_analysis, "prefill")
    try:
        _ = prefill_analysis.program.headline
    except SpecRefusal as exc:
        refusals.append(Refusal.of(exc, deployment="single",
                                   stage="model headline"))
    decode = Deployment(Program.of(resolved)).during("decode").analyze()
    refusals += _dispatch_gaps(decode, "decode")
    if ledger_built:
        try:
            physical_weight_leaves(prefill_analysis)
        except ValueError as exc:
            refusals.append(Refusal.of(exc, deployment="single",
                                       stage="weight identity"))
    return grouped_findings(refusals)

