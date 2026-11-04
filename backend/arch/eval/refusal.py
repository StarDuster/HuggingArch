"""What an analysis refusal is about: its graph node, and whose fix it is.

A refusal raised while the analysis handles one node — the walk placing it,
the comm lowering of its redistribute, the endpoints of its stored weight —
carries that node in a structured attribute, `analysis_node` (the node's
display path). The per-node loop sets it around whatever it calls
(`node_context`); the exception keeps its type, so a caller that catches a
specific refusal class is unchanged, and a reader takes the node from the
attribute, never from the message. The step of the analysis it was raised
in — the capacity entry point running, `stage_context` — is stamped the same
way (`analysis_stage`), and read back by `refusal_stage`.

A refusal whose cause is a declaration the spec writer can change is raised
as a `SpecRefusal`: it names the declaration and the fix. A refusal the engine
raises because it lacks a capability is an `EngineGap`: it names the
capability. A refusal whose cause is the deployment — a combination of
parallel knobs the walk does not run, whatever the model — is a
`DeploymentRefusal`: it names the knobs. A refusal raised as none of these
states no owner, and is not claimed for any side.
"""
from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator, Optional, Tuple

NODE_ATTR = "analysis_node"
STAGE_ATTR = "analysis_stage"


class LayoutRefusal(ValueError):
    """The base of refusals a caller may classify as 'this layout does not
    deploy'. Plain ValueError is an engine bug: it must stay loud instead of
    being downgraded to an unavailable sweep/Gallery degree."""


class SpecRefusal(LayoutRefusal):
    """An analysis refusal the spec resolves: `declaration` names the spec
    field at fault, `fix` says what to write. The message is the refusal as
    the engine states it."""

    def __init__(self, message: str, *, declaration: str, fix: str,
                 node: Optional[str] = None) -> None:
        super().__init__(message)
        self.declaration = declaration
        self.fix = fix
        if node is not None:
            vars(self)[NODE_ATTR] = node


class EngineGap(LayoutRefusal):
    """An analysis refusal the engine resolves: a construct the spec states
    correctly that the engine cannot yet evaluate. `capability` names what is
    missing, so the refusal reads as a known limit, not a spec error."""

    def __init__(self, message: str, *, capability: str,
                 node: Optional[str] = None) -> None:
        super().__init__(message)
        self.capability = capability
        if node is not None:
            vars(self)[NODE_ATTR] = node


class DeploymentRefusal(LayoutRefusal):
    """An analysis refusal the deployment resolves: the walk does not run
    this combination of parallel knobs (`knobs`, the `ParallelConfig` fields
    at fault) in this phase, whatever the model. Stated once, as a rule on
    the deployment, so the gate every request passes (`validate`) reports
    the refusal the walk raises rather than a copy of it."""

    def __init__(self, message: str, *, knobs: Tuple[str, ...],
                 phase: Optional[str] = None) -> None:
        super().__init__(message)
        self.knobs = tuple(knobs)
        self.phase = phase


@contextmanager
def node_context(display_path: str) -> Iterator[None]:
    """Name `display_path` on any refusal raised inside, unless a deeper
    per-node loop already named one — on it, or on a refusal it wraps."""
    try:
        yield
    except ValueError as exc:
        if refusal_node(exc) is None:
            vars(exc)[NODE_ATTR] = display_path
        raise


def refusal_node(exc: Optional[BaseException]) -> Optional[str]:
    """The node a refusal names, following its cause chain."""
    while exc is not None:
        node = vars(exc).get(NODE_ATTR)
        if node is not None:
            return node
        exc = exc.__cause__ or exc.__context__
    return None


@contextmanager
def stage_context(stage: str) -> Iterator[None]:
    """Name `stage` on any exception raised inside, unless an inner stage
    already named it. Usable as a decorator on a stage's entry point."""
    try:
        yield
    except Exception as exc:
        if refusal_stage(exc) is None:
            vars(exc)[STAGE_ATTR] = stage
        raise


def refusal_stage(exc: BaseException) -> Optional[str]:
    """The analysis stage a refusal was raised in: the innermost
    `stage_context` it left. The exception's own, not its cause's — a
    refusal re-raised in a later stage was raised there."""
    return vars(exc).get(STAGE_ATTR)


def spec_refusal(exc: Optional[BaseException]) -> Optional[SpecRefusal]:
    """The `SpecRefusal` in a refusal's cause chain, if any."""
    while exc is not None:
        if isinstance(exc, SpecRefusal):
            return exc
        exc = exc.__cause__ or exc.__context__
    return None


def engine_gap(exc: Optional[BaseException]) -> Optional[EngineGap]:
    """The `EngineGap` in a refusal's cause chain, if any."""
    while exc is not None:
        if isinstance(exc, EngineGap):
            return exc
        exc = exc.__cause__ or exc.__context__
    return None
