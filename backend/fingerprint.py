"""Engine and spec identity hashes — the one definition.

`engine_version()` is a semantic hash of the estimation engine's source and
all its data; `spec_fingerprint(resolved)` is a content hash of one resolved
spec. A cached result for one model is keyed on `result_version(resolved)`:
the engine's code, the data every model shares, and only the data that model
can read (`ModelFootprint`), so an edit to one model's components or to a
fusion plan another model cannot match leaves the other models' results
cached. Every result cache's key folds one of these in (the caches are
`backend.measure.cache`'s; the keys their callers') and the analysis replay memo of validate (`backend.inference.analysable`)
build their keys from here; `engine_version()` stays the stamp for what
spans every model (the gallery frontier request, `code_version`).
"""
from __future__ import annotations

from backend.arch.yaml_load import load_yaml
import hashlib
import json
import pathlib
from dataclasses import dataclass
from functools import cached_property, lru_cache
from typing import Any, ClassVar, Dict, FrozenSet, Iterator, Tuple


from backend.arch.eval.program import Program
from backend.arch.eval.transforms import (rebind_decode_variants,
                                          splice_sampled_head)
from backend.arch.loader import ResolvedSpec, strip_machine_paths
from backend.arch.ops import OP_REGISTRY
from backend.inference.framework_bindings import (FRAMEWORK_BINDINGS,
                                                  _FRAMEWORK_BINDINGS_DIR)
from backend.inference.fusion_plans import FUSION_SCHEMES
from backend.inference.materialize_cast import GRAFTED_OPS


# ---------------------------------------------------------------------------
# Versioning / keying — the single source of truth.
# ---------------------------------------------------------------------------


def _docstring_spans(tree: Any, lines: list) -> list:
    """(start, end) (lineno, CHAR col) spans of every module/class/function
    docstring. Docstrings are documentation, not semantics — a doc typo fix
    must not invalidate every cached result. (One engine file reads
    ``__doc__`` for a CLI help string — no cached *result* depends on it.)

    ast col offsets are UTF-8 BYTE offsets while tokenize / str indexing are
    char-based — without the conversion, a non-ASCII docstring over-blanks
    and erases the semantic code beside it from the hash."""
    import ast

    def _char_col(lineno: int, byte_col: int) -> int:
        return len(lines[lineno - 1].encode("utf-8")[:byte_col].decode("utf-8"))

    spans = []
    # Only a Module / ClassDef / FunctionDef holds a docstring, and Python's
    # grammar puts no statement inside an expression — so no def or class is
    # reachable through an `ast.expr`, and an expression subtree cannot hold
    # a docstring. Descending into one only costs: `ast.walk` visited all
    # 523k AST nodes of the engine trees to find 3,028 docstrings, where the
    # statements alone are 58k. Order is not part of the answer —
    # `normalized_py_source` only ever blanks these spans, and blanking is
    # order-independent.
    pending = [tree]
    while pending:
        node = pending.pop()
        pending.extend(child for child in ast.iter_child_nodes(node)
                       if not isinstance(child, ast.expr))
        if not isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        body = node.body
        if (
            body
            and isinstance(body[0], ast.Expr)
            and isinstance(body[0].value, ast.Constant)
            and isinstance(body[0].value.value, str)
        ):
            e = body[0].value
            spans.append((
                (e.lineno, _char_col(e.lineno, e.col_offset)),
                (e.end_lineno, _char_col(e.end_lineno, e.end_col_offset)),
            ))
    return spans


def normalized_py_source(src: str) -> str:
    """Semantic form of a Python source file: comments, docstrings, trailing
    whitespace and blank lines removed. Two files that differ only in those
    normalize identical — so a comment typo fix or a reformat no longer
    invalidates every cached engine result, while any change to names,
    literals, or structure still does. Lines that are (partly) inside a
    multi-line STRING LITERAL are content, not formatting: their text is
    kept byte-exact (a blank line inside ``X = \"\"\"a\\n\\nb\"\"\"`` is a
    different value than without it).

    Version-stable by construction: positions come from ``ast.parse`` +
    ``tokenize`` of the same source on the one supported interpreter
    (``requires-python``), and both agree on docstring spans and COMMENT
    tokens — a hard requirement now that cache keys are shared across
    machines over the Supabase tier. Unparseable source falls back to the raw bytes (a
    broken file still hashes distinctly)."""
    import ast
    import io
    import tokenize

    try:
        tree = ast.parse(src)
        raw_lines = src.splitlines()
        spans = _docstring_spans(tree, raw_lines)
        doc_spans = set(spans)
        # Interior of a multi-line string literal: those lines' whitespace IS
        # the value. Covers plain strings and f-strings (which lex as
        # FSTRING_START…FSTRING_END).
        protected: set = set()
        fstring_start = None
        for tok in tokenize.generate_tokens(io.StringIO(src).readline):
            if tok.type == tokenize.COMMENT:
                spans.append((tok.start, tok.end))
            elif tok.type == tokenize.STRING:
                if (tok.start, tok.end) not in doc_spans and tok.start[0] < tok.end[0]:
                    # From the opening line (its tail is literal content) to
                    # the line before the closing quotes.
                    protected.update(range(tok.start[0], tok.end[0]))
            elif tok.type == tokenize.FSTRING_START:
                fstring_start = tok.start[0]
            elif tok.type == tokenize.FSTRING_END and fstring_start is not None:
                if fstring_start < tok.end[0]:
                    protected.update(range(fstring_start, tok.end[0]))
                fstring_start = None
    except (SyntaxError, ValueError, tokenize.TokenError, IndentationError):
        return "\x00raw\x00" + src

    # Spanned text becomes spaces (so a docstring or comment contributes
    # nothing but the layout around it). Only the lines a span touches are
    # rewritten: of the engine's 105k lines, 21k are docstring and the rest
    # are returned as read, so splitting every line into a list of single
    # characters to blank those did the work of 4.9 MB to edit 0.8 MB.
    blanked: Dict[int, str] = {}
    for (l1, c1), (l2, c2) in spans:
        for ln in range(l1, l2 + 1):
            text = blanked.get(ln)
            if text is None:
                text = blanked[ln] = raw_lines[ln - 1]
            lo = c1 if ln == l1 else 0
            # Blanking never changes a line's length, so `len(text)` is the
            # raw line's length however many spans already landed on it.
            hi = min(c2 if ln == l2 else len(text), len(text))
            if hi > lo:
                blanked[ln] = text[:lo] + " " * (hi - lo) + text[hi:]
    out = []
    for i, raw in enumerate(raw_lines):
        text = blanked.get(i + 1, raw)
        if (i + 1) in protected:
            out.append(text)  # literal content — byte-exact, even blank
            continue
        stripped = text.rstrip()
        if stripped:
            out.append(stripped)
    return "\n".join(out)


#: `backend/`, under which the engine trees live.
_ROOT = pathlib.Path(__file__).resolve().parent


def _engine_files(suffixes: Tuple[str, ...]) -> Iterator[pathlib.Path]:
    """Every file of the engine trees with one of `suffixes`, in one order."""
    for sub in ("inference", "arch"):
        for p in sorted((_ROOT / sub).rglob("*")):
            if p.suffix in suffixes and p.is_file():
                yield p


@lru_cache(maxsize=1)
def _code_chunks() -> Tuple[bytes, ...]:
    """What `_hash_code` hashes, in order: each engine `.py`'s path and its
    normalized source. Normalizing parses every file (seconds), and both
    `engine_code_version` and `engine_version` hash it: read once per
    process, like the versions themselves."""
    chunks = []
    for p in _engine_files((".py",)):
        chunks.append(p.relative_to(_ROOT).as_posix().encode())
        raw = p.read_bytes()
        try:
            chunks.append(normalized_py_source(raw.decode("utf-8")).encode())
        except UnicodeDecodeError:
            # A non-UTF-8 engine file (encoding cookie) still hashes
            # distinctly — semantics we cannot parse we take verbatim.
            chunks.append(b"\x00raw-bytes\x00" + raw)
    return tuple(chunks)


def _hash_code(h: Any) -> None:
    for chunk in _code_chunks():
        h.update(chunk)


def _hash_data(h: Any, skipped: FrozenSet[str]) -> None:
    for p in _engine_files((".yaml", ".yml", ".csv")):
        rel = p.relative_to(_ROOT)
        if rel.parts[1] in skipped:
            continue
        h.update(rel.as_posix().encode())
        h.update(p.read_bytes())


@lru_cache(maxsize=1)
def engine_code_version() -> str:
    """Semantic hash of the engine's code alone: the normalized source of
    every `.py` under `backend/inference` and `backend/arch`. Global by
    nature — code is shared, and which functions one model's run reaches is
    not something a key can enumerate."""
    h = hashlib.sha256()
    _hash_code(h)
    return h.hexdigest()[:16]


@lru_cache(maxsize=1)
def shared_data_version() -> str:
    """Hash of the engine data every model reads, or may read in a way its
    spec does not name: the dims / roles / cache-family / migration
    registries, the quant schemes (a request's dtype override can reach any
    of them) and the GPU table. Everything outside `ModelFootprint.DATA_DIRS`
    lands here, so a new data file is shared until shown otherwise."""
    h = hashlib.sha256()
    _hash_data(h, ModelFootprint.DATA_DIRS)
    return h.hexdigest()[:16]


@lru_cache(maxsize=1)
def engine_version() -> str:
    """Semantic hash of the estimation/rendering engine: normalized source of
    every ``.py`` under ``backend/inference`` and ``backend/arch`` (see
    ``normalized_py_source``) plus the raw bytes of the shared engine data
    files. Any result-affecting edit (a new cost term, a KV formula, a walk
    fix, a component/dims/binding data change) flips this, so every cached
    result keyed on it is transparently superseded — while comment,
    docstring and formatting edits no longer do. A *genuinely global*
    signal: engine changes affect every model. Computed once per process —
    the source can't change under a live server, and ``ha restart``
    re-imports and recomputes."""
    h = hashlib.sha256()
    _hash_code(h)
    # Engine data (component library, dims/roles, quant schemes, fusion
    # plans, GPU table). Per-model specs (models/, snapshots/) invalidate one
    # model at a time via spec_fingerprint.
    _hash_data(h, frozenset({"models", "snapshots"}))
    return h.hexdigest()[:16]


def spec_fingerprint(resolved: Any) -> str:
    """Per-model spec identity: a content hash of a *resolved* spec — its own
    params + inlined components, config-independent. Identical specs collide (as
    they should); a refined or regenerated spec for one model flips only that
    model's fingerprint, so a new spec never busts another model's cache. Live:
    it reflects the loader's current state, so after an
    ``invalidate_default_loader()`` + re-match it moves on its own."""
    spec = resolved.spec if isinstance(resolved, ResolvedSpec) else resolved
    if spec is None:
        # A model with no spec has no spec fingerprint. Returning a sentinel
        # made "no spec" a cache key — and a transient failure upstream
        # (HF timeout) that left `resolved` None got persisted as an error
        # entry under it. The caller decides what to do without a key.
        raise ValueError("spec_fingerprint: no spec to fingerprint")
    # A spec that does not serialize is a bug in the schema, not a reason to
    # hash `repr(resolved)` (which carries an object address: a permanent miss).
    # `source_path` fields are absolute CHECKOUT paths — stripping them makes
    # the fingerprint reproducible across machines, which is the whole point
    # of the shared remote cache tier (CI warms, every deployment hits).
    blob = json.dumps(
        strip_machine_paths(spec.model_dump(mode="json")), sort_keys=True, default=str)
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


@dataclass(frozen=True)
class ModelFootprint:
    """The engine data one model's results read, beyond the code and the
    shared data: its spec, the decode twin its spec names, the fusion plans
    that can match its graph, and the binding entries for those plans.

    Each is enumerated from the resolved spec alone, and each inclusion is
    sound only because what is left out cannot reach the model:

    - the spec's `ArchSpec` carries every component it binds (the loader
      copies the transitive closure in) and their claims (`axioms`); the one
      component a run reads besides is a `decode_variant` twin, fetched from
      the library at decode (a sparse attention's at prefill too — the same
      twin, `phase_form`), so the decode-rebound spec is hashed too;
    - a plan fires only where every pattern node's op is a leaf of the
      program it is matched against (`eval.fusion`), and that program is the
      walked one — the spec or its twin, lowered, with the sampled head
      spliced — plus the collectives layout inserts, the activation casts
      `materialize_casts` grafts, and the fused ops a rewrite writes
      (`ops`). A plan with a node no op of `ops` satisfies cannot fire here,
      and neither can its binding entries.
    """

    resolved: Any

    #: The engine data directories whose files reach a result only through
    #: what this class hashes: specs and snapshots, the component library
    #: and its claims, fusion plans, framework bindings.
    DATA_DIRS: ClassVar[FrozenSet[str]] = frozenset({
        "models", "snapshots", "components", "claims", "fusion_plans",
        "framework_bindings"})

    @cached_property
    def specs(self) -> Tuple[Any, ...]:
        """The spec, and its decode form when it declares one."""
        twin = rebind_decode_variants(self.resolved)
        return (self.resolved,) if twin is None else (self.resolved, twin)

    @cached_property
    def walked_ops(self) -> FrozenSet[str]:
        """The leaf ops of every program this model's runs walk."""
        out = set()
        for spec in self.specs:
            for walked in (spec, splice_sampled_head(spec)):
                if walked is None:
                    continue
                program = Program.of(walked).lowered()
                out.update(node.op for node in program.nodes()
                           if node.op and not node.children)
        return frozenset(out)

    @cached_property
    def ops(self) -> FrozenSet[str]:
        """Every op a program this model is fused on can hold: the walked
        leaves, the collectives layout inserts, the activation casts and
        stored-transform ops a deployment can graft, and the fused ops a
        rewrite writes."""
        return (self.walked_ops | frozenset(FUSION_SCHEMES)
                | frozenset(name for name, spec in OP_REGISTRY.items()
                           if spec.is_comm_op)
                | GRAFTED_OPS)

    def admits(self, scheme: Any) -> bool:
        """Whether every node of `scheme`'s pattern has an op in `ops`."""
        pattern = scheme.naive_pattern
        forms = ((pattern.parallel,) if pattern.parallel is not None
                 else tuple(pattern.nodes.values()))
        return all(({form.op} if form.op else set(form.op_choices))
                   & self.ops for form in forms)

    @cached_property
    def plans(self) -> Dict[str, Any]:
        """The fusion plans that can match this model, as loaded."""
        return {name: scheme.model_dump(mode="json")
                for name, scheme in sorted(FUSION_SCHEMES.items())
                if self.admits(scheme)}

    @cached_property
    def bindings(self) -> Dict[str, Any]:
        """Per framework, its audit tag and the entries this model can
        reach, in the order the binding tries them. An entry is reached when
        its plan can match, or when a walked leaf's op already is its fused
        op (a row is labelled fused by that name)."""
        out: Dict[str, Any] = {}
        for framework, binding in sorted(FRAMEWORK_BINDINGS.items()):
            entries = _binding_entries(framework)
            out[framework] = {
                "audited_at": [binding.audited_at.repo,
                               binding.audited_at.tag,
                               binding.audited_at.commit],
                "entries": [[op, entries[op]]
                            for op in binding.fused_ops()
                            if op in self.plans or op in self.walked_ops],
            }
        return out

    @cached_property
    def digest(self) -> str:
        blob = json.dumps({
            "specs": [spec_fingerprint(spec) for spec in self.specs],
            "plans": self.plans,
            "bindings": self.bindings,
        }, sort_keys=True, default=str)
        return hashlib.sha256(blob.encode()).hexdigest()[:16]


@lru_cache(maxsize=None)
def _binding_entries(framework: str) -> Dict[str, Any]:
    """One binding file's entries as parsed: comments do not move them.

    A list-valued op is the multi-implementation spelling and stays a list in
    the fingerprint (including its order). The common single mapping remains
    a mapping, so existing footprints do not move merely because the loader
    learned the list spelling.
    """
    path = pathlib.Path(_FRAMEWORK_BINDINGS_DIR) / f"{framework}.yaml"
    with path.open(encoding="utf-8") as f:
        return {
            op: (raw if isinstance(raw, list) and len(raw) > 1
                 else (raw[0] if isinstance(raw, list) and raw else raw))
            for op, raw in ((load_yaml(f) or {}).get("op_implementations")
                            or {}).items()
        }


#: A model's footprint digest by what it is a function of: the fingerprints
#: of its spec and decode twin, and the config and dims its program is built
#: at. The registries it also reads are fixed for the process. Bounded: a
#: server resolves a fresh spec object per request.
_FOOTPRINTS: Dict[str, str] = {}
_FOOTPRINTS_MAX = 4096


def result_version(resolved: Any) -> str:
    """The engine identity one model's cached result is keyed on: the code,
    the shared data, and that model's own footprint (`ModelFootprint`)."""
    footprint = ModelFootprint(resolved)
    inputs = hashlib.sha256(json.dumps([
        [spec_fingerprint(spec) for spec in footprint.specs],
        resolved.config, resolved.dim_values,
    ], sort_keys=True, default=str).encode()).hexdigest()
    digest = _FOOTPRINTS.get(inputs)
    if digest is None:
        if len(_FOOTPRINTS) >= _FOOTPRINTS_MAX:
            _FOOTPRINTS.clear()
        digest = _FOOTPRINTS[inputs] = footprint.digest
    return hashlib.sha256("\x00".join((
        engine_code_version(), shared_data_version(),
        digest)).encode()).hexdigest()[:16]
