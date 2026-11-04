"""Dimension / view resolution — config + bindings → dim values / view values.

Extracted from loader.py, which mixed five concerns; this is the pure
config-walk alone, plus the one binding form that reads the checkpoint's
tensor header instead (`CheckpointHeader`). Loading / matching / synthesis
stay in loader.py, which imports resolve_dimensions / resolve_view from here.
One-directional: this layer depends on dims, eval.dim and hf.metadata, none
of which import it.
"""
from __future__ import annotations

import re
from typing import Any, Callable, Dict, Mapping, Optional, Tuple

import backend.arch.dims as dims

# Dim expressions evaluate in `eval.dim`. This module resolves config
# bindings against that funnel; it does not re-export the name.
from backend.arch.eval.dim import eval_dim_expr, names_of
# Bound as a module, not `from … import`: hf.metadata imports
# arch.quant_schemes, whose package init loads the loader, which imports this
# module — so whichever side is imported first sees the other half-built, and
# only attribute access at call time is safe in both orders.
import backend.hf.metadata as hf_metadata
from backend.arch.dtypes import canonical
from backend.arch.quant_schemes import (
    QUANT_SCHEMES, is_quantization_metadata)
from backend.hf.header import SafetensorsHeader


def config_path_value(path: str, config: Dict[str, Any]) -> Optional[Any]:
    """Walk `a.b.c` through nested dicts (or attributes) in their native
    type; None when any step is missing. The one walker behind both `dims:`
    (which then requires a number) and `view:` (which keeps the value).

    A numeric step indexes a list: `engram_num_embeddings.1` is the second
    entry of a per-layer list, so a value the config publishes per layer is
    read from it rather than copied into the spec as a number."""
    obj: Any = config
    for part in path.split("."):
        if isinstance(obj, (list, tuple)):
            obj = obj[int(part)] if part.isdigit() and int(part) < len(obj) else None
        elif isinstance(obj, dict):
            obj = obj.get(part)
        else:
            obj = getattr(obj, part, None)
        if obj is None:
            return None
    return obj


def resolve_view(view_bindings: Dict[str, str], config: Dict[str, Any]) -> Dict[str, Any]:
    """The spec's `view:` block against one config: for each key, the first
    config path with a value, kept in its native type (`rope_scaling` is a
    dict, `hidden_act` a string, `norm.eps` a float). A key none of whose
    paths this config publishes is absent from the result — the view is
    display, and a family spec names the paths its models publish, not a
    promise that every checkpoint has them."""
    out: Dict[str, Any] = {}
    for key, binding in view_bindings.items():
        for part in str(binding).split("??"):
            ref = part.strip()
            value = config_path_value(ref[len("config."):], config)
            if value is not None:
                out[key] = value
                break
    return out


_CONFIG_REF = re.compile(r"config\.(?P<path>[A-Za-z0-9_.]+)")


def language_config(dim_bindings: Mapping[str, str], config: Dict[str, Any]
                    ) -> Tuple[str, Dict[str, Any]]:
    """`(path, fields)`: where this config keeps the language model's own
    fields — `("", config)` for a text model, `("text_config",
    config["text_config"])` for a multimodal one. The spec states it: its
    `d_model` (the backbone's hidden size) binds a config path, and the
    mapping that path's field sits in is the backbone's config. Read with
    the binding's own `??` order, as `resolve_dimensions` reads it. A spec
    whose `d_model` reads no config path states no such mapping and is
    refused, naming the fix."""
    binding = str(dim_bindings.get("d_model", ""))
    for part in binding.split("??"):
        ref = _CONFIG_REF.search(part)
        if ref is None or config_path_value(ref["path"], config) is None:
            continue
        scope = ref["path"].rpartition(".")[0]
        fields = config_path_value(scope, config) if scope else config
        if isinstance(fields, dict):
            return scope, fields
    raise ValueError(
        f"dims.d_model: {binding or 'unbound'!s} reads no config field this "
        f"config sets, so the language model's config cannot be located. "
        f"Bind d_model to the config's hidden size (`config.hidden_size`, or "
        f"`config.text_config.hidden_size` for a multimodal config).")


def _resolve_config_ref(name: str, ref: str, config: Dict[str, Any]) -> Optional[Any]:
    """
    Resolve 'config.hidden_size' or 'config.text_config.hidden_size' from config dict.
    Returns None if not found. A value that IS there but is no number (a
    string such as "auto", a list) is not "not found": `??` would put the
    default in place of a fact the config does state, so it raises. A bool
    reads as 0 / 1 — a declared switch.
    """
    if not ref.startswith("config."):
        return None
    obj = config_path_value(ref[len("config."):], config)
    if obj is None:
        return None
    if not isinstance(obj, (int, float)):
        try:
            return int(obj)
        except (TypeError, ValueError):
            raise ValueError(
                f"dims.{name}: {ref} is {obj!r}, not an integer. If this field "
                f"is a string on some checkpoints, bind a path that reads an "
                f"integer, or bind presence with `present({ref})`.") from None
    return obj


#: `present(config.<path>)` — whether the config sets a field, as a 0 / 1
#: switch, for a step the implementation runs when the field `is not None`
#: whatever its value (a soft-cap constant, a scaling factor). The value is
#: not a dim: binding it would claim an integer, and reading "non-zero" off it
#: would drop the step at an explicit 0 that the implementation still runs.
_PRESENT_REF = re.compile(r"present\(config\.(?P<path>[A-Za-z0-9_.]+)\)")

def _config_presence(path: str, config: Dict[str, Any]) -> Optional[int]:
    """1 when the config sets `path` to a value, 0 when it sets it to null,
    None when it does not publish the field at all — so that a `??` fallback
    states the config class's default, exactly as for `config.<path>`."""
    parent_path, _, leaf = path.rpartition(".")
    parent = config_path_value(parent_path, config) if parent_path else config
    if not isinstance(parent, dict) or leaf not in parent:
        return None
    return 0 if parent[leaf] is None else 1


#: `len(config.<path>)` — how many entries a config list holds (Qwen3-VL's
#: `deepstack_visual_indexes`). Only a list counts; a config without the key
#: is unresolved, so a `??` fallback states the default.
_LEN_REF = re.compile(r"len\(config\.(?P<path>[A-Za-z0-9_.]+)\)")


def _config_list_length(name: str, path: str, config: Dict[str, Any]) -> Optional[int]:
    value = config_path_value(path, config)
    if value is None:
        return None
    if not isinstance(value, list):
        raise ValueError(
            f"dims.{name}: len(config.{path}) counts a list, and the config "
            f"sets {path} to {value!r}")
    return len(value)


def _resolve_constant(name: str, binding: str, config: Dict[str, Any],
                      resolved: Mapping[str, Any]) -> Optional[Any]:
    """A `constant`-kind symbol's value: the first `??` alternative that
    gives a number — a config path the config publishes, a literal, or an
    expression over symbols already bound (`vis_clamp_limit: clamp_limit`).
    None when none does: the symbol stays unbound rather than defaulted.

    A real number, kept as stated (`7.5` is a bound, not an extent). A field
    the config DOES set to something that is not a number is not "not
    published", and raises."""
    for part in str(binding).split("??"):
        ref = part.strip().strip("'\"")
        if ref.startswith("config."):
            value = config_path_value(ref[len("config."):], config)
            if value is None:
                continue
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(
                    f"dims.{name}: {ref} is {value!r}, not a number. A "
                    f"`constant` symbol binds a real number the config "
                    f"states.")
            return value
        value = eval_dim_expr(ref, dict(resolved))
        if names_of(value):
            # Reads a constant this config left unbound: unbound too.
            continue
        return int(value) if value == int(value) else float(value)
    return None


#: `checkpoint.<normalized tensor name>[<axis>]` — the one binding form whose
#: value is read from the safetensors header rather than config.json.
_CHECKPOINT_REF = re.compile(r"checkpoint\.(?P<tensor>[A-Za-z0-9_.]+)\[(?P<axis>\d+)\]")


class CheckpointHeader:
    """The checkpoint's tensor header as `dims:` reads it: a tensor name, as
    the spec composes it (`X` at each declared index), → the shape that
    checkpoint stores it at.

    For a tensor whose extent is a fact about the FILE rather than about the
    model — a table the exporting code happened to serialise, whose length
    the modeling code does not determine — the header is the declaration and
    a config formula can only coincide with it. Such a dim is bound
    `checkpoint.<name>[<axis>]` and read here, never derived.

    The header is fetched on first use, so resolving a spec that binds no
    checkpoint dim never touches it. When there is none to read — no model
    id, a custom model without a tensor index, a hub fetch that failed — the
    binding refuses; there is no fallback value, because any number put in
    its place is the guess the binding exists to replace.
    """

    def __init__(self, fetch: Callable[[], Optional[SafetensorsHeader]],
                 source: str):
        self._fetch = fetch
        self._header: Optional[SafetensorsHeader] = None
        self._fetched = False
        #: what the header is, for the refusal message
        self.source = source

    @classmethod
    def absent(cls, reason: str) -> "CheckpointHeader":
        return cls(lambda: None, reason)

    @classmethod
    def of(cls, model_id: Optional[str],
           header: Optional[SafetensorsHeader] = None) -> "CheckpointHeader":
        """The header a caller already holds, else the one `fetch_metadata`
        reads for `model_id` (custom index → snapshot → hub, the same order
        every other consumer of the header uses)."""
        if header is not None:
            return cls(lambda: header,
                       f"tensor metadata supplied for {model_id or 'this resolution'}")
        if model_id:
            return cls(
                lambda: hf_metadata.fetch_metadata(model_id).get("header"),
                f"{model_id}'s safetensors header")
        return cls.absent("resolved without a model id or tensor metadata")

    def unquantised_float_dtypes(self, quant_context: Any
                                 ) -> Dict[str, int]:
        """`{dtype: stored elements}` over the tensors this checkpoint holds
        at a plain float dtype (F64 / F32 / F16 / BF16) that no quant scheme
        of `quant_context` claims: not a scheme's metadata sidecar (scale,
        zero point), not stored at an integer or a
        quantised float (an FP8 / FP4 / packed weight is the scheme's).
        Empty when there is no header to read."""
        header = self._read()
        if not header:
            return {}
        methods = set()
        if quant_context is not None:
            methods |= {e.method for e in quant_context.per_role.values()
                        if e.method}
            methods |= {e.method for e in quant_context.overrides if e.method}
        schemes = [QUANT_SCHEMES[m] for m in methods if m in QUANT_SCHEMES]
        schemes += [QUANT_SCHEMES[other] for scheme in list(schemes)
                    for _field, _value, other in scheme.also_stores]
        out: Dict[str, int] = {}
        for group in header:
            name = group.template
            if any(is_quantization_metadata(name, s) for s in schemes):
                continue
            # An unknown spelling raises: no dtype is guessed past.
            dtype = canonical(str(group.dtype))
            if dtype not in ("F64", "F32", "F16", "BF16"):
                continue
            out[dtype] = out.get(dtype, 0) + int(group.elements)
        return out

    def _read(self) -> Optional[SafetensorsHeader]:
        if not self._fetched:
            self._header = self._fetch()
            self._fetched = True
        return self._header

    def extent(self, symbol: str, tensor: str, axis: int) -> int:
        """Axis `axis` of the one shape `tensor` is stored at."""
        header = self._read()
        if not header:
            raise ValueError(
                f"dims.{symbol} is bound to checkpoint tensor {tensor!r}, and "
                f"there is no tensor metadata to read it from ({self.source}). "
                f"The binding has no fallback: its value is whatever the "
                f"checkpoint stores.")
        selection = header.select(tensor)
        if not selection:
            raise ValueError(
                f"dims.{symbol} is bound to checkpoint tensor {tensor!r}, which "
                f"this checkpoint does not store ({self.source}; "
                f"{header.tensor_count} tensors).")
        shapes = selection.shapes
        if len(shapes) != 1:
            raise ValueError(
                f"dims.{symbol}: checkpoint tensor {tensor!r} is stored at "
                f"{len(shapes)} shapes {list(shapes)}; a dim is one extent.")
        (shape,) = shapes
        if axis >= len(shape):
            raise ValueError(
                f"dims.{symbol}: checkpoint tensor {tensor!r} has shape "
                f"{list(shape)}, no axis {axis}.")
        return int(shape[axis])


def resolve_dimensions(
    dim_bindings: Dict[str, str],
    config: Dict[str, Any],
    checkpoint: Optional[CheckpointHeader] = None,
    local: Optional[Mapping[str, Any]] = None,
) -> Dict[str, int]:
    """
    Resolve all dimension bindings, each after the dims it reads (a
    mapping's written order carries no meaning; among bindings whose inputs
    are ready it is kept). Every result is a whole number.

    Supports:
      - "config.hidden_size"          → direct config lookup
      - "config.head_dim ?? 'd_model // n_heads'"
                                      → try config, fallback to expression
      - "d_model // n_heads"          → pure expression over already-resolved dims
      - "128"                         → literal integer
      - "present(config.field) ?? 1"  → 1 when the config sets the field, 0
                                        when it sets null, else the fallback
      - "checkpoint.<tensor>[<axis>]" → that axis of the shape `checkpoint`
                                        stores the named tensor at; no `??`
      - "len(config.field) ?? 0"      → how many entries the config list
                                        holds, else the fallback
    """
    if checkpoint is None:
        checkpoint = CheckpointHeader.absent("resolved without a checkpoint")
    # `local`: the spec's own `dim_symbols`. A spec may bind dims.yaml's
    # symbols and its own, never another loaded spec's.
    known = dims.registered_for(local)
    unregistered = [n for n in dim_bindings if n not in known]
    if unregistered:
        raise ValueError(
            f"dims: binds symbol(s) {unregistered} that neither "
            f"backend/arch/dims.yaml nor this spec's `dim_symbols:` classifies. "
            f"A `dims:` entry says which config field gives a symbol its "
            f"VALUE; what the symbol IS — what it counts, which parallelism "
            f"may divide it — is a row. Use the dims.yaml symbol with the "
            f"same meaning if there is one; otherwise declare it under "
            f"`dim_symbols:` in this spec with `kind`, `shardable_by` (omit "
            f"for `whole`), `group` and `meaning`, the fields of a dims.yaml "
            f"row, grounded in the forward source."
        )

    resolved: Dict[str, int] = {}
    # Symbols of kind `constant` (a clamp bound, an activation's beta) are
    # real numbers, not extents: kept as the config states them rather than
    # held to a whole number, and a config that does not publish one leaves
    # it unbound — absent, not a default, so a consumer that has to compute
    # the function declines instead of computing another one.
    constants = {name for name, sym in {
        **dims.base_symbols(), **dims.local_symbols(local or {})}.items()
        if sym.kind == "constant"}

    for name, binding in _dependency_order(dim_bindings):
        if name in constants:
            value = _resolve_constant(name, binding, config, resolved)
            if value is not None:
                resolved[name] = value
            continue
        value = _resolve_single_dim(name, binding, config, resolved, checkpoint)
        if value is not None:
            resolved[name] = _whole(name, binding, value)
        else:
            raise ValueError(
                f"Cannot resolve dimension {name!r} from binding {binding!r}. "
                f"Already resolved: {resolved}"
            )

    _apply_derived_defaults(resolved, local)
    _refuse_out_of_bound(resolved, dim_bindings, local)
    return resolved


def _refuse_out_of_bound(resolved: Dict[str, int], dim_bindings: Mapping[str, str],
                         local: Optional[Mapping[str, Any]]) -> None:
    """Refuse a model whose dims break a bound the registry declares
    (`at_most:`): a top-k that picks more than it chooses among is not a
    model to evaluate, and evaluated anyway it is some other model."""
    rows = {**dims.base_symbols(), **dims.local_symbols(local or {})}
    problems = []
    for name, sym in rows.items():
        if not sym.at_most or name not in resolved \
                or not all(r in resolved for r in sym.bound_reads):
            continue
        bound = int(eval_dim_expr(sym.at_most, resolved))
        if resolved[name] > bound:
            written = dim_bindings.get(name)
            problems.append(
                f"`{name}` is {resolved[name]}"
                f"{f' (bound as `{written}`)' if written else ''}, more than "
                f"`{sym.at_most}` = {bound}: {sym.meaning.strip()} It is at "
                f"most `{sym.at_most}`; check the config fields both are "
                f"bound to")
    if problems:
        raise ValueError("dims: " + "; ".join(problems))


#: A config / checkpoint path inside a binding: its segments name fields, not
#: dims (`config.vision_config.hidden_size` does not read a dim `hidden_size`).
_PATH_REF = re.compile(r"\b(?:config|checkpoint)\.[A-Za-z0-9_.]+")


def references_config(binding: Any) -> bool:
    """Whether an authored dim binding reads a `config.` field.

    The one recogniser for a config path inside a binding (`_PATH_REF`,
    the grammar the resolution itself reads), so a display that lists the
    config-bound dims agrees with what resolution treated as a config read:
    `fooconfig.x` is not one, and a `checkpoint.` path is a different source.
    """
    return any(path.startswith("config.")
               for path in _PATH_REF.findall(str(binding)))


def _dependency_order(dim_bindings: Dict[str, str]):
    """The bindings in an order where each comes after the dims it reads.

    `dims:` is a mapping, not a program: a spec normalized with sorted keys
    lists `d_vis_merge: n_spatial_merge * n_spatial_merge * d_vis_model`
    before `d_vis_model`, and resolving in written order left it a free
    symbol. Among the ready bindings the written order is kept, so a spec
    already in dependency order resolves exactly as before. A cycle raises."""
    pending = dict(dim_bindings)
    reads = {
        name: {ident for ident in _IDENT.findall(_PATH_REF.sub(" ", str(b)))
               if ident in dim_bindings}
        for name, b in dim_bindings.items()
    }
    done: set = set()
    while pending:
        ready = [name for name in pending if reads[name] <= done]
        if not ready:
            raise ValueError(
                f"dims: {sorted(pending)} read each other in a cycle")
        for name in ready:
            done.add(name)
            yield name, pending.pop(name)


def _whole(name: str, binding: str, value: Any) -> int:
    """A dim's value as an int. An expression over a name nothing bound
    evaluates to a free symbol rather than failing; kept, it surfaced later as
    a projection error far from the binding that caused it."""
    try:
        whole = int(value)
    except TypeError:
        raise ValueError(
            f"dims.{name}: {binding!r} reads {value!r}, which names a dim no "
            f"binding gives a value") from None
    if whole != value:
        raise ValueError(f"dims.{name}: {binding!r} is {value}, not a whole number")
    return whole


# What an UNSTATED dim means. A property of the spec language, so it is settled
# once here — where `resolved.dim_values` is built — rather than by each consumer
# that needs a value the model happened not to write down.
#
# `n_expert_copies` was already defaulted here, with the reason spelled out: so
# "every path that reads `resolved.dim_values` (spec_tree / materialize / weight
# breakdown, not just the engine)" sees it. That reason covers all of them, and
# the rest had drifted into `SpecEngine.__init__` — which means any consumer that
# is not the engine saw a different dim environment for the same model.
# Every value a model need not write — a default (`Qb` = 0, `d_ffn_shared` =
# `d_ffn_expert`) or a derivation no spec may write (`n_route_targets` =
# `n_experts + n_zero_experts`) — is written in `dims.yaml` as `default:` or
# `derived:`, beside the symbol it belongs to, and read from there. A second list here would be a
# claim about the same symbol that nothing checks against the first, which is how
# `n_route_targets` came to be documented as `n_experts + n_zero_experts` while
# resolving to nothing at all. A literal is just a derivation with no inputs
# (`n_zero_experts: "0"`), ready on the first fixpoint pass.
#
# These MATERIALISE, they do not track: the expression is evaluated once, and
# the value thereafter stands alone. That is what lets expert parallelism
# divide `n_expert_copies` while the counts it derives from stay global — the
# dispatch denominator `n_route_targets` in
# `tokens_per_expert = B*s_ctx*n_experts_active*dp/n_route_targets`, and
# `n_experts` itself, which is what the model HAS rather than an extent.
_IDENT = re.compile(r"[A-Za-z_][A-Za-z_0-9]*")

#: What a derived expression may name that is not a dim — the sympy functions
#: `eval_dim_expr` understands. Everything else in one is a symbol, and the
#: derivation waits until that symbol has a value.
_DIM_EXPR_FUNCTIONS = frozenset({
    "ceiling", "Min", "Max", "floor", "min", "max", "minimum", "maximum",
    "sqrt", "abs",
})


def _apply_derived_defaults(resolved: dict,
                            local: Optional[Mapping[str, Any]] = None) -> None:
    """Fill in every unstated symbol `dims.yaml` defaults or derives.

    To a fixpoint, since one derivation may feed another: `n_route_targets`
    reads `n_zero_experts`, which most models do not state either. A derivation
    whose inputs the model never resolves is simply not
    that model's symbol, and is skipped rather than failed.
    """
    own = dims.local_symbols(local) if local else {}
    pending = {
        name: sym.when_unbound
        for name, sym in {**dims.base_symbols(), **own}.items()
        if sym.when_unbound and name not in resolved
    }
    while pending:
        ready = {
            name: expr for name, expr in pending.items()
            if all(
                ident in resolved
                for ident in _IDENT.findall(expr)
                if ident not in _DIM_EXPR_FUNCTIONS
            )
        }
        if not ready:
            return
        for name, expr in ready.items():
            resolved[name] = int(eval_dim_expr(expr, resolved))
            del pending[name]


def _resolve_single_dim(
    name: str, binding: str, config: Dict[str, Any], resolved: Dict[str, int],
    checkpoint: CheckpointHeader,
) -> Optional[int]:
    """Resolve a single dimension binding."""
    binding = binding.strip()

    if "checkpoint." in binding:
        ref = _CHECKPOINT_REF.fullmatch(binding)
        if ref is None:
            raise ValueError(
                f"dims.{name}: {binding!r} — a checkpoint binding is "
                f"`checkpoint.<normalized tensor name>[<axis>]` and nothing "
                f"else: no `??` (a fallback is a default for a fact only the "
                f"file has) and no arithmetic (bind the extent to its own "
                f"symbol and derive from that).")
        return checkpoint.extent(name, ref["tensor"], int(ref["axis"]))

    presence = _PRESENT_REF.fullmatch(binding)
    if presence is not None:
        return _config_presence(presence["path"], config)

    length = _LEN_REF.fullmatch(binding)
    if length is not None:
        return _config_list_length(name, length["path"], config)

    # Check for fallback operator ??
    if "??" in binding:
        parts = binding.split("??", 1)
        primary = parts[0].strip()
        fallback = parts[1].strip().strip("'\"")

        # Try primary
        val = _resolve_single_dim(name, primary, config, resolved, checkpoint)
        if val is not None:
            return val

        # Try fallback
        return _resolve_single_dim(name, fallback, config, resolved, checkpoint)

    # Direct config reference
    if binding.startswith("config."):
        val = _resolve_config_ref(name, binding, config)
        if val is None:
            return None
        # A dimension is a whole number. `int()` would turn a fractional
        # config value into a different one without a word — 32.5 heads into
        # 32, a 0.25 rotary factor into "no rotary". A continuous constant
        # that gates a step is bound by presence (`present(config.x)`), not
        # as a dim.
        if isinstance(val, float) and not val.is_integer():
            raise ValueError(
                f"dims.{name}: {binding} is {val!r}, not a whole number. A "
                f"dimension is an integer; truncating it to {int(val)} would "
                f"state a different fact. Bind the integer this value "
                f"determines (e.g. an expression over whole-number dims).")
        return int(val)

    # Literal integer
    try:
        return int(binding)
    except (ValueError, TypeError):
        pass

    # Expression referencing already-resolved dims
    try:
        return eval_dim_expr(binding, resolved)
    except (ValueError, TypeError):
        return None
