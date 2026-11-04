"""Parse a transformer block's forward() method into a canonical op list.

Pure static analysis — no model loading, no runtime. Walks the Python
AST, resolves variable dataflow (`residual = x; x = f(x); x = residual + x`)
into explicit SSA-style inputs on the emitted ops.

Depends only on stdlib + `.types.StructOp`. Knows nothing about the
spec side of the comparison.
"""
from __future__ import annotations

import ast
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Tuple

from .types import StructOp


def locate_layer_forward(
    files: List[Dict[str, str]],
    config: Mapping[str, Any],
    auto_class: Callable[[str], Optional[str]],
) -> Tuple[Optional[ast.FunctionDef], Tuple[str, ...], Optional[str]]:
    """The `forward` of the repeated decoder layer, located by following the
    model's own construction from its declared top class.

    The start is a declaration: `config.architectures[0]`, else a
    `config.auto_map` model entry whose module is one of `files`. From there
    the walk reads `__init__` bodies: a submodule built from the same config
    (`config`, or `config.text_config` — the text tower the spec describes;
    a vision tower is built from another sub-config and is not followed),
    either directly (`Cls(config)`, `Cls._from_config(config)`) or through
    a factory (`AutoModel.from_config(cfg)`: the class `cfg.auto_map` declares
    for that factory, else `auto_class(cfg.model_type)` from the transformers
    registry; neither naming one stops the walk), until an `nn.ModuleList` of a layer class
    (`ModuleList([Layer(...) for ...])`, `ModuleList(Layer(...) for ...)`) or
    a `self.<list>.append(Layer(...))` loop. A base class whose construction
    lives in source that is not among `files` stops the walk: its `__init__`
    may build the very list this one is looking for.

    Returns `(forward, trail, stop)`: `trail` is every step taken, from the
    declaration on; `stop` is None when `forward` was found and otherwise
    the reason the walk could not continue."""
    classes: Dict[str, Tuple[ast.ClassDef, Dict[str, Tuple[str, int]]]] = {}
    paths: Dict[str, str] = {}
    for f in files:
        try:
            tree = ast.parse(f.get("text", ""))
        except SyntaxError:
            continue
        imports = {alias.asname or alias.name: (node.module or "", node.level)
                   for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)
                   for alias in node.names}
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef) and node.name not in classes:
                classes[node.name] = (node, imports)
                paths[node.name] = f.get("path", "")

    trail: List[str] = []
    start = None
    archs = config.get("architectures") or []
    if archs and archs[0] in classes:
        start = archs[0]
        trail.append(f"config.architectures[0] = {start}")
    else:
        stems = {Path(p).stem for p in paths.values() if p}
        for key, ref in (config.get("auto_map") or {}).items():
            if not (key.startswith("AutoModel") and isinstance(ref, str)):
                continue
            module, _, cls = ref.rpartition("--")[2].rpartition(".")
            if module in stems and cls in classes:
                start = cls
                trail.append(f"config.auto_map.{key} = {ref}")
                break
    if start is None:
        return None, tuple(trail), (
            f"neither config.architectures[0] ({archs[0] if archs else None!r}) "
            f"nor a config.auto_map model class is defined in these files")

    def _method(cls_name: str, name: str):
        cls, _ = classes[cls_name]
        return next((item for item in cls.body
                     if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef))
                     and item.name == name), None)

    def _bases(cls_name: str) -> List[str]:
        cls, _ = classes[cls_name]
        return [ast.unparse(b) for b in cls.bases]

    def _unseen_base(cls_name: str) -> Optional[str]:
        """A base whose `__init__` is in another model's modeling source:
        imported from a `models.` package, a sibling model package
        (`..llama.modeling_llama`), or a sibling file of the repo
        (`.modeling_x`) — none of which is among `files`."""
        _, imports = classes[cls_name]
        for base in _bases(cls_name):
            if base in classes or base not in imports:
                continue
            module, level = imports[base]
            if (level == 0 and ".models." in f".{module}.") or level in (1, 2):
                return f"{base} (from {'.' * level}{module})"
        return None

    def _config_arg(call: ast.Call) -> Optional[str]:
        args = [*call.args[:1], *(kw.value for kw in call.keywords
                                   if kw.arg == "config")]
        return ast.unparse(args[0]) if args else None

    def _constructed(call: ast.Call, cfg: Mapping[str, Any]
                     ) -> Tuple[Optional[str], Optional[str]]:
        """`(class, stop)`: the class `call` constructs, if it is one this
        walk can name, or why the walk cannot say which class it is.

        A factory (`AutoModel.from_config(cfg)`) builds what `cfg` declares:
        its own `auto_map` entry for that factory when it has an `auto_map`
        (remote code registers itself there, ahead of the library), else the
        transformers registry's class for `cfg.model_type`."""
        func = call.func
        if isinstance(func, ast.Name) and func.id in classes:
            return func.id, None
        if (isinstance(func, ast.Attribute) and isinstance(func.value, ast.Name)
                and func.attr in ("from_config", "_from_config")):
            factory = func.value.id
            if factory in classes:
                return factory, None
            auto_map = cfg.get("auto_map")
            if isinstance(auto_map, Mapping):
                ref = auto_map.get(factory)
                if not isinstance(ref, str):
                    return None, (
                        f"{ast.unparse(call)} builds what its config's auto_map "
                        f"declares, and the auto_map has no {factory!r} entry")
                return ref.rpartition("--")[2].rpartition(".")[2], None
            model_type = cfg.get("model_type")
            target = auto_class(model_type) if model_type else None
            if target is None:
                return None, (
                    f"{ast.unparse(call)}: the transformers registry has no "
                    f"class for model_type {model_type!r}")
            return target, None
        return None, None

    unnamed: List[str] = []

    def _layer_classes(init: ast.FunctionDef) -> List[str]:
        found: List[str] = []
        for node in ast.walk(init):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            name = (func.attr if isinstance(func, ast.Attribute)
                    else func.id if isinstance(func, ast.Name) else "")
            if name not in ("ModuleList", "append"):
                continue
            for arg in node.args:
                elts = (arg.elts if isinstance(arg, (ast.List, ast.Tuple))
                        else [arg.elt] if isinstance(arg, (ast.ListComp, ast.GeneratorExp))
                        else [arg])
                for e in elts:
                    if not (isinstance(e, ast.Call) and isinstance(e.func, ast.Name)):
                        continue
                    if e.func.id in classes:
                        found.append(e.func.id)
                    elif name == "append" or arg is not e:
                        # A variable holding the class (`layer_cls(...)`):
                        # which one it is, is decided at run time.
                        unnamed.append(ast.unparse(e)[:60])
        return found

    layers: List[str] = []
    visited: set = set()
    frontier: List[Tuple[str, Mapping[str, Any]]] = [(start, config)]
    while frontier:
        cls_name, cfg = frontier.pop(0)
        if cls_name in visited:
            continue
        visited.add(cls_name)
        unseen = _unseen_base(cls_name)
        if unseen is not None:
            return None, tuple(trail), (
                f"{cls_name} builds on {unseen}, whose source is not among "
                f"these files")
        frontier.extend((b, cfg) for b in _bases(cls_name) if b in classes)
        init = _method(cls_name, "__init__")
        if init is None:
            continue
        params = [a.arg for a in init.args.args[1:]]
        own = params[0] if params else "config"
        layers.extend(f"{cls_name} → {layer}" for layer in _layer_classes(init))
        for node in ast.walk(init):
            if not (isinstance(node, ast.Assign) and isinstance(node.value, ast.Call)):
                continue
            arg = _config_arg(node.value)
            if arg == own:
                sub = cfg
            elif arg == f"{own}.text_config" and isinstance(cfg.get("text_config"), Mapping):
                sub = cfg["text_config"]
            else:
                continue
            target, why = _constructed(node.value, sub)
            if why is not None:
                trail.append(f"{cls_name}: {ast.unparse(node)[:120]}")
                return None, tuple(trail), why
            if target is None:
                continue
            trail.append(f"{cls_name}: {ast.unparse(node)[:120]}")
            if target not in classes:
                return None, tuple(trail), (
                    f"{target} is not defined in these files")
            frontier.append((target, sub))

    distinct = sorted({step.rsplit(" → ", 1)[1] for step in layers})
    if not distinct:
        return None, tuple(trail), (
            f"no ModuleList / append of a layer class in the classes reached "
            f"({', '.join(sorted(visited))})"
            + (f"; layers built through a variable: {'; '.join(unnamed)}"
               if unnamed else ""))
    if len(distinct) > 1:
        return None, tuple(trail), (
            f"more than one layer list is built: {'; '.join(layers)}")
    layer = distinct[0]
    trail.append(next(step for step in layers if step.endswith(f" → {layer}")))
    for cls_name in [layer, *(b for b in _bases(layer) if b in classes)]:
        fn = _method(cls_name, "forward")
        if fn is not None:
            trail.append(f"{cls_name}.forward ({paths[cls_name]})")
            return fn, tuple(trail), None
    unseen = _unseen_base(layer)
    return None, tuple(trail), (
        f"{layer} defines no forward of its own"
        + (f"; it inherits one from {unseen}, whose source is not among "
           f"these files" if unseen else ""))


def _call_submodule_name(call: ast.Call) -> Optional[str]:
    """If `call` is `self.<name>(...)`, return `<name>`. Else None.
    Rejects chained attribute calls (e.g. `self.norm.forward(...)`)."""
    func = call.func
    if isinstance(func, ast.Attribute) and isinstance(func.value, ast.Name) \
            and func.value.id == "self":
        return func.attr
    return None


def _referenced_names(node: ast.AST) -> List[str]:
    """Collect bare Name identifiers referenced inside `node` (not
    attribute accesses)."""
    return [sub.id for sub in ast.walk(node) if isinstance(sub, ast.Name)]


def _output_name(targets: List[ast.expr]) -> str:
    """Resolve the first bare-name target from an Assign's LHS.
    Handles `x = ...` and `x, _ = ...` (tuple destructuring keeps first)."""
    for t in targets:
        if isinstance(t, ast.Name):
            return t.id
        if isinstance(t, (ast.Tuple, ast.List)):
            for elt in t.elts:
                if isinstance(elt, ast.Name):
                    return elt.id
    return ""


def _classify_assign(value: ast.expr, output: str, line: int) -> Optional[StructOp]:
    """Classify the RHS of an assignment into a StructOp (or None if
    the RHS isn't a structural operation we track)."""
    if isinstance(value, ast.Call):
        submod = _call_submodule_name(value)
        if submod is None:
            return None  # non-self call — ignore
        inputs: List[str] = []
        for arg in value.args:
            inputs.extend(_referenced_names(arg))
        for kw in value.keywords:
            inputs.extend(_referenced_names(kw.value))
        return StructOp(kind="call", name=submod, inputs=inputs,
                         output=output, line=line)

    if isinstance(value, ast.BinOp) and isinstance(value.op, ast.Add):
        return StructOp(
            kind="add",
            inputs=_referenced_names(value.left) + _referenced_names(value.right),
            output=output,
            line=line,
        )

    if isinstance(value, ast.Name):
        return StructOp(kind="snapshot", inputs=[value.id],
                         output=output, line=line)
    return None


def extract_forward_ops(fn: ast.FunctionDef) -> List[StructOp]:
    """Emit canonical StructOps for a layer's `forward` (as located by
    `locate_layer_forward`).

    Handled statement kinds:
      - Assign            (x = expr)
      - AugAssign (+=)    (x += y)
      - Return            (end marker)
      - If / For / Try    (flagged as branch_detected, NOT recursed)
    """
    ops: List[StructOp] = []

    for stmt in fn.body:
        if isinstance(stmt, ast.Assign):
            out = _output_name(stmt.targets)
            op = _classify_assign(stmt.value, out, stmt.lineno)
            if op:
                ops.append(op)
        elif isinstance(stmt, ast.AugAssign) and isinstance(stmt.op, ast.Add):
            if isinstance(stmt.target, ast.Name):
                out = stmt.target.id
                ops.append(StructOp(
                    kind="add",
                    inputs=[out, *_referenced_names(stmt.value)],
                    output=out,
                    line=stmt.lineno,
                ))
        elif isinstance(stmt, ast.Return):
            inp_names = _referenced_names(stmt.value) if stmt.value else []
            ops.append(StructOp(kind="return", inputs=inp_names, line=stmt.lineno))
            break
        elif isinstance(stmt, (ast.If, ast.For, ast.Try)):
            # Only flag when the branch body contains a self.submodule
            # call — that's real architectural heterogeneity. Plain
            # `if mask is not None:` / `try/except` / shape-checking
            # branches carry no structural signal and would drown the
            # warning list in noise.
            if _has_structural_submodule_call(stmt):
                ops.append(StructOp(kind="branch_detected",
                                     name=type(stmt).__name__, line=stmt.lineno))
    return ops


def _has_structural_submodule_call(stmt: ast.AST) -> bool:
    """True if `stmt` (expected to be an If / For / Try) contains at
    least one `self.<name>(...)` call anywhere in its body or orelse."""
    for sub in ast.walk(stmt):
        if isinstance(sub, ast.Call) and _call_submodule_name(sub) is not None:
            return True
    return False
