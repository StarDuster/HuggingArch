"""The scoreboard: the handful of numbers that define "better" for this repo.

A finding list has no end — every deeper audit makes it longer. A counter
does: it has a definition, a starting value, a target, and a one-way gate.
Each counter below is one *shape* of debt this repo has decided is never
acceptable in new code (CLAUDE.md §1/§2), stated as an AST query so a commit
either leaves the number alone or lowers it. "Better" is these numbers going
down; "done" is these numbers at target.

Baselines live in `tests/meta/scoreboard_baseline.json` — declared, not
inferred, and lowered in the same commit that removes an instance
(`python -m tools.scoreboard --rebaseline`). The gate is a one-way ratchet:
new code never raises a baseline. The one exception is a recount — the
counter's definition is corrected to see instances it was missing, and the
baseline is raised once (`--rebaseline --allow-raise`) to the corrected
count, in that same commit, with every newly seen instance accounted for in
its message. The first recount: silent_except 83 -> 88 (handlers that only
log at debug level, and tuples of empty values, were invisible) and
name_lists 54 -> 57 (lists other batches added while the baseline was never
re-measured) are counter-correction backfill, not new debt allowed in. The
second recount — width_defaults 10 -> 23 (dtype fallbacks written in a
function body) and downstream_derivation 0 -> 39 (views/playground out of
scope; dims/config reads, dim-expression evaluation, empty-object parse
fallbacks and op-name-set branches the line regex could not see) — is the
same kind of backfill. The third — undecided_decisions 85 -> 104 — is 14
division sites the counter did not look at (`/` and `//` answer in BOTH
domains and answer differently, which is the same defect as a comparison
that only one domain can answer; every one of the 14 is named with its
reason in `tests/meta/test_symbolic_decisions.py`) plus 5 comparison sites
that accumulated while this baseline was never re-measured — the counter
read 90 against a floor of 85 before that commit. The fourth —
downstream_derivation 0 -> 5 — closes one blind spot that let the count
fall by rewriting rather than by moving a fact: a method whose whole answer
is a read of `dim_values` (`ResolvedSpec.get_dim`, and the `unbound` /
`dims_moved_from` wrappers since deleted) hid the read from a scan of the
attribute. On the code before the fixes that came with it the corrected
counter reads 7 (measured on this branch's base, 2b4dfa0de, with this file
alone taken from the branch); the five that remain are the editor's own two
resolutions, which no walk can answer. Every number here carries the commit
it was measured on, for the reason two of these rows drifted unnoticed
before: a count compared against a tree other than the one it was measured
on says nothing. `tests/meta/test_scoreboard.py`
holds every counter to its baseline; running this module prints the page,
rendered from the same counts the test reads.

Exemptions are declared per counter, each with a reason, and are excluded
from the count. Adding one is a reviewed decision, not a comment.
"""
from __future__ import annotations

import ast
import functools
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Tuple

REPO = Path(__file__).resolve().parents[1]
BACKEND = REPO / "backend"
KERNEL_BENCH = REPO / "kernel_bench"
BASELINE_PATH = REPO / "tests" / "meta" / "scoreboard_baseline.json"
START_DATE = "2026-09-06"


def _py_files(*roots: Path) -> Iterable[Path]:
    for root in roots:
        for p in sorted(root.rglob("*.py")):
            if "__pycache__" in p.parts:
                continue
            yield p


def _rel(p: Path) -> str:
    return str(p.relative_to(REPO))


def _parse(p: Path):
    try:
        return ast.parse(p.read_text(encoding="utf-8"))
    except (SyntaxError, UnicodeDecodeError):
        return None


# ---------------------------------------------------------------------------
# Counter 1: silent except handlers
# ---------------------------------------------------------------------------

#: Files whose silent handlers are a declared I/O boundary (probing a pid,
#: reading a lock file, a best-effort telemetry write). Each entry is a
#: judgement that the caller genuinely has nothing to say on failure; a parser
#: or a checker is never on this list.
SILENT_EXCEPT_EXEMPT: Dict[str, Tuple[str, int]] = {
    # key: `<repo-relative path>::<Qualified.name>` of the innermost function
    # holding the handler; value: (why the caller has nothing to say, how
    # many silent handlers that function holds TODAY). A handler beyond that
    # count is new and counts — an exemption is a judgement about the ones
    # that were reviewed, never a licence for the function.
    "backend/_agent_runner/locker.py::InMemoryLocker.release": ("releasing a lock the process may not hold", 1),
    "backend/spec_worker/runtime.py::release": ("semaphore release on agent exit", 1),
    "backend/inference/analysable.py::_locked": ("a non-blocking lock probe: a slot another process holds is the answer False, not a failure", 1),
    "backend/spec_worker/runtime.py::start_job._run": ("best-effort loader invalidation after an agent run", 1),
    # S2 (grok R1/R2 votes): I/O boundaries only, one function each.
    "backend/spec_worker/worker.py::_harvest_component_yamls": ("one unreadable component file must not abort publish", 1),
    "backend/spec_worker/worker.py::_publish_to_cache": ("writing the run log is best-effort", 1),
    "backend/spec_worker/worker.py::_maybe_snapshot": ("snapshot write is best-effort", 1),

    "backend/spec_worker/_op_chain.py::find_fresh_gap": ("directory scan of concurrently-written .gap.json files: a racing/partial entry is skipped, not a bug hidden", 2),
    "backend/spec_worker/overlay.py::is_user_spec_path": ("path-containment predicate: a path outside the overlay (or unresolvable) is by definition not a user spec", 1),
    "backend/service/auth.py::insert_analyze_log": ("analytics must not break a user-facing endpoint", 1),
    "backend/service/personal_tokens.py::verify": ("external credential check over the network", 1),
    "backend/service/personal_tokens.py::revoke": ("external credential revocation over the network", 1),
    "backend/measure/cache.py::DiskBackend.get": ("disk cache miss / unreadable entry", 1),
    "backend/measure/cache.py::DiskBackend.put": ("disk cache write is best-effort", 1),
    "backend/measure/cache.py::DiskBackend._prune": ("disk cache prune is best-effort", 1),
    "backend/measure/cache.py::DiskBackend.clear": ("disk cache clear is best-effort", 1),
    "backend/inference/sol_pool.py::SolPool.write": ("a failed SOL-pool write is a cache miss, never a failed estimate", 1),
    "backend/inference/sol_pool.py::SolPool.record": ("a missing or corrupt SOL record is a cache miss", 1),
    "backend/inference/sol_pool.py::SolPool.nbytes": ("an unmeasurable SOL bundle has no hot-tier size", 1),
    "backend/inference/sol_pool.py::SolPool._manifest": ("a missing or corrupt SOL manifest is a cache miss", 1),
    "backend/inference/sol_pool.py::SolPool._index": ("a missing or corrupt SOL index is a cache miss", 1),
    "backend/hf/_cache.py::cache_load_raw": ("disk cache miss, or an unreadable entry refetched", 2),
    "backend/hf/_cache.py::cache_store": ("disk cache write is best-effort", 1),
    # I/O probes whose failure IS the answer the docstring states.
    "backend/spec_worker/runtime.py::clear_job_log_sidecar": ("removing a sidecar that may already be gone", 1),
    "backend/spec_worker/runtime.py::machine_id": ("no /etc/machine-id is the stated answer: lock rows use the hostname", 1),
    "backend/spec_worker/runtime.py::owner_process_alive": ("a pid probe: a missing /proc entry is the answer False", 1),
    "backend/spec_worker/worker.py::_code_revision": ("git probe: no repository is the answer 'unknown'", 1),
    "backend/measure/pool.py::_cgroup_cpu_quota": ("cgroup probe: an unreadable or malformed cpu.max is the answer None (no quota), as the docstring states", 2),
    "backend/spec_worker/worker.py::RunRevision.changed_paths": ("git probe: no repository is the answer 'no paths known'", 1),
    "backend/spec_worker/_metadata.py::get_python_transformers_version": ("subprocess probe: a failed probe is the answer 'version unknown'", 1),
    "backend/spec_worker/_metadata.py::extract_model_card_highlights": ("optional prompt context fetched over the network", 1),
}


def _is_empty_value(v) -> bool:
    """`None`, a literal, an empty container, or a tuple of those."""
    if v is None or isinstance(v, ast.Constant):
        return True
    if isinstance(v, ast.Tuple) and v.elts:
        return all(_is_empty_value(e) for e in v.elts)
    if isinstance(v, (ast.Dict, ast.List, ast.Tuple, ast.Set)):
        return not (getattr(v, "elts", None) or getattr(v, "keys", None))
    if (isinstance(v, ast.Call) and isinstance(v.func, ast.Name)
            and v.func.id in ("dict", "list", "set", "tuple", "frozenset")):
        return not v.args
    return False


def _is_debug_log(s: ast.stmt) -> bool:
    """`<logger>.debug(...)`: a line nobody sees at the default level."""
    return (isinstance(s, ast.Expr) and isinstance(s.value, ast.Call)
            and isinstance(s.value.func, ast.Attribute)
            and s.value.func.attr == "debug")


def _is_silent_handler(handler: ast.ExceptHandler) -> bool:
    """Body is exactly one of: pass / continue / return <empty value>
    (`None`, a literal, an empty container, or a tuple of them), optionally
    after a debug-level log line — or that log line alone. A debug line is
    as silent as `pass` at the default level, so it does not take a handler
    off the count."""
    body = list(handler.body)
    if body and _is_debug_log(body[0]):
        body = body[1:]
        if not body:
            return True
    if len(body) != 1:
        return False
    s = body[0]
    if isinstance(s, (ast.Pass, ast.Continue)):
        return True
    if isinstance(s, ast.Return):
        return _is_empty_value(s.value)
    return False


def _silent_handlers_by_function(tree: ast.AST) -> List[Tuple[str, ast.ExceptHandler]]:
    """Every silent handler with the qualified name of its INNERMOST
    enclosing function (`Class.method`, `outer.inner`), or `<module>`."""
    out: List[Tuple[str, ast.ExceptHandler]] = []

    def visit(node: ast.AST, stack: List[str]) -> None:
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
                visit(child, [*stack, child.name])
            else:
                if isinstance(child, ast.Try):
                    name = ".".join(stack) if stack else "<module>"
                    out.extend((name, h) for h in child.handlers
                               if _is_silent_handler(h))
                visit(child, stack)

    visit(tree, [])
    return out


def silent_except_sites() -> List[str]:
    out = []
    for p in _py_files(BACKEND):
        rel = _rel(p)
        tree = _parse(p)
        if tree is None:
            continue
        seen: Dict[str, int] = {}
        for name, h in _silent_handlers_by_function(tree):
            key = f"{rel}::{name}"
            allowance = SILENT_EXCEPT_EXEMPT.get(key)
            if allowance is not None and seen.get(key, 0) < allowance[1]:
                seen[key] = seen.get(key, 0) + 1
                continue
            out.append(f"{rel}:{h.lineno}")
    return out


# ---------------------------------------------------------------------------
# Counter 2: module-level UPPER_SNAKE name-list constants
# ---------------------------------------------------------------------------

#: A vocabulary's ONE definition, consumed by a loader/validator that rejects
#: at load time. These are the declaration, not a second table of it.
NAME_LIST_EXEMPT: Dict[Tuple[str, str], str] = {
    ("backend/arch/dims.py", "MESHES"): "the mesh vocabulary; dims.World refuses a mesh outside it",
    ("backend/hf/model_card.py", "UNIT_SCALES"): "the unit vocabulary of a stated count; `_COUNT` is built from it and matches no other spelling",
    ("backend/arch/dims.py", "PHASES"): "the phase vocabulary; dims refuses a world naming another phase",
    ("backend/arch/dims.py", "_KIND_MESH"): "kind -> mesh law, checked against every dims.yaml symbol at load",
    ("backend/arch/schema/spec.py", "VIEW_KEYS"): "the view: vocabulary; ArchSpec._check_view refuses a key outside it at load",
    ("backend/arch/schema/markers.py", "BINDING_ONLY"): "a json_schema_extra marker, not a name list; pinned by test_loader.TestBlockGraphSourceSchema",
    ("backend/arch/schema/markers.py", "LOADER_DERIVED"): "a json_schema_extra marker, not a name list; pinned equal to the loader's source-key gate",
    ("backend/arch/schema/markers.py", "RETIRED_GRAPH_NODE_NAMING"): "retired field -> replacement; the loader refuses these keys naming the replacement",
    ("backend/arch/spec_parsing.py", "_RETIRED_CARRY_KEYS"): "retired authoring keys the loader refuses, naming the replacement",
    ("backend/arch/spec_parsing.py", "_RETIRED_KV_KEYS"): "retired authoring keys the loader refuses, naming the replacement",
    ("backend/arch/spec_parsing.py", "RETIRED_KV_REPLACEMENTS"): "retired KV key -> current authoring path; loader and validate findings share it",
    ("backend/arch/spec_parsing.py", "_RETIRED_IDENTITY_KEYS"): "retired authoring keys the loader refuses, naming the replacement",
    ("backend/arch/spec_parsing.py", "_BLOCK_GRAPH_SOURCE_KEYS"): "the authoring read set; an unknown key is refused at load with the allowed set named",
    ("backend/inference/sweep_grid.py", "SPEC_PRESETS"): "the speculative presets' one declaration; the Scaling tracks, /inference/defaults (pinned by test_inference_api) and web/src/compare.ts (pinned by test_sweep_grid) read it",
    ("kernel_bench/refs.py", "_BUILDER_PARAMS"): "the input-builder signature contract; register_ref refuses a builder whose parameters differ",
    ("backend/arch/dtypes.py", "DTYPES"): "the canonical dtype table — the declaration itself, which every dtype projection reads",
    ("backend/inference/benchmarks.py", "IDENTITY_COLUMNS"): "the corpus CSV schema; rows are validated against it on read and write",
    ("backend/inference/benchmarks.py", "MEASUREMENT_COLUMNS"): "the corpus CSV schema; rows are validated against it on read and write",
    ("backend/inference/benchmarks.py", "PROVENANCE_COLUMNS"): "the corpus CSV schema; rows are validated against it on read and write",
    ("backend/inference/benchmarks.py", "PHASE_VOCABULARY"): "closed vocabulary; the row validator refuses any other phase",
    ("backend/inference/benchmarks.py", "CROSS_CHECK_VOCABULARY"): "closed vocabulary; the row validator refuses any other value",
    ("backend/inference/benchmarks.py", "DTYPE_SLOTS"): "closed slot vocabulary; the row validator refuses any other slot",
    ("backend/arch/cache_families.py", "KINDS"): "cache-family kind vocabulary; the family loader refuses others",
    ("backend/arch/ops_base.py", "AXIS_USE_CATEGORIES"): "axis_use category vocabulary; AxisUseDecl refuses others, layout asserts the same tuple",
    ("backend/arch/ops_base.py", "LAYOUT_FAMILIES"): "layout family vocabulary; OpSpec refuses others, family_handler dispatches on exactly these",
    ("backend/arch/quant_schemes/__init__.py", "SIBLING_KINDS"): "the metadata-sibling kind vocabulary; the scheme parser refuses any other spelling at load",
    ("kernel_bench/providers/gemm.py", "_HUMMING_WEIGHT_DTYPE"): "a third-party provider's own dtype spellings at its interface; not a restatement of arch.dtypes",
    ("backend/arch/ops_base.py", "EDGE_KIND_VOCABULARY"): "closed edge-kind vocabulary; OpSpec.__post_init__ rejects others",
    ("backend/arch/ops_base.py", "INPUT_ROLE_VOCABULARY"): "closed operand-role vocabulary; OpSpec.__post_init__ rejects others",
    ("backend/arch/dims.py", "KINDS"): "dim-kind vocabulary; dims._load rejects others",
    ("backend/arch/dims.py", "COPY_UNITS"): "the display-noun vocabulary for a count of copies; DimSymbol.__post_init__ rejects any other word at load",
    ("backend/arch/roles.py", "KINDS"): "role-kind vocabulary; roles loader rejects others",
    ("backend/arch/eval/projection/__init__.py", "_COMPOSITION_ENTRIES"): "the eval-composition entry names re-exported lazily (PEP 562) from transforms; the set IS the re-export surface, not a classification roster",
    # --- F7 grok 裁定新增 18 条:声明表(有装载期完整性门),非第二张分类名单 ---
    ("backend/arch/dims.py", "_KINDS_WITHOUT_WORLD"): "kind->world 定律的一半;dims.py 装载期与 shardable_by 对账",
    ("backend/arch/law_text.py", "LAW_OPERATORS"): "法则运算符的唯一定义;未知名当运算符用会 raise(非运算符名当轴绑定成 Symbol,不 raise)",
    ("backend/arch/layout/foldlaw.py", "SCHEMA_MAPS"): "同态律 payload 变换闭集;ReductionHomomorphism.__post_init__ 拒未知",
    ("backend/arch/layout/axis_use.py", "REGISTERED_MERGE_STATISTICS"): "可走通用 Partial 的可合并统计闭集",
    ("backend/arch/layout/axis_use.py", "GEOMETRY_COUPLES"): "几何 op 耦合的声明;与 couples_on 有漂移门",
    ("backend/arch/layout/axis_use.py", "AXIS_USE"): "每 (op,axis) 的 axis_use 声明表;有完整性门,不是第二张分类名单",
    ("backend/arch/layout/foldlaw.py", "HOMOMORPHISM_REGISTRY"): "Partial 穿过律的登记表本身",
    ("backend/arch/layout/mesh.py", "REGION_AXES"): "网格分区轴的唯一定义;region_of 对跨区无间距则 raise",
    ("backend/arch/layout/mesh.py", "GLOBAL_OUTER_AXES"): "全区外层轴的唯一定义",
    ("backend/arch/layout/mesh.py", "SUBGROUP_AXES"): "子组世界的唯一定义",
    ("backend/arch/layout/mesh.py", "_MULTI_REGION_AXIS"): "跨区轴间距的唯一定义;缺则 raise",
    ("backend/arch/quant_schemes/__init__.py", "_KNOWN_QUANT_STATUSES"): "方案 valid_status 闭集;装载期拒未知",
    ("kernel_bench/quant.py", "FORMATS"): "kernel 权重格式声明;未知 raise",
    ("backend/service/routes/auth_session.py", "_ADMIN_RPC_ALLOWLIST"): "admin RPC 白名单本身;代理拒绝名单外的 RPC,是安全边界的声明而非分类名单",
    ("backend/service/routes/auth_session.py", "_SAFE_METHODS"): "RFC 9110 安全方法闭集;CSRF 守卫对其外的方法要求自定义头",
    ("kernel_bench/quant.py", "SCALE_ENCODINGS"): "kernel 标度编码闭集;未知 raise",
    ("backend/inference/benchmarks.py", "PHASE_BINDING_EXTRAS"): "CSV shape_json 允许键的相位组;校验器按组点名拒",
    ("backend/inference/benchmarks.py", "ENGINE_DIMS"): "CSV shape_json 允许键的引擎组;校验器按组点名拒",
    ("kernel_bench/smoke.py", "MINIMAL_DIMS"): "smoke 可合成轴的唯一定义;缺轴 -> no_cell 不猜",
    # --- F7 grok-clarify 裁定:#3/#5/#6 原判 CONVERT,落地证明非同一事实,撤回为 KEEP+豁免 ---
    ("backend/arch/schema/quant.py", "_ALLOWED_QUANT_CONTEXT_DTYPES"): "quant_context 的 dtype: / default_dtype: 作者可写集;装载期拒未知。不是 DTYPES 的投影",
    ("backend/arch/dim_resolution.py", "_DIM_EXPR_FUNCTIONS"): "eval_dim_expr 派生式的函数名声明。与 LAW_OPERATORS 是不同路径",
    ("backend/arch/layout/foldlaw.py", "PARTIAL_PRESERVING_VIEW_OPS"): "view-passthrough 律的声明表;HOMOMORPHISM_REGISTRY 从它构建",
    # --- Kernel Bench cross-check: a measured table, keyed by dtype ---
    ("kernel_bench/driver.py", "_XCHECK_BANDS"): "每 dtype 的交叉校验容差带本身（measured，tools/xcheck_floors.py 重调）：值是事实、不是名单，没有任何声明能派生出它；键在装载期按 dtypes.CORPUS_TOKENS 硬拒（否则一个拒不到的拼写会让那档 cell 落到 bf16 的带上）",
    ("kernel_bench/refs.py", "_RECORD_AMAX_FLOOR"): "每 dtype 的记录写入方 amax 下限本身（读自审计过的 writer 源码，cache format 不声明它）：值是事实、不是名单；键在装载期按 dtypes.CORPUS_TOKENS 硬拒（和 _XCHECK_BANDS 同形）",
    # --- Kernel Bench lake store: the tables' and the browse API's own definitions ---
    ("backend/service/bench/store.py", "JSON_COLUMNS"): "以 JSON 对象文本存的列;行校验器恰好解码这两列(_check_shape_json / _check_dtype_env_json)",
    ("kernel_bench/providers/__init__.py", "TORCH_FLOAT"): "audited torch floating dispatch set; a provider declares it only after its kernel was audited at those widths",
    ("backend/rendering/_node_build.py", "_DETAIL_SKIP_DIM_KEYS"): "the generic dim tooltip's display vocabulary; the detail renderer refuses to print these generic axes",
}

_UPPER = re.compile(r"^_?[A-Z][A-Z0-9_]+$")


def _is_str_container(node) -> bool:
    if (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
            and node.func.id in ("frozenset", "set", "tuple", "dict") and node.args):
        return _is_str_container(node.args[0])
    if isinstance(node, (ast.Set, ast.Tuple, ast.List)):
        return bool(node.elts) and all(
            isinstance(e, ast.Constant) and isinstance(e.value, str) for e in node.elts)
    if isinstance(node, ast.Dict):
        keys = [k for k in node.keys if k is not None]
        return bool(keys) and all(
            isinstance(k, ast.Constant) and isinstance(k.value, str) for k in keys)
    return False


def _own_body(fn) -> List[ast.stmt]:
    """`fn`'s statements, its docstring aside."""
    body = list(fn.body)
    if (body and isinstance(body[0], ast.Expr)
            and isinstance(body[0].value, ast.Constant)
            and isinstance(body[0].value.value, str)):
        body = body[1:]
    return body


def name_list_sites() -> List[str]:
    out = []
    for p in _py_files(BACKEND, KERNEL_BENCH):
        rel = _rel(p)
        tree = _parse(p)
        if tree is None:
            continue
        for n in tree.body:
            if not isinstance(n, (ast.Assign, ast.AnnAssign)) or n.value is None:
                continue
            tgt = n.targets[0] if isinstance(n, ast.Assign) else n.target
            if not (isinstance(tgt, ast.Name) and _UPPER.match(tgt.id)):
                continue
            if (rel, tgt.id) in NAME_LIST_EXEMPT:
                continue
            if _is_str_container(n.value):
                out.append(f"{rel}:{n.lineno} {tgt.id}")
    return out


# ---------------------------------------------------------------------------
# Counter 3: width literals as signature defaults
# ---------------------------------------------------------------------------

_WIDTH_ARG = re.compile(r"dtype|bytes|width|bpe", re.I)


def _dtype_vocabulary() -> Tuple[frozenset, frozenset]:
    """The dtype names and their widths, read off the dtype declaration
    (`arch/dtypes.py`) rather than listed here."""
    sys.path.insert(0, str(REPO))
    from backend.arch.dtypes import DTYPE_BYTES, accepted_dtype_names
    return accepted_dtype_names(), frozenset(DTYPE_BYTES.values())


def _key_name(e: ast.AST) -> str:
    """The name a fallback stands in for: `x.bytes_per_el`, `d["dtype"]`,
    `d.get("width")`, `bpe` — so a numeric default is judged by what it
    defaults, and `degree or 1` is not a width."""
    if isinstance(e, ast.Name):
        return e.id
    if isinstance(e, ast.Attribute):
        return e.attr
    if isinstance(e, ast.Subscript):
        s = e.slice
        return s.value if isinstance(s, ast.Constant) and isinstance(s.value, str) else _key_name(e.value)
    if isinstance(e, ast.Call):
        args = [a.value for a in e.args if isinstance(a, ast.Constant) and isinstance(a.value, str)]
        return " ".join([_key_name(e.func), *args])
    return ""


def _width_literal(d: ast.AST, key: str,
                   vocabulary: Tuple[frozenset, frozenset]) -> bool:
    """A dtype-name literal anywhere; a numeric width only for a width key."""
    names, widths = vocabulary
    if not isinstance(d, ast.Constant) or isinstance(d.value, bool):
        return False
    if isinstance(d.value, str):
        return d.value.lower() in names
    return d.value in widths and bool(_WIDTH_ARG.search(key))


def width_default_sites() -> List[str]:
    """Signature defaults, and the same default written in a function body:
    `x or "bf16"`, `d.get("kv", "bf16")`, `bytes_per_el or 2`."""
    dtypes = _dtype_vocabulary()
    out = []
    for p in _py_files(BACKEND):
        rel = _rel(p)
        tree = _parse(p)
        if tree is None:
            continue
        for n in ast.walk(tree):
            if isinstance(n, ast.BoolOp) and isinstance(n.op, ast.Or):
                d, left = n.values[-1], n.values[-2]
                if _width_literal(d, _key_name(left), dtypes):
                    out.append(f"{rel}:{n.lineno} {ast.unparse(left)} or {d.value!r}")
                continue
            if (isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
                    and n.func.attr == "get" and len(n.args) == 2
                    and isinstance(n.args[0], ast.Constant) and isinstance(n.args[0].value, str)):
                if _width_literal(n.args[1], n.args[0].value, dtypes):
                    out.append(f"{rel}:{n.lineno} .get({n.args[0].value!r}, {n.args[1].value!r})")
                continue
            if not isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            a = n.args
            positional = list(zip(a.args[len(a.args) - len(a.defaults):], a.defaults,
                                  strict=True))
            kwonly = list(zip(a.kwonlyargs, a.kw_defaults, strict=True))
            for arg, d in positional + kwonly:
                if d is None or not _WIDTH_ARG.search(arg.arg):
                    continue
                if _width_literal(d, arg.arg, dtypes):
                    out.append(f"{rel}:{n.lineno} {n.name}({arg.arg}={d.value!r})")
    return out


# ---------------------------------------------------------------------------
# Counter 4/5: import cycles and function-body imports (moved here from
# tests/meta/test_structure_debt.py so the test and the page read one query)
# ---------------------------------------------------------------------------

def _module_name(path: Path) -> str:
    parts = list(path.relative_to(REPO).with_suffix("").parts)
    name = ".".join(parts)
    return name[: -len(".__init__")] if name.endswith(".__init__") else name


def _is_type_checking(test: ast.AST) -> bool:
    """`if TYPE_CHECKING:` — annotations only, never executed at import."""
    return isinstance(test, ast.Name) and test.id == "TYPE_CHECKING"


def import_graph():
    """`(edges, lazy)` over `backend.*` modules. A lazy edge is the same
    dependency with the import moved into a function so the module can load,
    which is how a cycle stops being visible at import time.

    Imports under a module-level `if TYPE_CHECKING:` are not edges. They do
    not run, and counting them made three layout cycles out of
    annotation-only names."""
    edges: dict = {}
    lazy: dict = {}
    for path in _py_files(BACKEND):
        module = _module_name(path)
        tree = _parse(path)
        if tree is None:
            continue
        inside: set = set()
        typing_only: set = set()
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                inside.update(range(node.lineno, (node.end_lineno or node.lineno) + 1))
        for node in tree.body:
            if isinstance(node, ast.If) and _is_type_checking(node.test):
                typing_only.update(range(node.lineno, (node.end_lineno or node.lineno) + 1))
        for node in ast.walk(tree):
            if getattr(node, "lineno", None) in typing_only:
                continue
            targets: List[str] = []
            if isinstance(node, ast.Import):
                targets = [al.name for al in node.names if al.name.startswith("backend")]
            elif isinstance(node, ast.ImportFrom):
                if node.level:
                    base = module.split(".")[: -node.level]
                    targets = [".".join(base + ([node.module] if node.module else []))]
                elif node.module and node.module.startswith("backend"):
                    targets = [node.module]
            for target in targets:
                if not target.startswith("backend"):
                    continue
                edges.setdefault(module, set()).add(target)
                if node.lineno in inside:
                    lazy.setdefault(module, set()).add(target)
    return edges, lazy


def cycles(edges: dict, limit: int = 12) -> set:
    found: set = set()

    def walk(start: str, current: str, path: list) -> None:
        if len(path) > limit:
            return
        for nxt in sorted(edges.get(current, ())):
            if nxt == start and len(path) > 1:
                found.add(tuple(sorted(path)))
            elif nxt not in path and nxt in edges:
                walk(start, nxt, [*path, nxt])

    for module in sorted(edges):
        walk(module, module, [module])
    return found


def import_cycle_sites() -> List[str]:
    edges, _ = import_graph()
    return [" -> ".join(c) for c in sorted(cycles(edges))]


def lazy_edge_sites() -> List[str]:
    _, lazy = import_graph()
    return [f"{m} -> {t}" for m in sorted(lazy) for t in sorted(lazy[m])]


# ---------------------------------------------------------------------------
# Counter 6: defensive attribute probes across backend/arch
# ---------------------------------------------------------------------------

def defensive_access_sites() -> List[str]:
    """`getattr(x, "name", default)` / `hasattr` call sites under backend/arch.
    CLAUDE.md §2.7 bans them in eval; `test_eval_boundary` pins eval per
    module. This is the same question asked of the whole package, because
    layout's `getattr(rule, "parallel", None)` sat outside that scan."""
    out = []
    for p in _py_files(BACKEND):
        rel = _rel(p)
        tree = _parse(p)
        if tree is None:
            continue
        out.extend(f"{rel}:{n.lineno}" for n in ast.walk(tree)
                   if (isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
                       and n.func.id in ("getattr", "hasattr")))
    return out




# ---------------------------------------------------------------------------
# Counter 7: downstream systems deriving facts instead of reading Analysis
# ---------------------------------------------------------------------------
#
# eval (arch/eval + arch/layout, entered through transforms.Deployment) is the
# only producer of facts. inference / rendering / api / service / views /
# playground / kernel_bench consume the typed layers on `Analysis` and nothing else: not the op
# registry, not the spec's dims table, not the raw config, not an op name.
# A fact a consumer needs and cannot find on `Analysis` is materialized there
# first — never re-derived downstream. `test_eval_boundary` stops eval from
# looking outward; this is the other direction, which had no gate, and it is
# where 60 of the 66 restatements in the facts ledger live.

DOWNSTREAM_SCOPES = ("backend/inference", "backend/rendering", "backend/api",
                     "backend/service", "backend/analyzer.py", "backend/cli.py",
                     "backend/cli_bench.py", "backend/views", "backend/playground",
                     "kernel_bench")

#: Files that are themselves declarations consumed by the harness, not
#: derivations of engine facts.
DOWNSTREAM_EXEMPT: Dict[str, str] = {
    "kernel_bench/refs.py": "reference kernels read their own Invocation.dims (which dim a kernel takes), not any model's spec dims table",
    "kernel_bench/invocation.py": "the Invocation input channel: dims a kernel was asked to run, declared by the signature catalog",
    "kernel_bench/providers/gemm.py": "reads a Triton tile config dict keyed BLOCK_SIZE_*, not the HF config",
    "backend/inference/op_signatures_catalog.py": "kernel signature table: dim NAMES a kernel takes, not a derivation of any model's structure",
    "backend/inference/framework_bindings": "framework op-name tables: declares what vLLM/sglang call a kernel",
    "backend/inference/walk_registry.py": "walk-cache identity: names the raw config because a framework binding may read checkpoint fields the model footprint digest does not change",
}

#: Each pattern is one way of answering a fact downstream instead of reading
#: it off `Analysis`. Regex, applied per line.
#: Asking an OpSpec predicate (`has_weight()`, `is_comm_op`, `is_identity_source`)
#: is NOT on this list. Those are the declarations CLAUDE.md §1 tells a consumer
#: to ask instead of keeping a name list; counting them would penalize the
#: prescribed fix. What IS counted is rebuilding a walk result from them —
#: `kernel_bench/plan.py` re-walking `inputs_obj` to recover which node a GEMM's
#: weight came from is the join `traffic` already made, and belongs on the row.
DOWNSTREAM_PATTERNS: Tuple[Tuple[str, str], ...] = (
    # OpSpec.formula / .description are the op's own doc strings — asking for
    # them is display, not derivation. Everything else off the registry counts.
    (r"(OP_REGISTRY\b|\bget_op\()(?!.*\.(formula|description))", "asks the op registry"),
    (r"\bop(_name)? (==|!=) \"|\bop(_name)? (not )?in \(|\bop in _[A-Z_]+|\.op\.(startswith|endswith)\(", "branches on an op name or an op-name list"),
    (r"_?spec_dims\(|\(dims or \{\}\)\.get\(|dims(\.get\(|\[)\"n_experts\"|dim_values(\.get\(|\[)\"(E|K|I|n_experts|n_experts_active|n_route_targets)\"", "reads the dims table to infer MoE / expert structure"),
    (r"resolved\.config\b|\bconfig(\.get\(|\[)\"|\(config or \{\}\)\.get\(|quantization_config", "reads the raw HF config"),
    (r"\b_component_lib\b|\.blocks\.get\(", "reaches into the spec's component tables"),
    (r"legacy_copies_div|proportional_bytes", "keeps a fallback that answers when eval gave no placement"),
    (r"aggregation_id.*(\"expert\"|\"attn\"|\"moe\")|(\"expert\"|\"attn\"|\"moe\"|\"router\"|\"shared\") in (a|name|disp|agg|sid|child_name)\b", "classifies a node by a substring of its id"),
    (r"for ref in \(?node\.inputs_obj", "rebuilds a walk result by re-walking edges"),
)


def _call_name(n: ast.Call) -> str:
    f = n.func
    return f.id if isinstance(f, ast.Name) else (f.attr if isinstance(f, ast.Attribute) else "")


def _is_op_read(e: ast.AST) -> bool:
    """`x["op"]`, `x.op`, `x.get("op")`."""
    if isinstance(e, ast.Subscript):
        return isinstance(e.slice, ast.Constant) and e.slice.value == "op"
    if isinstance(e, ast.Attribute):
        return e.attr == "op"
    return (isinstance(e, ast.Call) and _call_name(e) == "get" and bool(e.args)
            and isinstance(e.args[0], ast.Constant) and e.args[0].value == "op")


def dim_table_wrappers_in(tree: ast.AST) -> set:
    """Functions of one module whose whole answer is a read of a
    resolution's dims table: one `return` over `.dim_values`
    (`ResolvedSpec.get_dim`). A constructor handed the table
    (`ResolvedSpec(…, dict(self.dim_values), …)`) builds, it does not read."""
    out = set()
    for fn in ast.walk(tree):
        if not isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        body = _own_body(fn)
        if len(body) != 1 or not isinstance(body[0], ast.Return) or body[0].value is None:
            continue
        value = body[0].value
        if (isinstance(value, ast.Call) and isinstance(value.func, ast.Name)
                and value.func.id[:1].isupper()):
            continue
        if any(isinstance(n, ast.Attribute) and n.attr == "dim_values"
               for n in ast.walk(value)):
            out.add(fn.name)
    return out


@functools.lru_cache(maxsize=None)
def _dim_table_wrappers() -> frozenset:
    """`dim_table_wrappers_in` over `backend/` outside eval. Calling one
    downstream is reading the table — the wrapper only renames the read.
    Derived from the definitions, so a new wrapper is seen without being
    listed; eval's own layers are what a consumer is told to read."""
    out: set = set()
    for p in _py_files(BACKEND):
        if _rel(p).startswith("backend/arch/eval"):
            continue
        tree = _parse(p)
        if tree is not None:
            out |= dim_table_wrappers_in(tree)
    return frozenset(out)


def _downstream_ast_sites(tree: ast.AST) -> Dict[int, str]:
    """The shapes a per-line regex cannot see or would mis-see, keyed by line.
    Matched on the attribute or call itself, never on a variable's name —
    except `.config`, which counts only off a name typed `ResolvedSpec`
    (`resolved.config` is the regex's), so a lake's or a sweep plan's own
    config never reads as the HF one."""
    module_names = {t.id for s in tree.body if isinstance(s, (ast.Assign, ast.AnnAssign))
                    for t in (s.targets if isinstance(s, ast.Assign) else [s.target])
                    if isinstance(t, ast.Name)}
    spec_types = {"ResolvedSpec"} | {
        a.asname for s in ast.walk(tree) if isinstance(s, ast.ImportFrom)
        for a in s.names if a.name == "ResolvedSpec" and a.asname}

    def names_spec(annotation: ast.AST) -> bool:
        """Whether an annotation names the resolved-spec type — as a name,
        an attribute, inside `Optional[...]`, or as a string forward ref."""
        for e in ast.walk(annotation):
            if isinstance(e, ast.Constant) and isinstance(e.value, str):
                try:
                    if names_spec(ast.parse(e.value, mode="eval")):
                        return True
                except SyntaxError:
                    continue
            elif ((isinstance(e, ast.Name) and e.id in spec_types)
                    or (isinstance(e, ast.Attribute) and e.attr in spec_types)):
                return True
        return False

    spec_names: set = set()
    for f in ast.walk(tree):
        if isinstance(f, (ast.FunctionDef, ast.AsyncFunctionDef)):
            a = f.args
            spec_names.update(x.arg for x in a.posonlyargs + a.args + a.kwonlyargs
                              if x.annotation is not None
                              and names_spec(x.annotation))
    hits: Dict[int, str] = {}
    for n in ast.walk(tree):
        why = None
        if isinstance(n, ast.Attribute) and n.attr == "dim_values":
            why = "reads the dims table"
        elif (isinstance(n, ast.Attribute) and n.attr == "config"
                and isinstance(n.value, ast.Name) and n.value.id in spec_names):
            why = "reads the raw HF config"
        elif isinstance(n, ast.Call):
            name = _call_name(n)
            key = n.args[1].value if (len(n.args) > 1 and isinstance(n.args[1], ast.Constant)) else None
            if name == "getattr" and key in ("dim_values", "config"):
                why = "reads the dims table" if key == "dim_values" else "reads the raw HF config"
            elif name in ("config_path_value", "eval_dim_expr"):
                why = "resolves a dim expression against the spec's dims / config"
            elif isinstance(n.func, ast.Attribute) and name in _dim_table_wrappers():
                why = "reads the dims table through a wrapper that only renames the read"
            elif (name == "loads" and n.args and isinstance(n.args[0], ast.BoolOp)
                    and isinstance(n.args[0].op, ast.Or)
                    and isinstance(n.args[0].values[-1], ast.Constant)
                    and n.args[0].values[-1].value == "{}"):
                why = "parses a missing value as an empty object"
        elif (isinstance(n, ast.BoolOp) and isinstance(n.op, ast.Or)
                and isinstance(n.values[0], ast.Call) and _call_name(n.values[0]) == "get"
                and n.values[0].args and isinstance(n.values[0].args[0], ast.Constant)
                and n.values[0].args[0].value == "node_dims"):
            why = "keeps a fallback that answers when the row carried no dims"
        elif (isinstance(n, ast.Compare) and _is_op_read(n.left)
                and any(isinstance(o, (ast.In, ast.NotIn)) for o in n.ops)
                and any(isinstance(c, ast.Name) and c.id in module_names for c in n.comparators)):
            why = "branches on an op name or an op-name list"
        if why and n.lineno not in hits:
            hits[n.lineno] = why
    return hits


def downstream_derivation_sites() -> List[str]:
    """One count per line. Import lines never count: naming a symbol is not
    using it — the use is what counts, wherever it is."""
    out = []
    compiled = [(re.compile(rx), why) for rx, why in DOWNSTREAM_PATTERNS]
    for scope in DOWNSTREAM_SCOPES:
        root = REPO / scope
        files = [root] if root.is_file() else list(_py_files(root))
        for p in files:
            rel = _rel(p)
            if any(rel.startswith(x) for x in DOWNSTREAM_EXEMPT):
                continue
            tree = _parse(p)
            hits: Dict[int, str] = {}
            imports: set = set()
            if tree is not None:
                hits = _downstream_ast_sites(tree)
                for n in ast.walk(tree):
                    if isinstance(n, (ast.Import, ast.ImportFrom)):
                        imports.update(range(n.lineno, (n.end_lineno or n.lineno) + 1))
            for i, line in _code_lines(p):
                if i in imports:
                    continue
                why = next((why for rx, why in compiled if rx.search(line)), None) or hits.get(i)
                if why:
                    out.append(f"{rel}:{i} [{why}]")
    return out


# ---------------------------------------------------------------------------
# Counter 9: a decision taken on an extent that may still be an expression
# ---------------------------------------------------------------------------
#
# An extent (`eval/values.Extent is eval/dim.Dim`) carries an expression and a
# number at once. Ask it a magnitude question with a Python operator and only
# the number answers: `Dim.__gt__` compares the value halves, `Dim.__bool__`
# tests the value half, `int(dim)` takes it. In a domain where the value half
# is itself a symbol the operator has no answer at all — sympy raises
# `TypeError` on `bool()` of an undecidable relation — and the ways of turning
# that non-answer back into a value are a closed set of shapes: catch the
# `TypeError`, test the truth, `or` a number in behind it, or take the smaller
# of two with the builtin.
#
# Division is the same defect without the refusal. `/` and `//` answer in both
# domains, and answer DIFFERENTLY: `128 / 128` on two Python ints is the float
# `1.0`, while the same division of two laws is sympy's exact `1`; `//` floors a
# number while `Dim.__floordiv__`'s expression half divides exactly. So the
# operator does not raise, it returns — one answer per binding, of two kinds,
# and the two readers of that row then disagree about what it says. Measured at
# `layout/physical.pairs_share`: a rank's share of the score pairs came out
# `1.0` in decode, rode into the law as a coefficient, and published the live
# pair count as the string "4224.00000000000" where the same law read at a
# point published the integer 4224. The verbs that answer once are in
# `eval/dim.py` — `exact_div` (declared exact, raises when the declaration is
# wrong), `ceil_div` / `floor_div` (a rounding the law intends), `ratio` (a
# proportion, which makes the caller say what an empty denominator means), and
# `exact` on both halves for a ratio of counts that must stay a `Rational`.
#
# Every one of those shapes decides something and leaves no record of having
# decided. The counter exists because exactly one of them survived a review
# (`traffic._wider` caught `TypeError` on `b > a` and kept `a`, billing the
# first of two incomparable cache spans) and was found by reading, not by a
# test. Reading does not scale; a count does.
#
# WHAT THE SCAN CAN AND CANNOT SEE. Python has no types here, so "could this
# be an extent" is approximated by an intraprocedural taint whose sources are
# read off declarations rather than listed:
#
#   * a call to a public name of the extent algebra (`eval/dim.py`,
#     `eval/values.py`) that does not DECLARE it returns a settled `int` /
#     `bool` / `str` — the surface is parsed out of those two modules, so
#     adding a helper there extends the scan for free;
#   * a parameter or variable annotated `Extent` / `Dim`;
#   * a call to a function in the scanned tree whose return annotation is one
#     of those;
#   * anything assigned, unpacked or iterated from the above.
#
# Against that, a value is NOT treated as one magnitude when the same function
# measures it, indexes it, iterates it, unpacks it or appends to it — a
# container is none of the things this counter is about. That test is
# structural too, not a vocabulary.
#
# So the classifier has false positives (`if params:` where `params` is a
# count the walk will never symbolise) and false negatives (an extent reaching
# a comparison through a field this scan cannot follow). Both are tolerable
# because `tests/meta/test_symbolic_decisions.py` pins the sites BY NAME
# beside this count: a false positive is written down once, and the gate's job
# is the next site, not this one.

ALGEBRA_MODULES = ("backend/arch/eval/dim.py", "backend/arch/eval/values.py")

#: Trees scanned. eval is where extents are made; layout is where they are
#: projected per rank, which is the other place a magnitude gets decided.
UNDECIDED_SCOPES = ("backend/arch/eval", "backend/arch/layout")

#: Return annotations that say "the symbolic question was settled inside me".
_SETTLED_RETURNS = {"bool", "str", "int", "float", "FrozenSet", "Dict",
                    "Mapping", "Set", "List", "None"}

#: Calls that build or measure a container, so their result is not one
#: magnitude. (`bool` is absent on purpose: `bool(extent)` is a site.)
_CONTAINER_CALLS = {"tuple", "list", "set", "dict", "frozenset", "sorted",
                    "zip", "enumerate", "reversed"}
_MEASURING_CALLS = _CONTAINER_CALLS | {"len", "any", "all", "sum"}
_SETTLING_CALLS = _MEASURING_CALLS | {"isinstance", "issubclass", "callable",
                                      "hasattr", "id", "repr", "str", "bool"}
_CONTAINER_METHODS = {"append", "extend", "pop", "items", "keys", "values",
                      "get", "setdefault", "add", "update", "join"}
#: Attributes of an extent that answer a question about its IDENTITY rather
#: than its magnitude — asking them is the thing this counter wants instead.
_IDENTITY_ATTRS = {"names", "free_symbols", "is_number", "is_integer",
                   "is_finite", "expr"}

_ORDER_OPS = (ast.Lt, ast.LtE, ast.Gt, ast.GtE)
_EQ_OPS = (ast.Eq, ast.NotEq)
_EXTENT_ANN = re.compile(r"\b(Extent|Dim)\b")
#: A return annotation that says "a SymPy object comes back": `sp.Basic`,
#: `sympy.Expr`, `sp.Symbol`.
_SYMPY_ANN = re.compile(r"^(sp|sympy)\.")
_PLURAL_ANN = re.compile(r"\b(Iterable|Sequence|Tuple|List|Dict|Mapping|Set)\b")


def _ann_is_extent(ann) -> bool:
    return ann is not None and bool(_EXTENT_ANN.search(ast.unparse(ann)))


def _returns_settled(ann) -> bool:
    if ann is None:
        return False
    return ast.unparse(ann).split("[")[0].strip() in _SETTLED_RETURNS


def _algebra_surface() -> frozenset:
    """The public names of the extent algebra that can still hand back an
    expression. Parsed from the algebra's own modules: the declaration is the
    definition, so there is no second list to drift."""
    out = set()
    for rel in ALGEBRA_MODULES:
        tree = _parse(REPO / rel)
        if tree is None:
            continue
        for n in tree.body:
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)):
                if not n.name.startswith("_") and not _returns_settled(n.returns):
                    out.add(n.name)
            elif isinstance(n, ast.ClassDef) and not n.name.startswith("_"):
                out.add(n.name)
            elif isinstance(n, ast.Assign):
                for t in n.targets:
                    if isinstance(t, ast.Name) and not t.id.startswith("_"):
                        out.add(t.id)
    return frozenset(out)


@functools.lru_cache(maxsize=None)
def _algebra_decisions() -> frozenset:
    """The public functions of the extent algebra that DECLARE they settle
    the question inside (`-> bool`, `-> int`, ...): `equal`, `less`,
    `is_zero`, `is_unit`. A call to one is an answer, made over every
    admitted point or recorded, and not a magnitude — whatever extents it
    was asked about."""
    out = set()
    for rel in ALGEBRA_MODULES:
        tree = _parse(REPO / rel)
        if tree is None:
            continue
        for n in tree.body:
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) \
                    and not n.name.startswith("_") \
                    and _returns_settled(n.returns):
                out.add(n.name)
    return frozenset(out)


@functools.lru_cache(maxsize=None)
def _algebra_sympy_carriers() -> frozenset:
    """The public algebra functions that DECLARE they hand back a SymPy object
    (`-> sp.Basic`, `-> sp.Symbol`): `exact`, `algebra`, `ratio_symbol`.

    An operator applied to one of those is SymPy's in either domain, and SymPy
    answers once: `exact(a) / exact(b)` is an exact `Rational` where `a / b`
    on the two Python ints underneath would have been a float. So a division
    through one of these is not the `bare-division` shape — which is why
    `exact` carries that annotation rather than `Any`."""
    out = set()
    for rel in ALGEBRA_MODULES:
        tree = _parse(REPO / rel)
        if tree is None:
            continue
        for n in tree.body:
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) \
                    and not n.name.startswith("_") and n.returns is not None \
                    and _SYMPY_ANN.match(ast.unparse(n.returns)):
                out.add(n.name)
    return frozenset(out)


def _sympy_aliases(tree) -> set:
    """How this module names sympy itself (`import sympy as sp`), so a
    constructor called on it — `sp.Rational(active)` — is read as the SymPy
    value it is."""
    out = set()
    for n in ast.walk(tree):
        if isinstance(n, ast.Import):
            for a in n.names:
                if a.name == "sympy":
                    out.add(a.asname or "sympy")
    return out


def _algebra_imports(tree) -> Tuple[set, set]:
    """`(names, module aliases)` through which this module reaches the
    algebra."""
    surface = _algebra_surface()
    names, mods = set(), set()
    for n in ast.walk(tree):
        if isinstance(n, ast.ImportFrom):
            mod = n.module or ""
            if mod.endswith(("eval.dim", "eval.values")) \
                    or mod in ("dim", "values"):
                for a in n.names:
                    if a.name in surface:
                        names.add(a.asname or a.name)
            else:
                for a in n.names:
                    if a.name in ("dim", "values"):
                        mods.add(a.asname or a.name)
        elif isinstance(n, ast.Import):
            for a in n.names:
                if a.name.endswith("eval.dim") or a.name.endswith("eval.values"):
                    mods.add(a.asname or a.name.split(".")[-1])
    return names, mods


def _container_evidence(fn) -> set:
    """The names this function treats as a container: it measures them,
    indexes them, iterates them, unpacks them, appends to them or asks
    membership of them. A magnitude is none of those."""
    out = set()

    def key(x):
        if isinstance(x, ast.Name):
            return x.id
        if isinstance(x, ast.Attribute) and isinstance(x.value, ast.Name):
            return f"{x.value.id}.{x.attr}"
        return None

    for n in _own_nodes(fn):
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) \
                and n.func.id in _MEASURING_CALLS:
            out.update(key(a) for a in n.args)
        elif isinstance(n, ast.Subscript):
            out.add(key(n.value))
        elif isinstance(n, (ast.For, ast.comprehension)):
            out.add(key(n.iter))
        elif isinstance(n, ast.Starred) or (isinstance(n, ast.Attribute) and n.attr in _CONTAINER_METHODS):
            out.add(key(n.value))
        elif isinstance(n, ast.Compare):
            for op, cmp in zip(n.ops, n.comparators, strict=False):
                if isinstance(op, (ast.In, ast.NotIn)):
                    out.add(key(cmp))
    out.discard(None)
    return out


def _is_container_expr(v) -> bool:
    if isinstance(v, (ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp,
                      ast.List, ast.Dict, ast.Set, ast.Tuple)):
        return True
    return (isinstance(v, ast.Call) and isinstance(v.func, ast.Name)
            and v.func.id in _CONTAINER_CALLS)


class _ExtentFlow:
    """Which of one function's names may still hold an expression."""

    def __init__(self, fn, names: set, mods: set, local: set,
                 enclosing: Optional["_ExtentFlow"] = None,
                 sympy_names: set = frozenset()) -> None:
        self.names, self.mods, self.local = names, mods, local
        self.surface = _algebra_surface()
        self.decisions = _algebra_decisions()
        self.carriers = _algebra_sympy_carriers()
        self.sympy_names = set(sympy_names)
        # Names holding a SymPy value, by the same fixpoint as the taint
        # below: `law = algebra(stored)` then `law / CACHE_LEN` is SymPy's
        # division, and the operand that says so is a name by then.
        self.exact = set(enclosing.exact) if enclosing else set()
        # A nested function closes over its parent's names, so the parent's
        # taint is this scope's starting point.
        self.scalar = set(enclosing.scalar) if enclosing else set()
        self.boxed = set(enclosing.boxed) if enclosing else set()
        self.measured = _container_evidence(fn)
        if enclosing:
            self.measured |= enclosing.measured
        for a in (list(fn.args.posonlyargs) + list(fn.args.args)
                  + list(fn.args.kwonlyargs)):
            if _ann_is_extent(a.annotation):
                plural = _PLURAL_ANN.search(ast.unparse(a.annotation))
                (self.boxed if plural else self.scalar).add(a.arg)
        # Assignment is a fixpoint: `a = f(x); b = a * 2` taints both, and
        # four passes cover any chain these modules actually write.
        for _ in range(4):
            before = len(self.scalar) + len(self.boxed) + len(self.exact)
            for n in _own_nodes(fn):
                if isinstance(n, ast.AnnAssign) and _ann_is_extent(n.annotation):
                    self._bind([n.target], boxed=False)
                if isinstance(n, (ast.For, ast.comprehension)):
                    if self.carries(n.iter):
                        self._bind([n.target], boxed=False)
                    continue
                targets, value = None, None
                if isinstance(n, ast.Assign):
                    targets, value = n.targets, n.value
                elif isinstance(n, (ast.AnnAssign, ast.NamedExpr)):
                    targets, value = [n.target], n.value
                if targets is None or value is None:
                    continue
                if self.sympy_valued(value):
                    for t in targets:
                        if isinstance(t, ast.Name):
                            self.exact.add(t.id)
                if self.carries(value):
                    self._bind(targets, boxed=_is_container_expr(value))
            if len(self.scalar) + len(self.boxed) + len(self.exact) == before:
                break

    def _bind(self, targets, boxed: bool) -> None:
        for t in targets:
            if isinstance(t, ast.Name):
                (self.boxed if boxed else self.scalar).add(t.id)
            elif isinstance(t, (ast.Tuple, ast.List)):
                self._bind(t.elts, boxed)
            elif isinstance(t, ast.Starred):
                self._bind([t.value], boxed)
            elif isinstance(t, ast.Subscript):
                # `out[key] = value` stores into the container `out`; the
                # key names a slot, and holds no extent for it.
                base = t.value
                while isinstance(base, ast.Subscript):
                    base = base.value
                if isinstance(base, ast.Name):
                    self.boxed.add(base.id)

    def decides(self, node) -> bool:
        """Whether this is a call to one of the algebra's decisions
        (`_algebra_decisions`), whose answer is settled."""
        if not isinstance(node, ast.Call):
            return False
        f = node.func
        if isinstance(f, ast.Name):
            return f.id in self.decisions
        return (isinstance(f, ast.Attribute) and isinstance(f.value, ast.Name)
                and f.value.id in self.mods and f.attr in self.decisions)

    def carries(self, node) -> bool:
        """Whether this expression may still hold an expression. A decision's
        answer does not, whatever it was asked about, so its arguments are
        not read."""
        stack = [node]
        while stack:
            n = stack.pop()
            if self.decides(n):
                continue
            stack.extend(ast.iter_child_nodes(n))
            if isinstance(n, ast.Call):
                f = n.func
                if isinstance(f, ast.Name) and (f.id in self.names
                                                or f.id in self.local):
                    return True
                if isinstance(f, ast.Attribute) and isinstance(f.value, ast.Name) \
                        and f.value.id in self.mods and f.attr in self.surface:
                    return True
            if isinstance(n, ast.Name) and (n.id in self.scalar
                                            or n.id in self.boxed):
                return True
        return False

    def sympy_valued(self, node) -> bool:
        """Whether this operand is ALREADY a SymPy object, so an operator on
        it is SymPy's and answers the same way in either domain.

        Read off declarations, like everything else here: a call to an algebra
        function that declares a SymPy return (`_algebra_sympy_carriers`), or
        a constructor on the module's own sympy alias (`sp.Integer(...)`).
        Arithmetic carries it — a SymPy value times an int is a SymPy value —
        so `sp.Integer(elements) * active / params` divides exactly. A nested
        division is still its own node and still scanned, so wrapping the
        result (`sp.floor(a / b)`) hides nothing: the `a / b` inside is where
        a float would be made."""
        if isinstance(node, ast.BinOp):
            return self.sympy_valued(node.left) or self.sympy_valued(node.right)
        if isinstance(node, ast.UnaryOp):
            return self.sympy_valued(node.operand)
        if isinstance(node, ast.Call):
            f = node.func
            if isinstance(f, ast.Name):
                return f.id in self.carriers
            if isinstance(f, ast.Attribute) and isinstance(f.value, ast.Name):
                return (f.value.id in self.sympy_names
                        or (f.value.id in self.mods and f.attr in self.carriers))
        return isinstance(node, ast.Name) and node.id in self.exact

    def magnitude(self, node) -> bool:
        """Whether this expression stands for ONE magnitude that may still be
        an expression — the only thing a comparison or a truth test can be
        asking about."""
        if _is_container_expr(node):
            return False
        if isinstance(node, ast.Call):
            f = node.func
            if isinstance(f, ast.Name) and f.id in _SETTLING_CALLS:
                return False
            if isinstance(f, ast.Attribute) and f.attr in _CONTAINER_METHODS:
                return False
        if isinstance(node, ast.Name) and (node.id in self.boxed
                                           or node.id in self.measured):
            return False
        if isinstance(node, ast.Attribute):
            if node.attr in _IDENTITY_ATTRS:
                return False
            if isinstance(node.value, ast.Name) \
                    and f"{node.value.id}.{node.attr}" in self.measured:
                return False
        return self.carries(node)


def _own_nodes(fn) -> Iterable[ast.AST]:
    """Every node of `fn`'s own body, NOT descending into a nested function.

    A nested function is scanned as itself (`_functions` yields it), so
    walking into it from the parent would count its sites twice — which is
    how `physical_cost._project_flops` first reported one comparison three
    times, once per enclosing scope."""
    stack = list(ast.iter_child_nodes(fn))
    while stack:
        node = stack.pop()
        yield node
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda,
                             ast.ClassDef)):
            continue
        stack.extend(ast.iter_child_nodes(node))


def _atoms(node, out: List) -> None:
    """A condition split into the parts that each decide on their own."""
    if isinstance(node, ast.BoolOp):
        for v in node.values:
            _atoms(v, out)
    elif isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not):
        _atoms(node.operand, out)
    else:
        out.append(node)


def _numeric_literal(v) -> bool:
    return (isinstance(v, ast.Constant) and not isinstance(v.value, bool)
            and isinstance(v.value, (int, float)))


def _functions(node, stack: List[str]) -> Iterable[Tuple[str, ast.AST]]:
    for child in ast.iter_child_nodes(node):
        if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
            qual = ".".join([*stack, child.name])
            yield qual, child
            yield from _functions(child, [*stack, child.name])
        elif isinstance(child, ast.ClassDef):
            yield from _functions(child, [*stack, child.name])
        else:
            yield from _functions(child, stack)


def undecided_decision_sites() -> List[str]:
    """Sites in eval / layout that decide a magnitude question with a Python
    operator on something that may still be an expression.

    One site per (function, shape, occurrence), so a site keeps its name when
    the lines around it move: `tests/meta/test_symbolic_decisions.py` pins
    them by that name.
    """
    out: List[str] = []
    for scope in UNDECIDED_SCOPES:
        for p in _py_files(REPO / scope):
            tree = _parse(p)
            if tree is None:
                continue
            rel = _rel(p)
            names, mods = _algebra_imports(tree)
            sympy_names = _sympy_aliases(tree)
            local = {n.name for n in ast.walk(tree)
                     if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
                     and _ann_is_extent(n.returns)}
            flows: Dict[str, _ExtentFlow] = {}
            for qual, fn in _functions(tree, []):
                parent = flows.get(qual.rpartition(".")[0])
                flow = _ExtentFlow(fn, names, mods, local, parent, sympy_names)
                flows[qual] = flow
                out.extend(f"{rel}::{qual}::{kind}"
                           for kind in _decisions(fn, flow))
    tally: Dict[str, int] = {}
    numbered = []
    for site in out:
        tally[site] = tally.get(site, 0) + 1
        numbered.append(site if tally[site] == 1 else f"{site}#{tally[site]}")
    return sorted(numbered)


def _decisions(fn, flow: "_ExtentFlow") -> List[str]:
    """The shapes, in source order. Each is a way of turning "cannot decide"
    into a value with nothing written down."""
    found: List[Tuple[int, int, str]] = []
    conditions: List[ast.AST] = []
    for n in _own_nodes(fn):
        if isinstance(n, (ast.If, ast.While, ast.IfExp, ast.Assert)):
            conditions.append(n.test)
        elif isinstance(n, ast.comprehension):
            conditions.extend(n.ifs)
        elif isinstance(n, ast.Call) and isinstance(n.func, ast.Name) \
                and n.func.id in ("min", "max") \
                and any(flow.carries(a) for a in n.args):
            # `min`/`max` pick by comparing, and hand back the one they
            # picked; `dim.smaller` / `dim.larger` answer over every point.
            found.append((n.lineno, n.col_offset, "builtin-minmax"))
        elif isinstance(n, ast.BinOp) and isinstance(n.op, (ast.Div, ast.FloorDiv)) \
                and any(flow.magnitude(side) for side in (n.left, n.right)) \
                and not any(flow.sympy_valued(side)
                            for side in (n.left, n.right)):
            # `/` and `//` answer in both domains and answer differently: two
            # ints divide to a float, two laws to an exact `Rational`, and
            # `Dim.__floordiv__` floors the value half while its expression
            # half divides exactly. `exact_div` / `ceil_div` / `floor_div` /
            # `ratio` each say which one the law meant, in both domains.
            found.append((n.lineno, n.col_offset, "bare-division"))
        elif isinstance(n, ast.BoolOp) and isinstance(n.op, ast.Or) \
                and _numeric_literal(n.values[-1]) \
                and any(flow.magnitude(v) for v in n.values[:-1]):
            found.append((n.lineno, n.col_offset, "or-default"))
        elif isinstance(n, ast.Try) and _body_decides(n, flow):
            found.extend((handler.lineno, handler.col_offset, "except-TypeError")
                         for handler in _type_error_handlers(n))
    seen = set()
    for test in conditions:
        atoms: List[ast.AST] = []
        _atoms(test, atoms)
        for atom in atoms:
            at = (atom.lineno, atom.col_offset)
            if at in seen:
                continue
            kind = _atom_kind(atom, flow)
            if kind is not None:
                seen.add(at)
                found.append((*at, kind))
    return [kind for _l, _c, kind in sorted(found)]


def _atom_kind(atom, flow: "_ExtentFlow"):
    if isinstance(atom, ast.Compare):
        operands = [atom.left, *atom.comparators]
        if any(isinstance(o, _ORDER_OPS) for o in atom.ops) \
                and any(flow.carries(c) for c in operands):
            return "order-decision"
        if any(isinstance(o, _EQ_OPS) for o in atom.ops) \
                and any(flow.magnitude(c) for c in operands):
            # `Dim.__eq__` compares the value halves, so two extents that
            # differ as expressions test equal.
            return "equality-decision"
        return None          # `is` / `in` ask identity, which is decidable
    return "zero-test" if flow.magnitude(atom) else None


def _type_error_handlers(node: ast.Try) -> List[ast.ExceptHandler]:
    out = []
    for h in node.handlers:
        caught = h.type
        if caught is None or (isinstance(caught, ast.Name) and caught.id == "TypeError") or (isinstance(caught, ast.Tuple) and any(
                isinstance(e, ast.Name) and e.id == "TypeError"
                for e in caught.elts)):
            out.append(h)
    return out


def _body_decides(node: ast.Try, flow: "_ExtentFlow") -> bool:
    """Whether the guarded body asks a magnitude question — a comparison, or
    a cast that takes one half. Those are what raise the `TypeError` the
    handler then answers in their place."""
    for stmt in node.body:
        for x in [stmt, *_own_nodes(stmt)]:
            if isinstance(x, ast.Compare) \
                    and any(isinstance(o, _ORDER_OPS + _EQ_OPS) for o in x.ops) \
                    and any(flow.carries(c) for c in [x.left, *x.comparators]):
                return True
            if isinstance(x, ast.Call) and isinstance(x.func, ast.Name) \
                    and x.func.id in ("int", "float", "round") \
                    and any(flow.carries(a) for a in x.args):
                return True
    return False


# ---------------------------------------------------------------------------
# Counter 10: a gate that stops gating where the domain is symbolic
# ---------------------------------------------------------------------------
#
# `ops_registry` checks an op's inputs with an `input_contract`, and reads the
# extents it checks through a helper declared `-> Optional[int]`: the number,
# "or None where the domain is symbolic" (`_extent`, and its duplicate
# `_known`). Every caller then guards on that None and SKIPS its check. The
# check is a gate; skipping it is the gate not gating, and nothing says so —
# the walk reports a clean contract either way.
#
# The fix is not a default: it is to carry the question instead of dropping
# it, which is what `dim.require_multiple`, `dim.require_equal` and
# `dim.require(sp.Le(…))` / `dim.require(sp.Ge(…))` are for (a condition
# recorded on the walk, refused at a point that breaks it). Until
# then the number is the record. `tests/meta/test_symbolic_decisions.py`
# (`ALLOWED_SKIPS`) lists each skip and the check it drops.
#
# The scan reads declarations: a "settling helper" is a function annotated
# `-> Optional[int]` whose body calls `value_of` (so it is de-symbolising an
# extent), and a skip is a `None` test on a value that came from one.

SETTLING_SCOPES = ("backend/arch",)


def _settling_helpers(tree) -> set:
    """Functions in this module that turn an extent into "a number, or None
    where the domain is symbolic" — read off the return annotation plus the
    `value_of` call, not a name."""
    out = set()
    for n in ast.walk(tree):
        if not isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        if n.returns is None or "Optional[int]" not in ast.unparse(n.returns):
            continue
        if any(isinstance(c, ast.Call) and isinstance(c.func, ast.Name)
               and c.func.id == "value_of" for c in ast.walk(n)):
            out.add(n.name)
    return out


def symbol_skipping_gate_sites() -> List[str]:
    """`None` guards on a de-symbolised extent: each one a check that does
    not run where the extent is an expression."""
    out: List[str] = []
    found: List[Tuple[int, int, str]] = []
    for scope in SETTLING_SCOPES:
        for p in _py_files(REPO / scope):
            tree = _parse(p)
            if tree is None:
                continue
            helpers = _settling_helpers(tree)
            if not helpers:
                continue
            rel = _rel(p)
            for qual, fn in _functions(tree, []):
                if fn.name in helpers:
                    continue
                settled = _settled_names(fn, helpers)
                for n in _own_nodes(fn):
                    if not isinstance(n, ast.Compare):
                        continue
                    if not any(isinstance(o, (ast.Is, ast.IsNot, ast.In,
                                              ast.NotIn)) for o in n.ops):
                        continue
                    operands = [n.left, *n.comparators]
                    if not any(isinstance(x, ast.Constant) and x.value is None
                               for x in operands):
                        continue
                    if any(_mentions(x, settled) for x in operands):
                        found.append((n.lineno, n.col_offset,
                                      f"{rel}::{qual}::skip"))
    # Named by (function, occurrence) rather than by line, so a site keeps
    # its name while the code around it moves.
    tally: Dict[str, int] = {}
    for _l, _c, site in sorted(found):
        tally[site] = tally.get(site, 0) + 1
        out.append(site if tally[site] == 1 else f"{site}#{tally[site]}")
    return sorted(out)


def _settled_names(fn, helpers: set) -> set:
    """Names in `fn` bound from a settling helper — directly, or through a
    comprehension over one (`[_extent(e) for e in shape]`)."""
    out = set()
    for n in _own_nodes(fn):
        targets, value = None, None
        if isinstance(n, ast.Assign):
            targets, value = n.targets, n.value
        elif isinstance(n, (ast.AnnAssign, ast.NamedExpr)):
            targets, value = [n.target], n.value
        if targets is None or value is None:
            continue
        if any(isinstance(c, ast.Call) and isinstance(c.func, ast.Name)
               and c.func.id in helpers for c in ast.walk(value)):
            for t in targets:
                for x in ast.walk(t):
                    if isinstance(x, ast.Name):
                        out.add(x.id)
    return out


def _mentions(node, names: set) -> bool:
    return any(isinstance(n, ast.Name) and n.id in names for n in ast.walk(node))


# ---------------------------------------------------------------------------
# The facts ledger — the positive half of the board
# ---------------------------------------------------------------------------
#
# The counters above go DOWN. They cannot tell a declarative repo from one
# that patched every number into place, because both read zero. This half
# goes UP: for each fact the engine needs, three observables —
#
#   declared:       there is exactly one place that states it (or none yet,
#                   in which case creating it is the first work item and the
#                   only kind of schema change this plan permits);
#   rederivations:  sites that restate it instead of asking — a name list,
#                   a proxy, an arithmetic recovery. Regex per fact, scoped;
#   gate:           a test that would go red if a second statement appeared.
#
# A fact is DONE when it is declared, has zero rederivations, and has a gate.
# Coverage is done facts / all facts. A patch that makes a number pass
# without moving a consumer onto the declaration leaves this half flat —
# which is the point.

@dataclass(frozen=True)
class Fact:
    key: str
    title: str
    #: `path:symbol` of the single declaration, or None when none exists yet.
    declared: "str | None"
    #: (regex, scope dirs relative to repo, excluded path substrings)
    rederive: Tuple[str, Tuple[str, ...], Tuple[str, ...]]
    #: pytest node id of the gate, or None.
    gate: "str | None" = None


FACTS: Tuple[Fact, ...] = (
    Fact("op_has_weight", "op 有没有存储权重", "backend/arch/ops.py:OpSpec.has_weight",
         (r"^\s*(_DIRECT_WEIGHT_OPS|_WEIGHT_OPS|PARAMETER_OPS)\s*[:=]", ("backend",), ()),
         gate="tests/backend/arch/test_component_rules_ask_the_registry.py::TestAttentionComponentNeedsACacheFamily::test_a_buffer_counts_as_a_stored_tensor"),
    Fact("op_arity", "op 声明了几个输入（零输入源）", "backend/arch/ops.py:OpSpec.arity",
         (r"^\s*_ZERO_INPUT_OPS\s*[:=]|is_identity_source\(node\.op\):\s*$", ("backend/arch/schema", "backend/arch/eval/evaluator.py"), ()),
         gate="tests/backend/arch/test_component_rules_ask_the_registry.py::TestZeroInputOpsRefuseInputs::test_parameter_tensor_with_in_is_refused"),
    Fact("op_stored_tensors", "op 在 checkpoint 存哪些张量（含 bias）", "backend/arch/ops.py:OpSpec.tensors",
         (r"^\s*(_OPS_WITH_DECLARED_BIAS|_OP_TENSOR_SUFFIXES)\s*[:=]|op_name in \(\"linear\"", ("backend",), ()),
         gate="tests/backend/arch/test_op_projections.py::TestStoredTensorsAreOneDeclaration::test_per_head_linear_declares_the_bias_its_params_count"),
    Fact("op_is_collective", "op 是不是集体通信", "backend/arch/ops.py:OpSpec.is_comm_op",
         (r"\"comm_world\" in m|endswith\(\"all_reduce\"\)|COMM_OPS = new Set", ("backend/inference", "backend/arch/layout/comm_cost.py"), ()),
         gate="tests/backend/inference/test_comm_meta_marks_collectives.py::TestCommMetaOnlyOnCollectives::test_every_comm_meta_row_is_a_declared_collective"),
    Fact("op_gemm_family", "op 属不属于 GEMM 家族", "backend/arch/ops.py:OpSpec.gemm (custom ops: CustomOpSpec.is_linear_map, joined in ops_base.declaration_is_gemm)",  # 先造：ops.py 上的 is_gemm_op 投影；OP_TO_SIGNATURE 是 kernel 目录，住在下游
         (r"^\s*(_GEMM_OPS|_GEMM_FAMILY)\s*[:=]", ("backend", "kernel_bench"), ()),
         gate="tests/backend/arch/test_op_projections.py::TestGemmIsAnAuthorFact::test_every_registered_op_measured_as_gemm_is_declared_gemm"),
    Fact("role_vocabulary", "role 词表", "backend/arch/roles.yaml (attributes: weight_class / ffn_gate)",
         (r"^\s*(_SLOT_ROLES|_FFN_GATE_ROLES|_KNOWN_ROLES)\s*[:=]|role: Optional\[Literal\[", ("backend",), ()),
         gate="tests/backend/arch/test_role_attributes.py::TestAttributesAreColumns::test_weight_classes_and_gate_roles"),
    Fact("dim_kinds", "dim 的 kind / 可切轴", "backend/arch/dims.yaml",
         (r"existing\.get\(\"worlds\"\)|\.get\(\"world\"\)", ("backend/arch/promote_op_proposal.py",), ()),
         gate="tests/backend/arch/test_promote_op_proposal_gate.py::TestProposalsSpeakTheRegistrysVocabulary::test_an_undeclared_world_is_refused_by_name"),
    Fact("dtype_names_and_widths", "dtype 的名字与宽度", "backend/arch/dtypes.py",
         (r"^\s*(_TORCH_DTYPE_NAMES|_DTYPE_DISPLAY|_DTYPE_TORCH_NAMES|_MAT_DTYPE_ALIAS|DTYPE_VOCABULARY|DTYPE_ALIASES|UNQUANTIZED|_HUMMING_WEIGHT_DTYPE)\s*[:=]|def dtype_bytes_to_name", ("backend", "kernel_bench"), ("backend/arch/dtypes.py", "kernel_bench/providers/gemm.py")),
         gate="tests/backend/arch/test_dtype_table.py::TestTheOldTablesAreColumns::test_the_corpus_vocabulary"),
    Fact("weight_bytes", "权重字节", "backend/arch/eval/analysis.py:Analysis.tensors",
         (r"\* sp\.Integer\(2\)|total_params \* 2|total_params \* dtype_bytes|bytes_per_param =", ("backend",), ("backend/arch/eval/analysis.py",)),
         "tests/backend/arch/eval/test_one_join.py::test_dashboard_weights_are_tensors_join"),
    Fact("hbm_access", "叶子的 HBM 访存", "backend/arch/eval/traffic.py:derive",
         (r"n\[\"access_bytes\"\] = \(", ("backend/rendering",), ()),
         "tests/backend/arch/eval/test_one_join.py::test_leaf_tooltip_access_matches_traffic"),
    Fact("tied_weight", "哪个头和哪个 embedding 共享权重", "spec yaml tied_weight → backend/arch/spec_tree/build.py:owns_weight",
         (r"tie_word_embeddings", ("backend/arch/eval", "backend/rendering", "backend/analyzer.py"), ()),
         "tests/backend/arch/eval/test_one_join.py::test_gemma2_tied_head_reads_the_embedding"),
    Fact("expert_bank", "一个叶子代表几份专家拷贝", "spec yaml 权重的 shape: 前导轴 n_expert_copies → backend/arch/eval/storage.py:leads_with_expert_axis",
         (r"ratio = params / elements", ("backend",), ()),
         "tests/backend/arch/eval/test_expert_pools.py::test_expert_parallelism_divides_each_pool_by_its_own_count"),
    Fact("hybrid_attention", "模型是不是混合注意力", "backend/arch/eval/program.py:Program.attention_mix（backbone 的注意力 / 序列混合器）；增长律是否不止一种是 backend/arch/eval/cache_view.py:CacheView.mixed_growth_laws",
         (r"len\(_variant_order\) > 1|len\(_seen\) > 1", ("backend",), ()),
         gate="tests/backend/arch/test_declared_identities.py::TestHybridIsTheBackbonesMixers::test_one_family_over_dense_and_moe_layers_is_not_hybrid"),
    Fact("moe_identity", "是不是 MoE / MoE 子块的角色", "backend/arch/eval/program.py:Program.has_dispatch_zone / routed_uids + roles.yaml `router`",
         (r"def _moe_child_class|def _coarse_role|\"n_experts\"\) or 0\) > 1", ("backend", "kernel_bench"), ()),
         gate="tests/backend/arch/test_declared_identities.py::TestDispatchZoneIsTheMoeFact::test_the_zone_reaches_the_bank_body_and_its_parameters"),
    Fact("quant_scheme_identity", "checkpoint 用的量化方案", "backend/arch/schema/spec.py:ArchSpec.quant_context (spec-gen 侧唯一入口 hf/snapshot.fetch_metadata_raw)",
         (r"resolve_scheme\(", ("backend/hf/metadata.py",), ()),
         gate="tests/backend/hf/test_header_only_metadata.py::TestHeadersAreEnough::test_the_hub_fetcher_takes_no_config"),
    Fact("supabase_mode", "部署配没配 Supabase / 开没开 auth", "backend/supabase_env.py (configured / jwt_verification_available / browser_auth_available)",
         (r"def auth_enabled\(|def _supabase_enabled\(|auth_on = bool\(|def configured\(", ("backend",), ("backend/supabase_env.py",)),
         gate="tests/backend/service/test_auth.py::TestSupabaseMode::test_three_questions_three_answers"),
    Fact("cache_family", "一个 cache 属于哪个 family（MLA/GQA/SWA/…）", "backend/arch/cache_families.yaml (kind per family)",
         (r"^\s*_SLIDING_LABELS\s*[:=]", ("backend",), ()),
         gate="tests/backend/arch/test_cache_family_registry.py::TestTheRegistry::test_the_corpus_families_are_registered_with_kinds"),
    Fact("custom_op_layout_family", "custom op 在 layout 里按哪族分派", "backend/arch/schema/custom_ops.py:CustomOpSpec.layout_family / layout_axis_use()",
         (r"^\s*(_LAW_SLOTS|_WEIGHT_OPS|AXIS_USE_CUSTOM)\s*[:=]|\"(mxfp4_fused_linear|packed_expert_linear|gla_simple_recurrence|qwen35_spatial_merge)\"", ("backend/arch/layout",), ()),  # _HANDLERS stays: the registered-op walk table; custom ops must not appear in it by name
         gate="tests/backend/arch/layout/test_custom_family_dispatch.py::TestFamilyComesFromTheDeclaration::test_the_three_families"),
    Fact("component_is_adapter", "一个 block 是不是别名包装（alias adapter）", "backend/arch/schema/graph.py:BlockGraph.alias_target (pre/post_norm 封装另问 BlockGraph.adapted_wrapper)",
         (r"def _is_pure_adapter|len\(nodes\) == 1:", ("backend/arch",), ("backend/arch/structure.py",)),
         gate="tests/backend/arch/test_declared_identities.py::TestAliasAdapterIsOneQuestion::test_type_contracts_and_the_loader_agree_on_every_component"),
    Fact("gpus_per_node", "一个节点几张卡", "backend/inference/gpu_specs.csv:gpus_per_node",
         (r"gpus_per_node: int = 8|gpus_per_node=8|return 8$|8 \* int\(parallel", ("backend",), ("gpu_specs.csv",)),
         gate="tests/backend/inference/test_gpu_specs.py::TestGPUSpecsLoader::test_gpus_per_node_refuses_a_missing_column"),
    Fact("config_field_optionality", "config 字段可不可以缺", "backend/arch/schema/model.py:BlockAssignment.config_list_optional (authored: `from: config.X ?? absent`)",
         (r"if not isinstance\(lst, list\):\s*\n\s*(return set\(\)|continue)", ("backend/arch/loader.py", "backend/arch/validate"), ()),
         gate="tests/backend/arch/test_config_list_optionality.py::TestAbsenceIsDeclared::test_an_undeclared_absent_list_is_refused"),
)


def _code_lines(p: Path) -> List[Tuple[int, str]]:
    """(lineno, line) for lines that are code — not `#` comments and not
    inside a string literal / docstring. Both queries below read this, so a
    docstring that explains why something must NOT be done never counts as
    doing it (`deployment_dtype.py` lost three points to its own warnings)."""
    try:
        src = p.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return []
    lines = src.splitlines()
    if p.suffix != ".py":
        return [(i, ln) for i, ln in enumerate(lines, 1) if not ln.strip().startswith("//")]
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return list(enumerate(lines, 1))
    in_string: set = set()
    for n in ast.walk(tree):
        if isinstance(n, ast.Constant) and isinstance(n.value, str) and n.end_lineno and n.end_lineno > n.lineno:
            in_string.update(range(n.lineno, n.end_lineno + 1))
        elif isinstance(n, ast.Expr) and isinstance(n.value, ast.Constant) and isinstance(n.value.value, str):
            in_string.update(range(n.lineno, (n.end_lineno or n.lineno) + 1))
    return [(i, ln) for i, ln in enumerate(lines, 1)
            if i not in in_string and not ln.strip().startswith("#")]


# ---------------------------------------------------------------------------
# Counter 8: `repeat:` left on a graph node
# ---------------------------------------------------------------------------

#: Where authored graphs live: the component library (plus adapters and
#: drafts) and the model specs, whose `blocks:` / `components:` bodies carry
#: inline graphs of their own.
_SPEC_YAML_ROOTS = ("backend/arch/components", "backend/arch/models")


def _graph_node_keys(doc: object, key: str) -> Iterable[Tuple[str, dict]]:
    """Every `(node name, node body)` a spec document declares under a
    `graph:`, at any depth, for nodes carrying `key`.

    Structural rather than textual, so a stack's `repeat:` (which lives in a
    `model:` list and means the model's own iteration) is never counted as a
    node's.
    """
    if isinstance(doc, dict):
        graph = doc.get("graph")
        if isinstance(graph, dict):
            for name, body in graph.items():
                if isinstance(body, dict) and key in body:
                    yield str(name), body
        for value in doc.values():
            yield from _graph_node_keys(value, key)
    elif isinstance(doc, list):
        for value in doc:
            yield from _graph_node_keys(value, key)


def graph_node_sites(key: str) -> List[str]:
    """`path::node` for every graph node declaring `key`."""
    import yaml

    out: List[str] = []
    for scope in _SPEC_YAML_ROOTS:
        for p in sorted((REPO / scope).rglob("*.yaml")):
            doc = yaml.safe_load(p.read_text(encoding="utf-8"))
            for name, _body in _graph_node_keys(doc, key):
                out.append(f"{_rel(p)}::{name}")
    return out


def repeat_node_sites() -> List[str]:
    return graph_node_sites("repeat")


def rederivation_sites(fact: Fact) -> List[str]:
    pattern, scopes, excluded = fact.rederive
    rx = re.compile(pattern)
    out = []
    for scope in scopes:
        root = REPO / scope
        files = [root] if root.is_file() else sorted(
            p for p in root.rglob("*") if p.suffix == ".py" and "__pycache__" not in p.parts
            and "node_modules" not in p.parts and "dist" not in p.parts)
        for p in files:
            rel = _rel(p)
            if any(x in rel for x in excluded):
                continue
            for i, line in _code_lines(p):
                if rx.search(line):
                    out.append(f"{rel}:{i}")
    return out


def measure_facts() -> Dict[str, List[str]]:
    return {f.key: rederivation_sites(f) for f in FACTS}


def fact_done(fact: Fact, sites: List[str]) -> bool:
    return fact.declared is not None and not sites and fact.gate is not None

# ---------------------------------------------------------------------------
# The board
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Counter:
    key: str
    title: str
    definition: str
    sites: Callable[[], List[str]]
    target: int = 0


COUNTERS: Tuple[Counter, ...] = (
    Counter("silent_except", "静默 except",
            "`backend/` 里 handler body 只有 pass / continue / return 空值（可先打一行 debug 日志）的 except（CLAUDE.md §1 静默兜底）",
            silent_except_sites),
    Counter("name_lists", "名单常量",
            "`backend/` + `kernel_bench/` 模块级 `UPPER_SNAKE` 字符串集合/元组/字典字面量（CLAUDE.md §1 第二张表）",
            name_list_sites),
    Counter("width_defaults", "宽度字面量默认",
            "`backend/` 里 dtype 名字面量（词表取自 `arch/dtypes.accepted_dtype_names()`）或 dtype 声明宽度列（`DTYPE_BYTES`）里的数字被当默认值：函数签名的形参默认、函数体里的 `x or \"bf16\"` / `d.get(k, \"bf16\")`；数字只在形参名 / 键名带 dtype/bytes/width/bpe 时算（`degree or 1` 不算）（CLAUDE.md §2.4 缺声明不给默认）",
            width_default_sites),
    Counter("import_cycles", "import 环",
            "`backend/` 内 ≤12 模块的 import 环（limit 4 只看得见 1/3——kimi R2）",
            import_cycle_sites),
    Counter("lazy_edges", "函数体内 backend import",
            "`backend/` 内一个模块在函数体里 import 另一个 backend 模块（CLAUDE.md §2.7）",
            lazy_edge_sites),
    Counter("defensive_access", "getattr/hasattr 探针",
            "`backend/` 全部 getattr / hasattr 调用点（CLAUDE.md §2.7 禁在 eval；inference 28 处、layout 的 `getattr(rule)` 都曾在扫描范围外）",
            defensive_access_sites),
    Counter("repeat_sites", "图节点上的 `repeat:`",
            "`backend/arch/components/**` + `backend/arch/models/**` 里还按拷贝数建模的并行 bank（`repeat:` 在图节点上）。"
            "shape 改造把每个 bank 换成权重的前导轴 + 一次 batched matmul，换完一个就少一个；"
            "链式拷贝（塔的 block 栈、MiMo 的 bank 累加）已经改用 `chain:`，不在这个数里。"
            "到 0 时这个关键字连同它的引擎分支一起删掉——这一列就是 shape 改造的验收信号",
            repeat_node_sites),
    Counter("downstream_derivation", "eval 之外自己造事实",
            "inference / rendering / api / service / views / playground / kernel_bench 里不读 `Analysis` 而去问 op 注册表、按 op 名或模块级 op 名集合分支（`c[\"op\"] in _X`）、读 dims 表（任何 `.dim_values`、`getattr(x, \"dim_values\")`、`config_path_value` / `eval_dim_expr`，以及调用 eval 之外整个答案就是一次 `dim_values` 读取的包装方法，如 `ResolvedSpec.get_dim`——包装集合从定义推出，不列名单）、读原始 config（`resolved.config`、类型为 `ResolvedSpec` 的名字的 `.config`、`getattr(x, \"config\")`）、保留\"eval 没给答案时\"的备胎（含 `json.loads(x or \"{}\")`、`row.get(\"node_dims\") or …`）、按 id 子串判类别的行；import 行不算（边界规则：eval 是事实唯一生产者）",
            downstream_derivation_sites),
    Counter("undecided_decisions", "拿 Python 运算符判一个还可能是表达式的量",
            "`backend/arch/eval` + `backend/arch/layout` 里对一个「可能还是表达式」的 extent 用 Python 运算符问大小的点："
            "`>` / `<` 比大小、`==` / `!=` 比相等、`if extent:` 问是不是零、`or 0` 垫一个数、`min` / `max` 挑一个、"
            "把 `TypeError` 接住当答案（`traffic._wider` 曾在这里悄悄计了两个不可比跨度中的第一个）、"
            "以及拿 `/` 或 `//` 除（这一条不是「拒绝后硬答」而是「两个域各答一个」：两个 Python int 相除是 float，"
            "两条式子相除是精确的 Rational，`layout.physical.pairs_share` 就因此把同一个 pair 数在一个读者那里"
            "发成 \"4224.00000000000\"、在另一个那里发成 4224。修法是改用声明意图的动词："
            "`exact_div`（声明整除，不整除就 raise）/ `ceil_div` / `floor_div`（法则要的取整）/ "
            "`ratio(part, whole, empty=…)`（比例，强迫调用方交代分母为零是什么意思）/ 两半都 `exact` 再除。"
            "凡经过声明返回 SymPy 的算子（`exact` / `algebra` / `ratio_symbol`）的除法本来就两域同答，不计）。"
            "判据是「这次判定有没有留下记录」：改成问 `dim` 的判定函数（over 所有点回答，或记成假设、或拒绝）就不算。"
            "分类器靠声明推导（算子表面解析自 `eval/dim.py` + `eval/values.py`，extent 身份看 `Extent` / `Dim` 注解），"
            "既有误报也有漏报——所以 `tests/meta/test_symbolic_decisions.py` 在这个数旁边逐点点名，新增一个点才是红",
            undecided_decision_sites),
    Counter("symbol_skipping_gates", "符号域里停止把关的门",
            "`backend/arch` 里一道检查因为 extent 是表达式而被跳过的点：把 extent 读成「数字，符号域给 None」的"
            "helper（声明 `-> Optional[int]` 且调 `value_of`）之后那个 `None` 守卫。"
            "门看见符号就放行，而走访报告的合约是干净的——修法不是给默认值，是把问题带下去"
            "（`dim.require_multiple` / `dim.require_equal`，不等式用 `dim.require(sp.Le/Ge(…))`：记成负载的定义域条件，在破坏它的点上拒绝）。"
            "逐点清单和每条跳掉的检查，见 `tests/meta/test_symbolic_decisions.py` 的 `ALLOWED_SKIPS`",
            symbol_skipping_gate_sites),
)


def measure() -> Dict[str, List[str]]:
    out = {c.key: c.sites() for c in COUNTERS}
    out.update({f"fact:{k}": v for k, v in measure_facts().items()})
    return out


def load_baseline() -> dict:
    return json.loads(BASELINE_PATH.read_text(encoding="utf-8"))


def render(measured: Dict[str, List[str]], baseline: dict) -> str:
    rows = []
    for c in COUNTERS:
        now = len(measured[c.key])
        b = baseline.get(c.key, {})
        start = b.get("start", now)
        floor = b.get("baseline", now)
        arrow = "↓" if now < start else ("=" if now == start else "↑")
        rows.append(f"| `{c.key}` | {c.title} | {c.definition} | {start} | **{now}** | {floor} | {c.target} | {arrow} |")
    head = (
        "# 记分板\n\n"
        f"起点 {START_DATE}。生成自 `python -m tools.scoreboard`，"
        "`tests/meta/test_scoreboard.py`（`regression`，不卡推送和部署）核对这页与计数一致。计数按模式匹配，是规则的代理指标：新增条目在评审时用 `--list` 判断是真问题还是误报。\n\n"
        "每个计数器是一种**永远不接受新增**的债的形状（CLAUDE.md §1/§2 的规则，写成 AST 查询）。"
        "\"越来越好\" = 这几列往下走；\"好了\" = 到目标。降一个数字的方式只有一种：删掉那个实例，并在同一个 commit 里把 baseline 改小。"
        "唯一例外是**重新计数**：计数器的定义被修正、看见了原先漏数的实例时，在同一个 commit 里一次性把 baseline 抬到修正后的数（`--rebaseline --allow-raise`），并在提交说明里逐条交代新看见的实例。"
        "首次重新计数：silent_except 83→88（只打 debug 日志的 handler、空值元组此前看不见）、name_lists 54→57（其他批次新增而 baseline 未重测的名单）是计数器修正的补账，不是放行新债。"
        "第二次重新计数：width_defaults 10→23（函数体里的 dtype 字面量兜底此前看不见）、downstream_derivation 0→39（views / playground 不在范围内；`.dim_values`、`getattr(x, \"config\")`、dims 表达式求值、`json.loads(x or \"{}\")`、`node_dims` 兜底、按模块级 op 名集合分支此前看不见），同样是补账。"
        "第四次重新计数：downstream_derivation 0→5（只把 `dim_values` 读取改个名的包装方法此前看不见）——这个盲区让数字靠改写而不是靠把事实挪进 eval 降下来，补上后如实抬高：本分支的基（2b4dfa0de，只把 `tools/scoreboard.py` 换成本分支的）上读 7，删掉那几个包装之后、在本分支 tip 上读 5。每个数都带着它是在哪个 commit 上量的——两次把别的树上的数拿来和这棵树比，就是这页之前失准的全部原因。\n\n"
        "| key | 计数器 | 定义 | 起点 | 现在 | 门（baseline） | 目标 | 趋势 |\n"
        "|---|---|---|---|---|---|---|---|\n"
    )
    frows = []
    done = 0
    for f in FACTS:
        sites = measured.get(f"fact:{f.key}", [])
        b = baseline.get(f"fact:{f.key}", {})
        start = b.get("start", len(sites))
        ok = fact_done(f, sites)
        done += ok
        frows.append(
            f"| `{f.key}` | {f.title} | {('`' + f.declared + '`') if f.declared else '**无 — 先造声明**'} "
            f"| {start} → **{len(sites)}** | {('`' + f.gate + '`') if f.gate else '无'} | {'✓' if ok else '·'} |")
    facts_table = (
        f"\n\n## 事实账本（正向：{done}/{len(FACTS)} 达标）\n\n"
        "**每个事实：一处声明、零处复述、一道门。** 达标 = 三项全有。上面的计数器量的是坏东西变少；"
        "这张表量的是声明变多——一个把数字凑到零的补丁在这张表上不会动，正是要的效果。"
        "\"声明式优先\"就是这张表的工作顺序：先给\"无\"那几行**造声明**（唯一允许改 schema 的地方），"
        "再把复述一处处换成问声明，最后加门。\n\n"
        "| key | 事实 | 声明处 | 复述处 起点→现在 | 门 | 达标 |\n|---|---|---|---|---|---|\n"
        + "\n".join(frows)
    )
    tail = (
        facts_table +
        "\n\n豁免是声明出来的，不是注释：见 `tools/scoreboard.py` 里每个计数器的 `*_EXEMPT`，每条带理由。\n"
        "\n**name_lists 的一个已知盲区，以及为什么这个数不能单独当门。** 这个计数器只扫模块级字面量，"
        "所以一张表挪进函数体、或一个模块常量由本模块一个只包一层字面量的函数造出，它都看不见——"
        "一个数于是可以靠换拼法降下来，而事实一步没动。2026-10-31 用一段 AST 探针在本分支的基 2b4dfa0de 上量过："
        "同口径下模块级读 51，连函数体内的表和包一层的常量一起读 84（那 84 里有 7 条是装载期自检的声明表、按名豁免；不豁免则 90），也就是 30 多处盲区实例。"
        "探针没有合并，因为它看见的是找到的那三种拼法、不是那一类债：表挪进类体、包装函数加一个形参、"
        "`_a() + _b()` 拼装、字典推导、按 `.items()` 查、同名绑定两次，六种写法一行就能绕开它——"
        "把 baseline 钉在这样一个数上，等于把门交给拼法。这一项的验收标准落到下半张**事实账本**的一行"
        "（一处声明、零处复述、一道门），不是把计数器的数凑小；要留住一处正当的表，用按站点的豁免 + 理由。"
        f"同一个原因，name_lists 现在 {len(measured['name_lists'])}、门 "
        f"{baseline['name_lists']['baseline']}：正当的表按站点豁免并写明理由；"
        "要抬这个门，必须连同上面那本盲区账一起做。\n"
        "\n**一个登记在册、而且任何计数器都看不见的代理条件：** `naive_pattern.runs(\"hadamard_transform\")`"
        "（`kernel_bench/providers/indexer.py:551`、`backend/inference/op_signatures_catalog.py:1740`）"
        "拿「plan 里有没有 Hadamard」当「框架这次 launch 省掉了 plan 的哪个成员」的开关，正是 CLAUDE.md §1.3 说的代理条件："
        "两者今天只在这两个 signature 上重合。诚实的形态是在 framework binding 上声明 `omits:`（省掉了 plan 的哪个成员），"
        "doc 文本和 oracle 的 canonicalize 都从它派生——要动 binding schema 和两个框架的全部条目，留给做那一步的人。"
        "第二句同样要说清：`downstream_derivation` 永远看不见它，因为 `op_signatures_catalog.py` 整文件在 `DOWNSTREAM_EXEMPT` 里，"
        "而那条文件级豁免发给的理由（「kernel signature table」）盖不住同一文件里按 op 名分支的这一点。\n"
        "\n不在板上、也不该在板上的：具名例外产生的**已知错数**——现均已收口：没有图读者的存储参数只能由 spec 的 `stored_without_graph_reader` 声明，`test_borrowed_weight_reads` 要求每个模型的无读者集合恰好等于声明集合，validate 对未声明或声明了却有读者的报可修复错误（全语料无 awaiting-op 孤儿权重）。收口："
        "Step-3.7 downsampler（`conv2d`）、gemma-4 位置表（`add_pe` component）、MiMo 二维窗口（`spatial_merge`）、buffer 读归属（`OpSpec.is_side_buffer` 声明）。\n"
        "\n一个**没有收口的缺口**，写在这里而不是假装它已经收了：`test_ep_as_count` 锁的是一条已发布总量上的恒等式——"
        "把每个 bank 的拷贝数 `n_expert_copies` 和它的路由分母 `n_route_targets` **同乘**一个因子，`total().flops` 一字不变、`total().params` 变大。"
        "（专家数并不是「不进计算」：它在计算里出现两次、符号相反而相消——bank 权重的前导专家轴是拷贝数，每个专家的行数 `M` 又把同一个数放在分母上。"
        "恒等式说的是这个相消成立，不是说单条表达式里没有专家数：一次 grouped launch 覆盖 `E_local` 个专家、做 `E_local × M × w` 的活，这条锁不禁止。）"
        "锁住的是 FLOPs 和 params，不是 bytes：**buffer 读归属按字节算还没有门**。"
        "要有门，得在同一层上断言「激活字节不变、权重字节变」，而这需要 `Analysis` 上有一个已发布的 traffic 总量——今天没有，"
        "自己在消费方凑一个聚合就是第二条推导路径。留给做 traffic 那一步的人。\n"
        "\n**同一把锁的第二个缺口：全局有门、per-rank 没门。** 那条恒等式只在已发布总量上成立。到了单个 rank 上，专家轴是 `E_local`、"
        "而 `M` 的分母仍是整池，于是只有 `E_local == E / ep` 恰好整除时才消得干净；而 per-rank extent 走的是决策 5 的 "
        "`padded_global_extent // gcd(extent, degree)`（`layout/resolver.shards_under_policy`、`layout/layout.py:206`），"
        "gcd 或 padding 一咬，`E_local × M` 就留下余项，per-rank 账和全局公式不再对得上——`test_ep_as_count` 断的是 `total()`，"
        "带着这个余项它照样绿。要有门，得有一个**不整除 EP degree** 的专家数（`gcd(E, ep) < ep`），并要求 per-rank 总和等于全局总量。"
        "这属于 launch-geometry 那一步（per-rank `E_local` 正是它的主体），在那里立。\n"
    )
    return head + "\n".join(rows) + tail


def main(argv: List[str]) -> int:
    measured = measure()
    if "--rebaseline" in argv:
        old = load_baseline() if BASELINE_PATH.exists() else {}
        keys = [c.key for c in COUNTERS] + [f"fact:{f.key}" for f in FACTS]
        # A rebaseline records what a commit SAVED. Raising a floor spends
        # it back, so that takes an explicit flag and names what it raises.
        raised = [f"{k}: {old[k]['baseline']} -> {len(measured[k])}" for k in keys
                  if k in old and len(measured[k]) > old[k]["baseline"]]
        if raised and "--allow-raise" not in argv:
            print("refusing to raise baselines (pass --allow-raise to do it on purpose):\n  "
                  + "\n  ".join(raised))
            return 1
        new = {k: {"start": old.get(k, {}).get("start", len(measured[k])),
                   "baseline": len(measured[k])} for k in keys}
        BASELINE_PATH.write_text(json.dumps(new, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"wrote {BASELINE_PATH.relative_to(REPO)}")
    baseline = load_baseline() if BASELINE_PATH.exists() else {}
    page = render(measured, baseline)
    if "--list" in argv:
        key = argv[argv.index("--list") + 1]
        print("\n".join(measured[key]))
        return 0
    print(page)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
