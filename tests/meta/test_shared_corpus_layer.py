"""The two rules that keep a shared walk from being mutated or re-walked.

WHAT THIS IS ABOUT, IN PLAIN TERMS
----------------------------------
Analysing a model here means resolving it, expanding its component graph,
building a `Program` and walking it. Measured on DeepSeek-V4-Flash: ~2 ms,
~140 ms, ~900 ms of sympy. Most test files do not sweep the corpus -- they
pick one or two real models to check one behaviour, which is right -- and
each of them paid all three again. `tests/snapshots.shared_program` shares
the first two for the process, and `tests/conftest.py`'s `walks` fixture
shares the walk with the rest of one test MODULE.

Sharing means several tests hold the SAME objects, and neither of the two
that matter is really immutable. So the sharing comes with three
obligations, and these checks are those obligations enforced rather than
written down and hoped for.

RULE 1 -- a sharing file does not walk privately by accident
    A file that takes `shared_walk` must not also call a bare `analyze()`.
    If it genuinely needs its own walk it says so with
    `own_walk(deployment, why=...)`, whose `why` sits at the call site.

RULE 2 -- a sharing file does not mutate the analysis it is given
    Two shapes, because an `Analysis` is not frozen and both reach every
    other test in the module:
      * assigning on it -- `analysis.domain = None`, a layer, any field --
        where `analysis` came from `shared_walk(...)`;
      * `Analysis.assumed()`, which writes the assumption ledger back when
        its scope closes (`self.assumptions = tuple(ledger.held)`).
    A file doing either must take `own_walk` instead of sharing. The first
    of the two was promised by this module before it was checked: a file
    whose whole body was `analysis = shared_walk(...)` followed by
    `analysis.domain = None` passed, because only `assumed()` was scanned
    for.

RULE 3 -- no test writes to a node of a shared program
    `Program` is a frozen dataclass, and that freezes the SHELL ONLY:
    `Program.ir` is a tree of ordinary mutable `SpecIRNode`s. A test handed
    a shared program could clear `node.kv_cached` or rewrite `node.repeat`
    to set up its own case and corrupt the program for every later holder
    in the process. Two did, and both have been rewritten to DECLARE their
    case in a spec, which is what they were really about.

    WHAT HOLDS THIS, EXACTLY: a syntax scan, and nothing else. It flags an
    attribute assignment on a name bound from `program.nodes()`, in a file
    that takes `shared_program`. It is NOT immutability and it is NOT a
    runtime guard — an earlier version of this branch wrapped
    `SpecIRNode.__setattr__` from here to enforce it and that was withdrawn
    as a hack (test code may not redefine engine behaviour, CLAUDE.md
    §2.7), so nothing stops such a write at runtime today. The durable
    guarantee is engine work: a frozen `SpecIRNode` with the walk building
    through a builder.

    The scan keys on where the object CAME FROM rather than on
    reading a test, because the node arrives through a helper, a
    comprehension or a `next(...)`. A scan would have both false positives
    (a test assigning `repeat` on an object of its own) and false
    negatives, and a rule that looks total and is not is worse than a
    narrow one. The runtime guard has neither failure mode for the case it
    covers.

WHY THERE IS NO LIST OF FILENAMES HERE
--------------------------------------
There was one, and it was wrong: a gate a newcomer satisfies by appending a
path is a suppressant, not a gate, and it is the second-table failure mode
CLAUDE.md §1.4 names. Both rules above are now properties of a file's own
syntax -- it either calls `analyze()` or it does not, it either opens
`assumed()` or it does not -- so the only way to go green is to change the
code. `own_walk` is the escape hatch and it costs a sentence of reasoning
at the point where the reasoning is needed.

WHAT THESE CANNOT SEE, STATED RATHER THAN GUESSED
-------------------------------------------------
Rule 2 catches an assignment on a name bound directly from
`shared_walk(...)`, and a DIRECT `assumed()`. It does not see either
reached indirectly: an analysis passed into a helper that writes to it, or
`inference.serving` / `inference.step_eval` / `inference.activation_peak`,
which each open the `assumed()` scope themselves -- so a test driving one
of those on a shared walk mutates it without naming `assumed()` anywhere.
Those three are the whole indirect set; a file that shares a walk and
drives one of them needs reading, not scanning.

A rule keeping `shared_walk` out of `tests/backend/inference/` was
considered and rejected, so nobody has to re-propose it. The blind spot is
not directory-shaped: it follows those three MODULES, and
`tests/backend/playground/` and `tests/backend/rendering/` drive them too,
so a directory rule would give false confidence about those while blocking
a directory that may be entirely fine. It would also be a second table --
"this directory is special" duplicating a fact that lives in the three
modules, wrong in both directions as soon as one of them stops opening the
scope or a fourth starts. The gate-shaped version, if one is ever wanted,
keys on the modules: a file that shares a walk AND reaches `serving` /
`step_eval` / `activation_peak`.

Rule 3 sees two shapes and no others: `for node in program.nodes():` and
`node = next(... .nodes() ...)`, followed by `node.attr = ...` in the same
file. It does NOT see a node reached through a helper that returns one, a
container mutated in place (`node.dims["x"] = 1`,
`node.children.append(...)`), a `delattr`, or a write spelled through
`node.__dict__` or `object.__setattr__`.

Those last two are not hypothetical: the suite uses both spellings today,
and none of the uses is a leak -- they write a Program's `stacks` SHELL, a
spec layer, or a `copy.copy`, never a node of a shared program. Find them by
grepping `tests/` for `__dict__` and for `object.__setattr__`, rather than
from a list here, because a count or a file list in this docstring is a
claim about other files that goes stale the moment one of them is edited --
which is how an earlier version of this paragraph came to say "four" when
there were six. None of them would be caught, and that is the honest size
of this rule: it stops the shape that actually bit us from coming back,
and it is not immutability.
The durable fix is a frozen `SpecIRNode` in the engine.
"""
from __future__ import annotations

import ast
import sys
import unittest
from pathlib import Path
from typing import Dict, List

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from tools.corpus_pins import curated
from tools.corpus_pins.corpus import ROOT

#: What makes a file a walk-sharing file, for rules 1 and 2, which are
#: about the shared `Analysis`. `shared_program` is not here because a file
#: may share a program and still walk privately; what it may not do is
#: write to that program's nodes, which rule 3 reads the file for.
SHARED = ("shared_walk",)

OWN_WALK = (
    "if this test needs a walk of its own, say why at the call site: "
    "`from tests.snapshots import own_walk` and "
    "`own_walk(deployment, why=\"...\")`. If it does not, take the "
    "module's shared one -- `shared_walk(Deployment("
    "shared_program(\"org/model\")).at(input_tokens=128))` -- which the "
    "rest of your file then reuses instead of walking again.")


def _calls(tree: ast.AST, attr: str) -> List[int]:
    """The line of every bare `<expr>.<attr>()` call."""
    return [node.lineno for node in ast.walk(tree)
            if isinstance(node, ast.Call) and not node.args
            and not node.keywords and isinstance(node.func, ast.Attribute)
            and node.func.attr == attr]


def _sharing_files(names: tuple = SHARED) -> Dict[Path, ast.AST]:
    """Every test module that takes one of `names`, parsed."""
    out = {}
    for path in curated.test_module_paths():
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except SyntaxError:          # a file being edited is not this gate's
            continue                 # business; the collector will say so
        found = {node.id for node in ast.walk(tree)
                 if isinstance(node, ast.Name)}
        found |= {node.name for node in ast.walk(tree)
                  if isinstance(node, ast.alias)}
        if found & set(names):
            out[path] = tree
    return out


class TestASharingFileDoesNotAlsoWalkPrivately(unittest.TestCase):
    def test_no_bare_analyze_where_walks_are_shared(self) -> None:
        offenders = [
            f"{path.relative_to(ROOT)}:{line}"
            for path, tree in _sharing_files().items()
            for line in _calls(tree, "analyze")]
        self.assertEqual(
            offenders, [],
            "these files share walks with the rest of their module but also "
            f"call `analyze()` directly, which walks again -- {OWN_WALK} "
            f"Sites: {offenders}")


class TestASharingFileDoesNotMutateWhatItIsGiven(unittest.TestCase):
    def test_no_attribute_write_on_a_shared_analysis(self) -> None:
        """The plainest way to mutate one, and the one the module promised
        to catch while only looking at `assumed()`: assigning a layer or a
        field on the analysis `shared_walk` handed you. `corpus.py` names
        this shape itself as the one requiring `own_walk`."""
        offenders = [
            f"{path.relative_to(ROOT)}:{line} writes `{attr}`"
            for path, tree in _sharing_files().items()
            for attr, line in _attribute_writes(
                tree, _bound_from(tree, "shared_walk("))]
        self.assertEqual(
            offenders, [],
            "these files write to the analysis `shared_walk` gave them, "
            "which every other test in the module is holding. An "
            f"`Analysis` is not frozen, so the write lands on them. {OWN_WALK} "
            f"Sites: {offenders}")

    def test_no_assumed_scope_where_walks_are_shared(self) -> None:
        offenders = [
            f"{path.relative_to(ROOT)}:{line}"
            for path, tree in _sharing_files().items()
            for line in _calls(tree, "assumed")]
        self.assertEqual(
            offenders, [],
            "these files share walks but open `Analysis.assumed()`, which "
            "writes the assumption ledger back to the analysis when the "
            "scope closes -- so the next test in the module reads an "
            "analysis carrying this one's assumptions. A consumer that "
            f"opens that scope needs its own walk: {OWN_WALK} "
            f"Sites: {offenders}")


class TestTheEscapeHatchIsReachable(unittest.TestCase):
    """The gate is only honest if the way out of it exists and is one
    import away; a rule whose remedy does not resolve is a rule people
    work around."""

    def test_own_walk_is_exported_and_demands_a_reason(self) -> None:
        import inspect

        from tests import snapshots

        self.assertIn("own_walk", snapshots.__all__)
        why = inspect.signature(snapshots.own_walk).parameters["why"]
        self.assertIs(why.default, inspect.Parameter.empty,
                      "`why` must be required, or the reason stops being "
                      "written down")
        self.assertIs(why.kind, inspect.Parameter.KEYWORD_ONLY)


class TestAResolvedSpecOffersNoWayToRewriteItsFacts(unittest.TestCase):
    """The first fact `corpus.py` rests the sharing on, as far as it is
    decidable.

    A shared resolution is safe while its facts are written once in
    `__init__` and its memo slots are each filled by exactly one place.
    Both slot sets are derived from the constructor rather than listed
    here, so there is nothing to keep in sync (see `_slots`).

    The MEMO rule is enforced exactly: those names exist nowhere else in
    the tree, so counting their writers has no false positives. The FACT
    rule is enforced only WITHIN the class -- that `ResolvedSpec` offers no
    method but `__init__` that rewrites one. It is not checked across the
    tree, because `spec`, `config` and `dim_values` are ordinary attribute
    names that dozens of unrelated objects also have
    (`backend/service/bench/lake.py` assigns a `.config`) and no scan here
    can tell which object a `.config = ...` is on without type inference.
    That the rest of `backend/` and `tools/` never assigns one was verified
    by reading, once, and `corpus.py` says so plainly rather than claiming
    a gate holds it."""

    #: Both sets are DERIVED from `ResolvedSpec.__init__`, which declares
    #: them by assigning them, so neither can drift from the class and
    #: neither is a second table to keep in sync.
    #:
    #: THE DEFAULT IS "FACT", AND THAT IS THE WHOLE DESIGN. A memo is only
    #: a slot the constructor assigns an explicitly EMPTY value to -- a
    #: bare `None` literal, which is what a memo that has not been filled
    #: yet looks like. Everything else the constructor assigns is a fact,
    #: however it was computed.
    #:
    #: The first version of this asked the opposite question -- "does the
    #: right-hand side mention a parameter?" -- and put everything else in
    #: `MEMOS`. Those are two different propositions that happen to agree
    #: on today's constructor: a slot computed from another slot
    #: (`self.derived = self.spec.name`) mentions no parameter, so it was
    #: filed as a memo, and the memo rule then ASKED FOR a second writer.
    #: One later `self.derived = ...` made it pass, silently, while every
    #: holder of the shared resolution saw a rewritten fact; with no
    #: rewriter at all it failed. Strictest where nothing is wrong, blind
    #: where something is. A parameter-mention heuristic is not a safe
    #: derivation of mutability, and an unsound derivation of the right
    #: shape is worse than the hand-written tuple it replaced.
    LOADER = ROOT / "backend" / "arch" / "loader.py"

    @staticmethod
    def _is_empty(value: ast.AST) -> bool:
        """Whether `value` is an explicitly empty memo initialiser: a bare
        `None`, or one annotated/parenthesised around a bare `None`."""
        return isinstance(value, ast.Constant) and value.value is None

    @classmethod
    def _slots(cls) -> tuple:
        """`(facts, memos)` read off `ResolvedSpec.__init__`.

        A slot assigned a bare `None` is a memo; EVERY other assignment is
        a fact, so a slot this cannot classify is treated as the thing that
        must not be rewritten."""
        init = next(node for node in cls._resolved_spec_class().body
                    if isinstance(node, ast.FunctionDef)
                    and node.name == "__init__")
        facts, memos = [], []
        for node in ast.walk(init):
            if not isinstance(node, (ast.Assign, ast.AnnAssign)):
                continue
            targets = (node.targets if isinstance(node, ast.Assign)
                       else [node.target])
            for target in targets:
                if not (isinstance(target, ast.Attribute)
                        and isinstance(target.value, ast.Name)
                        and target.value.id == "self"):
                    continue
                (memos if node.value is not None
                 and cls._is_empty(node.value) else facts).append(target.attr)
        return tuple(facts), tuple(memos)

    @property
    def FACTS(self) -> tuple:
        return self._slots()[0]

    @property
    def MEMOS(self) -> tuple:
        return self._slots()[1]

    @classmethod
    def _resolved_spec_class(cls) -> ast.ClassDef:
        tree = ast.parse(cls.LOADER.read_text(encoding="utf-8"))
        return next(node for node in ast.walk(tree)
                    if isinstance(node, ast.ClassDef)
                    and node.name == "ResolvedSpec")

    @staticmethod
    def _targets(scope: ast.AST) -> List[tuple]:
        """`(attribute, line)` for every attribute assignment in `scope`."""
        out = []
        for node in ast.walk(scope):
            if not isinstance(node, (ast.Assign, ast.AugAssign,
                                     ast.AnnAssign)):
                continue
            targets = (node.targets if isinstance(node, ast.Assign)
                       else [node.target])
            out.extend((target.attr, node.lineno) for target in targets
                       if isinstance(target, ast.Attribute))
        return out

    def test_only_the_constructor_writes_a_fact(self) -> None:
        offenders = [
            f"ResolvedSpec.{method.name}:{line} writes `{attr}`"
            for method in self._resolved_spec_class().body
            if isinstance(method, ast.FunctionDef)
            and method.name != "__init__"
            for attr, line in self._targets(method) if attr in self.FACTS]
        self.assertEqual(
            offenders, [],
            "`ResolvedSpec` grew a method that rewrites one of its own "
            "facts, which makes a shared resolution unsafe for every holder "
            "of it. To vary a resolution, return a NEW one the way "
            "`with_spec` does -- through `__init__`, so the spec-tree memos "
            f"start cold. {offenders}")

    def test_each_memo_slot_has_exactly_one_writer(self) -> None:
        """Checked over the whole tree, because these names are unique to
        `ResolvedSpec` and so a scan for them has no false positives."""
        found: Dict[str, List[str]] = {name: [] for name in self.MEMOS}
        for root in ("backend", "tools"):
            for path in sorted((ROOT / root).rglob("*.py")):
                tree = ast.parse(path.read_text(encoding="utf-8"))
                for attr, line in self._targets(tree):
                    if attr in found:
                        found[attr].append(f"{path.relative_to(ROOT)}:{line}")
        for name, writers in found.items():
            with self.subTest(memo=name):
                self.assertEqual(
                    len(writers), 2,
                    f"`{name}` is a memo of a resolution's facts, which is "
                    f"what makes sharing the resolution the same as sharing "
                    f"the memo. It should be declared once in "
                    f"`ResolvedSpec.__init__` and filled in exactly one "
                    f"other place, as an `if slot is None` memo; "
                    f"found {writers}")


def _bound_from(tree: ast.AST, source: str) -> set:
    """Names bound to something whose expression mentions `source`.

    Two shapes, which are the two ways a test gets hold of one of these:
    `for node in program.nodes():` / `node = next(... .nodes() ...)`, and
    `analysis = shared_walk(...)`."""
    out = set()
    for scope in ast.walk(tree):
        if (isinstance(scope, ast.For) and isinstance(scope.target, ast.Name)
                and source in ast.unparse(scope.iter)):
            out.add(scope.target.id)
        if isinstance(scope, ast.Assign) and source in ast.unparse(scope.value):
            out |= {target.id for target in scope.targets
                    if isinstance(target, ast.Name)}
    return out


def _attribute_writes(tree: ast.AST, names: set) -> list:
    """`(attribute, line)` for every `<name>.attr = ...` where `<name>` is
    in `names`."""
    out = []
    for node in ast.walk(tree):
        if not isinstance(node, (ast.Assign, ast.AugAssign, ast.AnnAssign)):
            continue
        targets = (node.targets if isinstance(node, ast.Assign)
                   else [node.target])
        out.extend((target.attr, node.lineno) for target in targets
                   if isinstance(target, ast.Attribute)
                   and isinstance(target.value, ast.Name)
                   and target.value.id in names)
    return out


class TestASharingFileDoesNotWriteToAProgramsNodes(unittest.TestCase):
    """Rule 3: nothing writes to a node of a shared program.

    `Program` is a frozen dataclass and that freezes the SHELL ONLY --
    `Program.ir` is a tree of ordinary mutable `SpecIRNode`s, so a shared
    `Program` is a graph every later holder can have rewritten under them.
    Two tests did it, and the damage landed on strangers: the cache term in
    `test_eval_traffic` reported 0 instead of 71303168 bytes, and
    `test_ep_as_count` before `test_program_moe`'s corpus sweep turned the
    sweep red while the reverse order was green. Both have been rewritten to
    DECLARE their case in a spec, which is what they were really about.

    WHAT THIS RULE IS AND IS NOT. It keys on where the object came from --
    a name bound by `for node in program.nodes()` or
    `node = next(... .nodes() ...)` -- and not on the field name, because
    keying on the field name is hopeless: `name`, `role` and `op` are
    `SpecIRNode` fields and also attributes of half the objects in the
    suite, and that rule produced seven hits of which seven were false
    (a component's `.role`, a local `.name`). This shape is narrow: it
    catches the two ways a test actually reaches a node and nothing else,
    and it does not see a node reached through a helper that returns one,
    or a container mutated in place.

    The guarantee this does NOT give, stated plainly: it is not immutability.
    That belongs to the engine -- a frozen `SpecIRNode` with the walk
    building through a builder -- and until then this rule plus the fact
    that no test needs to write to a node is what holds. An earlier version
    of this branch wrapped `SpecIRNode.__setattr__` from the test
    infrastructure to enforce it; that was rejected, correctly: test code
    may not redefine engine behaviour (CLAUDE.md §2.7), and a guard whose
    reach is bounded but whose shape reads as total is worse than a narrow
    rule that says so.
    """

    def test_no_node_of_a_shared_program_is_written(self) -> None:
        offenders = [
            f"{path.relative_to(ROOT)}:{line} writes `{attr}`"
            for path, tree in _sharing_files(("shared_program",)).items()
            for attr, line in _attribute_writes(
                tree, _bound_from(tree, ".nodes()"))]
        self.assertEqual(
            offenders, [],
            "these files take `shared_program` and then write to a node of "
            "it. `Program` is frozen only in its shell: its `ir` is a tree "
            "of mutable `SpecIRNode`s, so the write lands on every later "
            "holder in the process and the failure surfaces in somebody "
            "else's test. If you are setting up a case, DECLARE it in a "
            "spec instead — `test_ep_as_count.toy_resolved` and "
            "`test_eval_traffic._cacheless_attention_program` are the two "
            "worked examples, and both of those tests used to do this. If "
            "you truly need an edited graph, resolve it yourself with your "
            f"own loader. Sites: {offenders}")


if __name__ == "__main__":
    unittest.main()
