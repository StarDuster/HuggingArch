"""A test asks `tests.snapshots` for a snapshot; it does not decide for itself
what a missing one means.

`snapshot_config` / `resolve_snapshot` skip when the file is not checked out
and FAIL when it is present but broken. A hand-written
`self.skipTest("... snapshot ...")` behind an `is None` check collapsed the
two — a present-but-unresolvable snapshot skipped quietly — and there were
ninety of them. This gate walks the test tree's AST for `skipTest(...)` /
`pytest.skip(...)` calls — and `pytest.mark.skipif(..., reason=...)` decorators — whose message
mentions a snapshot and refuses any.
"""
from __future__ import annotations

import ast
from pathlib import Path

TESTS = Path(__file__).resolve().parents[1]
HELPER = TESTS / "snapshots.py"


def _skip_calls(tree: ast.AST):
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        fn = node.func
        name = fn.attr if isinstance(fn, ast.Attribute) else getattr(fn, "id", "")
        if name not in ("skipTest", "skip", "SkipTest", "skipif", "skipIf", "skipUnless"):
            continue
        # `pytest.mark.skipif(cond, reason="no X snapshot")` is the decorator
        # form of the same decision; the reason rides in a keyword.
        texts = list(node.args) + [kw.value for kw in node.keywords]
        for arg in texts:
            for sub in ast.walk(arg):
                if isinstance(sub, ast.Constant) and isinstance(sub.value, str) \
                        and "snapshot" in sub.value.lower():
                    yield node.lineno, sub.value
                    break


def test_no_test_decides_what_a_missing_snapshot_means():
    offenders = []
    for path in sorted(TESTS.rglob("test_*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for lineno, text in _skip_calls(tree):
            offenders.append(f"{path.relative_to(TESTS.parent)}:{lineno}: {text!r}")
    assert offenders == [], (
        "hand-written snapshot skips; ask tests.snapshots.snapshot_config / "
        "resolve_snapshot instead (missing -> skip, present-but-broken -> fail):\n"
        + "\n".join(offenders))


def test_the_helper_is_the_only_place_that_skips_on_a_missing_snapshot():
    tree = ast.parse(HELPER.read_text(encoding="utf-8"))
    raises = [n for n in ast.walk(tree) if isinstance(n, ast.Raise)]
    assert any("SkipTest" in ast.dump(r) for r in raises)
