"""The scoreboard sees a dims-table read however it is spelled.

`downstream_derivation` fell by rewriting rather than by moving a fact: the
`.dim_values` reads went through methods that only rename the read
(`ResolvedSpec.get_dim` and the `unbound` / `dims_moved_from` wrappers since
deleted). The counter derives that set of wrappers from their definitions,
so a new one is seen without being listed — which is what these cases pin,
together with the one shape that is a build and not a read.
"""
from __future__ import annotations

import ast
import sys
import textwrap
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from tools import scoreboard as sb


def test_a_method_that_only_renames_a_dims_read_is_seen():
    tree = ast.parse(textwrap.dedent("""
        class Resolved:
            def get_dim(self, expr):
                return eval_dim_expr(expr, self.dim_values)

            def unbound(self, names):
                return set(names) - set(self.dim_values)

            def with_spec(self, spec):
                return Resolved(spec, dict(self.dim_values))

            def layers(self, a):
                start = self.get_dim(a)
                return self.dim_values[start]
    """))
    assert sb.dim_table_wrappers_in(tree) == {"get_dim", "unbound"}
    assert "get_dim" in sb._dim_table_wrappers()


def test_the_counter_itself_reads_a_renamed_dims_read():
    """The wiring, not just the predicate.

    `dim_table_wrappers_in` above can be correct while the counter never
    asks it — and then deleting the hook drops the count without a single
    test going red, because the baseline ratchet only catches a RISE. So
    assert the live counter reports the wrapper call sites by the reason
    only the AST pass can give, which no regex in `DOWNSTREAM_PATTERNS`
    produces.
    """
    reason = "through a wrapper that only renames the read"
    sites = [s for s in sb.downstream_derivation_sites() if reason in s]
    assert sites, "the counter must read the wrapper set it derives"
    assert any("compose.py" in s for s in sites), sites
