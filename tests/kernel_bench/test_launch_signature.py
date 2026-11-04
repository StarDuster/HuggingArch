"""What a row says about the call that was timed (`Launch.describe`).

The column exists because a row could not show it: `moe_permute`'s cell
states a dispatch and the timed launch was an equal-length permutation, a
1.5x difference in traffic that only provider source revealed. These assert
the rendering's own rules, because the rules are what a reader will trust —
and none of them needs a GPU.
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from kernel_bench.providers import Launch


class _Fake:
    """A tensor as far as the renderer is concerned."""

    def __init__(self, shape, dtype="torch.bfloat16", contiguous=True):
        self.shape = shape
        self.dtype = dtype
        self._contiguous = contiguous

    def is_contiguous(self):
        return self._contiguous


def _gemm(a, b, out, *, recipe=None):
    return None


class TestTheRowNamesTheArguments(unittest.TestCase):
    def test_formal_names_come_from_the_callee(self):
        got = Launch(_gemm, _Fake((8, 16)), _Fake((4, 16)), _Fake((8, 4)),
                     recipe=(1, 32, 32)).describe()
        self.assertIn("a=bfloat16[8,16]", got)
        self.assertIn("b=bfloat16[4,16]", got)
        self.assertIn("out=bfloat16[8,4]", got)
        self.assertIn("recipe=(1,32,32)", got)

    def test_a_callee_with_no_signature_renders_positionally(self):
        """Most torch builtins have none, and a guess would be a second
        source for the thing the callee already knows."""
        got = Launch(len, _Fake((2, 3))).describe()
        self.assertIn("bfloat16[2,3]", got)
        self.assertNotIn("=bfloat16[2,3]", got)

    def test_var_positional_is_not_rendered_as_a_repeated_keyword(self):
        """`fn(args=…, args=…)` is not a call anyone could write, and one
        name bound to several values reads as a bug in the row."""
        def variadic(*args, flag=False):
            return None

        got = Launch(variadic, _Fake((1,)), _Fake((2,)), flag=True).describe()
        self.assertNotIn("args=", got)
        self.assertIn("flag=True", got)

    def test_a_c_function_takes_its_names_from_its_doc_line(self):
        """`inspect.signature` raises for most library calls measured here
        (`torch.index_add`, `torch.mul`), and CPython's convention for those
        is the documented first line."""
        class CFunc:
            __name__ = "index_add"
            __doc__ = ("index_add(input: Tensor, dim: int, index: Tensor, "
                       "source: Tensor, *, alpha=1) -> Tensor\n\nAdds…")

            def __call__(self, *a, **kw):
                return None

        got = Launch(CFunc(), _Fake((4, 8)), 0, _Fake((9,), "torch.int64"),
                     _Fake((9, 8))).describe()
        self.assertIn("input=bfloat16[4,8]", got)
        self.assertIn("dim=0", got)
        self.assertIn("index=int64[9]", got)
        self.assertIn("source=bfloat16[9,8]", got)

    def test_a_doc_that_is_prose_yields_no_names(self):
        """A first line that is not a signature must not be mined for one —
        a wrong name is worse than none."""
        class Prose:
            __name__ = "thing"
            __doc__ = "Return self[key]."

            def __call__(self, *a, **kw):
                return None

        got = Launch(Prose(), _Fake((2, 2))).describe()
        self.assertIn("bfloat16[2,2]", got)
        self.assertNotIn("=bfloat16[2,2]", got)

    def test_a_dispatch_key_keeps_its_value(self):
        """A dtype or a scalar type IS what the kernel dispatched on; folded
        to its type name, two different calls render identically."""
        class ScalarType:
            def __repr__(self):
                return "float4_e2m1f"

        got = Launch(_gemm, _Fake((1, 1)), _Fake((1, 1)), ScalarType()).describe()
        self.assertIn("float4_e2m1f", got)

    def test_a_tensor_says_when_it_is_not_contiguous(self):
        got = Launch(len, _Fake((4, 8), contiguous=False)).describe()
        self.assertIn(":strided", got)

    def test_a_raising_operand_does_not_sink_the_signature(self):
        """The column is evidence about the measurement, not the
        measurement: a renderer that trips must not cost a measured row."""
        class Hostile:
            @property
            def shape(self):
                raise RuntimeError("no shape for you")

            def __repr__(self):
                raise RuntimeError("no repr either")

        got = Launch(len, Hostile()).describe()
        self.assertIn("Hostile", got)

    def test_it_is_callable_and_calls_what_it_rendered(self):
        seen = {}

        def fn(a, b=None):
            seen["a"], seen["b"] = a, b
            return "ran"

        launch = Launch(fn, 1, b=2)
        self.assertEqual(launch(), "ran")
        self.assertEqual(seen, {"a": 1, "b": 2})

    def test_it_is_truncated_rather_than_unbounded(self):
        got = Launch(len, *[_Fake((123456, 234567))] * 60).describe()
        self.assertLessEqual(len(got), 600)
        self.assertTrue(got.endswith("…)"))


class TestOneObservationPerGroup(unittest.TestCase):
    """`kernel_symbol` is sampled per group, not per row (23% of a measured
    row's cost went to its profiler). The group has to keep the signal the
    column exists for: a substitution shows up per implementation, per
    phase, per precision and per SIZE — a dispatch heuristic picks
    differently at 1k and at 1M."""

    def _inv(self, **dims):
        from kernel_bench.invocation import Invocation
        return Invocation(
            signature="gemm", op="linear", dims=dims,
            dtype_env={"activation": "bf16", "output": "bf16"},
            phase="prefill", tp=1,
            physical_key=("gemm", repr(sorted(dims.items())), "bf16", "prefill"),
            model="m", spec_hash="h", node_spec_id="n", parallel="tp=1")

    def test_cells_of_one_size_band_share_an_observation(self):
        from kernel_bench.driver import _symbol_group
        a = _symbol_group(self._inv(d_out=4096, s_query=1024), "torch", "k")
        b = _symbol_group(self._inv(d_out=4096, s_query=1100), "torch", "k")
        self.assertEqual(a, b)

    def test_a_different_size_band_is_observed_again(self):
        from kernel_bench.driver import _symbol_group
        small = _symbol_group(self._inv(d_out=4096, s_query=1024), "torch", "k")
        large = _symbol_group(self._inv(d_out=4096, s_query=1048576), "torch", "k")
        self.assertNotEqual(small, large)

    def test_another_implementation_is_observed_again(self):
        from kernel_bench.driver import _symbol_group
        one = _symbol_group(self._inv(d_out=4096, s_query=1024), "torch", "k")
        two = _symbol_group(self._inv(d_out=4096, s_query=1024), "vllm", "k")
        self.assertNotEqual(one, two)


if __name__ == "__main__":
    unittest.main()
