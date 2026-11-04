"""Registering the providers is metadata only (docs/measured_calibration.md
§3.1, §8.3): `providers.discover()` imports no kernel
library — torch included — so the Kernel Bench service, which annotates
rows with the catalog's names and never runs a kernel, stays small. A
provider's library is imported where a case is built."""
from __future__ import annotations

import pathlib
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]

#: Libraries a provider builds on; none may load at discovery.
HEAVY = ("torch", "triton", "flashinfer", "vllm", "sglang", "sgl_kernel",
         "deep_gemm", "deepinfer", "flash_attn", "transformers")


def test_discovery_imports_no_kernel_library():
    code = (
        "import sys\n"
        f"sys.path.insert(0, {str(ROOT)!r})\n"
        "from kernel_bench import providers\n"
        "providers.discover()\n"
        "assert providers.all_providers()\n"
        f"print(sorted({{m.split('.')[0] for m in sys.modules}} & set({HEAVY!r})))\n")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                         check=True, timeout=120)
    assert out.stdout.strip() == "[]", out.stdout + out.stderr


def test_dispatch_capabilities_do_not_follow_the_unquantized_dtype_table():
    """A corpus dtype and a kernel's dispatch capability are separate facts."""
    code = (
        "import sys\n"
        f"sys.path.insert(0, {str(ROOT)!r})\n"
        "import backend.arch.dtypes as dtypes\n"
        "dtypes.UNQUANTIZED_TOKENS |= {'float128'}\n"
        "from kernel_bench.providers import TORCH_FLOAT\n"
        "assert TORCH_FLOAT == ('bf16', 'fp16', 'fp32'), TORCH_FLOAT\n")
    subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                   check=True, timeout=120)


def test_a_torch_unary_still_builds_its_function():
    """The unaries resolve their torch function when a case is built."""
    import pytest
    torch = pytest.importorskip("torch")
    from kernel_bench.providers import all_providers
    from kernel_bench.providers.norm_act import _torch_fn
    impls = {p.kernel_impl for p in all_providers() if p.library == "torch"}
    assert {"torch_silu", "torch_situ", "torch_normalize_sum_one"} <= impls
    x = torch.ones(2, 3)
    assert _torch_fn(lambda t, F: F.silu)(x).shape == x.shape
