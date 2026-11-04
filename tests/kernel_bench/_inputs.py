"""Build a reference's canonical inputs in a test.

`RefSpec.make_inputs(inv, device, gen)` draws from the cell's own generator
(`refs.input_generator`, seeded off the measurement key) — that is what makes
a measurement's data the cell's and nothing else's.

A test that is about shapes, dtypes or a reference's arithmetic states its
cell by hand and carries no planned key, so there is no cell seed to read.
Its data only has to be fixed, which is what this helper gives it: the cell's
own generator where the cell carries a key, and a stated seed where it does
not. The seed's own properties are asserted in `test_input_seeds.py`.
"""
from __future__ import annotations

from typing import Any, Dict

#: The seed a hand-stated test cell draws at. Any value does; it is written
#: down so two runs of a test see one batch of data.
TEST_SEED = 0


def build_inputs(spec: Any, inv: Any, device: str = "cpu") -> Dict[str, Any]:
    import torch
    from kernel_bench.refs import input_generator
    if inv.physical_key is not None:
        gen = input_generator(inv, device)
    else:
        rng_device = "cpu" if str(device).startswith("meta") else device
        gen = torch.Generator(rng_device).manual_seed(TEST_SEED)
    return spec.make_inputs(inv, device, gen)
