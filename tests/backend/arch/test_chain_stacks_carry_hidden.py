"""Every chain stack of the shared corpus declares its carried stream.

A stack threads its block's output into its next iteration only when it
declares so (`streams: { <name>: { carries: true } }`); there is no default.
`tools/chain_stacks.py` lists the stacks that repeat a block from the
authored YAML, independently of the loader, and each must carry a stream in
the spec the loader serves; what the author names it is not a fact checked
here.
"""
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from tools.chain_stacks import chain_stacks
from tools.corpus_pins.checks.chain_stacks import layer_at as _layer
from tools.corpus_pins.checks.chain_stacks import served_specs as _specs
from tools.corpus_pins.checks.chain_stacks import stack_violations


def test_every_chain_stack_carries_a_stream():
    stacks = chain_stacks()
    assert stacks
    assert stack_violations(stacks) == []


def test_a_base_variant_that_restates_its_model_carries_on_both():
    """`qwen3_next_mtp` inherits from `qwen3_next` but restates `model:`, so
    it owns its stacks; the checkpoint identity file that names it as its
    base and writes no `model:` serves the merged spec with the backbone and
    the MTP head each carrying `hidden`."""
    listed = {stack.path for stack in chain_stacks()
              if stack.file == "qwen3_next_mtp.yaml"}
    assert listed == {("model.layers",), ("mtp", "mtp.layers")}
    spec = _specs()["Qwen__Qwen3-Next-80B-A3B-Instruct"]
    for path in listed:
        assert _layer(spec, path).carried_stream is not None, path
