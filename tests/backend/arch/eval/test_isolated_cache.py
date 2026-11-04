"""A component analysed alone keeps the cache it keeps inside its model.

The domain fact: a component's `cache:` declaration belongs to the node that
invokes it — the attention layer a model block calls. Isolation (the
Playground's module bench, cache topology, axioms) runs the component as a
one-stage model whose block calls it at its call site (`CallSite`), so the
same declaration enters the program the same way. Per layer, the isolated
component and the model's call of it therefore agree on:

  - the cached tensors (name, row width, dtype, storage and read laws);
  - the KV bytes a layer writes, and the KV bytes a step reads — a decode
    step, and a prefill over a cached prefix;
  - the cache pedigree (which nodes carry cache-resident data, traced to
    which declared tensors), and which nodes `kv_access` marks as writers,
    readers and publishers.

Falsification targets: before isolation kept the call, an isolated attention
read no cache at all (`kv_reads` 0 — the cache was priced as edge spans, so a
sliding window read its whole span and a latent's value half was read twice),
and an encoder's attention was marked as keeping a cache nothing reads.

Each call is isolated at itself (`CallSite.calls`): from the frame the model
calls it in (its block's assignment dims, its enclosing calls' `dims:`) with its
own `dims:` (MiMo's MTP attention runs at `n_kv_heads_swa`), handed what the
model's walk hands it there.
"""
from __future__ import annotations

import sys
import unittest
from dataclasses import replace
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT))

from backend.arch.block_synthesis import binding_of
from backend.arch.eval import cache, traffic
from backend.arch.eval.dim import value_of
from backend.arch.eval.program import Program
from backend.arch.eval.transforms import (
    DECODE, CallSite, Deployment, isolate_resolved,
)
from tests.snapshots import resolve_snapshot, shared_loader

#: GQA, MLA, DSA, a recurrent state beside gated GQA, a sliding window, an
#: encoder attention, and a call whose own `dims:` rebind the component.
MODELS = (
    "Qwen/Qwen3-8B",
    "deepseek-ai/DeepSeek-V3",
    "deepseek-ai/DeepSeek-V3.2",
    "Qwen/Qwen3.5-35B-A3B",
    "mistralai/Mistral-7B-v0.1",
    "stepfun-ai/Step-3.7-Flash",
    "XiaomiMiMo/MiMo-V2.5",
)
BATCH = 2
CACHE_LEN = 8192
_LOADER = shared_loader()


def _phases(resolved):
    base = (Deployment(Program.of(resolved))
            .at(batch=BATCH, input_tokens=CACHE_LEN))
    return {
        "decode": base.during(DECODE, cache_len=CACHE_LEN),
        "prefill over a cached prefix": base.at(
            s_prefix_cached=CACHE_LEN // 2),
    }


def _path(node, holder) -> str:
    """`node`'s place inside the component `holder` calls, by local names —
    the same in the model and in isolation."""
    names = []
    while node is not holder:
        scope = node.component_scope
        names.append(node.name + (f"__{scope}" if scope else ""))
        node = node.parent
    return ".".join(reversed(names)) or "."


def _facts(analysis, holder) -> dict:
    """What the cache of the component `holder` calls is, per layer."""
    under = {}
    stack = [holder]
    while stack:
        node = stack.pop()
        under[node.uid] = node
        stack.extend(node.children)
    entry = cache.derive(analysis).get(holder)
    tensors = entry.tensors if entry is not None else ()
    reads = traffic.derive(analysis).values
    pedigree = cache.context_sources(cache.Edges(analysis))
    access = cache.kv_access(analysis).values
    return {
        "tensors": sorted(
            (t.name, str(value_of(t.elements)), t.dtype, str(t.stored),
             str(t.read), t.inner, t.recurrent) for t in tensors),
        "write bytes": int(sum(t.bytes_at(CACHE_LEN) for t in tensors)),
        "read bytes": int(sum(int(reads[uid].kv_reads)
                              for uid, node in under.items()
                              if not node.children and uid in reads)),
        "pedigree": sorted(
            (_path(under[uid], holder),
             tuple(sorted(_path(under[s], holder)
                          for s in seeds.seeds)))
            for uid, seeds in pedigree.items() if uid in under),
        "kv_access": sorted(
            (_path(under[uid], holder),
             tuple(sorted(r.tensor for r in a.writes)),
             tuple(sorted(r.tensor for r in a.reads)), a.publishes)
            for uid, a in access.items() if uid in under),
    }


class TestIsolatedComponentKeepsItsCache(unittest.TestCase):

    def test_isolated_equals_the_models_call(self) -> None:
        for model_id in MODELS:
            resolved = resolve_snapshot(model_id, _LOADER)
            for phase, deployment in _phases(resolved).items():
                whole = deployment.analyze()
                program = whole.program
                holders = [program.by_uid(uid)
                           for uid in cache.derive(whole).values]
                self.assertTrue(holders, model_id)
                for call in holders:
                    with self.subTest(model=model_id, phase=phase,
                                      component=call.op,
                                      block=call.template):
                        (site,) = (
                            site for site in CallSite.calls(resolved)
                            if site.invocation == program.block_of(call.uid)
                            and site.path == call.display_path)
                        isolated = isolate_resolved(
                            resolved, call.op, None, site)
                        alone = _phases(isolated)[phase].analyze()
                        (site_call,) = (node for node in alone.program.nodes()
                                        if node.provenance == "site")
                        self.assertEqual(_facts(alone, site_call),
                                         _facts(whole, call))

    def test_a_decode_step_reads_the_cache(self) -> None:
        """Where serving reads the cache, the isolated step does too."""
        resolved = resolve_snapshot("Qwen/Qwen3-8B", _LOADER)
        isolated = isolate_resolved(resolved, "gqa_qknorm_attention",
                                    site=CallSite.of(resolved,
                                                     "gqa_qknorm_attention"))
        step = _phases(isolated)["decode"].analyze()
        read = sum(int(t.kv_reads)
                   for t in traffic.derive(step).leaves().values.values())
        dims = resolved.dim_values
        # K and V, every position, every request.
        self.assertEqual(read, BATCH * CACHE_LEN * 2 * int(dims["n_kv_heads"])
                         * int(dims["d_head"]) * 2)

    def test_a_layer_of_a_stack_that_seeds_a_stream_isolates(self) -> None:
        """DeepSeek-V4.1's backbone seeds its hyper-connection mix stream
        from the `hc_seed` layer (`seed:`). A layer isolated alone keeps the
        stream it reads, seeded by the evaluator — not the seed, which names
        a layer the isolation does not have (it raised building the tree)."""
        resolved = resolve_snapshot("deepseek-ai/DeepSeek-V4.1-Flash", _LOADER)
        sites = [site for site in CallSite.calls(resolved)
                 if binding_of(resolved.spec, site.component, "dsv41_hc_block")]
        self.assertTrue(sites)
        for site in sites:
            isolated = isolate_resolved(resolved, "dsv41_hc_block", site=site)
            walk = Deployment(Program.of(isolated)).at(
                batch=1, input_tokens=2).analyze()
            self.assertGreater(sum(int(value_of(c.flops))
                                   for c in walk.cost.leaves().values.values()), 0)

    def test_a_site_without_a_component_has_no_block(self) -> None:
        """A call site built without its component (a test's hand-made one)
        runs in no block until isolation names what it calls."""
        site = CallSite(address="", role="attention", stack="", formats=None)
        with self.assertRaisesRegex(ValueError, "names no component"):
            site.block  # noqa: B018 — reading it is the act under test
        self.assertEqual(replace(site, component="mla_attention").block,
                         "mla_attention_site")


if __name__ == "__main__":
    unittest.main()
