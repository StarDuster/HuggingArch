"""A pooled indexer keeps one key per pool, and decode pools only the pool it
completes.

GLM-5.3-Flash's k-pool indexer and Qwen3.8-Flash-Next's QSA indexer score
pooled keys: compress_ratio consecutive keys folded into one. HF caches the
raw per-token keys and re-pools the whole cache every step; vLLM and sglang
cache the pooled key — one row per pool, written when the pool completes —
and keep only the pending pool's raw rows in a per-request ring. The specs
model the serving layout (the scores are bit-identical once a pool is
complete), so:

  * the cached index key is one d_indexer_head row per pool: ⌈s_kv/m⌉ rows
    stored, ⌊s_kv/m⌋ readable;
  * a decode step reads ⌊s_kv/m⌋ · d_indexer_head · bytes per element of
    index key per layer — not s_kv raw rows;
  * no op of the pool costs FLOPs that grow with the context at decode: the
    pool runs on the ring's compress_ratio rows, once per compress_ratio
    steps.

Which tensor is the indexer's cache and which nodes build it is read off the
components' `cache.entries` (`kind: index_key`), not off node names.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, Iterator, List, Set

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT))

from backend.arch.eval import cache, traffic
from backend.arch.eval.dim import value_of
from backend.arch.eval.program import Program
from backend.arch.eval.transforms import DECODE, Deployment
from tests.snapshots import resolve_snapshot, shared_loader

MODELS = ("zai-org/GLM-5.3-Flash", "Qwen/Qwen3.8-Flash-Next")
CONTEXT = 1 << 20


def _decode(model_id: str, cache_len: int) -> Any:
    resolved = resolve_snapshot(model_id, shared_loader())
    return (Deployment(Program.of(resolved))
            .at(batch=1, input_tokens=cache_len)
            .during(DECODE, cache_len=cache_len).analyze())


def _subtree(node: Any) -> Iterator[Any]:
    yield node
    for child in node.children or ():
        yield from _subtree(child)


def _key_side(analysis: Any) -> Dict[int, Any]:
    """Every leaf, inside a component that declares an `index_key` cache,
    on the path to a score's KEY operand — the operand a score op declares
    as `k` (`OpSpec.input_roles`). That is what builds the keys a decode
    step scores: the pool and the cache read beside the main attention's
    keys."""
    out: Dict[int, Any] = {}
    for holder in analysis.program.nodes():
        if not any(e.kind == "index_key" for e in (holder.kv_entries or ())):
            continue
        inside = {n.uid: n for n in _subtree(holder)}
        todo: List[Any] = []
        for node in inside.values():
            roles = (tuple(node.declaration.input_roles or ())
                     if node.declaration is not None else ())
            for position, ref in enumerate(node.inputs_obj or ()):
                if position < len(roles) and roles[position] == "k" \
                        and ref.src is not None:
                    todo.append(ref.src)
        seen: Set[int] = set()
        while todo:
            node = todo.pop()
            if node.uid in seen or node.uid not in inside:
                continue
            seen.add(node.uid)
            if not node.children:
                out[node.uid] = node
            todo.extend(node.children or ())
            todo.extend(ref.src for ref in node.inputs_obj or ()
                        if ref.src is not None)
    return out


@pytest.fixture(scope="module", params=MODELS)
def model(request):
    return request.param, _decode(request.param, CONTEXT)


def test_one_cached_index_key_per_pool(model):
    model_id, analysis = model
    dims = analysis.binding
    ratio, width = int(dims["compress_ratio"]), int(dims["d_indexer_head"])
    keys = [t for entry in cache.derive(analysis).values.values()
            for t in entry.tensors
            if t.kind == "index_key" and not t.recurrent]
    assert keys, model_id
    for tensor in keys:
        assert int(value_of(tensor.elements)) == width
        assert int(value_of(tensor.positions(CONTEXT))) == -(-CONTEXT // ratio)
        assert int(value_of(tensor.positions(CONTEXT, read=True))) == CONTEXT // ratio


def test_decode_reads_one_pooled_row_per_pool(model):
    """Per indexer layer, the step's index-key read is ⌊s_kv/m⌋ rows of
    d_indexer_head, each at the cached tensor's stored size per position."""
    model_id, analysis = model
    dims = analysis.binding
    ratio, width = int(dims["compress_ratio"]), int(dims["d_indexer_head"])
    reads = traffic.derive(analysis).values
    access = cache.kv_access(analysis).values
    entries = cache.derive(analysis)
    checked = layers = total = 0
    row_bytes = None
    for holder_uid, entry in entries.values.items():
        keys = [t for t in entry.tensors
                if t.kind == "index_key" and not t.recurrent]
        if not keys:
            continue
        (key,) = keys
        holder = analysis.program.by_uid(holder_uid)
        readers = [n for n in _subtree(holder)
                   if not n.children and n.uid in access
                   and any(r.tensor == key.name for r in access[n.uid].reads)]
        assert readers, (model_id, holder.display_path)
        read = sum(int(value_of(reads[n.uid].kv_reads)) for n in readers
                   if n.uid in reads)
        # One stored position's bytes, data and dynamic scales together
        # (`CachedTensor.position_bytes`): d_indexer_head elements of it.
        assert int(value_of(key.elements)) == width
        row_bytes = value_of(key.position_bytes)
        assert read == (CONTEXT // ratio) * row_bytes, (
            model_id, holder.display_path, [n.name for n in readers])
        copies = int(value_of(analysis.instances.values[holder_uid]))
        layers += copies
        total += read * copies
        checked += 1
    assert checked, model_id
    # The model's step: ⌊s_kv/m⌋ · d_indexer_head · bytes · indexer layers.
    assert total == (CONTEXT // ratio) * row_bytes * layers


def test_decode_pooling_does_not_grow_with_the_context(model):
    """Every op that builds the keys a decode step scores costs the same at
    a 64K and a 1M context: the step pools the ring and reads the cache; it
    does not re-pool the cache."""
    model_id, long_ctx = model
    short_ctx = _decode(model_id, 1 << 16)

    def flops(analysis) -> Dict[str, Any]:
        return {f"{node.display_path}#{uid}": value_of(analysis.cost.values[uid].flops)
                for uid, node in _key_side(analysis).items()
                if uid in analysis.cost.values}

    long, short = flops(long_ctx), flops(short_ctx)
    assert long, model_id
    grows = sorted(k for k in long if long[k] != short.get(k))
    assert grows == [], (model_id, grows)
