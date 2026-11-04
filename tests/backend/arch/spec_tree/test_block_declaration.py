"""Every `iter_blocks` record says which declaration of `spec.model` it came
from, so an edit can change that declaration alone."""
from __future__ import annotations

import pytest

from backend.arch.loader import SpecLoader
from backend.arch.spec_tree.build import iter_blocks
from tests.snapshots import resolve_snapshot, shared_loader


@pytest.fixture(scope="module")
def loader() -> SpecLoader:
    return shared_loader()


def _declaration(spec, meta):
    layer = spec.model[meta.layer_path[0]]
    for j in meta.layer_path[1:]:
        layer = layer.layers[j]
    if meta.assignment_index is None:
        return layer.template
    return layer.assignments[meta.assignment_index].template


@pytest.mark.parametrize("model_id", [
    "deepseek-ai/DeepSeek-V3",        # dense range + MoE rest + MTP template sub
    "moonshotai/Kimi-K3",             # vision group first, then 3 assignments
    "nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-BF16",  # MTP sub-layer with assignments
    "Qwen/Qwen3-Coder-Next",          # config_list assignments that select nothing
    "meta-llama/Llama-3.2-3B",        # one top-level template
])
def test_declaration_address_names_the_template(loader, model_id):
    resolved = resolve_snapshot(model_id, loader)
    for meta in iter_blocks(resolved):
        assert meta.layer_path, meta
        assert _declaration(resolved.spec, meta) == meta.template_name


def test_rest_assignment_index_is_its_declared_position(loader):
    # Qwen3-Coder-Next declares two config_list assignments (empty here), a
    # stride one, then `rest`; rest is resolved last but declared fourth.
    resolved = resolve_snapshot("Qwen/Qwen3-Coder-Next", loader)
    metas = [m for m in iter_blocks(resolved) if m.aux_kind is None]
    layer = resolved.spec.model[metas[0].layer_path[0]]
    rest = [i for i, a in enumerate(layer.assignments) if a.rest]
    assert [m.count for m in metas] == [12, 36]
    assert metas[-1].assignment_index == rest[0]
