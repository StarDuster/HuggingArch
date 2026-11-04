"""A model that publishes its context length has it bound as `s_max`.

The published context is a fact every page reads off the estimate (the
Inspector's "Max Context", the warning on a request longer than it, the
points a chart drops past it). It reaches them only through the spec's
`dims:` binding; a spec that forgets the binding silently publishes "no
limit" for a model whose config states one. dims.yaml names the config field
(`s_max.config`), so the check needs no per-model list.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import backend.arch.dims as dims
from backend.arch.dim_resolution import config_path_value, language_config
from backend.arch.snapshots import iter_snapshot_ids
from tests.snapshots import shared_loader, snapshot_config


def test_every_published_context_is_bound():
    field = dims.get("s_max").config
    assert field, "dims.yaml names no config field for s_max"
    loader = shared_loader()
    unbound = []
    for model_id in iter_snapshot_ids():
        config = snapshot_config(model_id)
        resolved = loader.match(config, model_id=model_id)
        if resolved is None:
            continue
        _scope, backbone = language_config(resolved.spec.dims, config)
        published = config_path_value(field, backbone)
        if published is None:
            continue
        bound = resolved.dim_values.get("s_max")
        if bound != published:
            unbound.append(f"{model_id}: config {field}={published}, s_max={bound}")
    assert not unbound, (
        "specs that do not bind the published context length "
        "(add `s_max: config.<path>` to the spec's dims:):\n  "
        + "\n  ".join(unbound))
