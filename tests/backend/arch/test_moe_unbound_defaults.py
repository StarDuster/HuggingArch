"""What a MoE dim is when the config or the spec does not say.

Two different facts, and they are kept apart:

* **A config field the config omits** falls to what HF's config class fills
  in for it. A `config.json` that does not publish `shared_expert_intermediate_
  size` is read by `Qwen3NextConfig`, whose default is the literal 512 — not
  the model's expert width. So a spec's fallback for a config field is that
  class's literal, cited (`config.shared_expert_intermediate_size ?? "512"`);
  a fallback to another dim (`?? d_ffn_expert`) is a guess that agrees with
  HF only where the two class defaults happen to coincide, and Qwen3.8-2.4T's
  expert width (2048) and Qwen3.8-Flash-Next's (640) are not 512.
* **A router's group counts have no default.** Whether a router limits its
  groups is a fact of the model's forward source, so `n_expert_groups` and
  `n_groups_active` are bound by every spec whose router reads them — the
  literal 1 with the source line that says there is no limit, or the config's
  `n_group` / `topk_group`. A default of 1 once silently dropped DeepSeek-V2's
  group limit from a spec that forgot them (f02a4ead1, which retired it; the
  migrator's `retired_defaults` table sends a stored spec back for them), so
  an unbound one is refused, naming the binding.
"""
from __future__ import annotations

import copy

import pytest

from backend.arch.dim_resolution import CheckpointHeader, resolve_dimensions
from tests.snapshots import (shared_loader, snapshot_config,
                             snapshot_header)
# Unused here, but `tools/corpus_pins/curated.walks_corpus` recognises a
# corpus reader by this name; the registry classifies SHARED_WIDTH through it.
from tests.snapshots import SNAPSHOT_DIR  # noqa: F401

# spec, the snapshot whose config it is read with (None: no snapshot), the
# config section, the shared-width field, and its HF config class default.
SHARED_WIDTH = [
    ("qwen3_next", "Qwen/Qwen3-Coder-Next", None,
     "shared_expert_intermediate_size", 512),          # configuration_qwen3_next.py:116
    ("qwen3_5_moe", "Qwen/Qwen3.5-35B-A3B", "text_config",
     "shared_expert_intermediate_size", 512),          # configuration_qwen3_5_moe.py:111
    ("Qwen__Qwen3.8-2.4T-A95B", "Qwen/Qwen3.8-2.4T-A95B", None,
     "shared_expert_intermediate_size", 512),          # configuration_qwen3_5_moe.py:111
    ("Qwen__Qwen3.8-Flash-Next", "Qwen/Qwen3.8-Flash-Next", "text_config",
     "shared_expert_intermediate_size", 512),          # configuration_qwen4_exp.py:130
    ("step3p7_flash", "stepfun-ai/Step-3.7-Flash", "text_config",
     "share_expert_dim", 1280),                        # configuration_step3p7.py:173
    # Step-3.5 is read by the same class; no snapshot, so Step-3.7's text
    # config stands in for its top-level one.
    ("step3p5", None, None, "share_expert_dim", 1280),
]


def _dims_with(spec_name, model_id, section, field):
    spec = shared_loader().get(spec_name)
    if model_id is None:
        config = copy.deepcopy(snapshot_config("stepfun-ai/Step-3.7-Flash")["text_config"])
    else:
        config = copy.deepcopy(snapshot_config(model_id))
    text = config[section] if section else config
    assert field in text, f"{model_id}: the snapshot publishes {field}"
    del text[field]
    text["moe_intermediate_size"] = 3 * text["moe_intermediate_size"] + 8
    header = (CheckpointHeader.of(model_id, snapshot_header(
        model_id.replace("/", "__", 1))) if model_id is not None else None)
    return resolve_dimensions(spec.dims, config, checkpoint=header,
                              local=spec.dim_symbols)


@pytest.mark.parametrize("spec_name, model_id, section, field, default", SHARED_WIDTH,
                         ids=[row[0] for row in SHARED_WIDTH])
def test_an_omitted_shared_width_is_the_config_classes_default(
        spec_name, model_id, section, field, default):
    dims = _dims_with(spec_name, model_id, section, field)
    assert dims["d_ffn_shared"] == default, (
        f"{spec_name}: a config without `{field}` resolved d_ffn_shared to "
        f"{dims['d_ffn_shared']} (d_ffn_expert {dims['d_ffn_expert']}); HF's "
        f"config class fills {default}")


def _without_group_counts() -> str:
    from tests.backend.arch.eval.test_frame_rederivation import MODEL
    model = MODEL.replace('  n_expert_groups: "1"\n  n_groups_active: "1"\n', "")
    assert model != MODEL
    return model


def test_a_router_whose_group_counts_are_unbound_is_refused():
    """The synthetic spec's `moe_ffn` router reads both group counts; with
    neither bound the walk refuses and says how to bind them."""
    from backend.arch.eval.program import Program
    from backend.arch.eval.transforms import Deployment
    from tests.backend.arch.test_derived_names_refused import _load

    resolved = _load(_without_group_counts())
    with pytest.raises(ValueError, match=r"unresolved dim n_expert_groups") as refused:
        Deployment(Program.of(resolved)).analyze()
    assert "`config.n_group` for a group-limited router" in str(refused.value)


def test_the_writer_gets_the_unbound_group_counts_as_an_error_to_repair():
    import tempfile
    from pathlib import Path
    from unittest import mock

    from backend.arch.validate import validate
    from tests.backend.arch.eval.test_frame_rederivation import CONFIG

    with tempfile.TemporaryDirectory() as tmp:
        (Path(tmp) / "models").mkdir()
        (Path(tmp) / "models" / "frames_toy.yaml").write_text(_without_group_counts())
        with mock.patch("backend.arch.validate._orchestrator._get_safetensors_data",
                        return_value={"total_params": 0, "model_storage_bytes": 0,
                                      "tensors": {}}), \
             mock.patch("backend.arch.validate._orchestrator._fetch_config",
                        return_value=dict(CONFIG)):
            result = validate("frames_toy", spec_name="frames_toy", overlay_dir=tmp)
    assert result["valid"] is False
    joined = "; ".join(str(e) for e in result["errors"])
    assert "n_expert_groups" in joined and "config.topk_group" in joined, joined


# -- a top-k picks at most what it chooses among ------------------------------

def test_a_top_k_larger_than_what_it_picks_from_is_refused():
    """`n_groups_active` <= `n_expert_groups` and `n_experts_active` <=
    `n_route_targets` are declared bounds (`at_most:` in dims.yaml). A
    config that breaks one is refused at resolution, naming the bound —
    evaluated anyway, a negative `n_expert_groups - n_groups_active` would
    switch the group steps ON."""
    base = {"d_model": "8", "n_experts": "8", "n_experts_active": "2",
            "n_expert_groups": "2", "n_groups_active": "1"}
    assert resolve_dimensions(base, {})["n_route_targets"] == 8
    with pytest.raises(ValueError, match=r"`n_groups_active` is 3.*`n_expert_groups` = 2"):
        resolve_dimensions({**base, "n_groups_active": "3"}, {})
    with pytest.raises(ValueError, match=r"`n_experts_active` is 9.*`n_route_targets` = 8"):
        resolve_dimensions({**base, "n_experts_active": "9"}, {})
    # The zero-compute destinations count: 9 of 8 + 2 is a legal top-k.
    assert resolve_dimensions({**base, "n_experts_active": "9",
                               "n_zero_experts": "2"}, {})["n_route_targets"] == 10


def test_the_writer_gets_the_bound_as_an_error_to_repair():
    import tempfile
    from pathlib import Path
    from unittest import mock

    from backend.arch.validate import validate
    from tests.backend.arch.eval.test_frame_rederivation import CONFIG, MODEL

    model = MODEL.replace('  n_groups_active: "1"\n', '  n_groups_active: "2"\n')
    assert model != MODEL
    with tempfile.TemporaryDirectory() as tmp:
        (Path(tmp) / "models").mkdir()
        (Path(tmp) / "models" / "frames_toy.yaml").write_text(model)
        with mock.patch("backend.arch.validate._orchestrator._get_safetensors_data",
                        return_value={"total_params": 0, "model_storage_bytes": 0,
                                      "tensors": {}}), \
             mock.patch("backend.arch.validate._orchestrator._fetch_config",
                        return_value=dict(CONFIG)):
            result = validate("frames_toy", spec_name="frames_toy", overlay_dir=tmp)
    assert result["valid"] is False
    joined = "; ".join(str(e) for e in result["errors"])
    assert "`n_groups_active` is 2" in joined and "`n_expert_groups` = 1" in joined, joined
