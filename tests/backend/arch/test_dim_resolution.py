"""A `checkpoint.<tensor>[<axis>]` dim is whatever that checkpoint stores.

Some checkpoints serialize a table whose length the modeling code does not
determine — Kimi-K2's `rotary_emb.inv_freq` is registered persistent=False,
runs at d_qk_rope/2 = 32, and is stored at 56 (Instruct, Base) and 128
(Thinking). A config formula can only coincide with such a length, so the
spec binds it to the header. These tests pin what that binding does with a
header, and that it refuses — never defaults — without one.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from backend.arch.dim_resolution import (CheckpointHeader, references_config,
                                         resolve_dimensions)
from backend.hf.header import SafetensorsHeader
from tests.snapshots import (resolve_snapshot, shared_loader, snapshot_config,
                             snapshot_header)

INV_FREQ = "model.layers.X.self_attn.rotary_emb.inv_freq"
BINDING = {"rope_inv_freq_len": f"checkpoint.{INV_FREQ}[0]"}


def _raw(*shapes):
    """A header storing the buffer in layer i at `shapes[i]`."""
    return SafetensorsHeader.from_tensors({
        f"model.layers.{i}.self_attn.rotary_emb.inv_freq": ("BF16", shape)
        for i, shape in enumerate(shapes)})


def _header(shape):
    return CheckpointHeader.of("synthetic/x", _raw(shape, shape))


def _stored_length(model_id: str) -> int:
    (shape,) = snapshot_header(model_id).select(INV_FREQ).shapes
    return shape[0]


@pytest.mark.parametrize("model_id", [
    "moonshotai/Kimi-K2-Instruct",
    "moonshotai/Kimi-K2-Base",
    "moonshotai/Kimi-K2-Thinking",
])
def test_kimi_k2_buffer_length_is_the_stored_one(model_id):
    """Same config arithmetic, different stored lengths: only the header
    tells 56 from 128."""
    resolved = resolve_snapshot(model_id)
    assert resolved.dim_values["rope_inv_freq_len"] == _stored_length(model_id)


def test_the_binding_reads_the_supplied_header():
    assert resolve_dimensions(BINDING, {}, _header([7]))["rope_inv_freq_len"] == 7


def test_supplied_header_is_read_instead_of_fetching():
    """A caller that already holds the header (analyze, validate) passes it;
    the loader then does not fetch one for the model id."""
    config = snapshot_config("moonshotai/Kimi-K2-Instruct")
    resolved = shared_loader().match(
        config, model_id="moonshotai/Kimi-K2-Instruct",
        header=_raw([9], [9]))
    assert resolved.dim_values["rope_inv_freq_len"] == 9


def test_no_header_refuses_rather_than_defaults():
    """Resolving without a model id or tensor metadata — a spec resolved by
    name, a custom model with no tensor index, a hub fetch that failed —
    has nothing to read, and there is no fallback value."""
    with pytest.raises(ValueError, match="no tensor metadata"):
        resolve_dimensions(BINDING, {})
    with pytest.raises(ValueError, match="no tensor metadata"):
        resolve_dimensions(BINDING, {}, CheckpointHeader.of(
            "synthetic/x", SafetensorsHeader(groups=())))
    loader = shared_loader()
    config = snapshot_config("moonshotai/Kimi-K2-Instruct")
    with pytest.raises(ValueError, match="rope_inv_freq_len"):
        loader.resolve_by_name("kimi_k2", config)
    with pytest.raises(ValueError, match="no tensor metadata"):
        loader.match(config, model_id="moonshotai/Kimi-K2-Instruct",
                     header=SafetensorsHeader(groups=()))


def test_a_header_without_the_tensor_refuses():
    with pytest.raises(ValueError, match="does not store"):
        resolve_dimensions(BINDING, {}, CheckpointHeader.of(
            "synthetic/x", SafetensorsHeader.from_tensors(
                {"model.norm.weight": ("BF16", [8])})))


def test_a_tensor_stored_at_two_shapes_refuses():
    with pytest.raises(ValueError, match="2 shapes"):
        resolve_dimensions(BINDING, {}, CheckpointHeader.of(
            "synthetic/x", _raw([8], [8], [16])))


@pytest.mark.parametrize("binding", [
    f"checkpoint.{INV_FREQ}[0] ?? 56",
    f"config.rope_len ?? checkpoint.{INV_FREQ}[0]",
    f"checkpoint.{INV_FREQ}[0] // 2",
    f"checkpoint.{INV_FREQ}",
])
def test_a_checkpoint_binding_stands_alone(binding):
    """No fallback (a default for a fact only the file has) and no
    arithmetic: the binding is the stored extent or nothing."""
    with pytest.raises(ValueError, match="checkpoint binding"):
        resolve_dimensions({"rope_inv_freq_len": binding}, {}, _header([7]))


def test_a_spec_without_a_checkpoint_binding_never_reads_the_header():
    """The header is read iff a binding asks for it."""
    def _unreadable():
        raise AssertionError("header read for a spec that binds no checkpoint dim")
    loader = shared_loader()
    config = snapshot_config("meta-llama/Llama-2-7b-hf")
    spec = loader.get("meta-llama__Llama-2-7b-hf") or loader.match(
        config, model_id="meta-llama/Llama-2-7b-hf").spec
    resolve_dimensions(spec.dims, config, CheckpointHeader(_unreadable, "test"))


def test_validate_reports_the_refusal_as_a_finding(monkeypatch):
    """The spec writer's contract: a dim that does not resolve comes back in
    `errors[]`, not as a crash."""
    from backend.arch.validate import _orchestrator as O
    # A readable checkpoint with no tensor metadata: validate refuses an
    # absent `model_storage_bytes` as infrastructure before reaching dims.
    monkeypatch.setattr(O, "_get_safetensors_data",
                        lambda model_id, overlay_dir=None: {"header": SafetensorsHeader(groups=()),
                                          "model_storage_bytes": 0})
    report = O.validate("moonshotai/Kimi-K2-Instruct")
    assert report["valid"] is False
    assert any("rope_inv_freq_len" in e and "no tensor metadata" in e
               for e in report["errors"])


# A config value bound to a dim becomes an integer. A fractional one has no
# integer that states the same fact, so the binding refuses instead of
# truncating: `num_attention_heads: 32.5` read as 32 would be a different
# model, where 32.0 reads as 32.

def test_a_whole_float_binds_to_its_integer():
    dims = resolve_dimensions({"n_heads": "config.num_attention_heads"},
                              {"num_attention_heads": 32.0})
    assert dims["n_heads"] == 32


def test_a_fractional_config_value_refuses_rather_than_truncates():
    with pytest.raises(ValueError, match="not a whole number"):
        resolve_dimensions({"n_heads": "config.num_attention_heads"},
                           {"num_attention_heads": 32.5})


# A config that publishes a value per layer publishes a list; a numeric step
# of the path reads one entry, so the spec reads it instead of copying the
# number (DeepSeek-V4.1's `engram_num_embeddings`, one table height per
# engram layer). An index past the end is a missing field, like any other.

def test_a_numeric_path_step_reads_one_entry_of_a_list():
    config = {"text_config": {"heads_per_layer": [8, 16]}}
    dims = resolve_dimensions(
        {"n_heads": "config.text_config.heads_per_layer.1"}, config)
    assert dims["n_heads"] == 16


def test_an_index_past_the_list_is_a_missing_field():
    dims = resolve_dimensions(
        {"n_heads": "config.heads_per_layer.2 ?? 4"},
        {"heads_per_layer": [8, 16]})
    assert dims["n_heads"] == 4


# `present(config.x)` is a 0 / 1 switch for a step the source runs when
# `x is not None`: the value never reaches the dim, so 0 and fractional values
# switch it on; null switches it off; an absent field takes the `??` fallback
# (the config class's default).

@pytest.mark.parametrize("config, expected", [
    ({"final_logit_softcapping": 30.0}, 1),
    ({"final_logit_softcapping": 0.5}, 1),
    ({"final_logit_softcapping": 0}, 1),
    ({"final_logit_softcapping": None}, 0),
    ({}, 7),
    ({"text_config": {"final_logit_softcapping": 0.0}}, None),
])
def test_presence_is_set_not_nonzero(config, expected):
    binding = {"n_logit_softcap": 'present(config.final_logit_softcapping) ?? "7"'}
    nested = {"n_logit_softcap": 'present(config.text_config.final_logit_softcapping) ?? "7"'}
    if expected is None:
        assert resolve_dimensions(nested, config)["n_logit_softcap"] == 1
        assert resolve_dimensions(nested, {})["n_logit_softcap"] == 7
    else:
        assert resolve_dimensions(binding, config)["n_logit_softcap"] == expected


def test_dims_resolve_whatever_order_they_are_written_in():
    """`dims:` is a mapping. A bundle normalized with sorted keys lists
    `d_vis_merge` before the two dims it multiplies; it must still resolve."""
    from backend.arch.dim_resolution import resolve_dimensions
    written = {
        "d_vis_merge": "n_spatial_merge * n_spatial_merge * d_vis_model",
        "d_vis_model": "config.vision_config.hidden_size",
        "n_spatial_merge": "config.vision_config.spatial_merge_size",
    }
    cfg = {"vision_config": {"hidden_size": 1152, "spatial_merge_size": 2}}
    assert resolve_dimensions(written, cfg)["d_vis_merge"] == 4608


def test_dims_that_read_each_other_are_refused():
    import pytest
    from backend.arch.dim_resolution import resolve_dimensions
    with pytest.raises(ValueError, match="cycle"):
        resolve_dimensions({"d_model": "n_heads * 2", "n_heads": "d_model // 2"}, {})
    with pytest.raises(ValueError, match="cycle"):
        resolve_dimensions({"d_model": "d_model + 1"}, {})


def test_a_path_segment_is_not_a_dim_and_a_quoted_fallback_is():
    """`config.d_head` names a field, not the dim `d_head`; the quoted
    fallback reads `d_model` and `n_heads`, so it resolves after them."""
    from backend.arch.dim_resolution import resolve_dimensions
    got = resolve_dimensions({
        "d_head": "config.d_head ?? 'd_model // n_heads'",
        "d_model": "config.hidden_size",
        "n_heads": "config.num_attention_heads",
    }, {"hidden_size": 64, "num_attention_heads": 4})
    assert got["d_head"] == 16


def test_an_expression_over_an_unbound_name_is_refused_at_its_binding():
    import pytest
    from backend.arch.dim_resolution import resolve_dimensions
    with pytest.raises(ValueError, match="no binding gives a value"):
        resolve_dimensions({"d_model": "n_heads * 2"}, {})


def test_true_division_that_does_not_divide_stays_exact():
    """`/` is true division: a quotient that does not divide is the exact
    Rational, tracked or not; `//` floors and a whole quotient is an int."""
    import sympy as sp
    from backend.arch.eval.dim import Dim, eval_dim_expr, value_of
    assert eval_dim_expr("s_ctx / m", {"s_ctx": 1, "m": 128}) == sp.Rational(1, 128)
    tracked = eval_dim_expr("s_ctx / m", {"s_ctx": Dim.symbol("s_ctx", 1), "m": 4})
    assert value_of(tracked) == sp.Rational(1, 4)
    assert eval_dim_expr("s_ctx // m", {"s_ctx": 7, "m": 2}) == 3
    whole = eval_dim_expr("s_ctx / m", {"s_ctx": 8, "m": 4})
    assert whole == 2 and isinstance(whole, int)


def test_a_spec_cannot_bind_the_scan_chunk():
    """The Mamba-2 scan chunk is the serving framework's choice
    (`Program.scan_chunk_size`), not a model dim: `dims.yaml`
    does not register it, so a spec that binds it is refused as an unknown
    symbol instead of being silently overridden by the deployment's."""
    with pytest.raises(ValueError, match=r"ssm_chunk_size.*neither"):
        resolve_dimensions({"ssm_chunk_size": "config.chunk_size"},
                           {"chunk_size": 128})


@pytest.mark.parametrize("binding, reads", [
    ("config.hidden_size", True),
    ('config.sliding_window ?? "128"', True),
    ("config.text_config.vocab_size // 2", True),
    ("fooconfig.x", False),
    ("checkpoint.model.embed.weight[0]", False),
    ("4096", False),
])
def test_references_config_is_the_resolution_grammar(binding, reads):
    """The config-dims card and resolution read config paths through one
    recogniser: a word that merely ends in `config` is not a config read,
    and a `checkpoint.` path is another source."""
    assert references_config(binding) is reads
