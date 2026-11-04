"""Every model layer that reads a tensor names it; only a zero-input op omits
`inputs:`. What the model receives from outside is named too
(`model_inputs:`), so a lookup reads declared token ids and an encoder its
declared features."""
from __future__ import annotations

import pytest
from pydantic import ValidationError

from backend.arch.schema import (ArchSpec, BlockGraph, GraphNode, ModelLayer,
                                 StreamSpec)

#: The token ids every model here reads.
_IDS = {"input_ids": {"kind": "tokens"}}

#: A stack a later layer reads by name carries its hidden state.
_HIDDEN = {"hidden": StreamSpec(carries=True)}

def _spec(**kwargs):
    blocks = kwargs.pop("blocks", None) or {
        "block": BlockGraph(
            graph={"n": GraphNode(op="rmsnorm", inputs=["x"], dim="d_model")},
            outputs=["n"],
        )
    }
    return ArchSpec(
        name="t",
        dims={"d_model": "8", "n_vocab": "16", "n_layers": "2"},
        blocks=blocks,
        **kwargs,
    )


def test_consumer_without_inputs_is_refused():
    with pytest.raises(ValidationError, match="must name it in `inputs:`"):
        _spec(model_inputs=_IDS, model=[
            ModelLayer(name="embed", op="embedding", inputs=["input_ids"],
                       boundary_role="input_embedding",
                       dims={"n_vocab": "n_vocab", "d_model": "d_model"}),
            ModelLayer(name="layers", template="block", repeat="n_layers"),
        ])


def test_source_op_may_omit_inputs():
    spec = _spec(model=[
        ModelLayer(name="gain", op="parameter_vector", dim="d_model"),
    ])
    assert spec.model[0].inputs in (None, [])


def test_authored_chain_loads():
    spec = _spec(model_inputs=_IDS, model=[
        ModelLayer(name="embed", op="embedding", inputs=["input_ids"],
                   boundary_role="input_embedding",
                   dims={"n_vocab": "n_vocab", "d_model": "d_model"}),
        ModelLayer(name="layers", template="block", repeat="n_layers",
                   inputs=["embed"], streams=_HIDDEN),
        ModelLayer(name="norm", op="rmsnorm", dim="d_model",
                   inputs=["layers"]),
    ])
    assert spec.model[1].inputs == ["embed"]


def test_group_inputs_must_be_named_by_a_child():
    with pytest.raises(ValidationError, match="no child"):
        _spec(model_inputs=_IDS, model=[
            ModelLayer(name="embed", op="embedding", inputs=["input_ids"],
                       boundary_role="input_embedding",
                       dims={"n_vocab": "n_vocab", "d_model": "d_model"}),
            ModelLayer(name="layers", template="block", repeat="n_layers",
                       inputs=["embed"], streams=_HIDDEN),
            ModelLayer(name="norm", op="rmsnorm", dim="d_model",
                       inputs=["layers"]),
            ModelLayer(
                name="mtp", group=True, aux_kind="mtp",
                inputs=["norm"],
                checkpoint={"prefix": "mtp", "index": "local"},
                layers=[
                    ModelLayer(name="mtp_inner", template="block",
                               repeat="1", inputs=["layers"]),
                ],
            ),
        ])
