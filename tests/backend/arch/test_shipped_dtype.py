"""The precision a model ships its unquantised tensors at is read off the
checkpoint, never off `config.torch_dtype`."""
from __future__ import annotations

import re

import pytest

from backend.arch.dtypes import (UnknownDtype, dtype_bytes, served_dtype,
                                 stated_default_dtype)
from backend.arch.schema.quant import QuantContext
from tests.snapshots import resolve_snapshot

_NO_LANGUAGE = ("", {})


def test_the_header_decides_and_the_config_only_cross_checks():
    dtype, source = stated_default_dtype(
        None, {"torch_dtype": "float16"}, _NO_LANGUAGE, {"BF16": 1000})
    assert dtype == "BF16"
    assert "config states F16" in source


def test_mixed_unquantised_dtypes_ask_for_a_declaration():
    with pytest.raises(ValueError, match=re.escape("quant_context.default_dtype")):
        stated_default_dtype(None, {"torch_dtype": "bfloat16"}, _NO_LANGUAGE,
                             {"BF16": 1000, "F32": 8})
    declared = QuantContext(default_dtype="BF16")
    assert stated_default_dtype(declared, {}, _NO_LANGUAGE,
                                {"BF16": 1000, "F32": 8})[0] == "BF16"


def test_a_declaration_the_header_does_not_store_is_refused():
    with pytest.raises(ValueError, match="stores no unquantised tensor"):
        stated_default_dtype(QuantContext(default_dtype="F16"), {},
                             _NO_LANGUAGE, {"BF16": 1000})


@pytest.mark.parametrize("snapshot, dtype", [
    ("google__gemma-2-9b", "F32"),
    ("meta-llama__Llama-3.2-1B", "BF16"),
    ("Qwen__Qwen3-VL-235B-A22B-Instruct", "BF16"),
])
def test_corpus_models_ship_what_their_header_stores(snapshot, dtype):
    assert resolve_snapshot(snapshot).shipped_dtype()[0] == dtype


@pytest.mark.parametrize("stored, served", [
    # The case the corpus has, and the one the rule exists for: gemma-2
    # stores float32, and nothing ships running 32 bits wide.
    ("F32", "BF16"),
    ("F64", "BF16"),
    # At the ceiling already: left exactly as stored, each at its own
    # dtype. A blanket "serve at BF16" would move Llama-2 to the other
    # 16-bit float — same width, but not the dtype it stores.
    ("F16", "F16"),
    ("BF16", "BF16"),
])
def test_nothing_ships_running_wider_than_16_bits(stored, served):
    assert served_dtype(stored) == served


@pytest.mark.parametrize("stored", ["F64", "F32", "F16", "BF16"])
def test_the_served_width_never_exceeds_the_stored_one(stored):
    """The monotone half of the rule, and the half that keeps it agreeing
    with "a BF16 checkpoint stays BF16": a loader that materialises fp32
    parameters for a 2-byte weight is not a serving width, and this
    function cannot express one — it only ever narrows."""
    assert dtype_bytes(served_dtype(stored)) <= dtype_bytes(stored)


def test_a_shipped_dtype_that_is_not_a_plain_float_is_refused():
    """`quant_context.default_dtype` names what the checkpoint stores its
    UNQUANTISED tensors at, so a scheme's width arriving there is a broken
    declaration rather than a narrow checkpoint. Answering BF16 for it —
    widening it, at that — is the plausible default §2.4 forbids."""
    with pytest.raises(UnknownDtype, match="cannot be stored at"):
        served_dtype("F8_E4M3")
    with pytest.raises(UnknownDtype, match="cannot be stored at"):
        served_dtype("I8")


def test_without_a_header_the_config_answers_and_says_so():
    dtype, source = stated_default_dtype(
        None, {"torch_dtype": "bfloat16"}, _NO_LANGUAGE, None)
    assert (dtype, source) == ("BF16", "config (no checkpoint header)")


def test_a_model_stating_no_precision_is_refused_not_assumed():
    from backend.arch.eval.formats import FormatPolicy
    from backend.arch.eval.refusal import SpecRefusal
    assert stated_default_dtype(None, {}, _NO_LANGUAGE, None) == (None, None)
    with pytest.raises(SpecRefusal, match=re.escape("quant_context.default_dtype")):
        FormatPolicy.of(None, None)
