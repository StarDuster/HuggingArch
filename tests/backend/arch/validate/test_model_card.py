"""The analysis's parameter counts against the model card's (`_model_card`).

DeepSeek-V4.1-Flash's generated spec matched its checkpoint byte for byte and
reported 214B activated parameters; the card states 8B in prefill and 16B in
decode. The card's counts are read as numbers (`hf.model_card`) and every
reading of a family is held against them.
"""
from __future__ import annotations

from types import SimpleNamespace

from backend.arch.validate import _model_card
from backend.hf.model_card import stated_counts

_V41_CARD = """\
DeepSeek-V4.1-Flash is a multimodal Mixture-of-Experts (MoE) model with 552B
backbone parameters that activate only 8B parameters per token during prefill
and 16B during decode.
Engram conditional memory (196B parameters, sparsely accessed via token-based lookup)
| | V4-Flash-Base | V4-Pro-Base | V4.1-Flash-Base |
|---|---|---|---|
| # Backbone Params | 284B | 1.6T | 552B |
| # Activated Params | 13B | 49B | 8B / 16B |
Trained on 32T tokens of text.
"""

B = 10 ** 9


def _report(total, active, prefill, aux=0, lookup=0, main_lookup=None):
    return SimpleNamespace(
        total_params=total, aux_params={"mtp": aux} if aux else {},
        aux_active_params={}, aux_prefill_active_params={},
        lookup_table_params=lookup,
        main_lookup_table_params=lookup if main_lookup is None else main_lookup,
        total_active_params=active,
        total_decode_active_params=active, total_prefill_active_params=prefill)


def _check(card, report, monkeypatch):
    monkeypatch.setattr(_model_card, "model_card", lambda _m: (card, None))
    return _model_card.run_model_card_check("m", report)


def test_counts_are_read_from_prose_and_tables_and_not_from_token_counts():
    read = {(s.kind, s.count // B) for s in stated_counts(_V41_CARD)}
    assert {("total", 552), ("active", 8), ("active", 16), ("total", 196),
            ("total", 284), ("total", 1600), ("active", 13),
            ("active", 49)} <= read
    assert not any(count == 32000 for _kind, count in read)


def test_a_count_dsv3_words_both_ways_is_read():
    card = "DeepSeek-V3, a MoE language model with 671B total parameters with 37B activated for each token."
    assert [(s.kind, s.count // B) for s in stated_counts(card)] == [
        ("total", 671), ("active", 37)]


def test_counting_lookup_tables_as_touched_is_reported(monkeypatch):
    """The generated V4.1 spec: 763B stored, 214B activated."""
    (finding,) = _check(_V41_CARD, _report(763 * B, 214 * B, 214 * B,
                                           lookup=196 * B), monkeypatch)
    assert finding.kind == "model_card_disagrees"
    assert finding.extra["family"] == "active"


def test_the_corrected_figures_agree(monkeypatch):
    """Backbone without lookup tables 552B, 16.8B per decoded token."""
    assert _check(_V41_CARD, _report(
        763 * B, int(16.8 * B), int(8.9 * B), aux=14 * B, lookup=197 * B),
        monkeypatch) == []


def test_an_aux_stacks_table_is_not_taken_from_the_main_stack_twice():
    """An MTP head's own table is in the aux stack's count already: the main
    stack without lookup tables subtracts only the main stack's."""
    readings = _model_card.readings(_report(
        700 * B, 40 * B, 40 * B, aux=12 * B, lookup=10 * B, main_lookup=9 * B))
    assert readings["total"]["main stack without lookup tables"] == 679 * B


def test_a_card_that_cannot_be_read_says_nothing(monkeypatch):
    monkeypatch.setattr(_model_card, "model_card", lambda _m: (None, "offline"))
    assert _model_card.run_model_card_check("m", _report(1, 1, 1)) == []


def test_a_rounded_count_stands_for_its_rounding(monkeypatch):
    """"3B activated" is 2.5B-3.5B: 3.87B is within 15% of that."""
    card = "Number of Parameters: 80B in total and 3B activated"
    assert _check(card, _report(80 * B, int(3.87 * B), int(3.56 * B)),
                  monkeypatch) == []


def test_a_dense_models_activated_count_is_its_total(monkeypatch):
    """A card listing its MoE siblings' activated counts says nothing about
    a dense model whose every reading activates the whole model."""
    card = ("| Model | Total params | Activated params |\n"
            "| E4B | 8B | 4B activated params |\n| 12B | 12B params | |")
    assert _check(card, _report(12 * B, int(11.9 * B), int(11.9 * B)),
                  monkeypatch) == []
