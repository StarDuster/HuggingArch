"""Readers of a stack or an appended stream whose syntax says one thing and
whose walk would do another.

A stack's name reads its final value — the carried stream after its last
iteration — and a model-level `op:` that names an appended stream without
`axis:` reads all its elements as one `[B, S, Sar, D]` history. Two misuses
loaded and walked without a finding before:

* a model-level `slice` / `identity` of a stack's name written to stand for
  an earlier layer's hidden state (Llava's `hidden_states[-2]`): it read the
  final value;
* a projector naming a multi-element appended stream without `axis:`: it ran
  on each element where the elements were meant to be joined.
"""
from __future__ import annotations

import os

import pytest

from backend.arch.eval.program import Program
from backend.arch.eval.transforms import Deployment
from tests.backend.arch.test_appended_streams import FEAT_READ, _analyze, _resolved
from tests.snapshots import resolve_snapshot, shared_loader, snapshot_config

APPENDED_HEADING = ("Appended streams: values a stack hands on after chosen "
                    "iterations")
MERGE = ("  - {{ name: merge, op: scatter_rows_replace, inputs: [embed, vis.blocks],\n"
         "      prompt_slots: {{ domain: pixel_values, defines: true, source_anchor: \"a test merge\" }} }}")
TOWER_END = "      - name: vis.mergers\n"


def _in_tower(layer: str):
    """`layer` as the vision group's stage after its blocks."""
    return lambda spec: spec.replace(TOWER_END,
                                     f"      - {layer}\n{TOWER_END}", 1)


def _select_from_tower(layer: str):
    """The tower's output is read through `layer` (`name: vsel`), the
    group's stage the merge reads."""
    def edit(spec):
        assert MERGE in spec
        spec = spec.replace(MERGE, MERGE.replace(
            "inputs: [embed, vis.blocks]", "inputs: [embed, vsel]"))
        return _in_tower(
            f"{{{{ name: vsel, {layer}, inputs: [vis.blocks] }}}}")(spec)
    return edit


def _without_deepstack(spec: str) -> str:
    """No decoder layer adds the mergers' outputs: a merge that hands over
    fewer rows than the tower holds is then the only edge onto the image's
    positions."""
    for line in (FEAT_READ, "        - {{ type: ds, {ds} }}\n",
                 "        ds: {{ template: inject }}\n"):
        assert line in spec
        spec = spec.replace(line, "", 1)
    return spec


def _walk(edit):
    _analyze(Program.of(_resolved("vsl", edit=edit)))


class TestAStackNameIsItsFinalValue:
    @pytest.mark.parametrize("layer", [
        "op: slice, dims: {{ in: d_model, out: d_model }}",
        "op: slice, dims: {{ out: d_model }}",
        "op: identity",
    ])
    def test_a_view_that_selects_nothing_is_refused(self, layer):
        with pytest.raises(Exception) as err:
            _walk(_select_from_tower(layer))
        text = str(err.value)
        assert "reads stack 'vis.blocks' by name and hands its value on unchanged" in text
        assert "from: hidden, after:" in text and APPENDED_HEADING in text

    def test_a_slice_along_an_iteration_axis_it_does_not_have_is_refused(self):
        with pytest.raises(Exception) as err:
            _walk(_select_from_tower(
                "op: slice, axis: 0, dims: {{ in: n_vis_layers, out: '1' }}"))
        assert "declares `in: n_vis_layers` (4), but the axis it narrows is B" \
            in str(err.value)

    def test_a_narrowing_of_the_final_value_is_read(self):
        select = _select_from_tower("op: slice, axis: 1, dims: {{ out: '1' }}")
        _walk(lambda spec: _without_deepstack(select(spec)))


class TestAHistoryGoesToAPortThatTakesOne:
    def test_a_projector_without_axis_is_refused(self):
        with pytest.raises(Exception) as err:
            _resolved("vsl", edit=_in_tower(
                "{{ name: proj, op: linear, inputs: [taps], "
                "dims: {{ in: '2 * d_model', out: d_model }}, "
                "weight_axes: [{{ replicated: true }}, {{ replicated: true }}] }}"))
        text = str(err.value)
        assert "names appended stream 'taps' in `inputs:` without `axis:`" in text
        assert "declare `axis:` on 'proj'" in text and APPENDED_HEADING in text

    def test_a_projector_that_joins_is_read(self):
        _walk(_in_tower(
            "{{ name: proj, op: linear, inputs: [taps], axis: -1, "
            "dims: {{ in: '2 * d_model', out: d_model }}, "
            "weight_axes: [{{ replicated: true }}, {{ replicated: true }}] }}"))


@pytest.mark.parametrize("model", [
    "moonshotai/Kimi-K3",           # attn_res_combine reads the stack and its history
    "XiaomiMiMo/MiMo-V2-Flash",     # MTP logits read a stack by name
    "zai-org/GLM-5.1",              # MTP logits read a repeated group by name
    "deepseek-ai/DeepSeek-V4-Flash",  # hc_head reads the stack
])
def test_corpus_readers_of_stacks_and_histories_still_walk(model):
    os.environ["HF_HUB_OFFLINE"] = "1"
    resolve_snapshot(model)
    resolved = shared_loader().match(snapshot_config(model), model_id=model)
    Deployment(Program.of(resolved)).at(batch=1, input_tokens=64).analyze()
