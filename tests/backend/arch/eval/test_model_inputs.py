"""What the model receives from outside its graph is declared, not guessed.

The token ids are `[B, s_ctx]`, the step's own. An image or an audio clip is a
`model_inputs:` entry with its own axes — items per request, then one item's
extents — which the workload gives (`Workload.media`) and only the encode
phase binds; an image's rows are its patch grid, cut by the `patchify` the
spec declares. A tensor nothing produced and nothing declared is refused,
never read off an op's signature.
"""
from __future__ import annotations

import os
import sys
import tempfile
import unittest
from pathlib import Path

import pytest
from pydantic import ValidationError

PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT))

from backend.arch.axis_propagation import infer_shape
from backend.arch.eval.dim import Dim, expr_of, value_of
from backend.arch.eval.program import Program
from backend.arch.eval.transforms import Deployment
from backend.arch.layout.layout import Block, Replicated
from backend.arch.eval.refusal import SpecRefusal
from backend.arch.loader import SpecLoader
from backend.arch.schema import BlockGraph, GraphNode
from backend.arch.shape_node import ShapeProbe
from tests.snapshots import shared_program, shared_walk

_ENCODER = """
match: { model_type: [_model_inputs] }
dims: { d_model: config.hidden_size, n_layers: "1", n_vocab: "16",
        n_vis_channels: "3", s_patch: "2" }
blocks:
  blk:
    graph:
      n: { op: rmsnorm, in: x, dim: d_model, weight_axes: [{ replicated: true }] }
    outputs: [n]
model_inputs:
  input_ids: { kind: tokens }
  pixel_values: { kind: image, axes: [n_img, img_h, img_w] }
model:
  - { name: embed, op: embedding, inputs: [input_ids], boundary_role: input_embedding,
      dims: { n_vocab: n_vocab, d_model: d_model },
      weight_axes: [{ replicated: true }, { replicated: true }] }
  - name: vis
    group: true
    aux_kind: vision
    layers:
      - { name: vis.patchify, op: patchify, inputs: [pixel_values],
          dims: { patch: s_patch, temporal: "1", channels: n_vis_channels, align: "2 * s_patch" },
          segment: item, source_anchor: "a test processor" }
      - { name: vis.patch, op: linear, inputs: [vis.patchify],
          dims: { in: "n_vis_channels * s_patch * s_patch", out: d_model },
          weight_axes: [{ replicated: true }, { replicated: true }] }
  - { name: merge, op: scatter_rows_replace, inputs: [embed, vis.patch],
      prompt_slots: { domain: pixel_values, defines: true, source_anchor: "a test merge" } }
  - { name: layers, template: blk, repeat: n_layers, inputs: [merge],
      streams: { hidden: { carries: true } } }
  - { name: head, op: linear, boundary_role: output_head, inputs: [layers],
      dims: { in: d_model, out: n_vocab },
      weight_axes: [{ replicated: true }, { replicated: true }] }
"""


def _load(text: str):
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as f:
        f.write(text)
        path = f.name
    loader = SpecLoader()
    try:
        return loader.resolve(loader.load_model_spec(path),
                              {"torch_dtype": "bfloat16", "hidden_size": 8,
                               "model_type": "_model_inputs"})
    finally:
        os.unlink(path)


def _node(analysis, name):
    (node,) = [n for n in analysis.program.nodes() if n.name == name]
    return node


_IMAGE = {"pixel_values": {"n_img": 2, "img_h": 8, "img_w": 12}}


class TestRowsComeFromTheMedia(unittest.TestCase):
    def test_an_images_rows_are_its_patch_grid_per_item(self):
        """Two 8x12 images in 2x2 patches: two segments of 4x6 patch rows,
        each row a patch's 3*2*2 values."""
        program = Program.of(_load(_ENCODER))
        analysis = shared_walk(Deployment(program).at(batch=1, media=_IMAGE).during(
            "encode", stack="vis"))
        patch = analysis.shapes.values[_node(analysis, "vis.patch").uid]
        self.assertEqual([value_of(d) for d in patch.ins[0]], [2, 24, 12])

    def test_the_media_fill_their_defining_edges_rows_on_the_encoder_side(self):
        """Two 8x12 images hand over 48 rows: the encoder's fact. The
        language model's step is its tokens alone, its merge handed no
        rows."""
        deployment = Deployment(Program.of(_load(_ENCODER))).at(
            batch=2, input_tokens=5, media=_IMAGE)
        self.assertEqual(deployment.media_tokens(), {"pixel_values": 48})
        self.assertEqual([t.rows for t in deployment.handed_over()["vis.patch"]],
                         [48])
        analysis = shared_walk(deployment)
        merge = analysis.shapes.values[_node(analysis, "merge").uid]
        self.assertEqual([value_of(d) for d in merge.ins[1]], [2, 0, 8])
        self.assertEqual(value_of(analysis.binding["s_ctx"]), 5)

    def test_no_media_is_zero_rows_and_the_language_model_alone(self):
        analysis = shared_walk(Deployment(Program.of(_load(_ENCODER))).at(
            batch=2, input_tokens=4))
        merge = analysis.shapes.values[_node(analysis, "merge").uid]
        self.assertEqual([value_of(d) for d in merge.ins[1]], [2, 0, 8])
        self.assertNotIn(_node(analysis, "vis.patch").uid,
                         analysis.cost.values)

    def test_a_lookups_leading_axes_are_the_ids(self):
        """The text embedding's batch and sequence are the token ids' own
        axes, under the identity the binding gives the step's tokens."""
        analysis = shared_walk(Deployment(Program.of(_load(_ENCODER))).at(
            batch=2, input_tokens=4))
        embed = analysis.shapes.values[_node(analysis, "embed").uid]
        self.assertIs(embed.out[0], embed.ins[0][0])
        self.assertIs(embed.out[1], embed.ins[0][1])
        self.assertEqual(expr_of(embed.out[1]),
                         expr_of(analysis.binding["s_ctx"]))


class TestMediaIsHeldToTheDeclaration(unittest.TestCase):
    def _encode(self, media, text=_ENCODER):
        return shared_walk(Deployment(Program.of(_load(text))).at(
            batch=1, media=media).during("encode", stack="vis"))

    def test_an_undeclared_input_is_refused(self):
        with self.assertRaisesRegex(ValueError, "name no media input"):
            self._encode({"pixel": {"n_img": 1, "img_h": 4, "img_w": 4}})

    def test_a_missing_axis_is_refused(self):
        with self.assertRaisesRegex(ValueError, r"missing \['img_w'\]"):
            self._encode({"pixel_values": {"n_img": 1, "img_h": 4}})

    def test_a_side_off_the_processors_factor_is_refused(self):
        with self.assertRaisesRegex(ValueError, r"height 6 is not a multiple of 4\b.*use 4 or 8"):
            self._encode({"pixel_values": {"n_img": 1, "img_h": 6, "img_w": 4}})

    def test_one_size_per_input(self):
        with self.assertRaisesRegex(ValueError, "one size per step"):
            self._encode({"pixel_values": {"n_img": 2, "img_h": [4, 8],
                                           "img_w": 4}})

    def test_a_fixed_axis_given_by_the_workload_is_refused(self):
        text = _ENCODER.replace(
            "pixel_values: { kind: image, axes: [n_img, img_h, img_w] }",
            'pixel_values: { kind: image, axes: [n_img, img_h, img_w], '
            'fixed: { img_h: "4" } }')
        self._encode({"pixel_values": {"n_img": 1, "img_w": 4}}, text)
        with self.assertRaisesRegex(ValueError, "the config fixes"):
            self._encode({"pixel_values": {"n_img": 1, "img_h": 4,
                                           "img_w": 4}}, text)

    def test_a_video_declares_its_frames(self):
        # A video's axes are its items, its sampled frames and its sides.
        with self.assertRaisesRegex(Exception, "`kind: video` input declares"):
            _load(_ENCODER.replace("kind: image", "kind: video"))

    def test_a_videos_frames_take_whole_temporal_patches(self):
        """`[N, n, T, H, W]` in temporal patches of `temporal` frames, the
        last frame repeated to a whole patch: 5 frames in pairs are 3
        patches, one attention segment each."""
        text = (_ENCODER
                .replace("pixel_values: { kind: image, axes: [n_img, img_h, img_w] }",
                         "pixel_values: { kind: video, axes: [n_img, frames, img_h, img_w] }")
                .replace('dims: { patch: s_patch, temporal: "1", channels: n_vis_channels, align: "2 * s_patch" },\n          segment: item',
                         'dims: { patch: s_patch, temporal: "2", channels: n_vis_channels, align: "2 * s_patch" },\n          segment: temporal_patch')
                .replace('dims: { in: "n_vis_channels * s_patch * s_patch", out: d_model }',
                         'dims: { in: "n_vis_channels * 2 * s_patch * s_patch", out: d_model }'))
        program = Program.of(_load(text))
        analysis = shared_walk(Deployment(program).at(batch=1, media={"pixel_values": {
            "n_img": 2, "frames": 5, "img_h": 8, "img_w": 12}}).during(
            "encode", stack="vis"))
        patch = analysis.shapes.values[_node(analysis, "vis.patch").uid]
        # 2 videos x 3 temporal patches, 4x6 patch rows, 3*2*2*2 values.
        self.assertEqual([value_of(d) for d in patch.ins[0]], [6, 24, 24])

    def test_the_vision_kind_is_retired(self):
        with self.assertRaisesRegex(Exception, "is now `kind: image`"):
            _load(_ENCODER.replace("kind: image", "kind: vision"))

    def test_an_encoder_node_reading_the_text_step_is_refused(self):
        text = _ENCODER.replace(
            '          dims: { in: "n_vis_channels * s_patch * s_patch", out: d_model },',
            '          dims: { in: "n_vis_channels * s_patch * s_patch", out: "s_ctx" },')
        with self.assertRaisesRegex(SpecRefusal, r"vis\.patch \(linear\) reads "
                                    r"\['s_ctx'\](.|\n)*model_inputs\.pixel_values"):
            self._encode(_IMAGE, text)


class TestPromptSlotsAreDeclared(unittest.TestCase):
    def test_an_edge_into_the_language_model_names_its_slots(self):
        with self.assertRaisesRegex(Exception, "declares no `prompt_slots`"):
            _load(_ENCODER.replace(
                ',\n      prompt_slots: { domain: pixel_values, defines: true, '
                'source_anchor: "a test merge" }', ''))

    def test_the_domain_is_a_media_input(self):
        for domain in ("pixels", "input_ids"):
            with self.assertRaisesRegex(Exception, "not a media input"):
                _load(_ENCODER.replace("domain: pixel_values",
                                       f"domain: {domain}"))

    def test_an_encoder_defines_its_slots(self):
        with self.assertRaisesRegex(Exception, "define no prompt positions"):
            _load(_ENCODER.replace("defines: true", "defines: false"))

    def test_a_domain_is_defined_once(self):
        with self.assertRaisesRegex(Exception, "has 2 defining edges"):
            _load(_ENCODER.replace(
                "  - { name: layers, template: blk, repeat: n_layers, inputs: [merge],",
                "  - { name: again, op: scatter_rows_add, inputs: [merge, vis.patch],\n"
                "      prompt_slots: { domain: pixel_values, defines: true, "
                "source_anchor: x } }\n"
                "  - { name: layers, template: blk, repeat: n_layers, inputs: [again],"))

    def test_slots_off_an_encoder_edge_are_refused(self):
        with self.assertRaisesRegex(Exception, "reads no encoder output"):
            _load(_ENCODER.replace(
                "  - { name: layers, template: blk,",
                "  - { name: other, op: rmsnorm, dim: d_model, inputs: [merge],\n"
                "      weight_axes: [{ replicated: true }],\n"
                "      prompt_slots: { domain: pixel_values, defines: false, "
                "source_anchor: x } }\n"
                "  - { name: layers, template: blk,"))

    def test_every_media_input_has_a_defining_edge(self):
        """A second image the encoder reads and no edge brings into the
        language model is refused: its prompt positions are undefined."""
        text = _ENCODER.replace(
            "  pixel_values: { kind: image, axes: [n_img, img_h, img_w] }",
            "  pixel_values: { kind: image, axes: [n_img, img_h, img_w] }\n"
            "  depth_values: { kind: image, axes: [n_depth, depth_h, depth_w] }"
        ).replace(
            "      - { name: vis.patch, op: linear, inputs: [vis.patchify],",
            "      - { name: vis.depth, op: patchify, inputs: [depth_values],\n"
            "          dims: { patch: s_patch, temporal: \"1\", channels: n_vis_channels, align: \"2 * s_patch\" },\n"
            "          segment: item, source_anchor: \"a test processor\" }\n"
            "      - { name: vis.patch, op: linear, inputs: [vis.patchify],")
        with self.assertRaisesRegex(Exception, r"media input\(s\) \['depth_values'\] "
                                    r"have no defining edge"):
            _load(text)

    def test_an_image_is_read_by_patchify(self):
        with self.assertRaisesRegex(Exception, "read by a `patchify` layer"):
            _load(_ENCODER.replace("op: patchify, inputs: [pixel_values]",
                                   "op: identity, inputs: [pixel_values]"))


_CROPS = _ENCODER.replace(
    "  pixel_values: { kind: image, axes: [n_img, img_h, img_w] }",
    "  pixel_values: { kind: image, axes: [n_img, img_h, img_w] }\n"
    "  crops: { kind: image, axes: [n_crops, crop_h, crop_w], "
    "fixed: { crop_h: \"4\", crop_w: \"4\" }, per_item_of: pixel_values, "
    "unmodeled: \"a second call of the tower, not expressible here\" }")


class TestUnmodeledInputs(unittest.TestCase):
    """An input the processor hands the model that the spec declares and
    the engine does not walk: stated with every request, refused any item."""

    def _media(self, crops=None):
        media = {"pixel_values": {"n_img": 1, "img_h": 8, "img_w": 8}}
        if crops is not None:
            media["crops"] = {"n_crops": crops}
        return Deployment(Program.of(_load(_CROPS))).at(batch=1, media=media)

    def test_a_request_without_items_of_it_is_exact(self):
        self.assertEqual(self._media(crops=0).media_tokens(),
                         {"pixel_values": 16})

    def test_its_items_are_stated_with_the_items_they_belong_to(self):
        with self.assertRaisesRegex(ValueError, "state them too"):
            self._media().media_tokens()

    def test_any_item_of_it_is_refused_with_the_reason(self):
        from backend.arch.eval.refusal import EngineGap
        with self.assertRaisesRegex(EngineGap, "2 item\\(s\\) of 'crops'"
                                    "(.|\\n)*a second call of the tower"):
            self._media(crops=2).media_tokens()

    def test_no_layer_reads_it(self):
        with self.assertRaisesRegex(Exception, "declared `unmodeled:` and read"):
            _load(_CROPS.replace("op: patchify, inputs: [pixel_values]",
                                 "op: patchify, inputs: [crops]"))

    def test_only_the_items_of_another_input_are_unmodeled(self):
        with self.assertRaisesRegex(Exception, "`unmodeled:` states why"):
            _load(_CROPS.replace(" per_item_of: pixel_values,", ""))


class TestPlacementByPosition(unittest.TestCase):
    """The batch is the request owner's, the rows are the sequence cut of
    the view that reads them, a row's width is whole."""

    def test_text_rows_follow_the_sequence_cut_and_encoder_rows_do_not(self):
        program = shared_program(
            "Qwen/Qwen3-VL-235B-A22B-Instruct")
        analysis = shared_walk(Deployment(program).at(batch=1, input_tokens=8).on(
            cp=2))
        embed = analysis.input_states[(
            _node(analysis, "language_model.embed_tokens").uid, 0)]
        self.assertIsInstance(embed.layout[1], Block)
        self.assertEqual(embed.layout[1].group.id, "cp")
        encode = shared_walk(Deployment(program).at(batch=1, media={
            "pixel_values": {"n_img": 1, "img_h": 64, "img_w": 64}}).on(
            cp=2).during("encode", stack="visual"))
        patch = encode.input_states[(
            _node(encode, "visual.patch_embed.proj").uid, 0)]
        self.assertIsInstance(patch.layout[1], Replicated)
        self.assertIsInstance(patch.layout[2], Replicated)


class TestNothingIsReadThatNothingNames(unittest.TestCase):
    def test_a_lookup_that_names_no_ids_is_refused(self):
        with self.assertRaisesRegex(Exception, "must name it in `inputs:`"):
            _load(_ENCODER.replace(
                "name: embed, op: embedding, inputs: [input_ids],",
                "name: embed, op: embedding,").replace(
                "  input_ids: { kind: tokens }\n", ""))

    def test_a_model_input_nothing_reads_is_refused(self):
        with self.assertRaisesRegex(Exception, "read by no layer"):
            _load(_ENCODER.replace(
                "  input_ids: { kind: tokens }\n",
                "  input_ids: { kind: tokens }\n"
                "  audio_codes: { kind: audio, axes: [n_aud, s_audio] }\n"))

    def test_media_is_read_by_an_encoder(self):
        with self.assertRaisesRegex(Exception, "only by the encoder group"):
            _load(_ENCODER.replace(
                "  - { name: merge, op: scatter_rows_replace",
                "  - { name: stray, op: patchify, inputs: [pixel_values],\n"
                "      dims: { patch: s_patch, temporal: \"1\", channels: n_vis_channels, align: s_patch },\n"
                "      segment: item, source_anchor: x }\n"
                "  - { name: merge, op: scatter_rows_replace"))

    def test_the_text_stack_reads_tokens(self):
        with self.assertRaisesRegex(Exception, "only by the encoder group"):
            _load(_ENCODER.replace("input_ids: { kind: tokens }",
                                   "input_ids: { kind: audio, axes: [n_aud, s_audio] }"))

    def test_a_width_outside_the_models_dims_is_refused(self):
        with self.assertRaisesRegex(Exception, "does not bind"):
            _load(_AUDIO.replace(_DECLARED, "width: d_elsewhere }"))


_LOOKUP_OF_A_SUM = (
    "  - { name: summed, op: sum_reduce, inputs: [embed], dims: { in: d_model } }\n"
    "  - { name: table, op: embedding, inputs: [summed],\n"
    "      dims: { n_vocab: n_vocab, d_model: d_model },\n"
    "      weight_axes: [{ replicated: true }, { replicated: true }] }\n")


def _analyze(text: str):
    return shared_walk(Deployment(Program.of(_load(text))).at(
        batch=1, input_tokens=4))


_AUDIO = _ENCODER.replace(
    "  input_ids: { kind: tokens }\n",
    "  input_ids: { kind: tokens }\n"
    "  frames: { kind: audio, axes: [n_aud, s_audio], width: \"n_vis_channels * s_patch * s_patch\" }\n"
).replace(
    "  - { name: merge, op: scatter_rows_replace, inputs: [embed, vis.patch],",
    "  - name: aud\n"
    "    group: true\n"
    "    aux_kind: audio\n"
    "    layers:\n"
    "      - { name: aud.proj, op: linear, inputs: [frames],\n"
    "          dims: { in: \"n_vis_channels * s_patch * s_patch\", out: d_model },\n"
    "          weight_axes: [{ replicated: true }, { replicated: true }] }\n"
    "  - { name: merge_audio, op: scatter_rows_replace, inputs: [embed, aud.proj],\n"
    "      prompt_slots: { domain: frames, defines: true, source_anchor: \"a test merge\" } }\n"
    "  - { name: merge, op: scatter_rows_replace, inputs: [merge_audio, vis.patch],")

_PROJ = (
    "      - { name: aud.proj, op: linear, inputs: [frames],\n"
    "          dims: { in: \"n_vis_channels * s_patch * s_patch\", out: d_model },\n"
    "          weight_axes: [{ replicated: true }, { replicated: true }] }\n")

_TOWER = (
    "  tower:\n"
    "    graph:\n"
    "      proj: { op: linear, in: x, dims: { in: \"n_vis_channels * s_patch * s_patch\", out: d_model },\n"
    "              weight_axes: [{ replicated: true }, { replicated: true }] }\n"
    "    outputs: [proj]\n")

_DECLARED = 'width: \"n_vis_channels * s_patch * s_patch\" }'
_TOO_NARROW = 'width: \"2 * s_patch * s_patch\" }'


class TestReadersTakeWhatTheInputIs(unittest.TestCase):
    """`model_inputs.width` is what an audio frame is; a reader that declares
    it takes another width, or a lookup that reads anything but ids, is a
    wrong spec, refused naming both sides."""

    def test_the_audio_frames_rows_are_its_clips_frames(self):
        analysis = shared_walk(Deployment(Program.of(_load(_AUDIO))).at(
            batch=2, media={"frames": {"n_aud": 3, "s_audio": 5}}).during(
            "encode", stack="aud"))
        proj = analysis.shapes.values[_node(analysis, "aud.proj").uid]
        self.assertEqual([value_of(d) for d in proj.ins[0]], [6, 5, 12])

    def test_a_width_its_reader_does_not_take_is_refused(self):
        _analyze(_AUDIO)
        with self.assertRaisesRegex(SpecRefusal, r"aud\.proj(.|\n)*"
                                    r"'frames'(.|\n)*2 \* s_patch"):
            _analyze(_AUDIO.replace(_DECLARED, _TOO_NARROW))

    def test_the_width_is_checked_through_a_components_formal(self):
        """The reader sits inside a component the stage calls; the formal
        that binds it carries the model input to it."""
        text = _AUDIO.replace("blocks:\n", "blocks:\n" + _TOWER, 1).replace(
            _PROJ,
            "      - { name: aud.proj, template: tower, inputs: [frames] }\n")
        _analyze(text)
        with self.assertRaisesRegex(SpecRefusal, r"proj(.|\n)*'frames'"):
            _analyze(text.replace(_DECLARED, _TOO_NARROW))

    def test_a_lookup_fed_an_activation_is_refused(self):
        text = _ENCODER.replace(
            "  - { name: merge, op: scatter_rows_replace",
            _LOOKUP_OF_A_SUM + "  - { name: merge, op: scatter_rows_replace")
        with self.assertRaisesRegex(SpecRefusal, r"table \(embedding\) looks "
                                    r"ids up(.|\n)*output of 'summed'"):
            _analyze(text)

    def test_a_lookup_of_an_input_with_a_width_is_refused(self):
        text = _AUDIO.replace(
            "      - { name: aud.proj, op: linear, inputs: [frames],",
            "      - { name: aud.table, op: embedding, inputs: [frames],\n"
            "          dims: { n_vocab: n_vocab, d_model: d_model },\n"
            "          weight_axes: [{ replicated: true }, { replicated: true }] }\n"
            "      - { name: aud.proj, op: linear, inputs: [frames],")
        with self.assertRaisesRegex(SpecRefusal, r"model input 'frames' "
                                    r"of width"):
            _analyze(text)

    def test_a_width_is_traced_as_the_walk_binds_a_colliding_name(self):
        """`inner` reads its formal `z`; the call passes `[good, x]`, so the
        expanded read keeps the caller's name `x`, which is also `inner`'s
        first formal. The caller-side name wins (`_callee_aliases`): `proj`
        reads the model input `x`, eight wide, and declares twelve."""
        text = _AUDIO.replace("blocks:\n", "blocks:\n" + (
            "  inner:\n"
            "    inputs: [x, z]\n"
            "    graph:\n"
            "      proj: { op: linear, in: z, dims: { in: \"n_vis_channels * s_patch * s_patch\", out: d_model },\n"
            "              weight_axes: [{ replicated: true }, { replicated: true }] }\n"
            "    outputs: [proj]\n"
            "  outer:\n"
            "    inputs: [good, x]\n"
            "    graph:\n"
            "      call: { op: inner, in: [good, x] }\n"
            "    outputs: [call]\n"), 1).replace(
            _PROJ,
            "      - { name: aud.pre, op: linear, inputs: [x],\n"
            "          dims: { in: \"2 * s_patch * s_patch\", out: \"n_vis_channels * s_patch * s_patch\" },\n"
            "          weight_axes: [{ replicated: true }, { replicated: true }] }\n"
            "      - { name: aud.proj, template: outer, inputs: [aud.pre, x] }\n").replace(
            "  frames: { kind: audio, axes: [n_aud, s_audio], "
            "width: \"n_vis_channels * s_patch * s_patch\" }",
            "  x: { kind: audio, axes: [n_aud, s_audio], width: \"2 * s_patch * s_patch\" }"
        ).replace("domain: frames", "domain: x")
        with self.assertRaisesRegex(SpecRefusal, r"proj \(linear\)(.|\n)*"
                                    r"model input 'x'"):
            _analyze(text)

    def test_a_lookup_is_traced_as_the_walk_binds_a_colliding_name(self):
        """`inner` looks up its formal `z`; the call passes `[input_ids,
        ids]`, and the caller-side name `ids` — a computed tensor — wins over
        `inner`'s formal `ids`, which the token ids bind."""
        text = _ENCODER.replace("blocks:\n", "blocks:\n" + (
            "  inner:\n"
            "    inputs: [ids, z]\n"
            "    graph:\n"
            "      table: { op: embedding, in: z, dims: { n_vocab: n_vocab, d_model: d_model },\n"
            "               weight_axes: [{ replicated: true }, { replicated: true }] }\n"
            "    outputs: [table]\n"
            "  outer:\n"
            "    inputs: [input_ids, ids]\n"
            "    graph:\n"
            "      call: { op: inner, in: [input_ids, ids] }\n"
            "    outputs: [call]\n"), 1).replace(
            "  - { name: merge, op: scatter_rows_replace",
            "  - { name: ids, op: sum_reduce, inputs: [embed], dims: { in: d_model } }\n"
            "  - { name: use, template: outer, inputs: [input_ids, ids] }\n"
            "  - { name: merge, op: scatter_rows_replace")
        with self.assertRaisesRegex(SpecRefusal, r"table \(embedding\) looks "
                                    r"ids up(.|\n)*'ids'"):
            _analyze(text)

    def test_the_check_reads_only_the_inputs_the_walk_bound(self):
        """A source op reads nothing even where its node carries an edge
        (`Evaluator._inputs` binds it none): the reader check asks the
        inputs the walk bound, not the node's refs. Here a reader with an
        edge onto a too-narrow model input, bound no input, is not
        refused."""
        program = Program.of(_load(_AUDIO.replace(_DECLARED, _TOO_NARROW)))
        proj = next(n for n in program.nodes() if n.name == "aud.proj")
        self.assertEqual(len(proj.inputs_obj), 1)
        program.check_model_input_reader(proj, {}, {proj.uid: 0})

    def test_a_shape_generic_reader_takes_any_width(self):
        """`identity` declares a rank, and its trailing axis is no width the
        node binds: it reads the frames as they are."""
        _analyze(_AUDIO.replace(_PROJ, (
            "      - { name: aud.id, op: identity, inputs: [frames] }\n"
            + _PROJ.replace("inputs: [frames]", "inputs: [aud.id]"))))

    def test_audio_codebook_ids_are_ids(self):
        """MiMo's speech embeddings look up `audio_codes`, ids with no width,
        as the processor pads them to whole groups (`audio.pad`)."""
        analysis = shared_walk(Deployment(shared_program(
            "XiaomiMiMo/MiMo-V2.5")).at(batch=1).during(
            "encode", stack="audio"))
        program = analysis.program
        (node,) = [n for n in program.nodes() if n.name == "speech_embeddings"]
        (pad,) = [n for n in program.nodes() if n.name == "audio.pad"]
        self.assertEqual(node.inputs_obj[0].src.uid, pad.uid)
        self.assertEqual(analysis.model_input_read(pad.uid, 0), "audio_codes")
        self.assertIsNone(program.model_inputs["audio_codes"].width)


class TestViewMergeWidth:
    def test_the_width_is_the_product_of_the_merged_axes(self):
        """No call-site width: the merged axis is the producer's last two
        axes multiplied, whatever they are named."""
        a, b = Dim.symbol("n_hidden_copies", 4), Dim.symbol("d_model", 6)
        ins = [(Dim.symbol("B", 2), Dim.symbol("s_ctx", 3), a, b)]
        out = infer_shape("view_merge", ShapeProbe(), ins, {"out": 999})
        assert [value_of(d) for d in out] == [2, 3, 24]
        assert out[0] is ins[0][0] and out[1] is ins[0][1]
        assert out[2] == a * b

    def test_a_call_site_width_is_refused_at_load(self):
        with pytest.raises(ValidationError, match="delete `out:`"):
            BlockGraph(inputs=["x"], graph={
                "m": GraphNode(op="view_merge", inputs=["x"],
                               dims={"out": "d_model"})}, outputs=["m"])

    def test_the_gate_reads_the_positions_the_dimmap_merges(self):
        """`registry_violations` probes the DimMap at every rank against
        the declared positions, so a drift between the two is caught."""
        from backend.arch.layout import axis_use
        from backend.arch.ops import OP_REGISTRY
        spec = OP_REGISTRY["view_merge"]
        assert axis_use._position_violations(
            "view_merge", spec, axis_use.GEOMETRY_POSITIONS["view_merge"]) == []
        assert axis_use._position_violations("view_merge", spec, (-3, -1))


class TestOutputsAreDeclared:
    def test_a_graph_without_outputs_is_refused(self):
        with pytest.raises(ValidationError, match="declares no `outputs:`"):
            BlockGraph(inputs=["x"], graph={
                "n": GraphNode(op="rmsnorm", inputs=["x"], dim="d_model")},
                outputs=[])

    def test_a_slice_that_states_no_window_is_refused(self):
        with pytest.raises(ValidationError, match=r"dims: \{ out: <kept width> \}"):
            BlockGraph(inputs=["x"], graph={
                "s": GraphNode(op="slice", inputs=["x"])}, outputs=["s"])


if __name__ == "__main__":
    unittest.main()
