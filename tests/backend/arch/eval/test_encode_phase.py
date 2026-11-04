"""An encoder is its own phase: what it hands the language model is what the
framework's processor and merger produce, and nothing of the language
model's step reaches it.

The formulas below are the frameworks' (each cited where it is asserted);
the engine reads none of them — it walks the encoder over the media's axes
through the `patchify` the spec declares, and these tests hold the outcome
to the source.
"""
from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT))

from backend.arch.eval.dim import value_of
from backend.arch.eval.program import Program
from backend.arch.eval.transforms import Deployment
from tests.snapshots import resolve_snapshot, shared_program, shared_walk

pytestmark = pytest.mark.regression  # resolves snapshots with encoders


def _program(model_id: str) -> Program:
    return shared_program(model_id)


def _handed(model_id: str, media: dict) -> dict:
    """Rows per request each defining edge hands over, by domain."""
    deployment = Deployment(_program(model_id)).at(batch=1, media=media)
    return deployment.media_tokens()


def _step_numbers(analysis) -> tuple:
    """A step's FLOPs, its activation reads and writes, and its lengths."""
    from backend.arch.eval import traffic
    leaves = traffic.derive(analysis).leaves()
    return (value_of(analysis.total().flops),
            sum(value_of(leaves[n].act_reads) for n in leaves),
            sum(value_of(leaves[n].act_writes) for n in leaves),
            value_of(analysis.binding["s_ctx"]),
            value_of(analysis.binding["s_kv"]))


def _image(h: int, w: int, n: int = 1) -> dict:
    return {"pixel_values": {"n_img": n, "img_h": h, "img_w": w}}


class TestHandoverIsTheFrameworks(unittest.TestCase):
    """Rows one image or clip occupies in the prompt, as the framework
    computes them (wrapper tokens excluded: the tokenizer writes those)."""

    def test_qwen_and_mimo_images_are_their_32_pixel_cells(self):
        # transformers processing_qwen3_vl.py:76-79: grid_thw.prod() //
        # merge_size**2 with 16-pixel patches; vLLM mimo_v2_omni.py:1028.
        for model_id in ("Qwen/Qwen3-VL-235B-A22B-Instruct",
                         "Qwen/Qwen3.5-35B-A3B", "Qwen/Qwen3.8-27B",
                         "Qwen/Qwen3.8-Flash-Next", "XiaomiMiMo/MiMo-V2.5",
                         "XiaomiMiMo/MiMo-V2.6-Pro-RL"):
            for h, w in ((672, 672), (1344, 672)):
                self.assertEqual(
                    _handed(model_id, _image(h, w))["pixel_values"],
                    (h // 32) * (w // 32), (model_id, h, w))

    def test_kimi_images_are_their_28_pixel_cells(self):
        # vLLM models/kimi_k3/common/mm_preprocess.py:66-78 (H/28)(W/28).
        for model_id in ("moonshotai/Kimi-K2.5", "moonshotai/Kimi-K3"):
            for h, w in ((672, 672), (1344, 672)):
                self.assertEqual(
                    _handed(model_id, _image(h, w))["pixel_values"],
                    (h // 28) * (w // 28))

    def test_glm_image_is_its_area_over_784(self):
        # transformers processing_glm5_next.py:63-66.
        for h, w in ((672, 672), (1344, 672)):
            self.assertEqual(
                _handed("zai-org/GLM-5.3-Flash", _image(h, w))["pixel_values"],
                h * w // 784)

    def test_hunyuan_image_has_a_newline_per_row_and_two_ends(self):
        # transformers processing_hunyuan_vl.py:89-96 hh*(ww+1)+2.
        for h, w in ((672, 672), (1344, 672)):
            self.assertEqual(
                _handed("tencent/HunyuanOCR", _image(h, w))["pixel_values"],
                (h // 32) * (w // 32 + 1) + 2)

    def test_gemma_image_is_its_48_pixel_cells(self):
        # transformers image_processing_gemma4_unified.py:71 side_mult 48.
        for h, w in ((672, 672), (816, 768)):
            self.assertEqual(
                _handed("google/gemma-4-12B-it", _image(h, w))["pixel_values"],
                (h // 48) * (w // 48))

    def test_step_image_without_crops_is_its_global_views_169_rows(self):
        # vLLM transformers_utils/processors/step3_vl.py:248-254: 169 rows
        # per 728x728 view; a request states its crops, here none.
        self.assertEqual(
            _handed("stepfun-ai/Step-3.7-Flash",
                    {"pixel_values": {"n_img": 2},
                     "patch_pixel_values": {"n_crops": 0}})["pixel_values"],
            2 * 169)

    def test_mimo_audio_is_its_clip_in_groups_of_four(self):
        # vLLM mimo_audio.py:1303-1337, 1376-1378: ceil(T / group_size).
        for model_id in ("XiaomiMiMo/MiMo-V2.5", "XiaomiMiMo/MiMo-V2.6-Pro-RL"):
            for frames in (125, 128):
                media = {"audio_codes": {"n_aud": 1, "s_audio": frames}}
                self.assertEqual(_handed(model_id, media)["audio_codes"],
                                 math.ceil(frames / 4))

    def test_gemma_audio_is_a_row_per_frame(self):
        # vLLM gemma4_unified.py:418-437: one row per unmasked frame.
        media = {"audio_features": {"n_aud": 2, "s_audio": 750}}
        self.assertEqual(
            _handed("google/gemma-4-12B-it", media)["audio_features"], 1500)


class TestVideos(unittest.TestCase):
    """A video runs the image tower's weights in a call of its own (a group
    tied to the image group), in temporal patches of its sampled frames,
    the last frame repeated to a whole patch."""

    def _video(self, model_id, name, frames, h, w, n=1):
        program = _program(model_id)
        deployment = Deployment(program).at(batch=1, media={name: {
            "n_vid": n, "vid_frames": frames, "vid_h": h, "vid_w": w}})
        return program, deployment

    def test_qwen_mimo_glm_videos_are_their_temporal_patches_cells(self):
        # vLLM v0.30.0 qwen3_vl.py:988-990 round_up(frames, Tp) // Tp and
        # :2363-2365 grid_t*h*w/merge^2; mimo_v2_omni.py:1386-1390;
        # glm4_1v.py:1928-1930. Tp = 2.
        for model_id, name, factor in (
                ("Qwen/Qwen3-VL-235B-A22B-Instruct", "pixel_values_videos", 32),
                ("Qwen/Qwen3.5-35B-A3B", "pixel_values_videos", 32),
                ("Qwen/Qwen3.8-27B", "pixel_values_videos", 32),
                ("Qwen/Qwen3.8-Flash-Next", "pixel_values_videos", 32),
                ("XiaomiMiMo/MiMo-V2.5", "video_pixel_values", 32),
                ("XiaomiMiMo/MiMo-V2.6-Pro-RL", "video_pixel_values", 32),
                ("zai-org/GLM-5.3-Flash", "pixel_values_videos", 28)):
            for frames in (4, 5):
                h, w = 7 * factor, 5 * factor
                _, deployment = self._video(model_id, name, frames,
                                            h, w, n=2)
                self.assertEqual(
                    deployment.media_tokens()[name],
                    2 * math.ceil(frames / 2) * (h // factor) * (w // factor),
                    (model_id, frames))

    def test_gemma_embeds_every_frame_as_an_image(self):
        # transformers 5.17 modeling_gemma4_unified.py:1161-1183: each frame
        # embedded, padded to 70 soft tokens, the padding stripped.
        _program, deployment = self._video(
            "google/gemma-4-12B-it", "pixel_values_videos", 5, 336, 240)
        self.assertEqual(deployment.media_tokens()["pixel_values_videos"],
                         5 * (336 // 48) * (240 // 48))
        (video,) = [e for e in deployment.encoders()
                    if e.phase.stack == "video"]
        rows = shared_walk(video).media_rows()["pixel_values_videos"]
        self.assertEqual((rows.segments, rows.rows, rows.valid_rows),
                         (5, 70, 35))

    def test_the_videos_step_is_its_own_whatever_images_come_along(self):
        # The video encoder walks beside the image tower whose weights it
        # runs, at no image: the request's images are the image step's.
        _program, alone = self._video(
            "Qwen/Qwen3.8-27B", "pixel_values_videos", 4, 224, 224)
        with_images = alone.at(media={
            **alone.workload.media,
            "pixel_values": {"n_img": 2, "img_h": 448, "img_w": 448}})

        def video(deployment):
            (step,) = [e for e in deployment.encoders()
                       if e.phase.stack == "video"]
            return shared_walk(step)

        self.assertEqual(value_of(video(with_images).total().flops),
                         value_of(video(alone).total().flops))

    def test_the_video_tower_holds_no_weights_of_its_own(self):
        from backend.arch.layout.persistent import (
            physical_artifacts, resident_weights)
        _program, deployment = self._video(
            "Qwen/Qwen3.8-27B", "pixel_values_videos", 4, 224, 224)
        image = deployment.at(media={"pixel_values": {
            "n_img": 1, "img_h": 224, "img_w": 224}})
        by = {e.phase.stack: shared_walk(e) for e in image.encoders()}
        video = {e.phase.stack: shared_walk(e)
                 for e in deployment.encoders()}["video"]
        own = video.by_stack()["video"]
        self.assertEqual(value_of(own.params), 0)
        self.assertEqual(resident_weights(
            physical_artifacts(video)).by_stack.get("video", 0), 0)
        # Two temporal patches of the image's grid: twice its tower work,
        # and the image tower it walks beside runs no item.
        self.assertEqual(value_of(own.flops),
                         2 * value_of(by["vision"].total().flops))
        self.assertEqual(value_of(video.total().flops), value_of(own.flops))


class TestItemsOfDifferentSizes(unittest.TestCase):
    """Items of different sizes run packed in one tower call; one walk binds
    one size per input, so a request giving several is refused with the
    repair (state each size as its own request), not priced by a sum."""

    def test_a_request_with_items_of_different_sizes_is_refused(self):
        from backend.arch.eval.refusal import EngineGap
        deployment = Deployment(_program("moonshotai/Kimi-K2.5"))
        mixed = deployment.at(batch=1, media={"pixel_values": [
            _image(672, 672)["pixel_values"],
            _image(1344, 672, n=2)["pixel_values"]]})
        with self.assertRaisesRegex(EngineGap, "items of different sizes"):
            mixed.media_tokens()


class TestStepCrops(unittest.TestCase):
    """Step-3.7's crops run the same tower a second time, with the same
    weights, at 504x504 (vLLM v0.30.0 models/step3_vl.py:678-692): 81 rows
    each, laid before the image's 169 global rows (:692-709)."""

    def _deployment(self, media, batch=1):
        return Deployment(_program("stepfun-ai/Step-3.7-Flash")).at(
            batch=batch, media=media)

    def _media(self, crops, n_img=1):
        return {"pixel_values": {"n_img": n_img},
                "patch_pixel_values": {"n_crops": crops}}

    def test_an_image_hands_over_its_crops_then_its_global_view(self):
        for crops in (0, 1, 4):
            self.assertEqual(
                self._deployment(self._media(crops, n_img=2)).media_tokens(),
                {"pixel_values": 2 * (crops * 81 + 169)})

    def test_the_crops_add_their_own_tower_runs_and_no_weight(self):
        from backend.arch.layout.persistent import (
            physical_artifacts, resident_weights)

        def walk(crops):
            (encode,) = self._deployment(self._media(crops)).encoders()
            return shared_walk(encode)

        none, four = walk(0), walk(4)
        self.assertGreater(value_of(four.total().flops),
                           value_of(none.total().flops))
        self.assertEqual(
            resident_weights(physical_artifacts(four)).resident,
            resident_weights(physical_artifacts(none)).resident)
        # Each crop is one more segment of 36 x 36 = 1296 patch rows.
        rows = four.media_rows()["patch_pixel_values"]
        self.assertEqual((rows.segments, rows.rows), (4, 1296))

    def test_an_image_states_its_crops(self):
        with self.assertRaisesRegex(ValueError, "state them too"):
            self._deployment({"pixel_values": {"n_img": 1}}).media_tokens()


class TestMiMoAudioPadding(unittest.TestCase):
    def test_the_padded_frames_are_looked_up(self):
        """A clip is padded to whole groups of four codec frames, repeating
        its last frame, before the lookup (vLLM mimo_audio.py:1303-1337):
        a 125-frame clip is looked up as 128 frames in every bank — the
        same reads and writes as a 128-frame clip."""
        from backend.arch.eval import traffic

        def lookup(frames):
            program = _program("XiaomiMiMo/MiMo-V2.5")
            analysis = shared_walk(Deployment(program).at(batch=1, media={
                "audio_codes": {"n_aud": 1, "s_audio": frames}}).during(
                "encode", stack="audio"))
            (node,) = [n for n in analysis.program.nodes()
                       if n.name == "speech_embeddings"]
            shape = analysis.shapes.values[node.uid]
            row = traffic.derive(analysis).leaves()[node.uid]
            return ([value_of(x) for x in shape.ins[0]],
                    [value_of(x) for x in shape.out],
                    value_of(row.act_reads), value_of(row.act_writes))
        ids, out, reads, writes = lookup(125)
        self.assertEqual((ids, out), ([1, 128], [1, 128, 1024]))
        self.assertEqual((ids, out, reads, writes), lookup(128))


class TestMiMoSpeechBankSum(unittest.TestCase):
    def test_each_bank_add_reads_its_lookup_and_the_running_sum(self):
        """`_apply_speech_embeddings` (modeling_mimo_v2.py:968-977) zeroes an
        accumulator and runs `out.add_(speech_embeddings[i](codes[:, :, i]))`
        once per bank: each add reads the bank's lookup and the
        accumulator's current value, and writes the accumulator."""
        from backend.arch.eval import traffic

        program = _program("XiaomiMiMo/MiMo-V2.5")
        analysis = shared_walk(Deployment(program).at(batch=1, media={
            "audio_codes": {"n_aud": 1, "s_audio": 128}}).during(
            "encode", stack="audio"))
        (node,) = [n for n in analysis.program.nodes()
                   if n.name == "speech_bank_sum"]
        shape = analysis.shapes.values[node.uid]
        row = traffic.derive(analysis).leaves()[node.uid]
        tensor = math.prod(value_of(x) for x in shape.out) * 2   # bf16
        self.assertEqual([value_of(x) for x in shape.ins[0]],
                         [value_of(x) for x in shape.out])
        self.assertEqual((value_of(row.act_reads), value_of(row.act_writes)),
                         (2 * tensor, tensor))


class TestMiMoMergerNorm(unittest.TestCase):
    def test_ln_q_is_a_layernorm_whose_checkpoint_holds_only_the_weight(self):
        """MiMoVisionPatchMerger builds `self.ln_q = nn.LayerNorm(context_dim,
        eps=1e-6)` (modeling_mimo_v2.py:772, applied at :780), and the
        checkpoint stores `visual.merger.ln_q.weight` with no `.bias`."""
        import json

        from tests.snapshots import snapshot_header, snapshot_path

        for model_id in ("XiaomiMiMo/MiMo-V2.5", "XiaomiMiMo/MiMo-V2.6-Pro-RL"):
            snap = json.loads(snapshot_path(model_id)
                              .read_text())
            self.assertIn("(ln_q): LayerNorm((1280,)", snap["model_str"])
            stored = {g.template for g in snapshot_header(model_id)
                      if g.template.startswith("visual.merger.ln_q.")}
            self.assertEqual(stored, {"visual.merger.ln_q.weight"}, model_id)
            program = _program(model_id)
            (node,) = [n for n in program.nodes()
                       if n.name == "visual.merger.ln_q"]
            self.assertEqual(node.op, "layernorm", model_id)
            deployment = Deployment(program).at(batch=1, media={
                "pixel_values": {"n_img": 1, "img_h": 448, "img_w": 448}})
            spec = {t.tensor.module + t.tensor.suffix
                    for encode in deployment.encoders()
                    for t in shared_walk(encode).tensors()
                    if t.tensor.module == "visual.merger.ln_q"}
            self.assertEqual(spec, stored, model_id)


class TestSlotsAreCountedOnce(unittest.TestCase):
    def test_deepstack_adds_onto_the_images_positions(self):
        """Qwen3-VL's three DeepStack edges land on the main merger's
        positions (vLLM qwen3_vl.py:2863-2902): the images fill their
        positions once."""
        program = _program("Qwen/Qwen3-VL-235B-A22B-Instruct")
        deployment = Deployment(program).at(
            batch=1, input_tokens=100, media=_image(672, 672, n=3))
        self.assertEqual(deployment.media_tokens(), {"pixel_values": 3 * 441})
        handed = deployment.handed_over()
        self.assertEqual([t.rows for t in handed["deepstack_feat"]],
                         [3 * 441] * 3)


class TestGemmaPadding(unittest.TestCase):
    def test_the_embedder_runs_on_the_padded_rows(self):
        """Every image is padded to max_soft_tokens = 280 rows and the
        projection runs on all of them (vLLM gemma4_unified.py:357-381); the
        handover is the image's own rows."""
        program = _program("google/gemma-4-12B-it")
        for h, w in ((672, 672), (816, 768)):
            analysis = shared_walk(Deployment(program).at(batch=1, media=_image(h, w)).during(
                "encode", stack="vision"))
            (proj,) = [n for n in analysis.program.walked_nodes()
                       if n.name == "embed_vision.embedding_projection"]
            shape = analysis.shapes.values[proj.uid]
            self.assertEqual(value_of(shape.out[1]), 280)

    def test_an_image_past_the_padding_is_refused(self):
        with self.assertRaisesRegex(ValueError, "more than the 280"):
            shared_walk(Deployment(_program("google/gemma-4-12B-it")).at(
                batch=1, media=_image(1344, 672)).during(
                "encode", stack="vision"))


class TestTheEncoderIsItsOwnPhase(unittest.TestCase):
    def test_encode_does_not_read_the_language_models_step(self):
        program = _program("moonshotai/Kimi-K2.5")
        media = _image(672, 672)
        costs = set()
        for deployment in (
                Deployment(program).at(batch=1, input_tokens=128, media=media),
                Deployment(program).at(batch=1, input_tokens=8192, media=media),
                Deployment(program).at(batch=1, input_tokens=128, media=media)
                .during("decode", cache_len=65536)):
            (encode,) = deployment.encoders()
            costs.add(value_of(shared_walk(encode).total().flops))
        self.assertEqual(len(costs), 1)

    def test_items_are_separate_segments(self):
        """n images of one size cost n times one: each attends within
        itself (one `cu_seqlens` segment per item)."""
        program = _program("Qwen/Qwen3-VL-235B-A22B-Instruct")

        def flops(n: int) -> int:
            return value_of(shared_walk(Deployment(program).at(
                batch=1, media=_image(672, 672, n=n)).during(
                "encode", stack="visual")).total().flops)
        self.assertEqual(flops(3), 3 * flops(1))

    def test_the_language_model_walks_no_encoder(self):
        program = _program("XiaomiMiMo/MiMo-V2.6-Pro-RL")
        media = {**_image(672, 672),
                 "audio_codes": {"n_aud": 1, "s_audio": 125}}
        for deployment in (
                Deployment(program).at(batch=1, input_tokens=64, media=media),
                Deployment(program).at(batch=1, input_tokens=64, media=media)
                .during("decode")):
            analysis = shared_walk(deployment)
            self.assertFalse(set(analysis.cost.values)
                             & set(program.encoder_uids))
            self.assertEqual(
                set(analysis.by_stack()),
                {s.name for s in program.stacks if not s.is_encoder})

    def test_media_move_no_language_model_number(self):
        """The language model's step is its tokens: the media a request
        carries are the encoders' own accounting, and no prefill or decode
        number moves with them."""
        for model_id in ("moonshotai/Kimi-K2.5", "XiaomiMiMo/MiMo-V2.6-Pro-RL",
                         "google/gemma-4-12B-it"):
            program = _program(model_id)
            media = {name: {axis: 1 if axis == entry.axes[0]
                            else 4 * 48 if entry.kind == "image" else 128
                            for axis in entry.axes
                            if axis not in (entry.fixed or {})}
                     for name, entry in program.model_inputs.items()
                     if entry.kind != "tokens"}
            for phase in ("prefill", "decode"):
                steps = [shared_walk(Deployment(program).at(
                    batch=2, input_tokens=100, output_tokens=8, media=m)
                    .during(phase)) for m in ({}, media)]
                self.assertEqual(*[_step_numbers(a) for a in steps],
                                 (model_id, phase))


class TestTheTowersAreTheirOwnWeights(unittest.TestCase):
    def test_the_language_models_weights_are_its_own_walks(self):
        """The whole checkpoint's composition, less the encoder rows, is the
        language model's walk: its parameters, its resident bytes and its
        stored parameter bytes."""
        from backend.arch.eval import capacity
        from backend.arch.eval.projection import project_model
        from backend.views.composition import ModelComposition
        for model_id in ("moonshotai/Kimi-K2.5", "XiaomiMiMo/MiMo-V2.5"):
            resolved = resolve_snapshot(model_id)
            whole = project_model(resolved, batch=1, input_tokens=8,
                                  checkpoint=True)
            language = project_model(resolved, batch=1, input_tokens=8)
            totals = ModelComposition(
                whole, capacity.weights(whole.analysis)).payload()["totals"]
            lm = totals["language_model"]
            self.assertEqual(lm["params"], language.total_params, model_id)
            self.assertEqual(lm["active_params"],
                             language.total_active_params, model_id)
            self.assertEqual(
                lm["storage_bytes"],
                int(capacity.weights(language.analysis).resident), model_id)
            self.assertEqual(
                whole.analysis.checkpoint_param_bytes(without_encoders=True),
                language.analysis.checkpoint_param_bytes(), model_id)
            self.assertLess(lm["params"], totals["params"], model_id)


class TestModalityEncodersAreDeclared(unittest.TestCase):
    """A modality encoder is a group that reads a media model input and
    names no backbone tensor (`ArchSpec.encoder_groups`). Anything else —
    DeepSeek-V4's compressors and indexers, a draft head, however it is
    named — is the language model's: in its step, its weights and its
    totals."""

    def test_the_encoders_are_the_groups_that_read_media(self):
        from tests.snapshots import SNAPSHOT_DIR, any_snapshot
        any_snapshot()
        for path in sorted(SNAPSHOT_DIR.glob("*.json")):
            resolved = resolve_snapshot(path.stem)
            program = Program.of(resolved)
            media = {name for name, entry in
                     (resolved.spec.model_inputs or {}).items()
                     if entry.kind != "tokens"}
            readers = resolved.spec.model_input_readers() if media else {}
            reading = {group.name for name in media
                       for _layer, group in readers[name] if group is not None}
            self.assertEqual(
                {s.name for s in program.stacks if s.is_encoder}, reading,
                path.stem)

    def test_deepseek_v4_compressors_and_indexers_are_the_language_models(self):
        from backend.arch.eval import capacity
        from backend.arch.eval.projection import project_model
        from backend.views.composition import ModelComposition
        for model_id in ("deepseek-ai/DeepSeek-V4-Flash",
                         "deepseek-ai/DeepSeek-V4-Pro"):
            resolved = resolve_snapshot(model_id)
            program = Program.of(resolved)
            self.assertFalse([s.name for s in program.stacks if s.is_encoder])
            modules = [n for n in program.nodes()
                       if n.op and ("compressor" in n.op or "indexer" in n.op)]
            self.assertTrue(modules, model_id)
            step = shared_walk(Deployment(program).at(batch=1, input_tokens=64))
            self.assertFalse([n.display_path for n in modules
                              if n.uid not in step.cost.values], model_id)
            whole = project_model(resolved, batch=1, input_tokens=8,
                                  checkpoint=True)
            totals = ModelComposition(
                whole, capacity.weights(whole.analysis)).payload()["totals"]
            self.assertEqual(totals["language_model"]["params"],
                             totals["params"], model_id)

    def test_deepseek_v41_has_one_encoder_its_vision_tower(self):
        """V4.1-Flash: the vision tower reads `pixel_values` and is the one
        encoder; the DSpark draft head and every compressor / CSA module run
        in the language model's step. Its image fills
        (H/42)(W/42) + H/42 + 2 positions (image_processor.py:3-6)."""
        from tests.backend.arch.spec_tree.test_inner_stream_reads import _v41
        resolved, _text = _v41()
        program = Program.of(resolved)
        self.assertEqual({s.name: s.is_encoder for s in program.stacks},
                         {"": False, "vision": True, "mtp": False})
        deployment = Deployment(program).at(batch=1, input_tokens=64, media={
            "pixel_values": {"n_img": 1, "img_h": 672, "img_w": 504}})
        self.assertEqual(deployment.media_tokens(),
                         {"pixel_values": 16 * 12 + 16 + 2})
        modules = [n for n in program.nodes() if n.op and any(
            key in n.op for key in ("compressor", "csa", "dspark"))]
        self.assertTrue(modules)
        step = shared_walk(deployment)
        self.assertFalse([n.display_path for n in modules
                          if n.uid not in step.cost.values])

    def test_a_group_named_encoder_that_reads_the_backbone_is_not_one(self):
        """The name says nothing: a head named `vision_encoder` that reads
        the backbone's hidden state is the language model's."""
        from tests.backend.arch.eval.test_model_inputs import _ENCODER, _load
        resolved = _load(_ENCODER + """
  - name: vision_encoder
    group: true
    aux_kind: mtp
    inputs: [layers]
    layers:
      - { name: vision_encoder.proj, op: linear, inputs: [layers],
          dims: { in: d_model, out: d_model },
          weight_axes: [{ replicated: true }, { replicated: true }] }
""")
        program = Program.of(resolved)
        self.assertEqual({s.name: s.is_encoder for s in program.stacks
                          if s.is_aux},
                         {"vis": True, "vision_encoder": False})

    def test_a_tower_that_also_reads_the_backbone_is_refused(self):
        """A group that reads its media and, through a nested layer, a
        tensor the language model produces is neither an encoder nor a
        head: refused at load, naming the read."""
        from tests.backend.arch.eval.test_model_inputs import _ENCODER, _load
        with self.assertRaisesRegex(Exception, r"its layer 'vis\.patch' reads "
                                    r"'embed', which the language model"):
            _load(_ENCODER.replace(
                "      - { name: vis.patch, op: linear, inputs: [vis.patchify],",
                "      - { name: vis.patch, op: add, inputs: [vis.patchify, embed],"))

    def test_a_name_an_encoder_layer_shares_resolves_where_the_walk_does(self):
        """A layer of the tower reading `foo`, with a language-model layer
        `foo` before the tower and a tower layer `foo` after the reader:
        the read resolves to the language model's (`layers_named`, earlier
        siblings then earlier top-level layers) and is refused."""
        from tests.backend.arch.eval.test_model_inputs import _ENCODER, _load
        text = _ENCODER.replace(
            "  - name: vis\n",
            "  - { name: foo, op: identity, inputs: [embed], dim: d_model }\n"
            "  - name: vis\n").replace(
            "      - { name: vis.patch, op: linear, inputs: [vis.patchify],",
            "      - { name: vis.mix, op: add, inputs: [vis.patchify, foo] }\n"
            "      - { name: foo, op: identity, inputs: [vis.mix], dim: d_model }\n"
            "      - { name: vis.patch, op: linear, inputs: [foo],")
        with self.assertRaisesRegex(Exception, r"its layer 'vis\.mix' reads "
                                    r"'foo', which the language model"):
            _load(text)

    def test_an_encoder_stage_reading_a_language_model_stream_is_refused(self):
        """A stage of the tower whose `reads_from` names a layer of the
        language model's stack reads a stream the language model writes:
        refused at load, naming the read (`layer_position`)."""
        text = """
match: { model_type: [_model_inputs] }
dims: { d_model: config.hidden_size, n_layers: "2", n_vocab: "16",
        n_vis_channels: "3", s_patch: "2" }
blocks:
  w:
    stream_out: { s: n }
    graph:
      n: { op: rmsnorm, in: x, dim: d_model, weight_axes: [{ replicated: true }] }
    outputs: [n]
  r:
    reads: [s]
    graph:
      g: { op: mul, in: [x, s], dim: d_model }
    outputs: [g]
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
    checkpoint: { prefix: vis.stage, index: local }
    layers:
      - { name: vis.patchify, op: patchify, inputs: [pixel_values],
          dims: { patch: s_patch, temporal: "1", channels: n_vis_channels, align: "2 * s_patch" },
          segment: item, source_anchor: "a test processor" }
      - { name: vis.patch, op: linear, inputs: [vis.patchify],
          dims: { in: "n_vis_channels * s_patch * s_patch", out: d_model },
          weight_axes: [{ replicated: true }, { replicated: true }] }
      - { name: vis.stage, template: r, repeat: n_layers, inputs: [vis.patch],
          streams: { hidden: { carries: true }, s: {} },
          reads_from: { s: { stack: layers, layer: "0" } } }
  - { name: merge, op: scatter_rows_replace, inputs: [embed, vis.stage],
      prompt_slots: { domain: pixel_values, defines: true, source_anchor: "a test merge" } }
  - name: layers
    repeat: n_layers
    inputs: [merge]
    streams: { hidden: { carries: true }, s: {} }
    per_layer:
      assign: [{ type: w, rest: true }]
      map: { w: { template: w } }
  - { name: head, op: linear, boundary_role: output_head, inputs: [layers],
      dims: { in: d_model, out: n_vocab },
      weight_axes: [{ replicated: true }, { replicated: true }] }
"""
        from tests.backend.arch.eval.test_model_inputs import _load
        with self.assertRaisesRegex(Exception, r"its layer 'vis\.stage' reads "
                                    r"stream 's' from layer 0 of 'layers'"):
            _load(text)

    def test_a_vision_group_naming_the_backbone_is_a_head(self):
        """`aux_kind` names what a stack is for and decides nothing: a group
        that names the backbone tensor it reads is the language model's
        head, whatever its kind."""
        from tests.backend.arch.eval.test_model_inputs import _ENCODER, _load
        resolved = _load(_ENCODER + """
  - name: side
    group: true
    aux_kind: vision
    inputs: [layers]
    layers:
      - { name: side.proj, op: linear, inputs: [layers],
          dims: { in: d_model, out: d_model },
          weight_axes: [{ replicated: true }, { replicated: true }] }
""")
        self.assertEqual({s.name: s.is_encoder for s in
                          Program.of(resolved).stacks if s.is_aux},
                         {"vis": True, "side": False})

    def test_a_tower_reading_no_media_input_is_refused(self):
        """An auxiliary group that names no `inputs:` and reads no media
        input is neither an encoder nor attached to the backbone."""
        from tests.backend.arch.eval.test_model_inputs import _ENCODER, _load
        with self.assertRaisesRegex(Exception, r"group\(s\) \['vis'\] name no "
                                    r"`inputs:` and read no media input"):
            _load(_ENCODER.replace(
                "  pixel_values: { kind: image, axes: [n_img, img_h, img_w] }\n",
                "").replace(
                "      - { name: vis.patchify, op: patchify, inputs: [pixel_values],\n"
                "          dims: { patch: s_patch, temporal: \"1\", channels: n_vis_channels, align: \"2 * s_patch\" },\n"
                "          segment: item, source_anchor: \"a test processor\" }\n",
                "").replace("inputs: [vis.patchify],", "").replace(
                '  - { name: merge, op: scatter_rows_replace, inputs: [embed, vis.patch],\n'
                '      prompt_slots: { domain: pixel_values, defines: true, '
                'source_anchor: "a test merge" } }\n', '').replace(
                "inputs: [merge],", "inputs: [embed],"))


class TestEveryEncoderIsBridged(unittest.TestCase):
    def test_each_encoder_stack_defines_prompt_positions(self):
        from tests.snapshots import SNAPSHOT_DIR, any_snapshot
        any_snapshot()
        for path in sorted(SNAPSHOT_DIR.glob("*.json")):
            program = shared_program(path.stem)
            encoders = {s.name for s in program.stacks if s.is_encoder}
            defining = {h.stack for h in program.handoffs if h.defines}
            self.assertEqual(encoders - defining, set(), path.stem)


if __name__ == "__main__":
    unittest.main()
