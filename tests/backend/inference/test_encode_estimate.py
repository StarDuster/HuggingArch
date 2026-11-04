"""The estimate's encode phase: each encoder priced over the request's media,
in its own step, and nothing of it in the language model's.

The media below are the tests' own reference inputs (no production path binds
a size): images whose sides are whole numbers of each model's processor
resize factor, and an audio clip.
"""
from __future__ import annotations

from typing import ClassVar

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from backend.api import app
from backend.arch.eval.dim import value_of
from backend.arch.eval.program import Program
from backend.arch.eval.transforms import Deployment
from tests.snapshots import resolve_snapshot, snapshot_config

pytestmark = pytest.mark.regression  # resolves snapshots with encoders

KIMI = "moonshotai/Kimi-K2.5"
GEMMA = "google/gemma-4-12B-it"
STEP = "stepfun-ai/Step-3.7-Flash"
IMAGE = {"pixel_values": {"n_img": 2, "img_h": 672, "img_w": 1344}}
#: Kimi-K2.5's 553 GiB of weights fit no single card, and a card that admits
#: no request publishes no step — the encode step included
#: (`capacity.without_step`). The endpoint checks below therefore deploy a
#: mesh that holds the model; its per-rank encode numbers are the
#: unsharded walk's, so the assertions read as they did at one rank.
KIMI_FITS = dict(num_gpus=16, parallel={"pp": 16})


def _client() -> TestClient:
    return TestClient(app)


def _estimate(model_id: str, **body) -> dict:
    snapshot_config(model_id)  # skip when the snapshot is not checked out
    response = _client().post("/inference/estimate", json={
        "model_id": model_id, "gpu_sku": "H100-SXM",
        "avg_input_tokens": 1024, "avg_output_tokens": 128, **body})
    assert response.status_code == 200, response.text
    return response.json()


class TestEncodeIsItsOwnAccount:
    def test_a_text_model_has_no_encode_phase(self):
        assert _estimate("meta-llama/Llama-3.2-1B")["encode"] is None

    def test_media_move_no_language_model_number(self):
        plain = _estimate(KIMI, **KIMI_FITS)
        with_media = _estimate(KIMI, media=IMAGE, **KIMI_FITS)
        for phase in ("prefill", "decode"):
            for key in ("flops_per_step", "t_sol_s", "mem_read_bytes"):
                assert (with_media["phases"][phase][key]
                        == plain["phases"][phase][key]), (phase, key)
        assert with_media["per_op_timing"] == plain["per_op_timing"]
        assert with_media["memory"] == plain["memory"]

    def test_the_encode_flops_are_the_encode_walks(self):
        # One rank, no fusion change to the FLOPs: the step's rows sum to
        # the encode walk's own total.
        encode = _estimate(KIMI, media=IMAGE, **KIMI_FITS)["encode"]
        (stack,) = encode["stacks"]
        walk = Deployment(Program.of(resolve_snapshot(KIMI))).at(
            batch=1, media=IMAGE).during("encode", stack=stack["stack"])
        assert stack["flops_per_rank"] == pytest.approx(
            float(value_of(walk.analyze().total().flops)))
        assert stack["t_sol"] > 0
        assert stack["t_sol_per_item"] == pytest.approx(stack["t_sol"] / 2)
        # The encoder's per-op table is a rolled-up fold like the language
        # model's, and its Total row is the stack's own step.
        total = stack["op_table_total"]
        assert total["t_sol_s"] == pytest.approx(stack["t_sol"])
        assert total["flops_per_rank"] == pytest.approx(stack["flops_per_rank"])
        assert total["access_bytes_per_rank"] == pytest.approx(
            stack["bytes_per_rank"])
        assert all("t_op_agg" in r for r in stack["flops_by_module"])

    def test_the_handoff_is_the_frameworks(self):
        # vLLM models/kimi_k3/common/mm_preprocess.py:66-78: (H/28)(W/28)
        # positions per image.
        encode = _estimate(KIMI, media=IMAGE, **KIMI_FITS)["encode"]
        (stack,) = encode["stacks"]
        per_image = (672 // 28) * (1344 // 28)
        assert stack["handoff_tokens_per_request"] == {
            "pixel_values": 2 * per_image}
        assert encode["media_tokens"] == {"pixel_values": 2 * per_image}
        rows = stack["media_rows"]["pixel_values"]
        assert rows["segments"] == 2  # one attention segment per image
        assert rows["rows_per_segment"] == (672 // 14) * (1344 // 14)

    def test_the_encode_batch_multiplies_the_items(self):
        one = _estimate(KIMI, media=IMAGE, **KIMI_FITS)["encode"]["stacks"][0]
        four = _estimate(KIMI, media=IMAGE, encode_batch_size=4,
                         **KIMI_FITS)["encode"]
        assert four["batch"] == 4
        assert four["stacks"][0]["items"] == 4 * one["items"]
        assert four["stacks"][0]["flops_per_rank"] == pytest.approx(
            4 * one["flops_per_rank"])


class TestEachEncoderRunsItsOwnMedia:
    def test_gemma_pads_an_image_to_its_soft_tokens(self):
        # transformers image_processing_gemma4_unified.py:352 pads to
        # num_soft_tokens = 280; the item hands over (H/48)(W/48).
        media = {"pixel_values": {"n_img": 1, "img_h": 672, "img_w": 672}}
        encode = _estimate(GEMMA, media=media)["encode"]
        by = {s["stack"]: s for s in encode["stacks"]}
        rows = by["vision"]["media_rows"]["pixel_values"]
        assert rows == {"segments": 1, "rows_per_segment": 280,
                        "valid_rows_per_segment": (672 // 48) ** 2}
        # No clip named: the audio encoder runs no item.
        assert by["audio"]["items"] == 0
        assert by["audio"]["t_sol"] is None
        assert by["audio"]["per_op_timing"] == []
        assert encode["subtotal"]["t_sol"] == pytest.approx(
            by["vision"]["t_sol"])

    def test_an_illegal_side_is_refused_with_the_legal_ones(self):
        snapshot_config(KIMI)
        response = _client().post("/inference/estimate", json={
            "model_id": KIMI, "gpu_sku": "H100-SXM",
            "media": {"pixel_values": {"n_img": 1, "img_h": 670,
                                       "img_w": 672}}})
        assert response.status_code == 400
        assert "644 or 672" in response.json()["detail"]


class TestMediaInputs:
    def test_kimi_sides_are_whole_resize_factors(self):
        snapshot_config(KIMI)
        body = _client().get("/inference/media-inputs",
                             params={"model_id": KIMI}).json()
        (image,) = body["inputs"]
        assert image["name"] == "pixel_values" and image["kind"] == "image"
        axes = {a["name"]: a for a in image["axes"]}
        assert axes["n_img"]["items"]
        # transformers kimi_k25/image_processing_kimi_k25.py:78-80:
        # factor = merge_kernel_size (2) * patch_size (14).
        assert axes["img_h"]["least"] == axes["img_w"]["least"] == 28
        assert body["encoders"] == [{"stack": "vision", "aux_kind": "vision",
                                     "inputs": ["pixel_values"]}]

    def test_step_fixes_its_view_and_states_its_crops_per_image(self):
        snapshot_config(STEP)
        body = _client().get("/inference/media-inputs",
                             params={"model_id": STEP}).json()
        by = {i["name"]: i for i in body["inputs"]}
        axes = {a["name"]: a for a in by["pixel_values"]["axes"]}
        # vLLM step3_vl.py:64-79: the global view is 728 x 728.
        assert axes["img_h"]["fixed"] == axes["img_w"]["fixed"] == 728
        crops = by["patch_pixel_values"]
        assert crops["per_item_of"] == "pixel_values"
        assert crops["unmodeled"] is None
        assert {a["name"]: a["fixed"] for a in crops["axes"]} == {
            "n_crops": None, "crop_h": 504, "crop_w": 504}

    def test_a_text_model_takes_no_media(self):
        body = _client().get(
            "/inference/media-inputs",
            params={"model_id": "meta-llama/Llama-3.2-1B"}).json()
        assert body["inputs"] == [] and body["encoders"] == []


class TestEncodeGraph:
    def test_the_graph_is_the_encoders_alone(self):
        snapshot_config(KIMI)
        response = _client().get("/inference/compute-graph", params={
            "model_id": KIMI, "phase": "encode", "stack": "vision",
            "media": json.dumps({"pixel_values": {
                "n_img": 1, "img_h": 672, "img_w": 672}})})
        assert response.status_code == 200, response.text
        body = response.json()
        assert body["phase"] == "encode" and body["stack"] == "vision"
        ids = json.dumps(body["graph"])
        assert "patchify" in ids
        assert "embed_tokens" not in ids and "lm_head" not in ids

    def test_an_encode_graph_names_its_encoder(self):
        response = _client().get("/inference/compute-graph", params={
            "model_id": KIMI, "phase": "encode"})
        assert response.status_code == 400


class TestInspectorEncoders:
    def test_an_encoders_flops_are_per_item_never_a_step_figure(self):
        snapshot_config(KIMI)
        body = _client().post("/analyze", json={"model_id": KIMI}).json()
        dash = body["dashboard"]
        assert dash["encoder_stacks"] == {"vision": "vision"}
        # The step's aux FLOPs are the language model's aux stacks alone.
        assert "vision" not in dash["aux_flops"]
        formula = dash["encoder_flops_per_item"]["vision"]
        assert formula["variables"] == ["img_h", "img_w"]
        assert formula["inputs"] == ["pixel_values"]


class TestTheEncodersOwnCapacity:
    def test_the_weights_are_the_pinned_per_rank_bytes(self):
        # Media-independent: the encode walk's resident ledger, equal to the
        # per-rank encoder weight pin. Only TP=8 is served here: at TP=1 the
        # weights fit no card, and a card that admits no request publishes no
        # step (`capacity.without_step`); the TP=1 pin is the layout golden's
        # (`test_encoder_weight_per_rank_golden`).
        golden = json.loads(
            (Path(__file__).resolve().parents[3]
             / "tests/golden/encoder_weight_per_rank.json").read_text())
        for tp in (8,):
            parallel = {"tp_attention": tp, "tp_dense_ffn": tp,
                        "tp_moe_expert": tp}
            for media in ({}, IMAGE):
                encode = _estimate(KIMI, num_gpus=tp, parallel=parallel,
                                   media=media)["encode"]
                (stack,) = encode["stacks"]
                assert stack["weight_bytes_per_rank"] == golden[
                    "moonshotai__Kimi-K2.5"][str(tp)]["vision"][
                        "bytes_per_rank"], (tp, media)

    def test_an_encoder_of_model_level_ops_has_their_peak(self):
        # gemma-4's audio encoder is one linear (HF Gemma4 audio projection):
        # its peak holds the clip it reads and the rows it writes, BF16.
        media = {"audio_features": {"n_aud": 1, "s_audio": 750}}
        encode = _estimate(GEMMA, media=media)["encode"]
        by = {s["stack"]: s for s in encode["stacks"]}
        width_in = by["audio"]["per_op_timing"][0]["shape_in"][-1]
        width_out = by["audio"]["per_op_timing"][0]["shape_out"][-1]
        assert by["audio"]["activation_peak"]["bytes_per_rank"] == (
            750 * (width_in + width_out) * 2)
        # No image: the vision encoder runs no step and holds no
        # activation, and still holds its weights.
        assert by["vision"]["activation_peak"] is None
        assert by["vision"]["weight_bytes_per_rank"] > 0

    def test_the_output_cache_is_vllms_budget(self):
        # vLLM v0.30.0 v1/core/encoder_cache_manager.py:293-340 and
        # config/scheduler.py:285: max(the scheduler's
        # max_num_batched_tokens, one item's tokens); Qwen3-VL caches its
        # main merger and three DeepStack levels side by side
        # (qwen3_vl.py:596, 887).
        qwen = "Qwen/Qwen3-VL-235B-A22B-Instruct"
        media = {"pixel_values": {"n_img": 1, "img_h": 1344, "img_w": 672}}
        item = (1344 // 32) * (672 // 32)
        hidden = snapshot_config(qwen)["text_config"]["hidden_size"]
        # 439 GiB of weights fit no single card; pp=8 holds them at tp=1, so
        # the per-rank cache below is the unsharded figure.
        fits = dict(num_gpus=8, parallel={"pp": 8})
        small = _estimate(qwen, media=media, prefill_chunk_size=512,
                          **fits)["encode"]["output_cache"]
        assert small["max_tokens_per_item"] == item
        assert small["tokens"] == item  # one image outgrows the budget
        assert small["bytes_per_token"] == 4 * hidden * 2
        assert small["bytes_per_rank"] == item * 4 * hidden * 2
        large = _estimate(qwen, media=media, prefill_chunk_size=8192,
                          **fits)["encode"]["output_cache"]
        assert large["tokens"] == 8192

    def test_media_move_no_language_model_capacity(self):
        plain = _estimate(KIMI)
        with_media = _estimate(KIMI, media=IMAGE)
        for key in ("memory", "max_batch_per_gpu", "max_tokens_per_gpu",
                    "activation_peak"):
            assert with_media[key] == plain[key], key


class TestItemsOfDifferentSizes:
    def test_a_request_with_items_of_different_sizes_is_refused(self):
        # One walk binds one size per input; the one packed call over
        # several sizes is not expressible yet, so the estimate says so.
        snapshot_config(KIMI)
        response = _client().post("/inference/estimate", json={
            "model_id": KIMI, "gpu_sku": "H100-SXM",
            "media": {"pixel_values": [
                {"n_img": 1, "img_h": 672, "img_w": 672},
                {"n_img": 2, "img_h": 1344, "img_w": 672}]}})
        assert response.status_code == 400
        assert "items of different sizes" in response.json()["detail"]


class TestVideos:
    QWEN = "Qwen/Qwen3.8-27B"
    VIDEO: ClassVar[dict] = {"pixel_values_videos": {"n_vid": 1, "vid_frames": 5,
                                     "vid_h": 448, "vid_w": 448}}

    def test_a_video_runs_the_image_towers_weights_in_its_own_step(self):
        encode = _estimate(self.QWEN, media=self.VIDEO)["encode"]
        by = {s["stack"]: s for s in encode["stacks"]}
        video = by["video"]
        assert video["reruns_weights_of"] == "vision"
        assert video["weight_bytes_per_rank"] == 0
        assert video["items"] == 1
        # 5 frames in pairs, the last repeated: 3 temporal patches of a
        # 14 x 14 merged grid (vLLM v0.30.0 qwen3_vl.py:988-990, 2363-2365).
        assert encode["media_tokens"] == {"pixel_values_videos": 3 * 14 * 14}
        assert video["media_rows"]["pixel_values_videos"]["segments"] == 3
        assert video["flops_per_rank"] > 0
        assert by["vision"]["items"] == 0

    def test_a_video_moves_no_language_model_number(self):
        plain = _estimate(self.QWEN)
        with_video = _estimate(self.QWEN, media=self.VIDEO)
        for phase in ("prefill", "decode"):
            for key in ("flops_per_step", "t_sol_s", "mem_read_bytes"):
                assert (with_video["phases"][phase][key]
                        == plain["phases"][phase][key]), (phase, key)
        assert with_video["memory"] == plain["memory"]
