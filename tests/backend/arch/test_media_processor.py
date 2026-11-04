"""A raw image's processed size, as each model's own processor resizes it
(`arch.media_processor`), read off the processor configuration its HF repo
ships (the snapshot's `processor`).

The expected sizes are the processors' formulas worked by hand at the
repo's parameters (each cited where it is asserted); a processed side is a
whole number of the resize factor the spec's `patchify` declares, which ties
the processor to the spec."""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from backend.api import app

from backend.arch.eval.dim import value_of
from backend.arch.media_processor import ImageResize, ProcessorRefusal
from backend.arch.snapshots import load_processor
from tests.snapshots import shared_program, snapshot_config

pytestmark = pytest.mark.regression  # resolves snapshots with encoders

#: Raw sizes (height, width) the tests resize: a landscape photo, a small
#: square, a large portrait.
RAW = ((1080, 1920), (224, 224), (4000, 3000))


def _resize(model_id: str) -> ImageResize:
    snapshot_config(model_id)  # skip when the snapshot is not checked out
    return ImageResize.of(load_processor(model_id))


class TestEachProcessorsRule:
    def test_qwen3_vl_rounds_then_meets_its_pixel_floor(self):
        # qwen2_vl/image_processing_qwen2_vl.py:62-88, factor 16 * 2,
        # min 65536 px: 1080 rounds to 34 * 32; 224^2 is under the floor
        # and grows by sqrt(65536 / 224^2) to 8 * 32.
        resize = _resize("Qwen/Qwen3-VL-235B-A22B-Instruct")
        assert resize.processed(1080, 1920) == (1088, 1920)
        assert resize.processed(224, 224) == (256, 256)

    def test_mimo_reads_its_min_pixels_over_the_default_size(self):
        # MiMo's min_pixels (3136) overrides size.shortest_edge (:108-118):
        # 224^2 needs no growth.
        assert _resize("XiaomiMiMo/MiMo-V2.5").processed(224, 224) == (224, 224)

    def test_hunyuan_reads_its_max_pixels_over_size(self):
        # hunyuan_vl/image_processing_hunyuan_vl.py:108-114: max_pixels
        # (16777216) overrides size.longest_edge (4194304); min 262144 px.
        resize = _resize("tencent/HunyuanOCR")
        assert resize.params["max_pixels"] == 16777216
        assert resize.processed(224, 224) == (512, 512)

    def test_kimi_pads_to_a_merge_window_without_resizing(self):
        # Kimi media_utils navit_resize_image: 77 * 137 patches is within
        # 16384, so no resize; each side padded up to a multiple of 28.
        assert _resize("moonshotai/Kimi-K2.5").processed(1080, 1920) == (1092, 1932)

    def test_gemma_fits_its_soft_token_budget(self):
        # gemma4_unified/image_processing_gemma4_unified.py:54-105:
        # sqrt(280 * 9 * 16^2 / (1080 * 1920)) scales the sides, floored to
        # multiples of 48: 12 * 48 and 22 * 48.
        assert _resize("google/gemma-4-12B-it").processed(1080, 1920) == (576, 1056)

    def test_glm_ceils_its_canvas_to_the_factor(self):
        # glm5_next/image_processing_glm5_next.py:59-120: ceil to 28, and
        # 2 * 1092 * 1932 is within 8000 tokens of 2 * 28^2 px.
        assert _resize("zai-org/GLM-5.3-Flash").processed(1080, 1920) == (1092, 1932)

    def test_a_repo_with_no_processor_configuration_is_refused(self):
        snapshot_config("stepfun-ai/Step-3.7-Flash")
        with pytest.raises(ProcessorRefusal, match="ships no processor"):
            ImageResize.of(load_processor("stepfun-ai/Step-3.7-Flash"))

    def test_an_undeclared_processor_class_is_refused(self):
        with pytest.raises(ProcessorRefusal, match="no resize rule"):
            ImageResize.of({"preprocessor_config.json": {
                "image_processor_type": "SomeOtherImageProcessor"}})


ENCODER_IMAGES = (
    "Qwen/Qwen3-VL-235B-A22B-Instruct", "Qwen/Qwen3.5-35B-A3B",
    "Qwen/Qwen3.8-27B", "Qwen/Qwen3.8-Flash-Next", "XiaomiMiMo/MiMo-V2.5",
    "XiaomiMiMo/MiMo-V2.6-Pro-RL", "google/gemma-4-12B-it",
    "moonshotai/Kimi-K2.5", "moonshotai/Kimi-K3", "tencent/HunyuanOCR",
    "zai-org/GLM-5.3-Flash")


@pytest.mark.parametrize("model_id", ENCODER_IMAGES)
def test_a_processed_side_is_a_whole_resize_factor_of_the_spec(model_id):
    """The processor and the spec's `patchify` agree on the resize factor:
    every processed side is a whole number of the `align` it declares."""
    program = shared_program(model_id)
    resize = _resize(model_id)
    for entry in program.model_inputs.values():
        if entry.kind != "image" or entry.unmodeled is not None:
            continue
        least = [int(value_of(entry.least[axis])) for axis in entry.axes[1:]]
        for raw in RAW:
            sides = resize.processed(*raw)
            assert all(side % unit == 0 for side, unit
                       in zip(sides, least, strict=True)), (raw, sides, least)


class TestProcessedMediaEndpoint:
    def test_raw_images_become_the_estimates_media(self):
        snapshot_config("moonshotai/Kimi-K2.5")
        body = TestClient(app).post(
            "/inference/processed-media", json={
                "model_id": "moonshotai/Kimi-K2.5",
                "raw": {"pixel_values": {"n_img": 2, "height": 1080,
                                         "width": 1920}}}).json()
        assert body["media"] == {"pixel_values": {
            "n_img": 2, "img_h": 1092, "img_w": 1932}}
        assert body["rules"]["pixel_values"]["params"]["patch_size"] == 14

    def test_a_raw_size_with_no_processor_rule_is_refused(self):
        # Step-3.7's repo ships no processor configuration: its global view
        # is fixed, but what the processor makes of a raw size (its crops)
        # is not known here, so a raw size is refused with the reason.
        snapshot_config("stepfun-ai/Step-3.7-Flash")
        response = TestClient(app).post(
            "/inference/processed-media", json={
                "model_id": "stepfun-ai/Step-3.7-Flash",
                "raw": {"pixel_values": {"n_img": 1, "height": 1080,
                                         "width": 1920}}})
        assert response.status_code == 400
        assert "ships no processor" in response.json()["detail"]

    def test_kimi_k3_cites_its_own_repo(self):
        snapshot_config("moonshotai/Kimi-K3")
        body = TestClient(app).post(
            "/inference/processed-media", json={
                "model_id": "moonshotai/Kimi-K3",
                "raw": {"pixel_values": {"n_img": 1, "height": 1080,
                                         "width": 1920}}}).json()
        source = body["rules"]["pixel_values"]["source"]
        assert source.startswith("moonshotai/Kimi-K3 repo media_utils.py")
        assert "Kimi-K2.5" not in source
