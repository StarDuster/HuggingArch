"""A raw image's size, as the model's processor resizes it.

A request's media are stated at the size the processor hands the encoder
(`Workload.media`, the processed sides `patchify` reads). This module is the
step before that: a model's HF repo ships its processor's configuration
(`preprocessor_config.json` / `processor_config.json`, kept verbatim in the
snapshot's `processor`), which names the processor class and its parameters;
each class resizes a raw image by its own rule, written here once with the
source it is copied from. A configuration that names no class this module
knows, or a model whose repo ships none, is refused with the reason — the
processed size is then the caller's to state, never inferred.

Nothing here reaches a walk: the result is a processed size, which the
workload then states.
"""
from __future__ import annotations

import math
from typing import Any, ClassVar, Dict, Mapping, Optional, Tuple


class ProcessorRefusal(ValueError):
    """The raw size cannot be converted: say why and what to state."""


class ImageResize:
    """One processor class's resize of a raw image, `(H, W)` in pixels, to
    the processed sides the encoder receives. Each subclass is one rule,
    copied from its `source`, answering for the processor classes a
    configuration names it by (`declared_as`); `of` picks the one a
    configuration declares."""

    declared_as: Tuple[str, ...] = ()
    source: str = ""
    _by_name: ClassVar[Dict[str, type]] = {}

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        for name in cls.declared_as:
            ImageResize._by_name[name] = cls

    def __init__(self, processor: str, params: Mapping[str, Any],
                 repo: Optional[str] = None) -> None:
        self.processor = processor
        self.params = dict(params)
        self.repo = repo

    @property
    def cited(self) -> str:
        """Where the rule is copied from."""
        return self.source

    @classmethod
    def read(cls, config: Mapping[str, Any]) -> Dict[str, Any]:
        raise NotImplementedError

    def resize(self, height: int, width: int) -> Tuple[int, int]:
        raise NotImplementedError

    @classmethod
    def of(cls, files: Optional[Mapping[str, Any]],
           repo: Optional[str] = None) -> "ImageResize":
        """The image resize the model's processor configuration declares
        (the snapshot's `processor`); `repo` names whose configuration it is,
        for a rule whose source is the repo's own code."""
        if not files:
            raise ProcessorRefusal(
                "the model's repo ships no processor configuration; state "
                "the processed size")
        name, config = _image_config(files)
        rule = cls._by_name.get(name)
        if rule is None:
            raise ProcessorRefusal(
                f"the repo's processor class {name!r} has no resize rule "
                f"here; state the processed size (rules: "
                f"{sorted(cls._by_name)})")
        return rule(name, rule.read(config), repo=repo)

    def processed(self, height: int, width: int) -> Tuple[int, int]:
        if height <= 0 or width <= 0:
            raise ProcessorRefusal(
                f"a raw image is a positive height and width, got "
                f"{height}x{width}")
        return self.resize(int(height), int(width))

    def payload(self) -> Dict[str, Any]:
        return {"processor": self.processor, "params": dict(self.params),
                "source": self.cited}


def _pixels(config: Mapping[str, Any], key: str, size_key: str) -> int:
    """`min_pixels` / `max_pixels`, which override the `size` dict's edge
    when given (transformers 5.17 qwen2_vl/image_processing_qwen2_vl.py:
    108-118; hunyuan_vl/image_processing_hunyuan_vl.py:108-114)."""
    if config.get(key) is not None:
        return int(config[key])
    return int(config["size"][size_key])


class Qwen2VLSmartResize(ImageResize):
    """Python `round` to the factor, then floor / ceil into the pixel
    budget; an aspect ratio above 200 is refused."""

    declared_as = ("Qwen2VLImageProcessor", "Qwen2VLImageProcessorFast")
    source = "transformers 5.17 qwen2_vl/image_processing_qwen2_vl.py:62-88"

    @classmethod
    def read(cls, config: Mapping[str, Any]) -> Dict[str, Any]:
        return {"factor": int(config["patch_size"]) * int(config["merge_size"]),
                "min_pixels": _pixels(config, "min_pixels", "shortest_edge"),
                "max_pixels": _pixels(config, "max_pixels", "longest_edge")}

    def resize(self, height: int, width: int) -> Tuple[int, int]:
        p = self.params
        factor, min_px, max_px = p["factor"], p["min_pixels"], p["max_pixels"]
        if max(height, width) / min(height, width) > 200:
            raise ProcessorRefusal(
                f"the processor refuses an aspect ratio above 200, got "
                f"{max(height, width) / min(height, width):.1f}")
        h = round(height / factor) * factor
        w = round(width / factor) * factor
        if h * w > max_px:
            beta = math.sqrt((height * width) / max_px)
            h = max(factor, math.floor(height / beta / factor) * factor)
            w = max(factor, math.floor(width / beta / factor) * factor)
        elif h * w < min_px:
            beta = math.sqrt(min_px / (height * width))
            h = math.ceil(height * beta / factor) * factor
            w = math.ceil(width * beta / factor) * factor
        return h, w


class HunYuanVLSmartResize(Qwen2VLSmartResize):
    """The same rule; `max_pixels` overrides `size.longest_edge`."""

    declared_as = ("HunYuanVLImageProcessor",)
    source = "transformers 5.17 hunyuan_vl/image_processing_hunyuan_vl.py:57-84, 108-114"


class Glm5NextCanvas(ImageResize):
    """The canvas `smart_resize` fits the image into at `num_frames =
    temporal_patch_size` (ceil to the factor, into the token budget by
    bisection), which `resize` pads the resized image to — the size the
    tower reads."""

    declared_as = ("Glm5NextImageProcessor",)
    source = ("transformers 5.17 glm5_next/image_processing_glm5_next.py:"
              "59-120, 148-183, 250")

    @classmethod
    def read(cls, config: Mapping[str, Any]) -> Dict[str, Any]:
        return {"factor": int(config["patch_size"]) * int(config["merge_size"])
                * int(config.get("patch_expand_factor", 1)),
                "temporal_patch_size": int(config["temporal_patch_size"]),
                "min_image_tokens": int(config["min_image_tokens"]),
                "max_image_tokens": int(config["max_image_tokens"])}

    def resize(self, height: int, width: int) -> Tuple[int, int]:
        p = self.params
        factor, frames = p["factor"], p["temporal_patch_size"]
        per_token = frames * factor ** 2
        min_px = p["min_image_tokens"] * per_token
        max_px = p["max_image_tokens"] * per_token

        def align(value: float) -> int:
            return math.ceil(value / factor) * factor

        h, w = align(height), align(width)
        if frames * h * w < min_px:
            scale = math.sqrt(min_px / (frames * height * width))
            h = align(max(1, math.ceil(height * scale)))
            w = align(max(1, math.ceil(width * scale)))
        if frames * h * w > max_px:
            if max_px < frames * factor ** 2:
                raise ProcessorRefusal(
                    f"max_image_tokens={p['max_image_tokens']} is below one "
                    f"aligned patch")
            low, high = 1, height
            h, w = factor, factor
            while low <= high:
                content_h = (low + high) // 2
                content_w = max(1, math.floor(width * content_h / height))
                if frames * align(content_h) * align(content_w) <= max_px:
                    h, w = align(content_h), align(content_w)
                    low = content_h + 1
                else:
                    high = content_h - 1
        return h, w


class MoonViTNavitResize(ImageResize):
    """Never upscaled, shrunk into the patch budget, then padded right and
    bottom to a whole merge window — the padded size is what the tower
    reads."""

    declared_as = ("media_proc.processors.moonvit.MoonViTMediaProcessorConfig",)
    source = "transformers 5.17 kimi_k25/image_processing_kimi_k25.py:50-82"

    @property
    def cited(self) -> str:
        """The repo's own `media_utils.py navit_resize_image` (Kimi-K2.5's
        :85-134, Kimi-K3's :109-156: the same code), then transformers'."""
        own = (f"{self.repo} repo media_utils.py navit_resize_image; "
               if self.repo else "")
        return own + self.source

    @classmethod
    def read(cls, config: Mapping[str, Any]) -> Dict[str, Any]:
        return {key: int(config[key]) for key in (
            "patch_size", "merge_kernel_size", "in_patch_limit",
            "patch_limit_on_one_side")}

    def resize(self, height: int, width: int) -> Tuple[int, int]:
        p = self.params
        patch, limit = p["patch_size"], p["in_patch_limit"]
        side = p["patch_limit_on_one_side"]
        factor = p["merge_kernel_size"] * patch
        patches = max(1, height // patch) * max(1, width // patch)
        scale = min(1.0, math.sqrt(limit / patches),
                    side * patch / height, side * patch / width)
        h = min(max(1, int(height * scale)), side * patch)
        w = min(max(1, int(width * scale)), side * patch)
        return (h + (factor - h % factor) % factor,
                w + (factor - w % factor) % factor)


class Gemma4AspectPreserving(ImageResize):
    """The largest sides, multiples of `pooling_kernel_size * patch_size`,
    within `max_soft_tokens * pooling_kernel_size**2` patches (upscaling a
    small image)."""

    declared_as = ("Gemma4UnifiedImageProcessor",)
    source = ("transformers 5.17 gemma4_unified/image_processing_gemma4_"
              "unified.py:54-105, 298")

    @classmethod
    def read(cls, config: Mapping[str, Any]) -> Dict[str, Any]:
        return {key: int(config[key]) for key in (
            "patch_size", "pooling_kernel_size", "max_soft_tokens")}

    def resize(self, height: int, width: int) -> Tuple[int, int]:
        p = self.params
        patch, kernel = p["patch_size"], p["pooling_kernel_size"]
        max_patches = p["max_soft_tokens"] * kernel ** 2
        target_px = max_patches * patch ** 2
        factor = math.sqrt(target_px / (height * width))
        unit = kernel * patch
        h = math.floor(factor * height / unit) * unit
        w = math.floor(factor * width / unit) * unit
        if h == 0 and w == 0:
            raise ProcessorRefusal(
                f"the processor resizes {height}x{width} to 0x0")
        longest = (max_patches // kernel ** 2) * unit
        if h == 0:
            h, w = unit, min(math.floor(width / height) * unit, longest)
        elif w == 0:
            w, h = unit, min(math.floor(height / width) * unit, longest)
        if h * w > target_px:
            raise ProcessorRefusal(
                f"the processor refuses {height}x{width}: {h}x{w} exceeds "
                f"{max_patches} patches")
        return h, w


def _image_config(files: Mapping[str, Any]) -> Tuple[str, Mapping[str, Any]]:
    """The image processor's class and configuration, as the repo's files
    declare them: `preprocessor_config.json` names it at its root (or, for a
    MoonViT, in `media_proc_cfg.config_type`); a `processor_config.json`
    nests it under `image_processor`."""
    pre = files.get("preprocessor_config.json")
    if pre is not None:
        if "media_proc_cfg" in pre:
            cfg = pre["media_proc_cfg"]
            return str(cfg.get("config_type")), cfg
        return str(pre.get("image_processor_type")), pre
    proc = files.get("processor_config.json")
    if proc is not None and "image_processor" in proc:
        cfg = proc["image_processor"]
        return str(cfg.get("image_processor_type")), cfg
    raise ProcessorRefusal(
        "the model's repo ships no image processor configuration "
        "(preprocessor_config.json / processor_config.json); state the "
        "processed size")


def processed_media(program: Any, files: Optional[Mapping[str, Any]],
                    raw: Mapping[str, Mapping[str, int]],
                    repo: Optional[str] = None,
                    ) -> Tuple[Dict[str, Dict[str, int]], Dict[str, Any]]:
    """`Workload.media` for raw images: per image input `raw` names, its
    items as given and its sides — the input's own side axes, in order
    (`ModelInputSpec.axes`) — as the processor resizes `{height, width}`.
    A raw size is converted only by the rule of the processor the model's
    repo declares: a model whose repo declares none (Step-3.7) is refused,
    even where the config fixes the sides — what else the processor makes
    of the raw size (Step's crops) is unknown without it. Returns the media
    and, per input, the rule that resized it. Only an image is converted
    here; any other input is stated processed."""
    media: Dict[str, Dict[str, int]] = {}
    rules: Dict[str, Any] = {}
    inputs = program.model_inputs
    resize: Optional[ImageResize] = None
    for name, given in raw.items():
        entry = inputs.get(name)
        if entry is None or entry.kind == "tokens":
            raise ProcessorRefusal(
                f"{name!r} is no media input of this model; its media inputs "
                f"are {sorted(n for n, e in inputs.items() if e.kind != 'tokens')}")
        if entry.kind != "image":
            raise ProcessorRefusal(
                f"{name!r} is {entry.kind} input; only an image's raw size "
                f"is converted — state it processed")
        items, *sides = entry.axes
        unknown = sorted(set(given) - {items, "height", "width"})
        if unknown or items not in given:
            raise ProcessorRefusal(
                f"raw[{name!r}] gives {sorted(given)}; give {items!r} and the "
                f"raw image's 'height' and 'width' in pixels")
        out = {items: int(given[items])}
        if "height" not in given or "width" not in given:
            raise ProcessorRefusal(
                f"raw[{name!r}] needs the raw 'height' and 'width'")
        if resize is None:
            resize = ImageResize.of(files, repo=repo)
        free = [axis for axis in sides if axis not in entry.fixed]
        if free:
            height, width = resize.processed(int(given["height"]),
                                             int(given["width"]))
            out.update({axis: value for axis, value
                        in zip(sides, (height, width), strict=True)
                        if axis not in entry.fixed})
            rules[name] = resize.payload()
        media[name] = out
    return media, rules
