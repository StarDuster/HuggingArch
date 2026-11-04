"""The decoder layer a block template is compared against is the one the
model's own construction builds, from its declared top class."""
from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT))

import backend.arch.validate  # noqa: F401  (import order of the package)
from backend.arch.structural import locate_layer_forward
from tests.snapshots import snapshot_config, snapshot_path


def _locate(name):
    snapshot_config(name)
    snap = json.loads(snapshot_path(name).read_text(encoding="utf-8"))
    return locate_layer_forward(snap["forward_source"]["files"], snap["config"],
                                lambda model_type: None)


class TestLayerLocation(unittest.TestCase):
    def test_a_vision_encoder_layer_is_not_the_text_block(self):
        """Kimi-K2.5's repo source holds the vision tower in
        `modeling_kimi_k25.py` and its text tower, DeepseekV3, in the sibling
        `modeling_deepseek.py`. The walk follows `language_model` to the
        text decoder layer, not the vision encoder's."""
        fn, trail, stop = _locate("moonshotai/Kimi-K2.5")
        self.assertIsNone(stop)
        self.assertIsNotNone(fn)
        self.assertIn("DeepseekV3DecoderLayer.forward", trail[-1])
        self.assertIn("modeling_deepseek.py", trail[-1])

    def test_the_text_tower_is_followed_past_the_vision_tower(self):
        """Kimi-K3 builds a vision encoder and a text model side by side;
        the text model is the one built from `config.text_config`."""
        _fn, trail, stop = _locate("moonshotai/Kimi-K3")
        self.assertIsNone(stop)
        self.assertIn("KimiDecoderLayer.forward", trail[-1])

    def test_inherited_construction_outside_the_files_stops(self):
        """MiMo-7B builds on Qwen2's classes from transformers; the layers are
        constructed there, and the MTP list it adds itself is not them."""
        fn, _trail, stop = _locate("XiaomiMiMo/MiMo-7B-RL")
        self.assertIsNone(fn)
        self.assertIn("Qwen2ForCausalLM", stop)


_FACTORY_SOURCE = """
class Top:
    def __init__(self, config):
        self.language_model = AutoModel.from_config(config.text_config)

class RemoteText:
    def __init__(self, config):
        self.layers = nn.ModuleList([RemoteLayer(config) for _ in range(2)])

class RemoteLayer:
    def forward(self, x):
        return x

class LibraryText:
    def __init__(self, config):
        self.layers = nn.ModuleList([LibraryLayer(config) for _ in range(2)])

class LibraryLayer:
    def forward(self, x):
        return x
"""


class TestFactoryConstruction(unittest.TestCase):
    """`AutoModel.from_config(cfg)` builds what `cfg` declares: its own
    `auto_map` entry ahead of the library registry's class for its
    model_type."""

    FILES = ({"path": "modeling_top.py", "text": _FACTORY_SOURCE},)

    def _locate(self, text_config):
        return locate_layer_forward(
            list(self.FILES), {"architectures": ["Top"], "text_config": text_config},
            {"lib_text": "LibraryText"}.get)

    def test_the_sub_configs_auto_map_wins_over_the_registry(self):
        _fn, trail, stop = self._locate(
            {"model_type": "lib_text",
             "auto_map": {"AutoModel": "modeling_top.RemoteText"}})
        self.assertIsNone(stop)
        self.assertIn("RemoteLayer.forward", trail[-1])

    def test_without_an_auto_map_the_registry_names_the_class(self):
        _fn, trail, stop = self._locate({"model_type": "lib_text"})
        self.assertIsNone(stop)
        self.assertIn("LibraryLayer.forward", trail[-1])

    def test_an_auto_map_without_the_factory_stops(self):
        fn, _trail, stop = self._locate(
            {"model_type": "lib_text",
             "auto_map": {"AutoConfig": "configuration_top.TextConfig"}})
        self.assertIsNone(fn)
        self.assertIn("no 'AutoModel' entry", stop)


if __name__ == "__main__":
    unittest.main()
