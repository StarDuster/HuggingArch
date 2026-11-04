"""model_info keeps one forward per distinct module class in the layers.

It used to classify a layer's children by class-name substring (`attn`,
`moe`, `mlp`): the first attention class won (`setdefault`), and a mixer
whose name said none of those (Qwen3-Next's GatedDeltaNet, Nemotron-H's
Mamba2 mixer) was dropped, so the "gold standard" forward excerpts of a
hybrid model left out half its layers.
"""
from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
nn = torch.nn

from backend.hf.model_info import _extract_source_info


class FullAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.q_proj = nn.Linear(4, 4)

    def forward(self, x):
        return self.q_proj(x)


class GatedDeltaNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.in_proj = nn.Linear(4, 4)

    def forward(self, x):
        return self.in_proj(x)


class Router(nn.Module):
    def forward(self, x):
        return x


class SparseBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.gate = Router()
        self.experts = nn.ModuleList([nn.Linear(4, 4) for _ in range(2)])

    def forward(self, x):
        return self.gate(x)


class Layer(nn.Module):
    def __init__(self, mixer):
        super().__init__()
        self.mixer = mixer
        self.mlp = SparseBlock()

    def forward(self, x):
        return self.mlp(self.mixer(x))


class Inner(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList(
            [Layer(GatedDeltaNet()), Layer(GatedDeltaNet()), Layer(FullAttention())])


class ForCausalLM(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = Inner()

    def forward(self, x):
        return x


def test_every_distinct_layer_class_gets_its_forward_by_name():
    forwards = _extract_source_info(ForCausalLM())["forward_methods"]
    assert set(forwards) == {"Model", "Layer", "GatedDeltaNet", "FullAttention",
                             "SparseBlock", "Router"}
    assert "self.in_proj(x)" in forwards["GatedDeltaNet"]
    assert "self.q_proj(x)" in forwards["FullAttention"]
