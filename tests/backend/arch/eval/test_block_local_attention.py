"""Attention confined to fixed groups, both directions inside a group.

MiMo-V2.5's audio encoder runs its local transformer on [T/G, G, D] groups of
`audio_config.group_size` = 4 frames with `is_causal=not input_full_attention`
= False (modeling_mimo_v2.py:538-551, 979-984; vLLM mimo_audio.py:1284-1289,
1304-1335): every frame scores exactly the G frames of its own group. Written
as the windowed ops at `s_window` = G, bidirectional, the work is the
block-diagonal form's, 2·B·H·S·G·d per score — not dense attention over the
sequence.
"""
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT))

from backend.arch.eval.transforms import Deployment
from tests.snapshots import shared_program, shared_walk, snapshot_config

STEM = "XiaomiMiMo__MiMo-V2.5"


def test_the_audio_encoder_scores_each_frame_against_its_group():
    config = snapshot_config(STEM)
    program = shared_program('XiaomiMiMo/MiMo-V2.5')
    tokens = 4096  # one clip of 4096 codec frames, a whole number of groups
    analysis = (shared_walk(Deployment(program)
                .at(batch=1, media={"audio_codes": {"n_aud": 1,
                                                    "s_audio": tokens}})
                .during("encode", stack="audio")))
    audio = config["audio_config"]
    group = audio["group_size"]
    heads = int(analysis.binding["n_audio_heads"])
    head_dim = int(analysis.binding["d_audio_head"])
    scores = [n for n in program.nodes()
              if n.op == "swa_attention_score" and "audio_encoder" in (n.display_path or "")]
    assert scores
    for node in scores:
        assert int(analysis.cost.values[node.uid].flops) == (
            2 * heads * tokens * group * head_dim)
