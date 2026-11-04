"""C6: splitting an encoder over TP never makes a rank hold more of a stack
or of a weight node than TP=1 does."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List

from tools.corpus_pins.check import CheckKind, CorpusCheck
from tools.corpus_pins.corpus import CorpusModel
from tools.corpus_pins.pins.encoder_weight import measure_snapshot


def grew(stem: str, row: Dict[str, Any]) -> List[str]:
    """The stacks and nodes of one measured encoder row that hold more per
    rank at TP=8 than at TP=1."""
    out: List[str] = []
    for stack, one in row["1"].items():
        eight = row["8"][stack]
        if eight["bytes_per_rank"] > one["bytes_per_rank"]:
            out.append(f"{stem}.{stack}: {one['bytes_per_rank']} -> {eight['bytes_per_rank']}")
        for node, bytes_one in one["nodes"].items():
            bytes_eight = eight["nodes"][node]
            if bytes_eight > bytes_one:
                out.append(f"{stem}.{stack}.{node}: {bytes_one} -> {bytes_eight}")
    return out


@dataclass(frozen=True)
class EncoderTpCheck(CorpusCheck):
    def violations(self, model: CorpusModel) -> List[str]:
        row = measure_snapshot(model)
        return grew(model.stem, row) if row else []


ENCODER_TP = EncoderTpCheck(
    id="C6",
    title="encoder TP=8 holds no more than TP=1",
    kind=CheckKind.ADMIT,
    owner_tests=(
        "tests/backend/arch/layout/test_encoder_weight_per_rank_golden.py::"
        "TestEncoderWeightPerRankGolden::test_tp8_never_holds_more_than_tp1",
    ),
)
