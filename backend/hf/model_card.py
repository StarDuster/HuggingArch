"""Parameter counts a model card states, as numbers.

A model card says "552B backbone parameters", "activate only 8B parameters
per token during prefill and 16B during decode", "671B total parameters with
37B activated", or carries a table row `# Activated Params | 13B | 49B | 8B`.
This reads those statements as `(kind, count)` pairs, so a check can hold
the analysis's figures against what the authors say.

A count is `active` when the nearest word that says which — "activ…" against
"total" / "backbone" — is the former, and `total` otherwise: "671B total
parameters with 37B activated" has each word beside its own count, and a
bare "196B parameters" states a size. A table row is what its label says.
A card also names its siblings' counts beside its own, and may say nothing:
the pairs are evidence for a warning, never a fact the engine computes
with.
"""
from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass
from typing import List, Optional, Tuple

logger = logging.getLogger(__name__)

#: Every spelling of a count's unit, lower-cased, and its scale.
UNIT_SCALES = {"t": 10 ** 12, "trillion": 10 ** 12, "b": 10 ** 9,
               "billion": 10 ** 9, "m": 10 ** 6, "million": 10 ** 6}
#: A count with its unit: `552B`, `1.6T`, `8 B`, `37 billion`.
_COUNT = re.compile(
    r"(?<![\w.])(\d+(?:\.\d+)?)\s*(" + "|".join(UNIT_SCALES) + r")\b",
    re.IGNORECASE)
#: How far either side of a count its context reaches, in characters.
_WINDOW = 60


@dataclass(frozen=True)
class StatedCount:
    kind: str           # "total" | "active"
    count: int
    #: Half the place of the last digit written: "3B" is 3B ± 0.5B, "3.3B"
    #: 3.3B ± 0.05B — a card rounds to the digits it writes.
    rounding: int
    text: str           # the line it was read from


_ACTIVE = re.compile(r"activ", re.IGNORECASE)
_TOTAL = re.compile(r"total|backbone", re.IGNORECASE)


def _kind(text: str, start: int, end: int) -> str:
    """Which family the nearest word that says so puts the count in."""
    def nearest(pattern: re.Pattern) -> float:
        return min((min(abs(m.start() - end), abs(start - m.end()))
                    for m in pattern.finditer(text)), default=float("inf"))
    return "active" if nearest(_ACTIVE) < nearest(_TOTAL) else "total"


def stated_counts(card: str) -> List[StatedCount]:
    """Every parameter count the card states, in reading order: a count
    whose line — or, in a table, whose row label — speaks of parameters,
    with its kind (`_kind`). A count followed by "tokens" is training data,
    not a parameter count."""
    out: List[StatedCount] = []
    for line in card.splitlines():
        cells = [c.strip() for c in line.split("|")] if "|" in line else None
        label = cells[1] if cells and len(cells) > 2 else None
        for match in _COUNT.finditer(line):
            after = line[match.end():match.end() + 12].lower()
            if "token" in after:
                continue
            context = (label if label and label != match.group(0).strip()
                       else line[max(0, match.start() - _WINDOW):
                                 match.end() + _WINDOW])
            if "param" not in context.lower():
                continue
            kind = (_kind(context, 0, 0) if context is label
                    else _kind(line, match.start(), match.end()))
            unit = match.group(2)
            scale = UNIT_SCALES[unit.lower()]
            digits = match.group(1)
            places = len(digits.split(".")[1]) if "." in digits else 0
            out.append(StatedCount(
                kind, int(float(digits) * scale),
                int(scale * 10 ** -places / 2), line.strip()[:200]))
    return out


def model_card(model_id: str) -> Tuple[Optional[str], Optional[str]]:
    """`(text, None)`, or `(None, why)` when the card cannot be read."""
    # The hub reads HF_HUB_OFFLINE once, at import; a caller that turns the
    # hub off for one run (`validate_offline`) sets it later, so it is asked
    # here, at the call: offline, only a cached card is read.
    offline = os.environ.get("HF_HUB_OFFLINE", "").lower() in ("1", "true", "yes")
    try:
        from huggingface_hub import hf_hub_download
        path = hf_hub_download(repo_id=model_id, filename="README.md",
                               local_files_only=offline)
        with open(path, encoding="utf-8") as f:
            return f.read(), None
    except Exception as exc:  # the hub raises its own error family
        logger.debug("model card for %s unreadable: %s", model_id, exc)
        return None, f"{type(exc).__name__}: {exc}"
