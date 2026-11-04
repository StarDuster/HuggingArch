"""Layer indices in words — the one phrasing every page uses for which
layers of a stack a block runs at (`block_label` tells two blocks of one
stack apart by it; the architecture view's cycle cards label their spans
with it)."""
from __future__ import annotations

from typing import List, Sequence


def layer_span(indices: Sequence[int]) -> str:
    """`indices` (ascending) as a reader reads them: consecutive runs as
    `a–b`, and five or more runs of one length at one stride as their
    first two and last with the stride — `6, 10, …, 74 (every 4)`,
    `3–5, 7–9, …, 75–77 (every 4)` — the rest listed."""
    runs: List[tuple] = []
    for index in indices:
        if runs and index == runs[-1][1] + 1:
            runs[-1] = (runs[-1][0], index)
        else:
            runs.append((index, index))

    def show(run: tuple) -> str:
        return str(run[0]) if run[0] == run[1] else f"{run[0]}–{run[1]}"

    parts: List[str] = []
    i = 0
    while i < len(runs):
        j = i
        if i + 1 < len(runs):
            length = runs[i][1] - runs[i][0]
            stride = runs[i + 1][0] - runs[i][0]
            while (j + 1 < len(runs)
                   and runs[j + 1][1] - runs[j + 1][0] == length
                   and runs[j + 1][0] - runs[j][0] == stride):
                j += 1
        if j - i >= 4:
            parts.append(f"{show(runs[i])}, {show(runs[i + 1])}, …, "
                         f"{show(runs[j])} (every {runs[i + 1][0] - runs[i][0]})")
            i = j + 1
        else:
            parts.append(show(runs[i]))
            i += 1
    return ", ".join(parts)
