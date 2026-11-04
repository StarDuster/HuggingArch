"""Periodic-span detection on a sequence of layer-template tokens.

Used by the layer-layout pass to fold `[gqa, swa, swa, gqa, swa, swa, ...]`
hybrid-attention sequences into a single "cycle" card on the architecture
diagram. Pure algorithm; no spec / dim / engine deps.
"""
from __future__ import annotations

from collections import Counter
from typing import Any, List, Optional, Tuple


def find_periodic_span(seq: List[Any]) -> Optional[Tuple[int, int, int]]:
    """Find a periodic span anchored on the *rarest* token's spacing.

    Mental model: a hybrid-attention model has a "rare event" attention
    type (e.g. full GQA in a SWA-dominant stack) that recurs every P
    layers; the rest is filler. Cycle = [rare_event, filler·(P-1)]. The
    cycle anchor is the first rare event in the steady-state run, the
    period is the dominant inter-occurrence gap, the cycle ends at the
    last filler after the last rare event in the run. Anything before
    the steady-state anchor (e.g. MiMo's layer-0 dense bootstrap, plus
    its short 4-SWA stretch before the first MoE GQA) automatically
    becomes a prefix run — no phase voting, no thresholds.

    Returns (start, end_inclusive, period) or None when `seq` lacks
    a rare-event-spacing structure (uniform sequence, no dominant
    gap, etc.). O(n) — one Counter pass plus one scan over rare-event
    positions.

    Caller must pass a sequence where each element captures the cycle
    KEY of that position (e.g. attention component name). Variations
    that aren't part of the cycle key — like dense vs MoE FFN at the
    same attention type — are surfaced separately by the caller using
    the per-position template, not folded into the period detection.
    """
    if not seq:
        return None
    counts = Counter(seq)
    if len(counts) < 2:
        return None  # uniform sequence — nothing to cycle through

    # Each token class is a candidate marker. Compute the best
    # period-spaced run for every class and return the longest. Picking
    # only the single rarest class breaks Ling-2.6-1T: the MTP layer
    # gives the rarest class a single occurrence, masking the real
    # `mla_attention_dense` cycle (10 occurrences spaced 8 apart).
    def _best_run_for_token(tok: Any) -> Optional[Tuple[int, int, int, int]]:
        positions = [i for i, t in enumerate(seq) if t == tok]
        if len(positions) < 3:
            return None
        gaps = [positions[i + 1] - positions[i] for i in range(len(positions) - 1)]
        period, period_count = Counter(gaps).most_common(1)[0]
        if period < 2 or period_count < 2:
            return None
        # Longest consecutive run of `period`-spaced positions. First
        # mismatching gap resets the run — that's how MiMo's irregular
        # layer-0 → layer-5 gap automatically excludes layer 0.
        best_idx, best_len = 0, 1
        cur_idx, cur_len = 0, 1
        for i in range(1, len(positions)):
            if positions[i] - positions[i - 1] == period:
                cur_len += 1
            else:
                cur_idx, cur_len = i, 1
            if cur_len > best_len:
                best_idx, best_len = cur_idx, cur_len
        if best_len < 2:
            return None
        start = positions[best_idx]
        last = positions[best_idx + best_len - 1]
        end = min(last + period - 1, len(seq) - 1)
        return (start, end, period, best_len)

    # The marker's spacing is only a guess at the period: the cycle is the
    # whole stretch between markers, and it repeats only where every
    # position does. Kimi-K3's MLA layer recurs every 4, but every third
    # stretch opens with a different KDA block, so its cycle is 12 long;
    # counted by the marker alone, 4 hid those layers behind a cycle card
    # that claimed they were plain KDA. Each candidate period is checked
    # cycle by cycle, and lengthened by whole multiples until the cycles
    # it covers repeat exactly.
    def _verified(start: int, end: int, marker_period: int
                  ) -> Optional[Tuple[int, int]]:
        best: Optional[Tuple[int, int]] = None
        span = end - start + 1
        period = marker_period
        while 2 * period <= len(seq) - start:
            pattern = seq[start:start + period]
            cycles = 1
            while (start + (cycles + 1) * period <= len(seq)
                   and seq[start + cycles * period:
                           start + (cycles + 1) * period] == pattern):
                cycles += 1
            if cycles >= 2 and (best is None
                                or cycles * period > best[0] * best[1]):
                best = (cycles, period)
            if best is not None and best[0] * best[1] >= span:
                break
            period += marker_period
        return best

    candidates: List[Tuple[int, int, int, int]] = []
    for c in (_best_run_for_token(t) for t in counts):
        if c is None:
            continue
        found = _verified(c[0], c[1], c[2])
        if found is None:
            continue
        cycles, period = found
        candidates.append((c[0], c[0] + cycles * period - 1, period, c[3]))
    if not candidates:
        return None

    # Pick the cycle with the most marker-event occurrences, then the
    # longest absolute span as a deterministic tiebreak.
    start, end, period, _ = max(candidates, key=lambda c: (c[3], c[1] - c[0]))
    return (start, end, period)
