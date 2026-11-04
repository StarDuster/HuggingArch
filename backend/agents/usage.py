"""Cross-backend token-usage extraction.

Every backend normalizes its usage fields through `extract_usage_events`
so stream_agent can display uniform token counts regardless of CLI
origin.
"""
from __future__ import annotations

from typing import List, Optional


def extract_usage_events(evt: dict) -> List[dict]:
    """Recursively find usage-like dicts inside an agent event and emit
    one normalized {"type": "usage", ...} per unique usage payload."""
    results: List[dict] = []
    seen = set()

    def visit(obj):
        if isinstance(obj, dict):
            usage = _normalize_usage_dict(obj)
            if usage:
                key = tuple(sorted(usage.items()))
                if key not in seen:
                    seen.add(key)
                    results.append({"type": "usage", **usage})
            for value in obj.values():
                visit(value)
        elif isinstance(obj, list):
            for item in obj:
                visit(item)

    visit(evt)
    return results


def _normalize_usage_dict(obj: dict) -> Optional[dict]:
    if not isinstance(obj, dict):
        return None

    input_tokens = _to_int(_first_present(obj, "input_tokens", "prompt_tokens", "inputTokens"))
    output_tokens = _to_int(_first_present(obj, "output_tokens", "completion_tokens", "outputTokens"))
    total_tokens = _to_int(_first_present(obj, "total_tokens", "totalTokens"))
    cache_creation = _to_int(_first_present(
        obj,
        "cache_creation_input_tokens",
        "cache_creation_tokens",
        "cacheCreationInputTokens",
    ))
    cache_read = _to_int(_first_present(
        obj,
        "cache_read_input_tokens",
        "cached_input_tokens",
        "cache_read_tokens",
        "cacheReadInputTokens",
    ))

    if all(v is None for v in (input_tokens, output_tokens, total_tokens, cache_creation, cache_read)):
        return None

    if total_tokens is None:
        # All four fields count as tokens processed (different billing
        # rates but all real). Anthropic's Message API doesn't emit
        # total_tokens, so we sum ourselves — must include cache tokens
        # or totals look absurd on cache-hit turns (e.g. in=1 out=48 but
        # cache_read=68k → real total is ~68K, not 49).
        subtotal = sum(
            v for v in (input_tokens, output_tokens, cache_creation, cache_read)
            if v is not None
        )
        total_tokens = subtotal if subtotal > 0 else None

    usage = {}
    if input_tokens is not None:
        usage["input_tokens"] = input_tokens
    if output_tokens is not None:
        usage["output_tokens"] = output_tokens
    if total_tokens is not None:
        usage["total_tokens"] = total_tokens
    if cache_creation is not None:
        usage["cache_creation_input_tokens"] = cache_creation
    if cache_read is not None:
        usage["cache_read_input_tokens"] = cache_read
    return usage


def _first_present(obj: dict, *keys: str):
    for key in keys:
        if key in obj:
            return obj.get(key)
    return None


def _to_int(value) -> Optional[int]:
    if value is None or isinstance(value, bool):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None
