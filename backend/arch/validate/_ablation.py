"""Guard-ablation switch for the (archived, out-of-tree) necessity study.

`HA_ABLATE_GUARDS` is a comma-separated list of guard *levers* to disable
inside `validate()`. When a lever is disabled its findings are dropped from
BOTH the gating `errors[]` list (so the publish gate can't catch them) AND the
per-guard warning fields (so the spec-gen agent's stderr feedback loop never
sees them) — the two channels a guard acts on.

The switch is inert when the env var is unset/empty: `parse_disabled()` returns
an empty set and every `lever_disabled(...)` call is False, so `validate()`
behaves byte-identically to production. This module exists only to support the
offline ablation harness and must never change default behavior.

Levers (aligned to validate()'s flow, grouped by ground-truth source):
  weight      — model-weight/checkpoint anchor: bytes headline + packing +
                per-tensor matching diagnostics + the per-rank weight ledger
                building. Implies `packing`.
    packing   — (sub) just the packing/quant-scheme count×dtype anchor (5j).
  shape       — shape + dim-map consistency (5d/5e).
  structural  — topology structural checks (5c non-source) + module-exec (5g)
                + the rotation-vs-rope-parameters check (`_rotary_positions`)
                + the unread-parameter check (`_unread_parameters`)
                + schedule-exhaustiveness (5i). Implies `schedule`.
    schedule  — (sub) per-layer config_list exhaustiveness (5i) and
                config-named param bindings (`ParamSpec.config`).
  source      — forward()-source graph diff (5c source-diff kinds) and
                cross-layer stream contracts (5i2).
  axiom       — source-locked axioms (5h) + KV/attention type contracts (5f).
                Implies `swa_window`.
    swa_window — (sub) just the gating SWA⟹capped-cache contract (the 128TB bug).
  analysis    — the spec analyses on the product's deployments: every
                reached op has a layout transition rule, the weight identity
                join builds at one GPU, and the injected analysis replay
                (`downstream_gates`) passes.
"""
from __future__ import annotations

import os

ENV_VAR = "HA_ABLATE_GUARDS"

# Kinds read off the forward() source (structural.check()'s comparison, the
# cache-write scan) — lever `source`. Everything else in structural_warnings
# is engine/spec topology — lever `structural`.
SOURCE_DIFF_KINDS = frozenset({
    "add_count_mismatch",
    "cache_write_quantised_unmodelled",
    "call_order_mismatch",
    "forward_block_unlocated",
    "forward_call_missing_in_spec",
    "forward_has_branches",
    "spec_call_not_in_forward",
})

# A parent lever implies its sub-levers.
_IMPLIES = {
    "weight": {"packing"},
    "structural": {"schedule"},
    "axiom": {"swa_window"},
}

_KNOWN = {"weight", "packing", "shape", "structural", "schedule",
          "source", "axiom", "swa_window", "analysis"}


def parse_disabled(env: str | None = None) -> frozenset:
    """Return the set of disabled levers (with parent→sub expansion).

    Empty when the env var is unset/empty → switch inert. Unknown tokens are
    ignored (so a typo can't silently disable a real guard)."""
    raw = os.environ.get(ENV_VAR, "") if env is None else env
    tokens = {t.strip().lower() for t in raw.split(",") if t.strip()}
    disabled = set(t for t in tokens if t in _KNOWN)
    for parent, subs in _IMPLIES.items():
        if parent in disabled:
            disabled |= subs
    return frozenset(disabled)


def lever_disabled(disabled: frozenset, lever: str) -> bool:
    return lever in disabled
