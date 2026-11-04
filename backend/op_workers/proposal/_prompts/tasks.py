"""Per-task prompt bodies for op-proposal worker.

Single task today: `propose_op`, fed a gap statement dict (read by the
caller from `logs/op_proposals/*.gap.json`) and produces a proposal
artifact.
"""
from __future__ import annotations

import json
import os

from .system import build_system_prompt
from ..runtime import (
    PROPOSALS_DIR_REL,
    GAP_DIR_REL,
)
from ...._agent_runner.source_paths import VLLM_SRC, SGLANG_SRC


def build_propose_op_task_body(op_name: str, gap: dict) -> str:
    """Build the task body. `gap` is the parsed JSON contents of the
    spec-worker's `<...>.gap.json` file."""
    target_dir_rel = f"{PROPOSALS_DIR_REL}/{op_name}"
    gap_json = json.dumps(gap, indent=2, ensure_ascii=False)
    return f"""# Task: propose primitive op `{op_name}`

The spec worker for model `{gap.get("context_model_id", "<unknown>")}`
hit a kernel it could not compose. It wrote the following gap statement:

```json
{gap_json}
```

## Workflow

1. **Verify the gap is real**. Walk the existing-primitives table in
   the system prompt above and confirm no combination composes the
   signature. If a combination DOES compose, write only `rationale.md`
   under `{target_dir_rel}/` explaining the composition path, then
   stop — no other artifacts.

2. **Find the kernel source**. Use Grep / Glob to locate the kernel in
   `{VLLM_SRC}` and `{SGLANG_SRC}` (and the
   HF transformers tree if installed under the project venv). The gap's
   `kernel_source_hint` is a starting pointer, not authoritative. You
   MUST land on an actual function definition before deriving FLOPs.

3. **Derive FLOPs and params** from the kernel source line-by-line. Do
   NOT use a paper / blog post's claimed cost. If the kernel calls into
   a fused CUDA op, read the wrapper Python's tile-loop accounting OR
   the reference math implementation (transformers usually has a slow
   reference path for testing).

4. **Pick `shape_kind`** from the nine enum values. Justify rejection
   of every adjacent value (e.g. why not `axis_replace`? why not
   `explicit`?). A wrong choice is silent and contaminates every
   downstream model.

5. **Bind dim symbols**. Reuse registered symbols where the semantics
   match exactly. Propose new symbols only when no existing symbol
   covers the meaning, and follow the family-prefix convention visible
   in the registry. A new symbol needs a `kind` — which parallelism
   divides it — as much as it needs a name; `whole` is the answer when
   none does, and it is a claim you must be able to defend.

6. **Write all six artifacts** under `{target_dir_rel}/`:
   - `gap.json` — verbatim copy of the gap statement above.
   - `patch.py` — structured (DIM_SECTION_COMMENT / NEW_DIM_SYMBOLS /
     OP_REGISTRATIONS) per the system prompt format.
   - `dims.yaml` — registry entries for the new symbols, a mapping of
     `<symbol>: {{kind, group, meaning}}` (optionally `config`, `derived`,
     `used_by`). Omit the file entirely if no new symbols.
   - `tests.py` — at least three pytest methods (registration, shape,
     FLOPs scaling).
   - `worked_example.yaml` — a component yaml that uses the new
     primitive to compose the blocking model's kernel.
   - `rationale.md` — the full narrative (kernel source citation,
     composition-rejection proof, shape_kind justification, FLOPs
     derivation, dim-symbol justification).

## Hard rules

- NEVER Read / Edit / Write outside `{target_dir_rel}/`, except
  read-only access to the gap.json file under `{GAP_DIR_REL}/`,
  `backend/arch/`, `tests/`, `{VLLM_SRC}/`,
  `{SGLANG_SRC}/`, and HF transformers source.
- NEVER edit `backend/arch/ops.py`, `backend/arch/dims.yaml`,
  or any file under `tests/`. The promote tool is the only path that
  modifies those.
- NEVER cite a blog post as the source for a cost formula.
- If you find a composition path that obsoletes the gap, write only
  `rationale.md` (no patch). The reviewer will reject the gap
  upstream — that's correct.

When the six artifacts are written (or the single rationale.md is
written for a rejected gap), stop. The promote tool reports results.
"""


def build_propose_op_prompt(op_name: str, gap: dict) -> str:
    system = build_system_prompt()
    body = build_propose_op_task_body(op_name, gap)
    return f"{system}\n\n---\n\n{body}"


def prep_from_gap(op_name: str, gap: dict) -> dict:
    """Prompt + prefetch logs from an already-parsed gap dict."""
    return {
        "prompt": build_propose_op_prompt(op_name, gap),
        "prefetch_logs": [
            f"Op-proposal worker — proposing primitive `{op_name}`",
            f"Blocking model: {gap.get('context_model_id', '<unknown>')}",
        ],
    }


def prepare_propose_op_prompt(op_name: str, gap_path: str) -> dict:
    """Read the gap file from disk and assemble prompt + prefetch logs.

    `gap_path` is repo-relative or absolute; the route handler is
    responsible for resolving and validating it before calling us.
    """
    with open(gap_path, encoding="utf-8") as f:
        gap = json.load(f)
    prompt = build_propose_op_prompt(op_name, gap)
    return {
        "prompt": prompt,
        "prefetch_logs": [
            f"Op-proposal worker — proposing primitive `{op_name}`",
            f"Gap statement: {os.path.basename(gap_path)}",
            f"Blocking model: {gap.get('context_model_id', '<unknown>')}",
        ],
    }
