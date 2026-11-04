"""Task body for op-review worker.

Single task: review the artifact directory for a given op_name and emit
`review.json` with a verdict.
"""
from __future__ import annotations

from pathlib import Path

from .components import build_artifact_listing
from .system import build_system_prompt
from ..runtime import PROPOSALS_DIR, PROPOSALS_DIR_REL
from ...._agent_runner.source_paths import VLLM_SRC, SGLANG_SRC


def build_review_task_body(op_name: str) -> str:
    artifact_dir_rel = f"{PROPOSALS_DIR_REL}/{op_name}"
    artifact_dir_abs = Path(PROPOSALS_DIR) / op_name
    artifact_listing = build_artifact_listing(artifact_dir_abs)

    return f"""# Task: review op proposal `{op_name}`

The proposer wrote artifacts to `{artifact_dir_rel}/`. Their contents
follow. You must judge whether the proposal is sound and write your
verdict to `{artifact_dir_rel}/review.json`.

## Artifact contents

{artifact_listing}

## Workflow

1. **Read all artifacts above carefully**. Cross-reference claims in
   `rationale.md` against the actual code in `patch.py` / `tests.py` /
   `worked_example.yaml`.

2. **Verify the source citation**. Use Read or Grep to confirm the
   kernel file path + symbol named in `rationale.md` actually exists.
   If it doesn't, that alone is a `reject`.

3. **Walk the composition-rejection argument**. Use the existing-
   primitives table in the system prompt above. For each primitive the
   rationale claims doesn't compose, do you agree? If you can construct
   a composition path the proposer missed, verdict is `reject` and
   `summary` should describe the composition path.

4. **Sanity-check FLOPs against kernel source**. Match the formula in
   `OP_REGISTRATIONS` line-by-line against the kernel. If the formula
   double-counts or omits a tile-loop bound, verdict is `reject`.

5. **Read every test in tests.py**. Anything that would pass against a
   trivially broken implementation is a red flag.

6. **Decide verdict and write `review.json`**. Use the schema in the
   system prompt. Be explicit in `summary` — your reasoning is the
   audit trail for the human reviewer if your verdict turns out wrong.

## Hard rules

- You may Read / Grep anywhere under `{VLLM_SRC}/`,
  `{SGLANG_SRC}/`, `backend/arch/`, `tests/`, and the
  artifact directory `{artifact_dir_rel}/`.
- You may NOT Write anywhere except `{artifact_dir_rel}/review.json`.
- You may NOT Edit any existing file. Even within the artifact
  directory.
- You may NOT invoke any promote / merge / git tool.

When `review.json` is written, stop. The route layer reads it and
takes the next action automatically.
"""


def build_review_prompt(op_name: str) -> str:
    system = build_system_prompt()
    body = build_review_task_body(op_name)
    return f"{system}\n\n---\n\n{body}"


def prepare_review_prompt(op_name: str) -> dict:
    """Assemble prompt + prefetch_logs for the SSE wrapper. No I/O beyond
    reading the artifact directory (which `build_artifact_listing` does
    internally)."""
    prompt = build_review_prompt(op_name)
    return {
        "prompt": prompt,
        "prefetch_logs": [
            f"Op-review worker — reviewing proposal `{op_name}`",
            f"Artifact dir: {PROPOSALS_DIR_REL}/{op_name}",
        ],
    }
