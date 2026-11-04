"""Writable fix task body: repair the SPECIFIC defects a prior NO_PASS review
named, or the refusals that stop a saved record from loading, editing only
this record's own draft model/component YAML.

The read-only counterpart is `_prompts/review.py`; this is its writable twin —
same ground truth (real modeling source + config + tensor patterns), but the
agent now makes the minimal scoped edits that turn the review's findings into a
PASS, then re-validates. Imports NOTHING from `backend.service`.
"""
from __future__ import annotations

from typing import Any, Optional, Sequence

from .review import _render_spec, _render_validate_summary
from .system import source_files_section
from .tasks import _PY, shell_rules
from ..overlay import _safe
from ..review_evidence import ReviewEvidence, SnapshotFacts


def _load_refusals_section(load_errors: Sequence[str]) -> str:
    """The refusals that stop the saved record from loading, as work order."""
    if not load_errors:
        return ""
    items = "\n".join(f"- {error}" for error in load_errors)
    return f"""## Load refusals — the saved spec cannot be loaded until these are resolved
The current schema refuses the saved spec with the errors below, so it is
seeded exactly as it was saved (nothing flattened, migrated or renamed).
Each error names its repair; resolve every one of them. A draft you rename
is a new file under `components/drafts/` plus the new name in the spec's
`drafts:` list and in every reference; the old file may stay, because a
draft file the spec's `drafts:` list does not name is neither loaded nor
saved. A declaration the error asks for is taken from the staged source,
config.json or the tensor table, never guessed.

{items}

"""


def _validation_errors_section(errors: Sequence[str]) -> str:
    """The errors a re-check of the saved spec found, as work order."""
    if not errors:
        return ""
    items = "\n".join(f"- {error}" for error in errors)
    return f"""## Validation errors — a re-check of the saved spec found these
The saved spec was re-validated against the current engine and checkpoint;
each error below names what to change. Resolve every one of them. The
deterministic validator result further down is this run's own run of the
same check.

{items}

"""


def _review_section(review: Optional[ReviewEvidence], *, refusals: bool,
                    validator: bool) -> str:
    """The review's findings after the evidence check. A review none of whose
    findings checked out orders nothing, and the section says so: the work
    order is then what this run's own checks found (the load `refusals` and
    the failing `validator` result), and the review's leads are only leads."""
    if review is None:
        return ""
    if review.gives_work_order():
        instead = ""
    else:
        found = [name for name, present in (
            ("the load refusals above", refusals),
            ("the errors in the deterministic validator result below",
             validator)) if present]
        instead = ("**No finding of this review checked out, so it orders no "
                   "edit.** ")
        instead += (f"The work order is {' and '.join(found)}; resolve every "
                    "one." if found else
                    "The load and the validator found nothing either: confirm "
                    "the leads below against the source, and if none is "
                    "confirmed, make no edits and stop.")
        instead += "\n\n"
    return f"""## The review's findings
Each finding's evidence was checked against this run's config.json,
checkpoint tensor index and staged source files before this prompt was
written. Reviews make mistakes; config.json, the checkpoint and the source
outrank any finding.

{instead}{review.work_order()}

"""


def build_fix_task_body(model_id: str, spec_yaml: str, new_component_yamls: Any,
                        validate_result: Optional[dict],
                        review: Optional[ReviewEvidence], *,
                        facts: SnapshotFacts,
                        target_root: Optional[str] = None,
                        source_files: Sequence[str] = (),
                        load_errors: Sequence[str] = (),
                        validation_errors: Sequence[str] = ()) -> str:
    """The fix task body. The record's current draft is seeded into the overlay
    (resolved via HA_SPEC_OVERLAY_DIR) and the agent edits it in place. The
    work order is the refusals that stop the saved record from loading
    (`load_errors`, when it is a `BlockedBundle`), then the NO_PASS review's
    findings after the evidence check (`review`, absent when Fix was opened
    by the refusals alone): verified findings are to be resolved, unverified
    ones confirmed first, and contradicted ones are named only with their
    contradiction — never with their suggested fix. A review none of whose
    findings is verified orders nothing; the validator's errors take its
    place beside the refusals (`_review_section`)."""
    identity_path = (
        f"{target_root}/models/{_safe(model_id)}.yaml" if target_root
        else f"models/{_safe(model_id)}.yaml"
    )
    if review is not None and load_errors:
        why = (f"The saved spec for `{model_id}` cannot be loaded under the "
               "current schema, and a prior adversarial review returned "
               "**NO_PASS** on it.")
    elif review is not None:
        why = (f"A prior adversarial review returned **NO_PASS** on the spec "
               f"for `{model_id}`.")
    elif validation_errors and not load_errors:
        why = (f"A re-check found validation errors in the saved spec for "
               f"`{model_id}`.")
    else:
        why = (f"The saved spec for `{model_id}` cannot be loaded under the "
               "current schema.")
    review_orders = review is not None and review.gives_work_order()
    # A review that orders nothing hands the work order to the validator.
    validator_orders = (review is not None and not review_orders
                        and bool(validate_result)
                        and not validate_result.get("valid"))
    orderers = [name for name, present in (
        ("the load refusals", bool(load_errors)),
        ("the re-check", bool(validation_errors)),
        ("the review", review_orders),
        ("the validator", validator_orders)) if present]
    named = (" and ".join(orderers) + " named" if orderers
             else "you confirm among the review's leads")
    review_section = _review_section(
        review, refusals=bool(load_errors), validator=validator_orders)
    return f"""# TASK: fix a generated spec for {model_id}

{why}
Your job is to apply the MINIMAL, SCOPED edits that resolve EXACTLY the defects
{named} — turning it into a spec that is a complete, semantically
faithful model of the real architecture in the ground truth above.

{_load_refusals_section(load_errors)}{_validation_errors_section(validation_errors)}{review_section}Checkpoint layer counts (prefix: stored layer indices):
```
{facts.render_layer_counts()}
```

{source_files_section(source_files)}
## The current draft (seeded into this run's overlay — edit it in place)
{_render_spec(spec_yaml, new_component_yamls)}

## Deterministic validator result (already run against the current draft)
```
{_render_validate_summary(validate_result)}
```

## Rules
- Edit ONLY the model identity YAML (`{identity_path}`) and its draft components
  (`components/drafts/*.yaml`). You may NOT touch anything else — no shared
  library components, no other models.
- Defects tagged `[library: …]` live in a shared library component: NOT yours
  to fix, and never worked around with a copied draft. Leave them; they stay
  on record in the review report and keep the record NO_PASS until the library
  is fixed. If every defect is library-level, make no edits and stop.
- Make the SMALLEST change that fixes each named defect. Do NOT rewrite the
  spec, do NOT restructure things the review did not flag, do NOT "improve"
  unrelated parts. Every edit must trace to a load refusal, a validator
  error, a work-order finding, or an unverified finding you confirmed in the
  source, config.json or the tensor table.
- Ground each edit in the real architecture (the transformers structure,
  `forward` bodies / `modeling_*.py`, config, tensor patterns shown above).
- A finding about a value that crosses layers is fixed with the form the
  system context's "`forward()` pattern → stream form" table gives for that
  pattern.
- After editing, run `{_PY} -m backend.inference.validate {model_id}` and iterate
  until it passes (0 hard errors) or reports `blocked_by_engine: true`. The
  seeded draft resolves automatically.
  {shell_rules(model_id)}

When `{_PY} -m backend.inference.validate {model_id}` reports no errors, every
load refusal and work-order finding is addressed, and every unverified lead is either fixed
(because you confirmed it) or left unchanged with the reason, you are done. A separate semantic review runs
afterward to confirm the fix — do not emit a verdict yourself.
"""
