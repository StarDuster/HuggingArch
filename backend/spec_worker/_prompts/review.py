"""Read-only review task body: judge an already-written spec against the model's
real architecture and end with a machine-parseable PASS / NO_PASS verdict.

This is the standalone, read-only form of the generate prompt's "Phase 3.5
forward-alignment reflection" — the same forward-fidelity standard
(`templates/forward_alignment.md`), applied as a review rather than in-loop. Imports NOTHING from `backend.service` (core↔service boundary).
"""
from __future__ import annotations

from typing import Any, Optional, Sequence

from .system import forward_alignment_doc, source_files_section
from .tasks import _PY, shell_rules
from ..review_evidence import SnapshotFacts


def _render_validate_summary(vr: Optional[dict]) -> str:
    """Compact rendering of the deterministic validate() result to fold into the
    prompt. A validate `valid: false` is strong NO_PASS evidence; `valid: true`
    is necessary but NOT sufficient (topology can be wrong while bytes match)."""
    if not vr:
        return "(deterministic validator did not run)"
    lines = [f"valid: {vr.get('valid')}"]
    errs = vr.get("errors") or []
    if errs:
        lines.append(f"errors ({len(errs)}):")
        lines += [f"  - {str(e)[:300]}" for e in errs[:12]]
    unm = vr.get("unmatched_hf_tensors") or []
    if unm:
        lines.append(f"unmatched_hf_tensors ({len(unm)}):")
        lines += [f"  - {str(t)[:200]}" for t in unm[:12]]
    pm = vr.get("param_mismatches") or []
    if pm:
        lines.append(f"param_mismatches ({len(pm)}):")
        lines += [f"  - {str(m)[:200]}" for m in pm[:8]]
    for k in ("shape_mismatches", "axiom_warnings", "type_contract_warnings"):
        w = vr.get(k) or []
        if w:
            lines.append(f"{k} ({len(w)}): {str(w[:4])[:300]}")
    mb = vr.get("model_storage_bytes") or {}
    if mb:
        lines.append(f"model_storage_bytes: {mb}")
    lines += _render_analysis_replay(vr)
    return "\n".join(lines)


#: The line a review report carries when every validate error is an
#: analysis engine refusal. Display only: what a record does next is decided
#: on its status (`GenerationStatus`), never on report text.
BLOCKED_BY_ENGINE_LINE = ("blocked_by_engine: every error is an analysis "
                          "engine refusal; no spec edit resolves it")


def _render_analysis_replay(vr: dict) -> list:
    """The analysis replay as a table: one row per deployment validate ran
    the analysis on, with the node each refusal names."""
    analysis = vr.get("analysis")
    if not analysis:
        return []
    lines = [f"analysis replay: {analysis}"]
    if vr.get("blocked_by_engine"):
        lines.append(BLOCKED_BY_ENGINE_LINE)
    rows = (vr.get("analysis_replay") or {}).get("deployments") or []
    if rows:
        lines.append("| deployment | parallel | result |")
        lines.append("|---|---|---|")
        for row in rows:
            refusals = row.get("refusals") or []
            result = "ok" if not refusals else "; ".join(
                f"{r.get('kind')} at `{r.get('node')}` ({r.get('stage')})"
                for r in refusals)
            lines.append(f"| {row.get('label')} | {row.get('parallel')} | {result} |")
    for refusal in vr.get("analysis_refusals") or []:
        fix = f" — fix: {refusal['fix']}" if refusal.get("fix") else ""
        lines.append(f"  - {refusal.get('kind')} node `{refusal.get('node')}`"
                     f"{fix}"[:400])
    return lines


def _render_spec(spec_yaml: str, new_component_yamls: Any) -> str:
    parts = ["### model spec\n```yaml\n" + (spec_yaml or "").strip() + "\n```"]
    for comp in (new_component_yamls or []):
        if not isinstance(comp, dict):
            continue
        path = (comp.get("path") or "component").strip()
        body = (comp.get("body") or "").strip()
        parts.append(f"### draft component `{path}`\n```yaml\n{body}\n```")
    return "\n\n".join(parts)


def evidence_rules(facts: SnapshotFacts) -> str:
    """The evidence contract `review_evidence` enforces, and the checkpoint
    layer counts it checks `count:` tags against."""
    return f"""## Evidence (checked mechanically before your verdict is stored)
Admissible evidence is exactly: the staged source files listed below,
config.json and the checkpoint tensor table (both in the ground truth above).
A model card, paper, README or your memory of another framework is not
evidence; a claim only they support is not a defect.

Write evidence as backtick tags, exactly in the forms below. After you
finish, a checker compares every tag with this run's snapshot:
  `src: modeling_x.py:120-122 "self.scaling = self.head_dim**-0.5"`
      a staged file, the line range, and the code quoted verbatim from those
      lines (whitespace-insensitive; `...` elides). Write the line numbers
      as digits (`:120`, `:120-122`, never `:L120`), one quoted excerpt per
      tag, and quote marks inside the code as they are, unescaped
  `cfg: text_config.sliding_window = 128`
      a config.json key path and the value it holds. A bare key also matches
      nested keys of the same name, so a key that occurs more than once
      cannot be checked: write `config.rope_theta` for the top-level one.
      `= absent` for a key config.json omits (then cite the
      configuration-class default with `src:`)
  `ckpt: model.layers.X.mlp.gate_proj.weight = [2048, 6144]`
      a tensor's full path from the table and its stored shape; `= absent`
      for one the checkpoint does not have
  `count: model.mtp.layers = 3`
      how many layer indices the checkpoint stores under a prefix (see
      below); `= 0` when it stores none

What the checker enforces:
  - Every Table F row carries a `src:` tag whose quote is found at the cited
    lines (an n/a row may carry a `cfg:` tag instead).
  - Every defect carries such a `src:` tag, and every number it states (a
    size, count, window, layer index, shape or constant) must be the value of
    one of its verified tags or appear in one of its verified quotes.
  - One top-level Defects bullet is one defect: one repair whose evidence
    stands on its own. The checker classifies each bullet as a whole, so do
    not bundle repairs; one bad tag would hold back the others with it.
  - A tag that contradicts the snapshot (wrong value, wrong shape, a quote
    that is not in the file) marks the defect contradicted: Agent Fix sees
    only the false tag and what the snapshot says instead.
  - A defect whose evidence does not check out reaches Agent Fix only as a
    title and the places to look, never with its text or fix.
  - A finding or row is verified only when every check in it verifies:
    one tag that cannot be checked (an ambiguous key, a bare tensor name, a
    file that is not staged, an untagged `file.py:L` or `key = value`
    mention) keeps the whole finding unverified. Put only checkable evidence
    in a finding.
  - A PASS is recorded only when every Table F row is verified and its
    status is exactly match, constant not expressible or n/a (a reason may
    follow in parentheses), Table S is present with no UNSOURCED row, and
    every defect listed is verified. Otherwise the review is recorded as an
    error.

The deterministic validator's errors reach Agent Fix verbatim. Do not list
them as defects and do not guess their cause.

Checkpoint layer counts (prefix: stored layer indices):
```
{facts.render_layer_counts()}
```
"""


def build_review_task_body(model_id: str, spec_yaml: str, new_component_yamls: Any,
                           validate_result: Optional[dict], *,
                           facts: SnapshotFacts,
                           target_root: Optional[str] = None,
                           source_files: Sequence[str] = ()) -> str:
    """The review task body. The spec under review is embedded inline and also
    seeded into the overlay (resolved via HA_SPEC_OVERLAY_DIR) so the agent MAY
    re-run `python -m backend.inference.validate <model_id>` /
    `python -m backend.arch.structure <model_id>` to check its own reasoning.
    `source_files` are the full source files staged into the run overlay
    (`overlay.stage_forward_source`), listed so citations can carry line numbers.
    `facts` is the snapshot the report's evidence tags are checked against."""
    return f"""# TASK: review a generated spec for {model_id}

You are an ADVERSARIAL reviewer. A spec for `{model_id}` has already been
written. Judge whether it is a COMPLETE and SEMANTICALLY FAITHFUL model of the
real architecture shown in the ground truth above (the transformers structure,
the `forward` bodies / `modeling_*.py` source, config, and tensor patterns).

This is READ-ONLY. Do NOT write or edit any file. You MAY run, for evidence:
  - `{_PY} -m backend.inference.validate {model_id}`
  - `{_PY} -m backend.arch.structure {model_id}`
  - `{_PY} -m backend.arch.component_reuse <a draft component yaml>` — which
    existing components already hold that draft's graph, by share of its
    edges; 0.85 and above is the draft restating one of them
(the spec under review is seeded into this run's overlay, resolved automatically).
{shell_rules(model_id)}

## The spec under review
{_render_spec(spec_yaml, new_component_yamls)}

## Deterministic validator result (already run)
```
{_render_validate_summary(validate_result)}
```
The deterministic validator proves tensors map + shapes/params resolve. It does
NOT prove forward fidelity. Treat `valid: false` as strong NO_PASS evidence.
`valid: true` alone is NOT enough to PASS — the topology can be wrong while the
numbers happen to line up.

## Forward alignment (the core of this review)
{forward_alignment_doc()}

{source_files_section(source_files)}
{evidence_rules(facts)}
## Mandatory two-way mapping (goes INSIDE the report block)
Get the spec's compiled components from `{_PY} -m backend.arch.structure {model_id}`. Rows follow the granularity rule above: one per distinct
component, one per layer-assignment rule, auxiliary stacks separately.

Table F — forward → spec, in execution order:
`| # | forward step | source | spec node(s) | tags | status |`. `source` holds
the row's `src:` tag (and any `cfg:` / `ckpt:` / `count:` tags for the numbers
the row states). Status ∈
match / MISMATCH / UNMAPPED / constant not expressible (per the two-condition
rule; name the op checked). Non-steps may appear as `n/a (reason)`.

Table S — spec → forward: `| component / node | op | forward step # |`, or
UNSOURCED when no forward step justifies it.

Symbols the spec declares under `dim_symbols:` (new dimensions, promoted into
dims.yaml with the model) get one Table S row each, `dim_symbols.<name>`,
whose source tag shows what the number counts and whether the forward source
splits it across ranks; the row's `kind`, `shardable_by`, `group` and
`meaning` match that source.

## Verdict rules
NO_PASS if ANY of:
  - a forward step is UNMAPPED (no spec node) or MISMATCH (wrong op, wrong
    edge, wrong switch / score function / layer pattern);
  - a spec node is UNSOURCED (invented computation);
  - a checklist item is left without a citation, or a framework disagreement
    is not resolved by a spec comment naming the reference;
  - the deterministic validator says `valid: false`, or a checkpoint tensor
    family is missing / invented, or layer counts / dims do not resolve to the
    config's real values.
PASS only when both tables are complete and every row is match, constant
not expressible, or n/a.

## Output contract (MANDATORY — must be the LAST thing you emit)
End your response with exactly one report block in this shape, the verdict on
its own line as the last line inside the block:

===REVIEW REPORT START===
<Table F, Table S, then `### Defects`: one top-level bullet per MISMATCH /
UNMAPPED / UNSOURCED row or other NO_PASS reason, one repair per bullet — a
short bold title naming the defect (not the fix), the evidence tags, the spec
node, the cost tag, then `Fix:`; prefix
`[library: <component file>]` when the defect lives in a shared
`components/library/` component rather than this record's own YAML; write
`None.` and nothing else under `### Defects` when there are none.
Observations that are not defects (limitations of this review, harmless
deviations) go in a separate `### Notes` section after the Defects list, never
inside it: every bullet under `### Defects` is a defect, a PASS that lists
any defect is recorded as an error, and so is a Defects section that says
`None.` and then continues>
VERDICT: {{verdict}}
===REVIEW REPORT END===

where {{verdict}} is the single token PASS or NO_PASS per the rules above
(on NO_PASS, list every defect). Emit this block exactly once, as the very
last thing; do not write the verdict token anywhere earlier in your response.
"""
