"""System prompt — the reusable context shown to every spec-gen / spec-
refine task.

Template-driven: the prompt body lives in `templates/system.md` (with
`{ops_doc}` / `{components}` / `{schema_doc}` / `{examples}` placeholders)
and the YAML schema doc lives in `templates/schema.md`. Editing either
is a markdown change; this module just reads + formats them.

Why .md not Python:
  • Triple-quoted strings inline indentation, escaping rules (`{{` for
    literal `{`), and break IDE syntax highlighting.
  • Docs-as-data lets a maintainer diff the prompt without scanning
    around 360 lines of f-string brackets.
"""
from __future__ import annotations

import os
import sys
from functools import lru_cache

from .components import (
    read_formal_bases,
    read_all_components,
    read_example_specs,
    build_primitive_ops_doc,
    schema_registry_tables,
)


_TEMPLATES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "templates")


@lru_cache(maxsize=2)
def _load_template(name: str) -> str:
    """Read a markdown template file once. lru_cache keeps subsequent
    calls free even on long-running servers."""
    with open(os.path.join(_TEMPLATES_DIR, name), encoding="utf-8") as f:
        return f.read().rstrip()


def schema_doc() -> str:
    """Return the YAML schema documentation block, its registry tables
    filled from the registries (`components.schema_registry_tables`).

    The template is not `str.format`-ed — its YAML examples are full of
    literal braces — so each placeholder is replaced by name, and one the
    template no longer carries is an error rather than a table that
    silently stops reaching the prompt."""
    doc = _load_template("schema.md")
    for placeholder, table in schema_registry_tables().items():
        if placeholder not in doc:
            raise ValueError(
                f"templates/schema.md lost its {placeholder} placeholder")
        doc = doc.replace(placeholder, table)
    return doc


def forward_alignment_doc() -> str:
    """The forward-alignment standard shared by the generate / refine
    self-check and the read-only review (one text, three consumers)."""
    return _load_template("forward_alignment.md")


def source_files_section(source_files) -> str:
    """List the full source files staged for this run (see
    `overlay.stage_forward_source`) so forward-alignment citations can carry
    `file:line`. Prep refuses to start an agent without them
    (`_ground_truth.ForwardSourceUnavailable`)."""
    vllm = [p for p in source_files if "/source/vllm/" in str(p)]
    modeling = [p for p in source_files if p not in vllm]
    listed = "\n".join(f"  - `{p}`" for p in modeling)
    section = ("## Full source files (read these; cite `file:line`)\n"
               "The complete modeling / configuration files, including code the "
               "ground-truth excerpts above omit:\n" + listed + "\n")
    if vllm:
        section += (
            "\n## vLLM's implementation (reference)\n"
            "vLLM's model files for this checkpoint's architectures, with its "
            "MTP draft model where vLLM has one. Write against the modeling "
            "files above; where they have no forward for a part the checkpoint "
            "stores (an MTP layer transformers drops on load, a weight only a "
            "serving framework reads), vLLM's implementation is the source to "
            "follow. Cite it like the modeling files, by its path under "
            "`source/` (`vllm/models/<model>/nvidia/mtp.py:381`):\n"
            + "\n".join(f"  - `{p}`" for p in vllm) + "\n")
    return section


def build_system_prompt() -> str:
    """Build the reusable system context (components + ops + schema +
    examples). Cheap on subsequent calls — only the live OP_REGISTRY
    snapshot + component manifest are recomputed; the static templates
    are cached."""
    tmpl = _load_template("system.md")
    # Phase-4 validate-iteration cap. Production default is 40 (template) — a
    # generous runaway backstop; the escape hatch still tells the agent to stop
    # early once it plateaus, so a capable agent exits well before 40. A tight
    # cap (was 4) penalizes the most prompt-compliant agent (guard-ablation
    # study). Overridable via HA_SPEC_MAX_ITERS for experiments.
    _iters = (os.environ.get("HA_SPEC_MAX_ITERS") or "").strip()
    if _iters.isdigit() and _iters != "0" and _iters != "40":
        tmpl = tmpl.replace("max 40 iterations", f"max {_iters} iterations")
        tmpl = tmpl.replace("after **40\niterations**", f"after **{_iters}\niterations**")
    return tmpl.format(
        ops_doc=build_primitive_ops_doc(),
        formal_bases=read_formal_bases(),
        components=read_all_components(),
        schema_doc=schema_doc(),
        examples=read_example_specs(),
        python=sys.executable,
    )
