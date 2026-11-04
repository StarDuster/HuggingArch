"""Prompt-construction subsystem.

The system prompt + per-task body builders live here. Public entry points
(`prepare_*_prompt`, `build_custom*_prompt`) are re-exported from the parent
`spec_worker` package; this sub-package's modules are internal.

Layout:
  components.py — discover + summarize component library + example specs
                  + primitive op registry doc
  system.py     — reusable system prompt (template + .md content)
  tasks.py      — per-task bodies (generate / refine / custom) + the
                  public build_*_prompt and prepare_*_prompt entry points
"""
from .components import (
    read_all_components,
    read_example_specs,
    build_primitive_ops_doc,
)
from .system import build_system_prompt
from .tasks import (
    build_custom_prompt,
    build_custom_refine_prompt,
    prepare_generate_prompt,
    prepare_refine_prompt,
    prepare_custom_generate_prompt,
    prepare_custom_refine_prompt,
    build_generate_task_body,
    build_refine_task_body,
    build_unknown_structure_guidance,
)

__all__ = [
    "build_custom_prompt",
    "build_custom_refine_prompt",
    "build_generate_task_body",
    "build_primitive_ops_doc",
    "build_refine_task_body",
    "build_system_prompt",
    "build_unknown_structure_guidance",
    "prepare_custom_generate_prompt",
    "prepare_custom_refine_prompt",
    "prepare_generate_prompt",
    "prepare_refine_prompt",
    "read_all_components",
    "read_example_specs",
]
