"""Self-register a spec under its exact model_id.

The loader resolves a checkpoint to its spec by ``model_id`` — an exact lookup,
no ``model_type`` scoring (see ``backend.arch.loader.SpecLoader.match``). A spec
authored with only ``match.model_type`` therefore never resolves on its own: the
id index has no entry for it. This stamps ``match.model_id: ["<id>"]`` in when
the author declared none, pinning the spec to THIS model rather than a same-
``model_type`` sibling (Pro vs Flash, a model vs its mirrors).

Pure, idempotent text transform. It lives in this leaf module (no intra-package
deps) so every producer/consumer of a stored spec_yaml shares ONE implementation:
the spec-worker at generate time, the overlay writer that seeds review / analyze /
refine / fix runs, and the service layer that pushes a contribution PR. Keeping
one copy is what stops the "generate stamps but review/PR don't" drift that let
old ``spec_generations`` rows fail to resolve on review.
"""
from __future__ import annotations

from backend.arch.yaml_load import load_yaml
import yaml


def stamp_match_model_id(spec_yaml_text: str, model_id: str) -> str:
    """Inject ``match.model_id: ["<id>"]`` when the spec declares none.

    Comment- and format-preserving: a single line added inside the top-level
    ``match:`` block (after ``model_type`` when present, else as its first key),
    and only when no ``model_id:`` is already there. Respects an author-declared
    ``model_id`` and leaves inline / unparseable match blocks untouched. Never
    returns YAML that fails to re-parse — falls back to the original.
    """
    try:
        parsed = load_yaml(spec_yaml_text)
    except yaml.YAMLError:
        return spec_yaml_text
    if not isinstance(parsed, dict):
        return spec_yaml_text                    # non-mapping YAML — leave it
    match = parsed.get("match")
    if not isinstance(match, dict) or match.get("model_id"):
        return spec_yaml_text
    out, in_match, inserted = [], False, False
    for line in spec_yaml_text.splitlines(keepends=True):
        if inserted:
            out.append(line)
            continue
        stripped = line.strip()
        if stripped == "match:" and line[:1] not in (" ", "\t"):
            out.append(line if line.endswith("\n") else line + "\n")
            in_match = True
            # No model_type to anchor after (e.g. a match block keyed on some
            # other field): insert model_id as the block's first child now, so a
            # match block without model_type still gets pinned.
            if not match.get("model_type"):
                out.append(f'  model_id: ["{model_id}"]\n')
                inserted = True
        elif in_match and stripped.startswith("model_type:"):
            indent = line[:len(line) - len(line.lstrip())]
            # Guarantee the anchor line ends cleanly, so a file whose LAST line
            # is `model_type:` (no trailing newline) doesn't merge the two into
            # broken YAML that would then feed the loader directly.
            out.append(line if line.endswith("\n") else line + "\n")
            out.append(f'{indent}model_id: ["{model_id}"]\n')
            inserted = True
        else:
            out.append(line)
    if not inserted:
        return spec_yaml_text
    stamped = "".join(out)
    # Safety net: never hand downstream YAML we can't parse back. If the edit
    # somehow produced invalid YAML, keep the original untouched.
    try:
        load_yaml(stamped)
    except yaml.YAMLError:
        return spec_yaml_text
    return stamped
