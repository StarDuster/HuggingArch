"""Reviewer prompt components — artifact listing + ops_doc + glossary.

Re-uses `op_workers.proposal._prompts.components` for the
registry-snapshot helpers so the reviewer sees identically what the
proposer was working against. New helper: `build_artifact_listing(op_dir)`
dumps every file in the proposal artifact directory verbatim so the
reviewer reads the whole submission in one prompt.
"""
from __future__ import annotations

from pathlib import Path

from backend.op_workers.proposal._prompts.components import (
    build_existing_ops_doc,
    read_dim_registry,
    read_shape_kind_taxonomy,
)


_ARTIFACT_ORDER = [
    "gap.json",
    "patch.py",
    "glossary.md",
    "tests.py",
    "worked_example.yaml",
    "rationale.md",
]


def build_artifact_listing(op_dir: Path | str) -> str:
    """Dump every artifact file under `op_dir/` as a fenced block.

    Files are emitted in a stable order matching the proposer's
    contract (gap → patch → glossary → tests → worked_example →
    rationale). Anything else in the directory is appended at the
    end so unexpected files surface to the reviewer instead of
    being silently dropped.
    """
    op_dir = Path(op_dir)
    if not op_dir.is_dir():
        return f"(artifact directory not found: {op_dir})"

    files = {f.name for f in op_dir.iterdir() if f.is_file()}
    ordered = [n for n in _ARTIFACT_ORDER if n in files]
    extras = sorted(files - set(_ARTIFACT_ORDER))

    chunks: list[str] = []
    for name in ordered + extras:
        path = op_dir / name
        try:
            body = path.read_text(encoding="utf-8")
        except OSError as e:
            body = f"(read failed: {e})"
        ext = path.suffix.lstrip(".") or "text"
        chunks.append(f"### `{name}`\n\n```{ext}\n{body}\n```\n")

    if not chunks:
        return f"(artifact directory is empty: {op_dir})"
    return "\n".join(chunks)


__all__ = [
    "build_artifact_listing",
    "build_existing_ops_doc",
    "read_dim_registry",
    "read_shape_kind_taxonomy",
]
