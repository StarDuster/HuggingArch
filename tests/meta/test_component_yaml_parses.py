"""Every checked-in component YAML must parse.

A malformed component library file is read by two consumers —
`backend.arch.op_names` (feeds `universe()`) and the spec-writer manifest
(`backend.spec_worker._prompts.components`). If it silently fails to parse,
the component vanishes from both: a spec that references it then looks like
it uses an *unknown op*, with the real cause (a corrupt file) hidden.

Gate the invariant at the source so corruption is loud in CI, not silently
absorbed at runtime.
"""
import glob
import pathlib
import re

import pytest
import yaml

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
_COMPONENTS = sorted(
    glob.glob(str(_REPO_ROOT / "backend/arch/components/**/*.yaml"), recursive=True)
)


def test_there_are_components_to_check():
    # Guard against a path regression silently emptying the parametrization.
    assert _COMPONENTS, "no component YAMLs found — has the components dir moved?"


@pytest.mark.parametrize(
    "path", _COMPONENTS, ids=[pathlib.Path(p).name for p in _COMPONENTS]
)
def test_component_yaml_parses(path):
    yaml.safe_load(pathlib.Path(path).read_text())


_DOCUMENTS = sorted(
    glob.glob(str(_REPO_ROOT / "backend/**/*.yaml"), recursive=True)
    + glob.glob(str(_REPO_ROOT / "backend/**/*.yml"), recursive=True)
)


def _as_written(value):
    """A parsed document with every mapping's written order made part of
    its value."""
    if isinstance(value, dict):
        return [("mapping", [(k, _as_written(v)) for k, v in value.items()])]
    if isinstance(value, list):
        return [_as_written(v) for v in value]
    return (type(value).__name__, value)


@pytest.mark.parametrize(
    "path", _DOCUMENTS,
    ids=[str(pathlib.Path(p).relative_to(_REPO_ROOT)) for p in _DOCUMENTS],
)
def test_libyaml_reads_each_document_as_written(path):
    """The backend reads its YAML through libyaml (`yaml_load.load_yaml`):
    each document it ships reads to the same values, in the same written
    order, as under PyYAML's own reader."""
    from backend.arch.yaml_load import load_yaml

    text = pathlib.Path(path).read_text()
    assert _as_written(load_yaml(text)) == _as_written(yaml.safe_load(text))


def _nodes(node):
    """A composed document's nodes as a splice reads them: kind, tag,
    marks, style and flow style (a plain scalar's style and a block
    collection's flow style are None in one reader, '' and False in the
    other: each is false in both) and value."""
    out = []
    pending = [node] if node is not None else []
    while pending:
        n = pending.pop()
        out.append((type(n).__name__, n.tag,
                    (n.start_mark.index, n.start_mark.line, n.start_mark.column),
                    (n.end_mark.index, n.end_mark.line, n.end_mark.column),
                    getattr(n, "style", None) or None,
                    getattr(n, "flow_style", None) or None,
                    n.value if isinstance(n, yaml.ScalarNode) else None))
        if isinstance(n, yaml.SequenceNode):
            pending.extend(reversed(n.value))
        elif isinstance(n, yaml.MappingNode):
            pending.extend(x for pair in reversed(n.value) for x in reversed(pair))
    return out


@pytest.mark.parametrize(
    "path", _DOCUMENTS,
    ids=[str(pathlib.Path(p).relative_to(_REPO_ROOT)) for p in _DOCUMENTS],
)
def test_libyaml_composes_each_document_as_written(path):
    """A written spec spliced back reads its nodes through libyaml
    (`yaml_load.compose_yaml`): the same nodes, tags and marks as PyYAML's
    own reader composes."""
    from backend.arch.yaml_load import compose_yaml

    # The text as the splice reads it: ended by a line break.
    text = pathlib.Path(path).read_text()
    text = text if text.endswith("\n") else text + "\n"
    assert _nodes(compose_yaml(text)) == _nodes(
        yaml.compose(text, Loader=yaml.SafeLoader))


_YAML_READS = re.compile(
    r"\b_?yaml\.(?:safe_load|load|load_all|safe_load_all|full_load|"
    r"full_load_all|unsafe_load|unsafe_load_all|compose|compose_all|parse|"
    r"scan)\s*\("
    r"|\b_?yaml\.\w*Loader\b|\bSafeConstructor\b"
    r"|^\s*from\s+yaml(?:\.\w+)?\s+import\b.*\b(?:\w*load\w*|compose\w*|"
    r"\w*Loader|\w*Constructor)\b", re.M)


def test_the_backend_reads_yaml_only_through_yaml_load():
    """Every YAML document the backend reads goes through
    `backend/arch/yaml_load.py`: no other module of `backend/` loads,
    composes or constructs YAML itself."""
    own = _REPO_ROOT / "backend" / "arch" / "yaml_load.py"
    offenders = [
        f"{path.relative_to(_REPO_ROOT)}:{text[:m.start()].count(chr(10)) + 1}"
        for path in sorted((_REPO_ROOT / "backend").rglob("*.py"))
        if path != own
        for text in (path.read_text(encoding="utf-8"),)
        for m in _YAML_READS.finditer(text)]
    assert offenders == []
