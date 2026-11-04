"""An IR leaf carries its op declaration; consumers read it, not the registry.

`SpecIRNode.declaration` is resolved once when the tree is built — the
spec's custom op first, else the registered primitive. Rendering
(`_node_build`, `_shapes`) and phases read the field; a leaf whose name is
neither is refused at build, not rendered by a lookup that returned None.
"""
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from backend.arch.loader import SpecLoader
from backend.arch.ops import OP_REGISTRY, OpSpec
from backend.arch.schema.custom_ops import CustomOpSpec
from backend.arch.spec_tree import iter_tree
from tests.snapshots import resolve_snapshot

MODELS = PROJECT_ROOT / "backend" / "arch" / "models"


def _walk(nodes):
    for n in nodes:
        yield n
        yield from _walk(n.children or [])


def _all_nodes(resolved):
    tree = iter_tree(resolved)
    yield from _walk(tree.top_level)
    for _meta, nodes in tree.blocks:
        yield from _walk(nodes)


def test_primitive_leaves_carry_the_registry_object_itself():
    r = resolve_snapshot("deepseek-ai/DeepSeek-V3")
    seen = 0
    for n in _all_nodes(r):
        if n.is_component:
            assert n.declaration is None, n.display_path
        elif isinstance(n.declaration, OpSpec):
            assert n.declaration is OP_REGISTRY[n.op], n.display_path
            seen += 1
    assert seen > 100


def test_custom_op_leaves_carry_the_spec_declaration():
    # gpt-oss's packed-expert customs: a spec with custom ops AND a snapshot.
    r = resolve_snapshot("openai/gpt-oss-20b")
    assert r.spec.custom_ops, "gpt-oss no longer declares custom ops — pick a spec that does"
    customs = [n for n in _all_nodes(r) if isinstance(n.declaration, CustomOpSpec)]
    assert customs, "a spec with custom ops has no custom leaf"
    for n in customs:
        assert n.declaration is r.spec.custom_ops[n.op]


def test_a_leaf_naming_no_declaration_is_refused_at_build(tmp_path):
    text = (MODELS / "llama.yaml").read_text()
    assert "op: rmsnorm" in text
    (tmp_path / "llama.yaml").write_text(text.replace("op: rmsnorm", "op: frobnicate_norm", 1))
    loader = SpecLoader()
    # The tree builder's refusal, not an earlier schema rejection.
    with pytest.raises(ValueError, match="is neither a registered primitive"):
        spec = loader.load_model_spec(str(tmp_path / "llama.yaml"))
        cfg = resolve_snapshot("meta-llama/Llama-3.2-1B").config
        iter_tree(loader.resolve(spec, cfg))
