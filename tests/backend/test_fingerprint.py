"""Which edits move which model's cached results (`result_version`).

A cached result for one model is keyed on the engine code, the data every
model shares, and the data that model can read (`ModelFootprint`): its spec
with the components it binds, the decode twin its spec names, the fusion
plans that can match its graph and the binding entries for them. These tests
make real edits on a copy of the engine tree and check that each moves
exactly the models that read it, and that what the footprint leaves out is
never read: every op of every program fusion is matched against is in the
footprint's vocabulary.
"""
from __future__ import annotations

import shutil
from pathlib import Path

import pytest

import backend.arch.eval.transforms as transforms
import backend.fingerprint as fp
from backend.fingerprint import engine_version, normalized_py_source
from backend.arch.loader import SpecLoader
from backend.inference.framework_bindings import FRAMEWORK_BINDINGS, ServingDeployment
from backend.inference.fusion_plans import _load_fusion_plans_dir
from backend.inference.parallel_config import ParallelConfig
from backend.inference.pipeline import analyze
from tests.snapshots import resolve_snapshot, shared_loader, snapshot_config

DEEPSEEK = "deepseek-ai/DeepSeek-V3"   # MLA with a decode twin, MoE router
KIMI = "moonshotai/Kimi-K2-Instruct"   # MLA, the same library component
QWEN = "Qwen/Qwen3-8B"                 # dense GQA
MODELS = (DEEPSEEK, KIMI, QWEN)


class _Engine:
    """A copy of `backend/{arch,inference}` the fingerprint reads instead of
    the checkout, with the loader and registries pointed at it."""

    def __init__(self, root: Path, monkeypatch: pytest.MonkeyPatch):
        self.root = root
        for sub in ("arch", "inference"):
            shutil.copytree(Path(fp._ROOT) / sub, root / sub, ignore=shutil.ignore_patterns(
                "__pycache__", "snapshots"))
        self.monkeypatch = monkeypatch
        monkeypatch.setattr(fp, "_ROOT", root)
        monkeypatch.setattr(fp, "_FRAMEWORK_BINDINGS_DIR",
                            str(root / "inference" / "framework_bindings"))
        self.reload()

    def reload(self) -> None:
        """Re-read the copy: the loader, the plans, every memo."""
        for cached in (fp.engine_code_version, fp.shared_data_version,
                       fp._code_chunks, fp._binding_entries):
            cached.cache_clear()
        fp._FOOTPRINTS.clear()
        self.loader = SpecLoader(base_dir=str(self.root / "arch"))
        self.monkeypatch.setattr(transforms, "get_default_loader",
                                 lambda: self.loader)
        self.monkeypatch.setattr(fp, "FUSION_SCHEMES", _load_fusion_plans_dir(
            str(self.root / "inference" / "fusion_plans")))

    def edit(self, rel: str, old: str, new: str) -> None:
        path = self.root / rel
        text = path.read_text(encoding="utf-8")
        assert old in text, (rel, old)
        path.write_text(text.replace(old, new, 1), encoding="utf-8")
        self.reload()

    def versions(self) -> dict:
        out = {}
        for model in MODELS:
            resolved = self.loader.match(snapshot_config(model), model_id=model)
            out[model] = fp.result_version(resolved)
        return out


@pytest.fixture
def engine(tmp_path, monkeypatch):
    yield _Engine(tmp_path / "backend", monkeypatch)
    for cached in (fp.engine_code_version, fp.shared_data_version,
                   fp._code_chunks, fp._binding_entries):
        cached.cache_clear()
    fp._FOOTPRINTS.clear()


def _moved(before: dict, after: dict) -> set:
    return {model for model in MODELS if before[model] != after[model]}


def test_a_component_edit_moves_only_the_models_that_bind_it(engine):
    before = engine.versions()
    engine.edit("arch/components/library/mla_attention.yaml",
                "description: \"Multi-head Latent Attention",
                "description: \"Multi-head latent attention")
    assert _moved(before, engine.versions()) == {DEEPSEEK, KIMI}


def test_a_comment_edit_moves_nothing(engine):
    before = engine.versions()
    engine.edit("arch/components/library/mla_attention.yaml",
                "description:", "# a note\ndescription:")
    engine.edit("inference/fusion_plans/moe_fused_gate.yaml",
                "fused_op:", "# a note\nfused_op:")
    assert _moved(before, engine.versions()) == set()


def test_a_decode_twin_edit_moves_the_models_whose_spec_names_it(engine):
    """The absorbed MLA is not among the components a spec binds: it is
    fetched from the library at decode, by the name the spec declares."""
    before = engine.versions()
    engine.edit("arch/components/library/mla_attention_absorbed.yaml",
                "description: \"MLA decode-time absorbed form.",
                "description: \"MLA absorbed form at decode.")
    assert _moved(before, engine.versions()) == {DEEPSEEK, KIMI}


def test_a_plan_only_a_router_matches_moves_only_routed_models(engine):
    before = engine.versions()
    engine.edit("inference/fusion_plans/moe_fused_gate.yaml",
                "    topk_scale:     { op: scale_const,       inputs: [topk_norm.out] }\n",
                "")
    engine.edit("inference/fusion_plans/moe_fused_gate.yaml",
                "output: topk_scale.out", "output: topk_norm.out")
    assert _moved(before, engine.versions()) == {DEEPSEEK, KIMI}


def test_a_binding_entry_edit_moves_only_the_models_its_plan_matches(engine):
    before = engine.versions()
    engine.edit("inference/framework_bindings/vllm.yaml",
                "source_anchor: \"vllm._custom_ops.grouped_topk",
                "source_anchor: \"vllm grouped_topk")
    assert _moved(before, engine.versions()) == {DEEPSEEK, KIMI}


def test_shared_data_and_code_edits_move_every_model(engine):
    before = engine.versions()
    engine.edit("arch/roles.yaml", "kind:", "kind: ")
    shared = engine.versions()
    assert _moved(before, shared) == set(MODELS)
    engine.edit("arch/eval/cache_view.py", "return n * self.scale",
                "return self.scale * n")
    assert _moved(shared, engine.versions()) == set(MODELS)


@pytest.mark.parametrize("model_id", [
    DEEPSEEK,
    "zai-org/GLM-5.2",                    # DSA, MTP
    "Qwen/Qwen3.5-35B-A3B",               # gated delta net + full attention, MTP
    "Qwen/Qwen3-VL-235B-A22B-Instruct",   # vision tower, deepstack
    # KDA, attention residuals; minutes-long
    pytest.param("moonshotai/Kimi-K3", marks=pytest.mark.regression),
    # compressed attention, HC streams; minutes-long
    pytest.param("deepseek-ai/DeepSeek-V4-Flash", marks=pytest.mark.regression),
    "openai/gpt-oss-20b",                 # sliding window with sinks
])
def test_every_op_fusion_meets_is_in_the_vocabulary(model_id):
    """Plans the footprint leaves out are plans none of whose node ops can
    appear. That holds only if the vocabulary covers every leaf of the program
    fusion is matched against, in both phases, unsharded and sharded."""
    resolved = resolve_snapshot(model_id, shared_loader())
    vocabulary = fp.ModelFootprint(resolved).ops
    binding = FRAMEWORK_BINDINGS["vllm"].deployed(
        ServingDeployment.at(gpu_sku="H100-SXM", activation_dtype="BF16"))
    for parallel in ({}, {"tp_attention": 8, "tp_dense_ffn": 8,
                          "tp_moe_expert": 8}, {"ep_moe": 8, "dp_attention": 8},
                     {"pp": 2}):
        for phase in ("prefill", "decode"):
            result = analyze(resolved, parallel=ParallelConfig(**parallel),
                             framework_binding=binding, batch_size=2,
                             avg_input_tokens=64, avg_output_tokens=0, phase=phase, breakdown=True)
            seen = {node.op for node in result.pre_program.nodes()
                    if node.op and not node.children}
            assert seen <= vocabulary, (parallel, phase, seen - vocabulary)


def test_every_file_left_to_the_footprint_is_one_it_reads():
    """The data directories `ModelFootprint.DATA_DIRS` carves out of the
    shared hash hold only what the footprint hashes per model: specs,
    components and their claims, fusion plans, bindings. A registry file
    dropped among them would reach a model without moving its version.
    A component's claims are looked up by identity whether it is a library
    component, a draft (it reaches only a spec that lists it under
    `drafts:`, whose `ArchSpec` then carries it) or a block written inline
    in a model spec (`spec_parsing._attach_claims`); the spec carries each."""
    from backend.arch.loader import get_default_loader
    from backend.inference.fusion_plans import FUSION_SCHEMES
    loader = get_default_loader()
    components = (loader.component_names() | frozenset(loader._draft_lib)
                  | {name for spec in loader.all_specs().values()
                     for name in (spec.blocks or {})})
    for path in fp._engine_files((".yaml", ".yml", ".csv")):
        rel = path.relative_to(fp._ROOT)
        if rel.parts[1] not in fp.ModelFootprint.DATA_DIRS:
            continue
        where = rel.parts[1:-1]
        if where[0] in ("models", "snapshots"):
            assert path.suffix == ".yaml" and len(where) == 1, rel
        elif where[0] in ("components", "claims"):
            assert path.stem in components, rel
        elif where[0] == "fusion_plans":
            assert path.stem in FUSION_SCHEMES, rel
        else:
            assert path.stem in FRAMEWORK_BINDINGS, rel


# The version stamps every result cache's key folds (moved with the cache
# machinery to backend.measure.cache; the stamps stay here).
def test_engine_version_is_a_stable_hex():
    engine_version.cache_clear()
    v1 = engine_version()
    v2 = engine_version()
    assert v1 == v2 and len(v1) == 16 and int(v1, 16) >= 0


_SRC = '''"""Module docstring with a typo: teh engine."""


def f(x):
    """Compute the thing."""  # trailing comment
    # a comment with a tpyo
    y = x + 1  # inline comment
    return y
'''


def test_normalized_source_ignores_comments_docstrings_formatting():
    base = normalized_py_source(_SRC)
    # comment typo fix + trailing whitespace + extra blank lines
    cosmetic = _SRC.replace("tpyo", "typo").replace(
        "y = x + 1", "y = x + 1   \n\n")
    assert normalized_py_source(cosmetic) == base
    # docstring edit only
    docs = _SRC.replace("teh engine", "the engine").replace(
        "Compute the thing.", "Compute the thing, faster.")
    assert normalized_py_source(docs) == base


def test_normalized_source_moves_on_semantic_change():
    base = normalized_py_source(_SRC)
    assert normalized_py_source(_SRC.replace("x + 1", "x + 2")) != base
    assert normalized_py_source(_SRC.replace("def f(", "def g(")) != base
    # a string LITERAL is semantics, not a comment
    assert normalized_py_source(_SRC + "\nK = 'v'\n") != base


def test_normalized_source_falls_back_on_unparseable():
    broken = "def f(:\n"
    out = normalized_py_source(broken)
    assert out.startswith("\x00raw\x00") and broken in out


def test_normalized_source_preserves_multiline_string_content():
    """Lines inside a multi-line string literal are VALUE, not formatting —
    an interior blank line or trailing whitespace must move the hash
    (dual-review finding: the first version rstripped/dropped them, so a
    literal edit silently kept the old cache key)."""
    a = normalized_py_source('X = """a\n\nb"""\n')
    b = normalized_py_source('X = """a\nb"""\n')
    assert a != b                       # interior blank line is content
    c = normalized_py_source('X = """a\n\nc"""\n')
    assert a != c                       # interior text is content
    d = normalized_py_source('X = """a  \n\nb"""\n')
    assert a != d                       # trailing ws on a literal's first line
    # …while a SINGLE-line string's surroundings still normalize away
    e = normalized_py_source('X = "a"  # note\n')
    f = normalized_py_source('X = "a"\n')
    assert e == f


def test_normalized_source_finds_docstrings_however_deeply_nested():
    """A docstring is stripped wherever a def or class can appear — nested in
    another def, in a class, and inside an `if` / `try` / `with` / `match`
    body. `_docstring_spans` reaches them by walking STATEMENTS only (no
    statement lives inside an expression, so expression subtrees are skipped
    and can't hold one); this is the test that the statement walk reaches
    every holder the full node walk did."""
    nested = (
        'def outer():\n'
        '    """OUTER"""\n'
        '    def inner():\n'
        '        """INNER"""\n'
        '    class C:\n'
        '        """KLASS"""\n'
        '        async def m(self):\n'
        '            """METHOD"""\n'
        '    if flag:\n'
        '        def branch():\n'
        '            """BRANCH"""\n'
        '    try:\n'
        '        def tried():\n'
        '            """TRIED"""\n'
        '    except E:\n'
        '        def caught():\n'
        '            """CAUGHT"""\n'
        '    finally:\n'
        '        with open(p) as h:\n'
        '            def held():\n'
        '                """HELD"""\n'
        '    match flag:\n'
        '        case 1:\n'
        '            def cased():\n'
        '                """CASED"""\n'
        '    return inner\n'
    )
    out = normalized_py_source(nested)
    for word in ("OUTER", "INNER", "KLASS", "METHOD", "BRANCH", "TRIED",
                 "CAUGHT", "HELD", "CASED"):
        assert word not in out, f"{word} survived normalization"
    # …and the code around them did not get blanked with them.
    for kept in ("def outer():", "class C:", "return inner", "case 1:"):
        assert kept in out
    # A bare string expression that is NOT the first statement of its suite
    # is a value, not a docstring — and a string inside an expression is too.
    assert "KEPT" in normalized_py_source(
        'def f():\n    x = 1\n    "KEPT"\n    return x\n')
    assert "KEPT" in normalized_py_source('@d(lambda: "KEPT")\ndef f():\n    """D"""\n')


def test_normalized_source_non_ascii_docstring_keeps_neighbor_code():
    """ast col offsets are UTF-8 BYTE offsets; the blanking loop works in
    CHARS. Without the conversion, a non-ASCII docstring over-blanks and
    erases the semantic statement beside it from the hash."""
    base = normalized_py_source('def f():\n    """héllo"""\n    x = 1\n    return x\n')
    assert "x = 1" in base
    changed = normalized_py_source('def f():\n    """héllo"""\n    x = 2\n    return x\n')
    assert changed != base              # the code edit is visible in the hash
    # and the docstring edit still isn't
    docs = normalized_py_source('def f():\n    """héllo!"""\n    x = 1\n    return x\n')
    assert docs == base
