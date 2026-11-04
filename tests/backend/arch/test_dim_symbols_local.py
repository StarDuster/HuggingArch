"""A spec declares the dim symbols it introduces (`dim_symbols:`).

The rows have dims.yaml's fields and pass the constructor dims.yaml is loaded
with. Whether a spec may bind a name is asked of dims.yaml and that spec's own
rows only; the runtime projections (world, shard policy) see every live
spec's rows, for as long as the loader that loaded the spec holds it.
"""
from __future__ import annotations

import gc
import json
import threading
from pathlib import Path

import pytest

from backend.arch import dims
from backend.arch import loader as loader_mod
from backend.arch.eval.transforms import tp_divides
from backend.arch.layout.persistent import physical_weight_leaves
from backend.arch.loader import SpecLoader
from backend.fingerprint import spec_fingerprint
from backend.inference.parallel_config import ParallelConfig
from backend.inference.step_eval import evaluate_step

ARCH = Path(__file__).resolve().parents[3] / "backend" / "arch"
CONFIG = json.loads(
    (ARCH / "snapshots" / "meta-llama__Llama-3.2-1B.json").read_text())["config"]

ROWS = {
    "n_test_rows": {"kind": "vocab", "shardable_by": ["attention"],
                    "group": "per_arch_one_offs",
                    "meaning": "Rows of a test lookup table."},
    "d_test_width": {"kind": "whole", "group": "per_arch_one_offs",
                     "meaning": "Width of one test table row."},
}

_DECLARATION = """
dim_symbols:
  n_test_rows:
    kind: {kind}
{shardable}    group: per_arch_one_offs
    meaning: {meaning}
  d_test_width:
    kind: whole
    group: per_arch_one_offs
    meaning: Width of one test table row.
"""


def _model(model_id: str, *, declare: bool = True, kind: str = "vocab",
           meaning: str = "Rows of a test lookup table.") -> str:
    text = (ARCH / "models" / "llama.yaml").read_text()
    shardable = "    shardable_by: [attention]\n" if kind != "whole" else ""
    head = f"match:\n  model_id: [{model_id}]\n"
    if declare:
        head += _DECLARATION.format(kind=kind, shardable=shardable,
                                    meaning=meaning)
    text = text.replace("match:\n  model_type: [llama]\n", head)
    text = text.replace(
        "  s_max: config.max_position_embeddings\n",
        "  s_max: config.max_position_embeddings\n"
        '  n_test_rows: "1024"\n  d_test_width: "64"\n')
    return text.replace(
        "  - { name: norm, op: rmsnorm,",
        "  - { name: test_table, op: embedding, inputs: [input_ids], dims: { n_vocab: n_test_rows, "
        "d_model: d_test_width }, weight_axes: [{ world: attention, parts: "
        "n_test_rows, fit: pad }, { replicated: true }] }\n"
        "  - { name: norm, op: rmsnorm,")


def _overlay(tmp_path: Path, **models: str) -> Path:
    root = tmp_path / "models"
    root.mkdir(parents=True, exist_ok=True)
    for stem, text in models.items():
        (root / f"{stem}.yaml").write_text(text)
    return root


@pytest.fixture(autouse=True)
def _no_leaked_layers():
    yield
    gc.collect()
    assert "n_test_rows" not in dims.symbols()


# -- the rows ----------------------------------------------------------------

def test_a_row_is_checked_by_the_registry_constructor():
    with pytest.raises(ValueError, match=r"dim_symbols\.n_x: .*worlds: \["):
        dims.local_symbols({"n_x": {"kind": "head_count",
                                    "shardable_by": ["nowhere"],
                                    "group": "per_arch_one_offs",
                                    "meaning": "x"}})
    with pytest.raises(ValueError, match=r"groups: \["):
        dims.local_symbols({"n_x": {"kind": "whole", "group": "nope",
                                    "meaning": "x"}})
    with pytest.raises(ValueError, match="kind 'count'"):
        dims.local_symbols({"n_x": {"kind": "count", "group": "per_arch_one_offs",
                                    "meaning": "x"}})


def test_a_registered_name_is_refused_with_its_meaning():
    with pytest.raises(ValueError, match=r"'n_vocab' is registered.*Vocab size"):
        dims.local_symbols({"n_vocab": dict(ROWS["d_test_width"])})


def test_rows_say_nothing_the_engine_or_promotion_writes():
    for field in ("used_by", "engine_supplied"):
        with pytest.raises(ValueError, match="remove"):
            dims.local_symbols({"n_x": {**ROWS["d_test_width"], field: "x"}})


def test_a_derivation_names_only_known_symbols():
    dims.local_symbols({"n_x": {**ROWS["d_test_width"],
                                "derived": "n_layers * d_test_width"},
                        "d_test_width": ROWS["d_test_width"]})
    with pytest.raises(ValueError, match=r"\['n_nowhere'\]"):
        dims.local_symbols({"n_x": {**ROWS["d_test_width"],
                                    "derived": "n_nowhere + 1"}})


def test_a_row_is_derived_or_defaulted_never_both():
    """`derived:` is a name no spec writes, derived again in every frame;
    `default:` fills a name a spec may write. A row saying both would leave
    the loader's refusal and the frame's re-derivation to disagree about it —
    in dims.yaml (the constructor) and in a spec's own rows alike."""
    with pytest.raises(ValueError, match="both `derived:` and `default:`"):
        dims.DimSymbol(name="n_x", kind="whole", group="per_arch_one_offs",
                       meaning="x", derived="n_layers", default="1")
    with pytest.raises(ValueError, match="both `derived:` and `default:`"):
        dims.local_symbols({"n_x": {**ROWS["d_test_width"],
                                    "derived": "n_layers", "default": "1"}})
    for field in ("derived", "default"):
        assert getattr(dims.local_symbols(
            {"n_x": {**ROWS["d_test_width"], field: "n_layers"}})["n_x"],
            field) == "n_layers"


def test_a_name_a_law_cannot_read_as_an_axis_is_refused():
    with pytest.raises(ValueError, match="not usable as an axis name"):
        dims.local_symbols({"beta": dict(ROWS["d_test_width"])})


# -- layers ------------------------------------------------------------------

def test_a_layer_is_runtime_only_and_leaves_the_base_alone():
    misses = dims._load_base.cache_info().misses
    labels = dims.world_by_label()
    token = dims.declare_local("a", ROWS)
    try:
        assert dims.get("n_test_rows").shardable_by == ("attention",)
        assert "n_test_rows" in tp_divides()   # read after import, sees it
        assert "n_test_rows" not in dims.registered_for(None)
        assert "n_test_rows" in dims.registered_for(ROWS)
        assert "n_test_rows" not in dims.meanings()
        assert "n_test_rows" not in dims.render_markdown()
        assert dims.world_by_label() is labels
    finally:
        dims.withdraw_local(token)
    assert dims._load_base.cache_info().misses == misses
    assert dims.get("n_test_rows") is None


def test_two_layers_may_word_a_symbol_differently_but_not_count_it_differently():
    first = dims.declare_local("a", ROWS)
    try:
        second = dims.declare_local("b", {
            "n_test_rows": {**ROWS["n_test_rows"], "meaning": "other prose"}})
        dims.withdraw_local(second)
        with pytest.raises(ValueError, match="classifies 'n_test_rows' differently") as e:
            dims.declare_local("b", {"n_test_rows": {
                "kind": "whole", "group": "per_arch_one_offs",
                "meaning": "private wording"}})
        assert "Rows of a test lookup table" not in str(e.value)
    finally:
        dims.withdraw_local(first)


def test_conflicting_declarations_racing_install_one():
    outcomes: list = []
    barrier = threading.Barrier(8)

    def declare(i: int) -> None:
        row = (ROWS["n_test_rows"] if i % 2 else
               {"kind": "whole", "group": "per_arch_one_offs", "meaning": "w"})
        barrier.wait()
        try:
            outcomes.append(("ok", row["kind"], dims.declare_local(str(i), {"n_test_rows": row})))
        except ValueError:
            outcomes.append(("refused", row["kind"], None))

    threads = [threading.Thread(target=declare, args=(i,)) for i in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    kinds = {kind for status, kind, _ in outcomes if status == "ok"}
    for _, _, token in outcomes:
        dims.withdraw_local(token)
    assert len(kinds) == 1


# -- through the loader -------------------------------------------------------

def test_a_declared_symbol_shards_as_its_row_says(tmp_path):
    root = _overlay(tmp_path, test__a=_model("test/a"))
    loader = SpecLoader(overlay_model_dirs=[str(root)])
    resolved = loader.resolve(loader.get("test__a"), CONFIG)

    def table_bytes(tp: int) -> int:
        analysis = evaluate_step(
            resolved, ParallelConfig(tp_attention=tp, tp_dense_ffn=tp,
                                     tp_moe_expert=tp),
            phase="prefill", batch=1, input_tokens=16, output_tokens=0).analysis
        leaves = physical_weight_leaves(analysis)
        uid = next(n.uid for n in analysis.program.nodes()
                   if n.name == "test_table")
        return leaves[uid].resident_data_bytes_per_rank

    assert table_bytes(1) == 1024 * 64 * 2
    assert table_bytes(2) == table_bytes(1) // 2


def test_a_spec_cannot_bind_another_specs_symbol(tmp_path):
    root = _overlay(tmp_path, test__a=_model("test/a"), test__b=_model("test/b", declare=False))
    loader = SpecLoader(overlay_model_dirs=[str(root)])
    loader.resolve(loader.get("test__a"), CONFIG)
    # test__b binds n_test_rows; only test__a's layer classifies it.
    with pytest.raises(ValueError, match=r"\['n_test_rows', 'd_test_width'\].*dim_symbols"):
        loader.resolve(loader.get("test__b"), CONFIG)
    del loader


def test_the_layers_leave_with_their_loader(tmp_path):
    root = _overlay(tmp_path, test__a=_model("test/a"))
    loader = SpecLoader(overlay_model_dirs=[str(root)])
    assert dims.get("n_test_rows") is not None
    del loader
    gc.collect()
    assert dims.get("n_test_rows") is None


def test_a_rolled_back_overlay_takes_its_layer_back(tmp_path):
    overlay = tmp_path / "overlay"
    models = overlay / "models"
    models.mkdir(parents=True)
    (models / "test__a.yaml").write_text(_model("test/a"))
    (models / "test__z.yaml").write_text("match: {model_id: [test/z]}\nmodel: 3\n")
    loader = SpecLoader()
    assert loader.attach_overlay(str(overlay)) is False
    assert dims.get("n_test_rows") is None
    del loader


def test_a_library_model_declares_none(tmp_path, monkeypatch):
    root = _overlay(tmp_path, test__a=_model("test/a"))
    monkeypatch.setattr(loader_mod, "_LIBRARY_MODELS_DIR", str(root))
    loader = SpecLoader(overlay_model_dirs=[str(root)])
    assert loader.get("test__a") is None
    assert any("library model declares no `dim_symbols`" in f[-1]
               for f in loader._model_load_failures)


def test_an_unvalued_declaration_is_refused(tmp_path):
    text = _model("test/a").replace('  n_test_rows: "1024"\n', "")
    loader = SpecLoader(overlay_model_dirs=[str(_overlay(tmp_path, test__a=text))])
    assert loader.get("test__a") is None
    assert any("declared but never given a value" in f[-1]
               for f in loader._model_load_failures)


def test_the_fingerprint_covers_the_classification(tmp_path):
    one = _overlay(tmp_path / "1", test__a=_model("test/a"))
    two = _overlay(tmp_path / "2", test__a=_model(
        "test/a", meaning="Rows of a test lookup table, reworded."))
    first = spec_fingerprint(SpecLoader(overlay_model_dirs=[str(one)]).get("test__a"))
    gc.collect()
    second = spec_fingerprint(SpecLoader(overlay_model_dirs=[str(two)]).get("test__a"))
    assert first != second


# -- one spec's symbols never reach another's outputs -----------------------------

def test_a_derived_symbol_stays_in_the_spec_that_declares_it(tmp_path):
    derived = _model("test/a").replace(
        "    meaning: Width of one test table row.\n",
        "    meaning: Width of one test table row.\n"
        "  n_test_twice:\n"
        "    kind: whole\n"
        "    group: per_arch_one_offs\n"
        "    meaning: Twice the layer count.\n"
        "    derived: n_layers * 2\n", 1)
    root = _overlay(tmp_path, test__a=derived, test__b=_model("test/b"))
    loader = SpecLoader(overlay_model_dirs=[str(root)])
    a = loader.resolve(loader.get("test__a"), CONFIG)
    b = loader.resolve(loader.get("test__b"), CONFIG)
    assert a.dim_values["n_test_twice"] == 2 * a.dim_values["n_layers"]
    assert "n_test_twice" not in b.dim_values
    del loader, a, b


def test_a_custom_op_parameter_is_checked_against_its_own_specs_symbols():
    from backend.arch.ops_base import reject_divisible_parameter_names
    from backend.arch.schema.custom_ops import CustomOpSpec
    op = CustomOpSpec(signature={"x": ["B", "n_test_rows"]}, shape_kind="preserve",
                      couples_on=[], description="fixture")
    token = dims.declare_local("another spec", ROWS)
    try:
        reject_divisible_parameter_names("op", op)        # not this spec's
        with pytest.raises(ValueError, match="n_test_rows"):
            reject_divisible_parameter_names("op", op, local=ROWS)
    finally:
        dims.withdraw_local(token)


def test_a_spec_that_fails_to_lower_holds_no_layer(tmp_path):
    broken = _model("test/a").replace("\ndim_symbols:\n",
                                      "\ndrafts: [no_such_draft]\ndim_symbols:\n", 1)
    loader = SpecLoader(overlay_model_dirs=[str(_overlay(tmp_path, test__a=broken))])
    assert loader.get("test__a") is None
    assert dims.get("n_test_rows") is None
    del loader
