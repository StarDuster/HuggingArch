"""`python -m tools.corpus_admit verify`, the corpus-pin check every CI lane
runs: missing keys, orphan keys, README counts; and with `--changed`, the
changed models measured against the files."""
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from tools.corpus_pins import verify
from tools.corpus_pins.corpus import Corpus, CorpusModel
from tools.corpus_pins.pins.checkpoint_reconciliation import CHECKPOINT_RECONCILIATION
from tools.corpus_pins.pins.readme_counts import README_COUNTS, counts
from tools.corpus_pins.pins.sol_surface import SOL_SURFACE
from tools.corpus_pins.pins.spec_match import SPEC_MATCH

_STEM = "meta-llama__Llama-3.2-1B"


def test_the_tree_verifies():
    found = verify.verify()
    assert found == {"keys": [], "aggregate": []}


def test_a_snapshot_without_its_key_is_missing():
    doc = SPEC_MATCH.load()
    del doc[_STEM]
    assert SPEC_MATCH.missing_keys(doc, CorpusModel(_STEM)) == [_STEM]
    sol = {k: v for k, v in SOL_SURFACE.load().items() if not k.startswith(_STEM + ".json|")}
    assert SOL_SURFACE.missing_keys(sol, CorpusModel(_STEM)) == [
        f"{_STEM}.json|eight", f"{_STEM}.json|single"]
    rec = {k: v for k, v in CHECKPOINT_RECONCILIATION.load().items() if not k.endswith("/" + _STEM)}
    assert CHECKPOINT_RECONCILIATION.missing_keys(rec, CorpusModel(_STEM))


def test_a_key_for_a_snapshot_that_is_gone_is_an_orphan():
    stems = set(Corpus.stems())
    assert not SPEC_MATCH.key_exists("org__gone", stems, set())
    assert not SOL_SURFACE.key_exists("org__gone.json|single", stems, set())
    assert SOL_SURFACE.key_exists(f"{_STEM}.json|single", stems, set())


def test_a_stale_count_beside_a_fresh_one_is_red():
    texts = README_COUNTS.texts()
    n = counts()["families"]
    stale = dict(texts, **{"README.md": texts["README.md"] + f"\n{n - 1} architecture families\n"})
    assert any("occurs 2 times" in p for p in README_COUNTS.verify(stale, counts()))
    wrong = dict(texts, **{"README.md": texts["README.md"].replace(
        f"{n} architecture families", f"{n + 1} architecture families")})
    assert any("states families" in p for p in README_COUNTS.verify(wrong, counts()))


def test_changed_models_are_the_spec_gates_selection():
    models = verify.affected_models([f"backend/arch/snapshots/{_STEM}.json"])
    assert [m.stem for m in models] == [_STEM]


def test_a_changed_model_is_measured_against_the_files():
    assert verify.value_problems(CorpusModel(_STEM)) == []
