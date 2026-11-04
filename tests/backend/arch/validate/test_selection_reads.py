"""A score that reads through a selection reads a bounded set of positions
(`_selection_reads`), and a cached tensor takes its read law from the kernel
that scores it (`cache._scoring_law`).

GLM-5.3-Flash lists its indexer's dense score before the DSA score that reads
its MLA latent; the latent took the first kernel's law and was read whole
under a top-k score. The scoring law fixes it; the check is what would have
caught it.

Both phases now run the absorbed graph (`transforms.phase_form`), where the
latent reaches its score through views and the reader's own law answers
(`cache._consumer_law`); the authored graph reached it through `kv_b_proj`,
which only the scoring law sees past. The check is falsified on the law
itself: a latent left with a dense read under the selected score.
"""
from __future__ import annotations

from dataclasses import replace

import pytest
import sympy as sp

from backend.arch.eval import cache as cache_mod
from backend.arch.eval.program import Program
from backend.arch.eval.transforms import Deployment
from backend.arch.validate._selection_reads import KIND, run_selection_read_check
from tests.snapshots import resolve_snapshot, shared_loader

MODEL = "zai-org/GLM-5.3-Flash"


@pytest.fixture(scope="module")
def resolved():
    return resolve_snapshot(MODEL, shared_loader())


def _analysis(resolved):
    return Deployment(Program.of(resolved)).at(input_tokens=64).analyze()


def _latent(analysis):
    return [tensor for (_holder, _cache, tensor)
            in cache_mod.resident(analysis).values()
            if tensor.name == "kv_a_layernorm"]


def test_a_latent_is_read_by_the_law_of_the_kernel_that_scores_it(resolved):
    analysis = _analysis(resolved)
    latents = _latent(analysis)
    assert latents
    for tensor in latents:
        # The DSA score reads top-k (2048) plus the pool tail it expands to.
        assert sp.simplify(tensor.read - sp.Min(2051, cache_mod.CACHE_LEN)) == 0
        assert cache_mod.law_grows(tensor.stored)
    assert run_selection_read_check(analysis) == []


def test_a_selected_score_over_a_dense_read_law_is_refused(resolved, monkeypatch):
    """A latent whose law reads every position — the indexer's dense score's,
    as the anchor used to hand it — under the selected score: the check names
    the selected score."""
    law_for = cache_mod._law_for

    def dense_latent(analysis, component, node, anchor):
        law = law_for(analysis, component, node, anchor)
        if node.name != "kv_a_layernorm":
            return law
        return replace(law, read=cache_mod.CACHE_LEN, pairs=None)

    monkeypatch.setattr(cache_mod, "_law_for", dense_latent)
    findings = run_selection_read_check(_analysis(resolved))
    assert findings
    assert {f.kind for f in findings} == {KIND}
    assert all(f.node.split(".")[-1].startswith("qk_score") for f in findings)
    assert all("kv_a_layernorm" in f.detail for f in findings)
