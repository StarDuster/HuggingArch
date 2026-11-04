"""A pipeline stage caches its own layers, and only its own.

The weight tree learned that `pp` selects a stage rather than dividing the
model; the KV cache did not, so a rank at `pp=8` was told it held an eighth of
the weights and all twenty-eight layers of cache. Closing that gap turned out
to need three things this file pins, because each was wrong in a way that
over-reported what a deployment could serve:

  * a stage's cache is its LAYERS' cache. Summed over stages it must come back
    to the whole model — no more and no less. Keying that split by template
    NAME instead of by block ordinal made GLM-5.1's stages sum to 1.96x the
    model, because it declares one MoE template twice;
  * WHICH stage binds is a question about the workload. A hybrid backbone's
    stages differ in how many of their layers cap at a window, and a cap only
    shows once requests are long enough to reach it — so the stage picked
    workload-free for `max_tokens` is not the stage that limits a batch.
    MiMo-V2.5 at `pp=8` reported 3770 concurrent requests where its worst
    stage could hold 2182;
  * when NO stage fits, the binding stage is the one furthest from fitting.
    Ranking those by free-bytes-over-cache — a deficit divided by a rate —
    named Kimi-K3's most nearly feasible stage as the reason nothing fitted.

What is deliberately NOT asserted is that the binding stage holds at least an
even share of the cache. It need not: which stage runs out first depends on its
weights too, so a stage with fewer layers than average can still be the first
to go. DeepSeek-V3 at `pp=2` binds its 30-layer stage over its 31-layer one.
"""
from __future__ import annotations

import json
import sys
from typing import NamedTuple

import pytest

from tests.snapshots import snapshot_config
from functools import lru_cache
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from backend.arch.spec_tree import BlockMeta, iter_blocks
from backend.arch.layout.stages import StagePlacement
from backend.inference.capacity import (
    CapacityKnobs,
    _binding_key,
    _capacity_preparation,
    estimate_capacity,
)
from backend.inference.parallel_config import ParallelConfig
from backend.inference.weight_breakdown import (
    _reject_ambiguous_template_names,
)

SNAPSHOT_DIR = PROJECT_ROOT / "backend" / "arch" / "snapshots"

#: The shapes the split has to survive. Llama is homogeneous, so every stage's
#: cache is proportional to its layer count; gpt-oss alternates SWA and GQA by
#: declared layer index, so stages of equal size cache different amounts;
#: MiMo-V2.5 is the hybrid whose window cap only appears at length; V3 is
#: front-loaded MLA; GLM-5.1 is the duplicate-template-name model. Llama comes
#: first because it is the cheapest: the default run takes a parametrised
#: test's first case (tests/conftest.py), and gpt-oss and GLM-5.1 keep their
#: own unswept tests in it.
_MODELS = (
    "meta-llama__Llama-3.2-3B",
    "openai__gpt-oss-20b",
    "XiaomiMiMo__MiMo-V2.5",
    "deepseek-ai__DeepSeek-V3",
    "zai-org__GLM-5.1",
)

#: Finest split first. A stage count is not a structural case of its own: the
#: partition obeys one law at 2, 4 and 8 stages, and 8 is where that law is
#: hardest — the most cuts, the uneven remainder (28 layers over 8 stages,
#: 61 over 8), and the split GLM-5.1's duplicate template name doubled. 4 and
#: 2 are the same law with fewer cuts.
_PP = (8, 4, 2)

_GPU = "H200"
_IN, _OUT = 8192, 256
_L = _IN + _OUT


class _Case(NamedTuple):
    """One (model, stage count) the partition has to survive."""

    model: str
    pp: int


@lru_cache(maxsize=None)
def _config(model: str):
    path = SNAPSHOT_DIR / f"{model}.json"
    if not path.exists():
        return None
    return json.loads(path.read_text())["config"]


@lru_cache(maxsize=None)
def _resolved(model: str):
    """Resolve the spec once so capacity never re-matches (single entry)."""
    from backend.arch.loader import get_default_loader
    return get_default_loader().match(
        _config(model), model_id=model.replace("__", "/", 1))


_CASES = [pytest.param(_Case(model, pp), id=f"{model}-pp{pp}")
          for model in _MODELS for pp in _PP]


@lru_cache(maxsize=None)
def _prep(model: str, pp: int):
    return _capacity_preparation(
        config=_config(model), gpu_sku=_GPU,
        knobs=CapacityKnobs(
            resolved=_resolved(model), selection_mode="theoretical_only",
            parallel=ParallelConfig(pp=pp)))


@lru_cache(maxsize=None)
def _estimate(model: str, pp: int) -> dict:
    return estimate_capacity(
         config=_config(model), gpu_sku=_GPU, resolved=_resolved(model),
        selection_mode="theoretical_only", avg_input_tokens=_IN,
        avg_output_tokens=_OUT, parallel=ParallelConfig(pp=pp))


@lru_cache(maxsize=None)
def _stage_footprints(model: str, pp: int, length: int) -> tuple:
    """Each stage's per-request cache bytes at `length`.

    Built from the pp=1 reader on purpose — a second route to the same number.
    `capacity` reaches it through `_StageProfile`; this reaches it by asking
    the whole model's cache view for the layers the partition says the stage
    holds. A disagreement then means the two disagree, not that one formula was
    written down twice.
    """
    whole = _prep(model, 1)
    placement = StagePlacement.of(whole.kv_reader.source_analysis, pp)
    return tuple(whole.kv_reader.on(placement.view(st))
                 .storage_bytes_at(length)
                 for st in range(pp))


class TestStagesPartitionTheCache:
    """A stage holds some of the layers, so it holds some of the cache."""

    @pytest.mark.parametrize("case", _CASES)
    def test_stages_sum_to_the_whole_model(self, case):
        snapshot_config(case.model)
        whole = _prep(case.model, 1)
        ref = whole.kv_reader.storage_bytes_at(_L)
        stages = _stage_footprints(case.model, case.pp, _L)
        assert stages
        assert round(sum(stages) / ref - 1.0, 9) == 0, (
            "every layer's cache lives on exactly one stage")

    @pytest.mark.parametrize("case", _CASES)
    def test_no_stage_holds_more_than_the_model(self, case):
        snapshot_config(case.model)
        whole = _prep(case.model, 1)
        ref = whole.kv_reader.storage_bytes_at(_L)
        for st, got in enumerate(_stage_footprints(case.model, case.pp, _L)):
            assert got <= ref * (1 + 1e-9), st

    @pytest.mark.parametrize("case", _CASES)
    def test_every_block_ordinal_is_held_by_exactly_one_stage_per_layer(self, case):
        """The counts are keyed by block ordinal, never by template name:
        GLM-5.1 once declared its MoE template twice, and keyed by NAME both
        invocations were handed the name's whole share, so the stages summed
        to 1.96x the model. Summed over stages, every invocation's layers
        are its count."""
        snapshot_config(case.model)
        analysis = _prep(case.model, 1).kv_reader.source_analysis
        placement = StagePlacement.of(analysis, case.pp)
        whole = placement.block_counts(None)
        for ordinal, count in whole.items():
            assert sum(placement.block_counts(st)[ordinal]
                       for st in range(case.pp)) == count

    def test_glm_5_1_stages_conserve_the_cache(self):
        snapshot_config("zai-org__GLM-5.1")
        whole = _prep("zai-org__GLM-5.1", 1)
        ref = whole.kv_reader.storage_bytes_at(_L)
        assert round(
            sum(_stage_footprints("zai-org__GLM-5.1", 8, _L)) / ref - 1.0,
            9) == 0


class TestReportedNumbersAreOneStages:
    """A budget from one rank over a footprint from another is nobody's
    answer, so every reported quantity has to come from the same stage."""

    @pytest.mark.parametrize("case", _CASES)
    def test_per_request_bytes_is_some_stages(self, case):
        snapshot_config(case.model)
        stages = _stage_footprints(case.model, case.pp, _L)
        assert stages
        got = _estimate(case.model, case.pp)["kv_cache"]["per_request_kv_bytes"]
        assert any(abs(got - x) <= 1e-6 * max(1.0, x) for x in stages), (
            f"{got} is no stage's footprint: {stages}")

    @pytest.mark.regression  # minutes-long
    @pytest.mark.parametrize("case", _CASES)
    def test_max_batch_is_the_minimum_over_stages(self, case):
        """The deployment serves what its worst rank serves.

        Recomputed here from the stage profiles rather than from the reported
        number's own path, because the defect this catches was a stage chosen
        by the right rule for the wrong question.
        """
        snapshot_config(case.model)
        prep = _prep(case.model, case.pp)
        assert prep.pp_stages
        worst = min(
            int(prof.free_bytes(prep.usable_bytes)
                // prof.reader.storage_bytes_at(_L))
            if (prof.free_bytes(prep.usable_bytes) > 0
                and prof.reader.storage_bytes_at(_L) > 0)
            else 0
            for prof in prep.pp_stages)
        assert _estimate(case.model, case.pp)["max_batch_per_gpu"] == worst

    @pytest.mark.parametrize("case", _CASES)
    def test_replication_does_not_depend_on_pp(self, case):
        # How many ranks hold a COPY of the cache is a dp/tp question. Slicing
        # the backbone into stages changes which layers a rank holds, not how
        # many times the model's cache is duplicated across the cluster.
        snapshot_config(case.model)
        base = _estimate(case.model, 1).get("kv_replication_factor")
        assert (_estimate(case.model, case.pp).get("kv_replication_factor")
                == base)


class TestHybridStagesDifferAtLength:
    """gpt-oss alternates SWA and GQA by declared layer index.

    Its per-token slopes are identical, so every three-layer stage looks the
    same until requests are long enough for the windowed layers to cap. That
    is what makes the binding stage a workload question rather than a property
    of the partition.
    """

    MODEL = "openai__gpt-oss-20b"

    def test_equal_sized_stages_cache_different_amounts(self):
        snapshot_config(self.MODEL)
        stages = _stage_footprints(self.MODEL, 8, _L)
        assert stages
        assert max(stages) > min(stages) * 1.5, (
            "SWA and GQA layers do not cost the same at 8k — if they do, this "
            "model is no longer the hybrid case this file needs")

    def test_the_binding_stage_is_the_expensive_one(self):
        snapshot_config(self.MODEL)
        stages = _stage_footprints(self.MODEL, 8, _L)
        got = _estimate(self.MODEL, 8)["kv_cache"]["per_request_kv_bytes"]
        assert round(got / max(stages) - 1.0, 6) == 0


class TestBindingKey:
    """Fitting and not fitting are not the same question."""

    def test_fitting_beats_not_fitting(self):
        assert _binding_key(-1.0, 1e9) < _binding_key(1.0, 1e18)

    def test_among_fitting_stages_the_smallest_count_binds(self):
        assert _binding_key(100.0, 10.0) < _binding_key(100.0, 5.0)

    def test_among_overflowing_stages_the_deepest_binds(self):
        # The regression: -226/0.045 is more negative than -255/0.054, so
        # dividing named the stage CLOSEST to fitting as the binding one.
        deepest = _binding_key(-255.0, 0.054)
        shallow = _binding_key(-226.0, 0.045)
        assert deepest < shallow
        assert min([shallow, deepest]) == deepest

    def test_a_free_term_never_runs_out(self):
        assert _binding_key(1.0, 0.0) == (1.0, float("inf"))

    def test_the_terms_are_compared_in_order(self):
        # Equal on tokens, decided on requests-at-full-context: the gpt-oss
        # case, where identical slopes hide a 2x difference in stored bytes.
        assert _binding_key(100.0, 1.0, 2.0) < _binding_key(100.0, 1.0, 1.0)


class TestTemplateNameMustIdentifyAnInvocation:
    """The weight side is still keyed by name, and now says so.

    It cannot be keyed by ordinal without an invocation tag on the leaf, so it
    refuses the stack it cannot describe instead of silently doubling it.
    """

    def test_two_live_invocations_of_one_name_raise(self):
        with pytest.raises(ValueError) as ctx:
            _reject_ambiguous_template_names([
                BlockMeta(template_name="moe", count=18, stack_name="layers"),
                BlockMeta(template_name="moe", count=57, stack_name="layers"),
            ])
        assert "invoked 2 times" in str(ctx.value)

    def test_a_zero_layer_duplicate_is_fine(self):
        # GLM-5.1 and GLM-5.2's actual shape: the name is declared twice but
        # only one invocation has layers, so it still names one thing.
        _reject_ambiguous_template_names([
            BlockMeta(template_name="moe", count=0, stack_name="layers"),
            BlockMeta(template_name="moe", count=75, stack_name="layers"),
        ])

    @pytest.mark.parametrize("model", ["zai-org__GLM-5.1", "zai-org__GLM-5.2"])
    def test_the_corpus_models_that_duplicate_a_name_still_load(self, model):
        snapshot_config(model)
        _reject_ambiguous_template_names(
            list(iter_blocks(_prep(model, 1).resolved)))


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
