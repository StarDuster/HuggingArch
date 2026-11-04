"""`Analysis.copy_factors` — the named counts a weight leaf is repeated by.

A leaf's `multiplicity` is a product of counts and each one was declared by
something: a block template's invocation says how many layers it runs, each
enclosing `repeat:` says how many copies of its node exist. The storage
tooltip prints them (`× 61 layers`), so somebody has to say
which count is which — and the consumer used to decide by substring:

    unit = "experts" if "expert" in raw else "copies"

That is right about `n_expert_copies` by accident of spelling, right about
`n_shared_experts` for the wrong reason, and calls every layer loop and every
speech bank "copies". `TestTheBugItReplaces` writes out the 54 leaves across
five models where the guess and the declarations disagree today, with the
word each one gains.

The noun is a declaration (`dims.yaml`'s `unit:`), not something `kind` can
carry: `expert_copies` drags `shardable_by: [moe_copies]` with it and expert
parallelism does not shard `n_shared_experts`, so spelling the shared
experts' kind to fix a word would break their sharding.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, Set, Tuple

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT))

from tests.snapshots import (
    SNAPSHOT_DIR,
    resolve_snapshot,
    shared_program,
    shared_walk)


def _factors(stem: str) -> Tuple[Any, Dict[int, Tuple[Any, ...]]]:

    program = shared_program(stem)
    return program, dict(program.copy_factors.values)


def _by_path(stem: str) -> Dict[str, Tuple[Any, ...]]:
    program, values = _factors(stem)
    return {program.by_uid(uid).display_path: factors
            for uid, factors in values.items()}


def _declared_count_symbols(stem: str) -> Set[str]:
    """Every symbol that NAMES a count in `stem`: a node's `repeat:` /
    `chain:`, a block invocation's own count, or its enclosing group's.
    Enumerated from the walk, never from a list — the list in the field's
    own design note was already stale when it was written (it named
    `n_deepstack` where the corpus also has `n_audio_layers`)."""
    from backend.arch.eval.copies import _named

    program = shared_program(stem)
    found: Set[str] = set()
    for node in program.nodes():
        symbol = _named(node.repeat_raw)
        if symbol:
            found.add(symbol)
    for template in program.blocks:
        meta = template.meta
        for raw in (meta.count_raw, meta.outer_repeat_raw):
            symbol = _named(raw)
            if symbol:
                found.add(symbol)
    return found


class TestTheProductIsTheMultiplicity:
    """Rule one, and the reason the consumer never decomposes: the factors
    multiply to the count the tree already carries, so a reader checks a
    product instead of reconciling a remainder."""

    @pytest.mark.parametrize("stem", [
        "Qwen__Qwen3-30B-A3B", "deepseek-ai__DeepSeek-V2-Lite",
        "openai__gpt-oss-20b", "XiaomiMiMo__MiMo-V2.5",
        "nvidia__NVIDIA-Nemotron-3-Ultra-550B-A55B-BF16",
        "moonshotai__Kimi-K3",
    ])
    def test_every_leaf_multiplies_back(self, stem):
        program, values = _factors(stem)
        for uid, factors in values.items():
            node = program.by_uid(uid)
            product = 1
            for factor in factors:
                product *= factor.count
            assert product == int(node.multiplicity or 1), (
                f"{stem} {node.display_path}: {factors}")


class TestWhatEachFactorCites:
    def test_an_expert_bank_is_repeated_by_its_layers_only(self):
        """A bank's experts are the leading axis of its weight
        (`[n_expert_copies, out, in]`), not a copy of the leaf: the only
        count it is repeated by is the layer loop."""
        factors = _by_path("Qwen__Qwen3-30B-A3B")["mlp.experts.gate_proj"]
        assert [(f.symbol, f.count, f.kind, f.unit) for f in factors] == [
            ("n_layers", 48, "layer_count", "layers"),
        ]

    def test_the_kind_is_the_registrys_word_unremapped(self):
        """`n_shared_experts` is kind `whole` and unit `experts`: the two
        fields answer different questions and neither is derived from the
        other. Spelling its kind `expert_copies` to get the noun would hand
        it `shardable_by: [moe_copies]`, and expert parallelism does not
        shard the shared experts."""
        import backend.arch.dims as dim_registry

        factors = _by_path(
            "deepseek-ai__DeepSeek-V2-Lite")["mlp.shared_experts.gate_proj"]
        shared = next(f for f in factors if f.symbol == "n_shared_experts")
        assert (shared.kind, shared.unit) == ("whole", "experts")
        for stem in ("Qwen__Qwen3-30B-A3B", "XiaomiMiMo__MiMo-V2.5"):
            _program, values = _factors(stem)
            for factors in values.values():
                for factor in factors:
                    if factor.symbol is None:
                        continue
                    declared = dim_registry.get(factor.symbol)
                    assert factor.kind == declared.kind, factor
                    assert factor.unit == (declared.unit or None), factor

    def test_a_count_no_symbol_named_cites_none(self):
        """DeepSeek-V2-Lite's 26 MoE layers come from an assignment's layer
        indices, so there is no symbol to cite — and the factor still knows
        it counts layers, because that is what a block invocation's count
        IS. `symbol=None` is the honest half, `layers` is not a guess."""
        factors = _by_path("deepseek-ai__DeepSeek-V2-Lite")["input_layernorm"]
        assert [(f.symbol, f.count, f.kind, f.unit) for f in factors] == [
            (None, 26, "layer_count", "layers")]

    def test_a_leaf_nothing_repeats_has_an_empty_tuple(self):
        assert _by_path("Qwen__Qwen3-30B-A3B")["embed_tokens"] == ()

    def test_a_groups_outer_repeat_is_its_own_factor(self):
        """Step-3.7-Flash is the corpus's one group whose `repeat:` is above
        one: `repeat: n_mtp_layers` around its MTP stack. The factor cites
        that declaration rather than the product."""
        from backend.arch.eval.copies import _block_factors

        program = shared_program("stepfun-ai__Step-3.7-Flash")
        outer = [template for template in program.blocks
                 if int(template.meta.outer_repeat or 1) > 1]
        assert len(outer) == 1, [t.meta.stack_name for t in outer]
        meta = outer[0].meta
        factors = _block_factors(program, outer[0].roots[0])
        assert [(f.symbol, f.count, f.unit) for f in factors] == [
            ("n_mtp_layers", 3, "layers")]
        # Its stack's own count is 1 here, so there is one factor. The law
        # that makes a second one appear is the division below, which holds
        # on every invocation in the corpus — a model declaring both a group
        # repeat and a stack count gets two factors with no new code.
        assert int(meta.count) == int(meta.outer_repeat)

    @pytest.mark.parametrize("stem", [
        "Qwen__Qwen3-30B-A3B", "XiaomiMiMo__MiMo-V2.5",
        "stepfun-ai__Step-3.7-Flash",
        "nvidia__NVIDIA-Nemotron-3-Ultra-550B-A55B-BF16",
    ])
    def test_a_block_count_divides_into_its_group_and_its_stack(self, stem):

        program = shared_program(stem)
        for template in program.blocks:
            meta = template.meta
            outer = int(meta.outer_repeat or 1)
            assert int(meta.count) % outer == 0, (stem, meta.stack_name)


class TestTotalityOverWeightLeaves:
    """`()` is "not repeated", so an ABSENT entry must mean "not a weight
    leaf" and nothing else. A layer that used both spellings for one thing
    could not refuse a leaf repeated 61 times whose factors went missing."""

    @pytest.mark.parametrize("stem", [
        "Qwen__Qwen3-30B-A3B", "XiaomiMiMo__MiMo-V2.6-Pro-RL",
        "moonshotai__Kimi-K3",
    ])
    def test_every_on_disk_leaf_has_an_entry(self, stem):
        from backend.arch.spec_tree.projections import iter_leaves

        resolved = resolve_snapshot(stem)
        _program, values = _factors(stem)
        assert set(values) == {leaf.ir_node.uid
                               for leaf in iter_leaves(resolved)}

    def test_the_walks_storage_layer_is_not_the_domain(self):
        """A walk is per phase and per stage, so it stores fewer leaves than
        the checkpoint ships — MiMo-V2.6 walks 41 of its 73. Keying the
        layer on the walk would have left the audio and MTP leaves with no
        factors at all, and the breakdown is about the whole checkpoint."""
        from backend.arch.eval.transforms import Deployment

        program = shared_program("XiaomiMiMo__MiMo-V2.6-Pro-RL")
        analysis = (shared_walk(Deployment(program)
                    .at(batch=1, input_tokens=128)))
        stored = set(analysis.storage.values)
        covered = set(analysis.copy_factors.values)
        assert stored < covered, (len(stored), len(covered))


class TestTheUnitVocabularyIsClosedAndUsed:
    def test_an_unregistered_noun_is_refused_at_load(self):
        from backend.arch.dims import COPY_UNITS, DimSymbol

        with pytest.raises(ValueError, match="unit"):
            DimSymbol(name="n_probe", kind="whole", meaning="probe",
                      group="backbone", unit="instances")
        assert "instances" not in COPY_UNITS

    def test_there_is_no_generic_copies_entry(self):
        """Every word in the vocabulary has a user. A generic `copies` had
        none — every symbol the corpus repeats wants `layers`, `experts` or
        `banks` — and an unreachable entry is an invitation, this one
        especially: `copies` is the wrong label the substring guess
        produces."""
        import backend.arch.dims as dim_registry
        from backend.arch.dims import COPY_UNITS

        declared = {symbol.unit
                    for symbol in dim_registry.symbols().values()
                    if symbol.unit}
        assert declared == COPY_UNITS, (
            "a vocabulary entry with no symbol declaring it, or a symbol "
            "declaring a word the vocabulary does not hold")
        assert "copies" not in COPY_UNITS

    def test_a_symbol_with_no_declared_noun_publishes_none(self):
        """The no-fallback rule, pinned directly because the corpus cannot
        show it: every symbol the corpus repeats declares a unit (the gate
        below is what keeps that true), so a fallback would never fire and
        would sit there unobserved until the day a spec declared its own
        count symbol.

        `n_experts_active` is registered, is not a count of copies, and
        declares no unit — exactly the shape of a symbol a future spec might
        repeat. Its factor must come out None, so a reader shows
        `× 8 n_experts_active` rather than inventing a noun for it.
        """
        import backend.arch.dims as dim_registry
        from backend.arch.eval.copies import _factor

        assert dim_registry.get("n_experts_active").unit == ""
        factor = _factor("n_experts_active", 8)
        assert factor.symbol == "n_experts_active"
        assert factor.unit is None
        assert factor.kind == dim_registry.get("n_experts_active").kind

    @pytest.mark.parametrize("stem", [
        "Qwen__Qwen3-30B-A3B", "XiaomiMiMo__MiMo-V2.6-Pro-RL",
        "moonshotai__Kimi-K2.5", "Qwen__Qwen3-VL-235B-A22B-Instruct",
        "nvidia__NVIDIA-Nemotron-3-Ultra-550B-A55B-BF16",
    ])
    def test_a_symbol_that_names_a_count_declares_its_noun(self, stem):
        import backend.arch.dims as dim_registry

        missing = sorted(
            symbol for symbol in _declared_count_symbols(stem)
            if not (dim_registry.get(symbol) is not None
                    and dim_registry.get(symbol).unit))
        assert missing == [], (
            f"{stem}: {missing} name a count the breakdown prints and "
            f"declare no `unit:` in dims.yaml, so it would print the "
            f"symbol's own name")


#: The leaves where the substring guess and the declarations disagree, one
#: per shape of disagreement, as `(stem, display path, count, old, new)`.
#: Measured over the whole corpus: 54 leaves in five models, all of them a
#: layer loop or a speech bank the guess called "copies".
_REPLACED = [
    ("moonshotai__Kimi-K2.5",
     "vision_tower.encoder.blocks.norm0", 27, "copies", "layers"),
    ("XiaomiMiMo__MiMo-V2.5",
     "audio_encoder.input_local_transformer.layers.input_layernorm",
     6, "copies", "layers"),
    ("stepfun-ai__Step-3.7-Flash",
     "vision_model.transformer.resblocks.ln_1", 47, "copies", "layers"),
    ("deepseek-ai__DeepSeek-V4.1-Flash",
     "vision.blocks.norm1", 32, "copies", "layers"),
    ("XiaomiMiMo__MiMo-V2.5", "speech_embeddings", 20, "copies", "banks"),
]


class TestTheBugItReplaces:
    """The leaves where the substring guess and the declarations disagree,
    written out with the word each gains."""

    @pytest.mark.parametrize("stem,path,count,old,new", _REPLACED)
    def test_the_declared_noun_replaces_the_guess(self, stem, path, count,
                                                  old, new):
        factors = _by_path(stem)[path]
        assert [(f.count, f.unit) for f in factors] == [(count, new)], (
            f"{path} should print `× {count} {new}`, where the substring "
            f"guess answered `{old}`")


@pytest.mark.regression
class TestTheWholeCorpus:
    """The two laws over every snapshot: the product, and a declared noun
    for every symbol that names a count. Marked `regression` because it
    walks the corpus."""

    def test_every_leaf_of_every_model_multiplies_back(self):
        offenders = []
        for path in sorted(SNAPSHOT_DIR.glob("*.json")):
            program, values = _factors(path.stem)
            for uid, factors in values.items():
                node = program.by_uid(uid)
                product = 1
                for factor in factors:
                    product *= factor.count
                if product != int(node.multiplicity or 1):
                    offenders.append(
                        f"{path.stem} {node.display_path}: {factors}")
        assert offenders == [], "\n".join(offenders)

    def test_every_count_symbol_in_the_corpus_declares_a_noun(self):
        import backend.arch.dims as dim_registry

        offenders = []
        for path in sorted(SNAPSHOT_DIR.glob("*.json")):
            for symbol in sorted(_declared_count_symbols(path.stem)):
                declared = dim_registry.get(symbol)
                if declared is None or not declared.unit:
                    offenders.append(f"{path.stem}: {symbol}")
        assert offenders == [], "\n".join(sorted(set(offenders)))
