"""Where each stack attaches is declared, not synthesised.

A `group:` entry with an `aux_kind` is a stack of its own — its own residual,
its own checkpoint namespace, its own cost bucket. Two kinds of stack live
there and they differ in exactly one way: a prediction head (MTP, speculative)
reads a hidden state the backbone produced, and an encoder (vision, audio)
reads an image or a waveform and nothing the backbone has.

That difference used to be answered from `aux_kind`, which says what a stack IS
and was being asked what it READS. The kind was right and the NODE was still
missing, so the attachment point was searched for: the backbone's last output at
the residual's width. On LongCat that search returned `mtp.norm` — the head's
own output norm — and the head spent every evaluation reading a tensor it
produces itself. Same width, so nothing complained.
"""
from __future__ import annotations

import sys
import unittest
import pytest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT))

from backend.arch.loader import SpecLoader
from tools.corpus_pins.checks.chain_stacks import served_specs
from tools.corpus_pins.checks.stack_attachment import (
    PREDICTION_KINDS,
    aux_groups,
    attached_encoders,
    undeclared_heads,
    unresolved_reads,
)

pytestmark = pytest.mark.regression  # walks every snapshot



def _single_rank_rule():
    """The rule a bare `Evaluator` walk runs under: one rank, no worlds."""
    from backend.arch.layout.parallel_config import ParallelConfig
    from backend.arch.layout.propagation import LayoutPropagation

    return LayoutPropagation.of(ParallelConfig(), "prefill")


def _specs():
    return sorted(served_specs().items())


class TestStackAttachment(unittest.TestCase):

    def test_every_prediction_head_declares_what_it_reads(self):
        """An MTP head names the node it attaches to.

        Not `aux_kind: mtp` plus a search — the name. Thirteen of the fifteen
        heads in the corpus used to leave this out, and what filled the gap was
        a width match against the backbone's stages.
        """
        heads = [(name, g) for name, spec in _specs()
                 for g in aux_groups(spec) if g[1] in PREDICTION_KINDS]
        self.assertTrue(heads, "corpus has no prediction heads to check")
        undeclared = [(name, layer) for name, spec in _specs()
                      for layer in undeclared_heads(spec)]
        self.assertEqual(undeclared, [],
                         "prediction head(s) with no `inputs:` — the stack "
                         "would attach to nothing")

    def test_encoders_declare_nothing(self):
        """A vision or audio tower reads an image or a waveform.

        The absence is the statement, and it is the one thing that distinguishes
        an encoder from a head now that the kind is not consulted. An encoder
        that acquired an `inputs:` would be claiming to read a hidden state.
        """
        encoders = [(name, g) for name, spec in _specs()
                    for g in aux_groups(spec) if g[1] not in PREDICTION_KINDS]
        self.assertTrue(encoders, "corpus has no encoder stacks to check")
        attached = [(name, found) for name, spec in _specs()
                    for found in attached_encoders(spec)]
        self.assertEqual(attached, [])

    def test_declared_attachment_names_an_earlier_layer(self):
        """The node a head reads is produced before the head runs.

        The evaluator raises on a name that resolves to nothing rather than
        dropping it, so this is the same fact checked one layer earlier and
        against every spec at once — including the ones no snapshot exercises.
        A name may also be an appended stream an earlier stack appends to
        (Kimi-K3's `output_attn_res` reads the whole `block_residual`), or a
        stage inside an earlier group: a group is a stack of its own and the
        evaluator resolves `inputs:` against every stage of the stacks walked
        before (Qwen3-VL's `embed_merge` reads `visual.merger.linear_fc2`).
        """
        for spec_name, spec in _specs():
          with self.subTest(spec=spec_name):
            self.assertEqual(
                unresolved_reads(spec), [],
                f"{spec_name}: a layer reads a name no earlier top-level "
                f"layer or group stage produces")


class TestCrossStackReads(unittest.TestCase):
    """A stage's `inputs:` may name another stack's stage (Kimi's decoder
    reads the vision tower). Two rules make that safe: a read of an
    encoder's output is a handoff the encoder's own walk produces, and an
    unproduced name is an error everywhere — not just on aux stacks, which
    is the hole the decoder's `vision` input fell through."""

    def _kimi_program(self):
        import json
        from pathlib import Path
        from backend.arch.eval.program import Program
        snap = Path("backend/arch/snapshots/moonshotai__Kimi-K2.5.json")
        config = json.loads(snap.read_text())["config"]
        resolved = SpecLoader().match(config, model_id=snap.stem.replace("__", "/", 1))
        return Program.of(resolved), resolved

    def test_the_producer_stack_hands_its_output_to_the_reader(self):
        """The decoder reads `vision` from the tower's encode walk: the
        language model's walk does not run the tower, the read is a declared
        handoff onto the image's prompt positions, and the encode walk runs
        the tower alone."""
        from backend.arch.eval.program import Handoff
        prog, _resolved = self._kimi_program()
        self.assertEqual([s.name for s in prog.walk_stacks], [""])
        self.assertEqual(prog.handoffs, (Handoff(
            name="vision", stack="vision", domain="pixel_values",
            defines=True, stream=False),))
        self.assertEqual([s.name for s in prog.encoding("vision").walk_stacks],
                         ["vision"])

    def test_an_unproduced_name_is_an_error_not_a_drop(self):
        from dataclasses import replace
        from backend.arch.eval.evaluator import Evaluator
        from backend.arch.eval.transforms import Deployment, Workload
        from backend.arch.eval.program import Program
        prog, resolved = self._kimi_program()
        # What the tower hands over, from a program of its own: the one
        # under test is edited below, before anything walks it.
        handed = Deployment(Program.of(resolved)).handed_over()
        first = next(s for s in prog.stacks if not s.is_encoder)
        # The first stage that reads another stage (the embedding reads the
        # token ids, which the walk checks as a model input).
        stage_names = {st.name for st in first.stages}
        target = next(st for st in first.stages
                      if set(st.upstream) & stage_names)
        bad = replace(target, upstream=("no_such_tensor",))
        stacks = tuple(
            replace(first, stages=tuple(bad if st is target else st
                                        for st in first.stages))
            if s is first else s
            for s in prog.stacks)
        prog.__dict__["stacks"] = stacks
        with self.assertRaisesRegex(ValueError, "named nothing evaluated"):
            Evaluator(prog, rule=_single_rank_rule(),
                      workload=Workload(), handed=handed).run(
                Deployment(prog).binding())


if __name__ == "__main__":
    unittest.main()
