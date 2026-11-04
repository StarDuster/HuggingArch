"""Every MTP stack reads the next token's embedding.

An MTP layer predicts token t+2 from two things: the backbone's hidden state
at t and the embedding of token t+1. vLLM's and sglang's MTP layers all take
both (`eh_proj(cat[enorm(inputs_embeds), hnorm(previous_hidden_states)])` —
deepseek_mtp.py, mimo_v2_mtp.py, step3p5_mtp.py, nemotron_h_mtp.py,
bailing_moe_v3_mtp.py, hy_v3_mtp.py, sglang hunyuan_v4_nextn.py, …). Eight
specs once wired `enorm` to the hidden state instead, so the embedding never
entered the draft: every shape still matched, so no byte or FLOP total moved,
and only the dataflow was wrong.

The check reads the dataflow the walk bound (`Analysis.occurrence_graph`),
not names. A token embedding is a node that reads a `kind: tokens` model
input (the backbone's `input_embedding`, or a private embedding inside the
MTP stack). The MTP stack must contain one, or read one directly — through
an edge from outside the stack whose source is such a node. Reading only the
backbone's output does not count, even though that output descends from the
embedding: the hidden state is the other operand.

No declaration says which of the two entry norms is `enorm`, so the test
cannot ask that the embedding reaches that norm in particular: it asks that
the embedding enters the stack and is read there."""
from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from backend.arch.eval.program import Program
from backend.arch.eval.transforms import DECODE, PREFILL, Deployment
from backend.arch.snapshots import iter_snapshot_ids
from tests.snapshots import resolve_snapshot, shared_loader

pytestmark = pytest.mark.regression  # walks every snapshot with an MTP stack




def _reads_tokens(program, node) -> bool:
    tokens = {name for name, entry in program.model_inputs.items()
              if entry.kind == "tokens"}
    return any(ref.src is None and ref.external_sid in tokens
               for ref in node.inputs_obj or ())


def mtp_token_readers(program, phase: str = PREFILL) -> list:
    """The token-embedding nodes the MTP stacks contain or read directly, in
    the program `phase` walks. That is `analysis.program`, not `program`: a
    phase runs its components' own form (`transforms.phase_form` — a sparse
    MLA prefills absorbed), so its walk is over a rebound program whose uids
    are not the ones handed in."""
    analysis = Deployment(program).at(batch=1, input_tokens=16,
                                      media={}).during(phase).analyze()
    program = analysis.program
    graph = analysis.occurrence_graph
    sources = defaultdict(set)
    for sink, _index, _k, source, _port in graph.local:
        sources[sink].add(source)
    for edge in graph.pairs:
        sources[edge.sink.uid].add(edge.source.uid)
    mtp = {uid for uid, _iterations in program.occurrence_domains()
           if program.evaluating_stage(uid)[0].aux_kind == "mtp"}
    assert mtp, f"{program.spec.name}: the walk evaluates no MTP node"
    candidates = mtp | {source for uid in mtp for source in sources[uid]}
    return sorted(program.where(uid) for uid in candidates
                  if _reads_tokens(program, program.by_uid(uid)))


# Every snapshot, by file name only; which ones the test applies to is
# asked of the resolved spec in the test, so collection resolves nothing.
@pytest.mark.parametrize("model_id", sorted(iter_snapshot_ids()))
def test_every_mtp_stack_reads_the_next_token_embedding(model_id):
    resolved = resolve_snapshot(model_id, shared_loader())
    if not any(layer.aux_kind == "mtp" for layer in resolved.spec.model):
        pytest.skip("no MTP stack")
    program = Program.of(resolved)
    for phase in (PREFILL, DECODE):
        assert mtp_token_readers(program, phase), (f"{phase}: "
            f"{model_id}: no MTP node reads the token embedding — the stack "
            f"sees only the backbone's hidden state, so the next token never "
            f"enters the draft (`enorm` must read the embedding, `hnorm` the "
            f"hidden state)")
