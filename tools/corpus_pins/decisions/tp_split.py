"""D1: how an external framework splits a stack over TP, and where the spec
knowingly departs from it.

`tests/golden/pins/tp_split_decisions.json` keys `{stem: {stack: entry}}`
(the backbone is the stack named ""). An entry says:
  verdict     framework_split | framework_replicated;
  encoder     whether the stack is an encoder (the encoder golden reads those);
  source      the framework classes that decide it;
  nodes       `{node: {framework, decided_by, spec?, reason?}}` for every
              weight node a candidate-introduced graph defines: `framework`
              is the framework's split and `decided_by` which source said it
              (`framework_split.SOURCE_KINDS`), `spec` (with `reason`) a
              registered departure from it; `null` for an entry admitted
              before node tables existed, which keeps its source and its
              stack-level verdict only.
Nodes from library components keep the library's declaration, which the
corpus goldens already pin. A candidate-introduced graph is the model file
itself (its root entries and inline `blocks:`) and the drafts it lists.

A stack with no entry yet is written from the framework sources the admitter
staged (`framework_split.py`: vLLM first, then the publisher's own reference
implementation): if every node's answer is settled and agrees with the
measurement, the entry is the record, naming the sources that decided and
citing each node's assignment. A node no source settles, that two sources
settle differently, or that a source splits differently from the
measurement, leaves the stack a person's to write.

Each registered node is classified from two independent signals at TP=8:
its placement on the TP world (the rank's runtime shape against the global
shape) and its resident bytes against TP=1. Two signals that disagree are
`inconsistent (classification)` for a person to read. Agreeing signals must
equal `spec` when registered, else `framework`.
"""
from __future__ import annotations

import functools
import json
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Set, Tuple

from tools.corpus_pins.corpus import ROOT, CorpusModel
from tools.corpus_pins.decision import Decision, DecisionState, Verdict

ARTIFACT = "tests/golden/pins/tp_split_decisions.json"
VERDICTS = ("framework_split", "framework_replicated")
SPLIT, REPLICATED = "split", "replicated"
#: the key each node entry states the framework's own split under
FRAMEWORK = "framework"


def load() -> Dict[str, Dict[str, Any]]:
    return json.loads((ROOT / ARTIFACT).read_text(encoding="utf-8"))


def encoder_entries(verdict: str) -> Dict[Tuple[str, str], str]:
    """`{(stem, stack): source}` of the encoder entries with `verdict`."""
    return {(stem, stack): entry["source"]
            for stem, stacks in load().items()
            for stack, entry in stacks.items()
            if entry["encoder"] and entry["verdict"] == verdict}


def defining_graph(node: Any, templates: Mapping[str, str]) -> str:
    """The graph that declares a node: the innermost component it sits in,
    else its block template's component, else "" for an entry of the model
    file itself. A node's `template` is its invocation's label, which adds
    the stack (`hy_v3_mtp_block@mtp`) and the history it reads
    (`[Sar=3]`); `templates` maps a label to the template it invokes, and a
    synthesized template is named `<component>__<params>`
    (`block_synthesis.synthesize`)."""
    if node.component_path:
        return node.component_path[-1][1]
    if node.template:
        return templates[node.template].split("__", 1)[0]
    return ""


def candidate_graphs(model: CorpusModel) -> Set[str]:
    """The graphs the model's admission brings: the model file ("") and its
    inline `blocks:`, and the drafts it lists."""
    import yaml
    doc = yaml.safe_load(model.spec_file().read_text(encoding="utf-8")) or {}
    return {"", *(doc.get("blocks") or {}), *(doc.get("drafts") or ())}


def node_table(model: CorpusModel) -> Dict[str, Dict[str, Any]]:
    """`{stack: {node: row}}` (see `_node_table`), measured once per model
    per process: the tree does not change under one admission."""
    return _node_table(model.stem)


@functools.lru_cache(maxsize=4)
def _node_table(stem: str) -> Dict[str, Dict[str, Any]]:
    model = CorpusModel(stem)
    """`{stack: {node: row}}` for the candidate-introduced weight nodes: TP=1
    and TP=8 bytes per rank, and the two classifications."""
    from backend.arch.eval.program import Program
    from tools.corpus_pins.pins.encoder_weight import weight_leaves
    resolved = model.resolved()
    graphs = candidate_graphs(model)
    program = Program.of(resolved)
    templates = {meta.label: meta.template_name for meta, _roots in program.ir.blocks}
    defined = {}
    for node in program.nodes():
        defined.setdefault(node.aggregation_id, defining_graph(node, templates))
    _, one = weight_leaves(resolved, 1)
    _, eight = weight_leaves(resolved, 8)
    table: Dict[str, Dict[str, Any]] = {}
    for name in sorted(one):
        if defined[name] not in graphs:
            continue
        a, b = one[name], eight[name]
        # placement: the TP=8 rank holds a slice of the global tensor (or of
        # its instances); bytes: the TP=8 rank holds less than TP=1
        placed = (SPLIT if (b.runtime_shape != b.global_shape
                            or b.runtime_instances != b.global_instances)
                  else REPLICATED)
        by_bytes = (SPLIT if b.resident_bytes_per_rank < a.resident_bytes_per_rank
                    else REPLICATED if b.resident_bytes_per_rank == a.resident_bytes_per_rank
                    else "grew")
        table.setdefault(a.stack, {})[name] = {
            "tp1_bytes": int(a.resident_bytes_per_rank),
            "tp8_bytes": int(b.resident_bytes_per_rank),
            "placement": placed,
            "bytes": by_bytes,
        }
    return table


def architectures(model: CorpusModel) -> List[str]:
    return list(model.config.get("architectures") or ())


def judge_stack(recorded: Mapping[str, Any], measured: Mapping[str, Mapping[str, Any]]) -> List[str]:
    """Why a written entry disagrees with the node table; empty when every
    candidate node is registered and classified as registered."""
    problems = []
    if recorded["verdict"] not in VERDICTS:
        problems.append(f"verdict {recorded['verdict']!r} is not one of {VERDICTS}")
    nodes = recorded["nodes"]
    if nodes is None:
        # admitted before node tables existed: the stack-level verdict is
        # the encoder golden's to check
        return problems
    problems += [f"{name}: not registered" for name in sorted(set(measured) - set(nodes))]
    problems += [f"{name}: registered but not a candidate weight node"
                 for name in sorted(set(nodes) - set(measured))]
    for name in sorted(set(nodes) & set(measured)):
        row, entry = measured[name], nodes[name]
        if row["placement"] != row["bytes"]:
            problems.append(
                f"{name}: inconsistent (classification): placement {row['placement']}, "
                f"bytes {row['bytes']} ({row['tp1_bytes']} -> {row['tp8_bytes']})")
            continue
        expected = entry.get("spec", entry[FRAMEWORK])
        if row["placement"] != expected:
            problems.append(
                f"{name}: measured {row['placement']} ({row['tp1_bytes']} -> "
                f"{row['tp8_bytes']} B), registered {expected}"
                + ("" if "spec" in entry else " (the framework's); no departure registered"))
    return problems


@dataclass(frozen=True)
class TpSplitDecision(Decision):
    def judge_model(self, model: CorpusModel) -> Dict[str, Verdict]:
        """One verdict per stack with candidate-introduced weight nodes."""
        table = node_table(model)
        recorded = load().get(model.stem, {})
        out: Dict[str, Verdict] = {}
        for stack, rows in sorted(table.items()):
            suggestion = {"nodes": rows, "architectures": architectures(model)}
            if stack not in recorded:
                out[stack] = Verdict(DecisionState.PENDING, [], suggestion)
                continue
            problems = judge_stack(recorded[stack], rows)
            out[stack] = Verdict(
                DecisionState.INCONSISTENT if problems else DecisionState.CONSISTENT,
                problems, suggestion)
        return out

    def judge(self, model: CorpusModel, measured: Dict[str, Any],
              oracle: Any = None) -> Verdict:
        """The model's D1 state. With `oracle` (a `framework_split.Oracle`), a
        stack with no entry whose every node the oracle settles as measured
        is consistent, and the suggestion carries the entries to write
        (`written_from_framework`); every pending stack carries the oracle's
        answer per node."""
        per_stack = self.judge_model(model)
        written: Dict[str, Any] = {}
        if oracle:
            table, encoders = node_table(model), encoder_stacks(model)
            for stack, verdict in list(per_stack.items()):
                if verdict.state is not DecisionState.PENDING:
                    continue
                answers = {n: oracle.answer(n) for n in table[stack]}
                verdict.suggestion[FRAMEWORK] = {
                    n: {"placement": a.placement, "citation": a.citation, "why": a.why,
                        "decided_by": a.decided_by} for n, a in answers.items()}
                entry = framework_entry(table[stack], answers, stack in encoders, oracle)
                if entry is not None:
                    written[stack] = entry
                    per_stack[stack] = Verdict(DecisionState.CONSISTENT, [], verdict.suggestion)
        states = {v.state for v in per_stack.values()}
        state = (DecisionState.NOT_NEEDED if not per_stack
                 else DecisionState.INCONSISTENT if DecisionState.INCONSISTENT in states
                 else DecisionState.PENDING if DecisionState.PENDING in states
                 else DecisionState.CONSISTENT)
        suggestion = {stack: {"state": v.state.value, **v.suggestion}
                      for stack, v in per_stack.items()}
        if written:
            suggestion["written_from_framework"] = {
                "asked": oracle.staged, "entries": written}
        return Verdict(state,
                       [f"{stack or '(backbone)'}: {line}" for stack, v in per_stack.items()
                        for line in v.detail],
                       suggestion)


def framework_entry(rows: Mapping[str, Mapping[str, Any]], answers: Mapping[str, Any],
                    encoder: bool, oracle: Any) -> Optional[Dict[str, Any]]:
    """The entry the framework source writes for one stack, or None unless
    every node is settled, classified consistently, and split as measured."""
    for name, row in rows.items():
        answer = answers[name]
        if (answer.placement is None or row["placement"] != row["bytes"]
                or answer.placement != row["placement"]):
            return None
    placements = {a.placement for a in answers.values()}
    return {
        "encoder": encoder,
        "nodes": {n: {FRAMEWORK: answers[n].placement,
                      "decided_by": answers[n].decided_by} for n in sorted(rows)},
        "citations": {n: answers[n].citation for n in sorted(rows)},
        "source": oracle.source({a.decided_by for a in answers.values()}),
        "verdict": "framework_split" if SPLIT in placements else "framework_replicated",
    }


def render_with(model: CorpusModel, entries: Mapping[str, Any]) -> bytes:
    """The decisions file with `entries` added to the model's stacks."""
    doc = load()
    doc[model.stem] = {**doc.get(model.stem, {}), **entries}
    return (json.dumps(doc, indent=2, sort_keys=True) + "\n").encode("utf-8")


def encoder_stacks(model: CorpusModel) -> Set[str]:
    from backend.arch.eval.program import Program
    return {st.name for st in Program.of(model.resolved()).stacks if st.is_encoder}


TP_SPLIT = TpSplitDecision(
    id="D1", title="TP split semantics", artifacts=(ARTIFACT,), gates_g4=True)
