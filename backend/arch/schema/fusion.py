"""Declarative operator-fusion patterns (FusionOpSpec / NaivePattern)."""
from __future__ import annotations

from typing import (Any, Dict, FrozenSet, List, Mapping, Optional, Tuple)

from pydantic import (BaseModel, model_validator)

from backend.arch.ops import OP_REGISTRY
from backend.arch.ops_base import INPUT_ROLE_VOCABULARY


def _registered(owner: str, ops: List[str]) -> None:
    """Every op a pattern names is a registered primitive: a pattern matches
    the graph's ops by name, so a name the registry lacks matches nothing."""
    unknown = [op for op in ops if op not in OP_REGISTRY]
    if unknown:
        raise ValueError(
            f"{owner}: {unknown} name no registered op; a pattern matches "
            f"registered ops by name.")




class PatternNode(BaseModel):
    """One node in an explicit DAG-shaped naive pattern.

    Fields:
      op          - op name to match (mutually exclusive with `op_choices`).
      op_choices  - list of acceptable op names; the matcher accepts any
                    one of them. Used for op-family patterns (e.g. an
                    activation slot that accepts silu / gelu / gelu_tanh).
                    Mutually exclusive with `op`.
      inputs      - ordered list of input references; each is either
                    "external.<name>" (an external input that flows into
                    the pattern from outside) or "<pattern_node>.<port>"
                    (an internal edge — port "out" for the main output,
                    or a secondary port name). The order matters: it
                    lines up with the SpecIRNode.resolved_inputs order
                    at the matched node.
      optional_inputs - trailing `external.<name>` refs the candidate may
                    carry beyond `inputs`. Same slot, different arity:
                    softmax vs softmax-with-sink share one pattern node
                    because serving still runs one attention kernel.
                    Internal edges cannot be optional — a missing internal
                    producer is a different subgraph.
      feeds       - an operand this node's output becomes: an input role
                    (`OpSpec.input_roles` vocabulary — q, k, v, …). The
                    matcher collects, over every downstream path, the role
                    held at the first role-declaring op on that path, and a
                    candidate binds when `feeds` is anywhere in that union
                    (so a shared K/V rope satisfies `feeds: k`). Two
                    candidates that both satisfy it are an ambiguity the
                    matcher refuses rather than resolves by order. A
                    candidate whose union also holds an operand ANOTHER node
                    of the pattern `feeds` binds neither: a tensor that
                    becomes both q and k is neither side (DeepSeek-V4's MTP
                    `enorm` / `hnorm` against `fused_q_kv_rmsnorm`). This is
                    what tells apart pattern nodes the edges alone cannot:
                    `fused_qk_rope`'s two ropes have the same op and no edge
                    between them, and only `feeds` says which one is Q.
      optional    - the node may be absent: a candidate without it binds
                    with the node's one pattern reader reading the node's
                    one internal input in its place. The node-level
                    counterpart of `optional_inputs` — the same kernel run
                    with one step folded away: the DSA indexer weights its
                    heads (`mul` by W) before the head sum, the QSA indexer
                    sums them unweighted, and both are one MQA-logits
                    launch. An external read only by an absent node is
                    absent from the match. An optional node is a pass-through
                    on one internal edge: it has exactly one internal input,
                    and either exactly one reader in the pattern and no port,
                    or no reader at all — a SINK, the last step of the chain
                    (a router's routed-scaling multiply, which some
                    checkpoints do not have) — whose ports, when it is
                    absent, name its internal input instead.
    """
    op: Optional[str] = None
    op_choices: Optional[List[str]] = None
    inputs: List[str] = []
    optional_inputs: List[str] = []
    feeds: Optional[str] = None
    optional: bool = False

    @model_validator(mode="after")
    def _validate(self) -> "PatternNode":
        if self.op is None and not self.op_choices:
            raise ValueError(
                "PatternNode: must set exactly one of `op` (single op name) "
                "or `op_choices` (list of acceptable op names)."
            )
        if self.op is not None and self.op_choices:
            raise ValueError(
                "PatternNode: `op` and `op_choices` are mutually exclusive."
            )
        _registered("PatternNode", [self.op] if self.op else self.op_choices)
        for ref in self.optional_inputs:
            if not ref.startswith("external."):
                raise ValueError(
                    f"PatternNode.optional_inputs: {ref!r} must be "
                    f"'external.<name>' (optional slots cannot be internal "
                    f"edges)."
                )
        if self.feeds is not None and self.feeds not in INPUT_ROLE_VOCABULARY:
            raise ValueError(
                f"PatternNode.feeds: {self.feeds!r} is not an input role; "
                f"valid: {sorted(INPUT_ROLE_VOCABULARY)}."
            )
        return self

    def signature(self) -> Tuple:
        """What a node is before its edges: op constraint, arity, and the
        operand it feeds. Two pattern nodes can only be swapped by an
        automorphism when these agree."""
        return (self.op, tuple(sorted(self.op_choices or ())),
                len(self.inputs), len(self.optional_inputs), self.feeds,
                self.optional)

    def internal_inputs(self) -> List[str]:
        """The `<pattern_node>.<port>` refs among `inputs`."""
        return [ref for ref in self.inputs if not ref.startswith("external.")]

    def accepts_op(self, op_name: str) -> bool:
        """Return True if `op_name` matches this pattern node's op constraint."""
        if self.op is not None:
            return op_name == self.op
        return op_name in (self.op_choices or [])



class ParallelPatternForm(BaseModel):
    """Variable-arity sibling-match form for `NaivePattern`.

    Matches **any** `count` ≥ `min_count` SpecIRNodes that:
      - have the declared `op` (or one of `op_choices`);
      - share the SAME external inputs (a single external producer for
        each `external.<name>` reference);
      - are siblings (none consumes another's output).

    The matched siblings are interchangeable — the matcher returns ONE
    `PatternMatch` per maximal sibling set (no factorial permutation
    explosion). The graft produces `count` positional output ports
    `output_0`, ..., `output_{count-1}`; each original sibling's downstream
    consumers are rewired to the fused port at the sibling's
    deterministic position (sorted by spec_id).

    Use the `parallel.<i>` reference form in `NaivePattern.io_mapping`
    to point a fused output at the i-th matched sibling.

    Fields:
      op             - op name every sibling must match. Mutually exclusive with op_choices.
      op_choices     - list of acceptable op names (op-family form).
      shared_inputs  - list of external-input references that all
                       siblings consume in the same positions.
                       Each entry is "external.<name>"; the matcher
                       binds <name> at apply time.
      min_count      - smallest acceptable count (default 2 — single-sibling
                       fusion is degenerate).
      max_count      - largest acceptable count (None = unlimited).
    """
    op: Optional[str] = None
    op_choices: Optional[List[str]] = None
    shared_inputs: List[str]
    min_count: int = 2
    max_count: Optional[int] = None

    @model_validator(mode="after")
    def _validate(self) -> "ParallelPatternForm":
        if self.op is None and not self.op_choices:
            raise ValueError(
                "ParallelPatternForm: must set exactly one of `op` or "
                "`op_choices`."
            )
        if self.op is not None and self.op_choices:
            raise ValueError(
                "ParallelPatternForm: `op` and `op_choices` are mutually "
                "exclusive."
            )
        _registered("ParallelPatternForm",
                    [self.op] if self.op else self.op_choices)
        if not self.shared_inputs:
            raise ValueError(
                "ParallelPatternForm: `shared_inputs` must list at least "
                "one external reference (the matched siblings all consume "
                "the same external producer in each declared position)."
            )
        for ref in self.shared_inputs:
            if not ref.startswith("external."):
                raise ValueError(
                    f"ParallelPatternForm: `shared_inputs[*]` entry "
                    f"{ref!r} must be 'external.<name>'."
                )
        if self.min_count < 2:
            raise ValueError(
                f"ParallelPatternForm: `min_count={self.min_count}` must "
                f"be ≥ 2 (single-sibling 'fusion' is degenerate)."
            )
        if self.max_count is not None and self.max_count < self.min_count:
            raise ValueError(
                f"ParallelPatternForm: `max_count={self.max_count}` must "
                f"be ≥ min_count={self.min_count}."
            )
        return self

    def accepts_op(self, op_name: str) -> bool:
        if self.op is not None:
            return op_name == self.op
        return op_name in (self.op_choices or [])



class NaivePattern(BaseModel):
    """The naive op subgraph that a FusionOpSpec rewrites away.

    Two mutually-exclusive authoring forms:

      (1) Explicit DAG — `nodes: Dict[str, PatternNode]`. Each pattern
          node declares its op (or `op_choices` for op-family slots) and
          its inputs (external refs + internal-edge refs). Used for
          chains, fan-in, fan-out, and DAG patterns alike.

      (2) Variable-arity sibling form — `parallel: ParallelPatternForm`.
          Matches **any** `count` ≥ min_count interchangeable siblings of the
          same op (or op family) sharing the same external inputs. The
          matched siblings are symmetric — the matcher returns one
          PatternMatch per maximal sibling set (no factorial permutations).
          Fused outputs are positional (`output_0`, ..., `output_{count-1}`);
          downstream consumers are rewired by spec_id sort order, so the
          rewrite is permutation-stable under IR reordering.

    Pattern matching is by op identity plus connectivity, never by node name;
    see `backend/arch/eval/fusion.py`.

    `io_mapping` keys are fused-operand names (input operand names +
    `output` for the primary fused output). Values resolve into the
    matched subgraph:
      - "<pattern_node>.<port>" — references a port on a pattern node
        (port "out" or "" denotes the main output). Used by nodes form.
      - "external.<name>" — references an external input. Used for
        fused-op INPUTS whose source is outside the matched subgraph.

    A port names a pattern node, so the node must be one the matcher can
    tell apart from every other. When the pattern has a symmetry — a
    relabelling of its nodes that preserves every op constraint, edge and
    `feeds` — that moves a port's node, the matcher could bind either
    candidate to that port and the name it shows would be a coin toss. The
    loader rejects such a pattern; `feeds` is how an author breaks the
    symmetry.
    """
    # Exactly one of `nodes` / `parallel` is set.
    nodes: Optional[Dict[str, "PatternNode"]] = None
    parallel: Optional["ParallelPatternForm"] = None
    io_mapping: Dict[str, str]

    @model_validator(mode="after")
    def _validate(self) -> "NaivePattern":
        has_nodes = self.nodes is not None and len(self.nodes) > 0
        has_parallel = self.parallel is not None
        if has_nodes and has_parallel:
            raise ValueError(
                "NaivePattern: `nodes` and `parallel` are mutually "
                "exclusive — set exactly one."
            )
        if not has_nodes and not has_parallel:
            raise ValueError(
                "NaivePattern: must set either `nodes` (explicit DAG) or "
                "`parallel` (variable-arity sibling form)."
            )
        if not self.io_mapping:
            raise ValueError(
                "NaivePattern: `io_mapping` must map at least one fused "
                "operand to its corresponding pattern edge."
            )
        # Validate explicit-nodes form: every internal-edge reference
        # must resolve to a declared pattern node + port.
        if has_nodes:
            for pn_name, pn in self.nodes.items():
                for inp in pn.inputs:
                    if inp.startswith("external."):
                        continue
                    if "." not in inp:
                        raise ValueError(
                            f"NaivePattern.nodes[{pn_name!r}].inputs: "
                            f"reference {inp!r} must be 'external.<name>' "
                            f"or '<pattern_node>.<port>'."
                        )
                    ref_node, _port = inp.rsplit(".", 1)
                    if ref_node not in self.nodes:
                        raise ValueError(
                            f"NaivePattern.nodes[{pn_name!r}].inputs: "
                            f"reference {inp!r} names pattern node "
                            f"{ref_node!r} but only "
                            f"{sorted(self.nodes)!r} are declared."
                        )
            self._check_optional_nodes()
            self._reject_interchangeable_ports()
        return self

    def runs(self, op: str) -> bool:
        """Whether a node of this subgraph runs `op` — the question a
        signature's doc and its providers both ask of the plan (DeepSeek-V4's
        indexer query names a Hadamard rotation, V4.1's does not), answered
        once off the declared nodes."""
        return op in self.ops()

    def _check_optional_nodes(self) -> None:
        """An optional node can be elided only where eliding it leaves one
        well-formed subgraph: it stands on one internal edge (one internal
        input), so its reader can read through it. Inside the chain it has
        one reader and names no port, so the fused op's outputs exist either
        way; as a sink it has no reader, and a port that names it names its
        internal input when it is absent (`port_node`)."""
        nodes = self.nodes or {}
        readers: Dict[str, int] = {}
        for pn in nodes.values():
            for ref in pn.internal_inputs():
                head = ref.rsplit(".", 1)[0]
                readers[head] = readers.get(head, 0) + 1
        named = {target.partition(".")[0] for target in self.io_mapping.values()}
        for name, pn in nodes.items():
            if not pn.optional:
                continue
            if len(pn.internal_inputs()) != 1:
                raise ValueError(
                    f"NaivePattern.nodes[{name!r}]: an optional node has "
                    f"exactly one internal input (the edge its reader reads "
                    f"through when it is absent); it has "
                    f"{pn.internal_inputs()!r}.")
            if readers.get(name, 0) == 0:
                continue                # a sink: its ports fall through
            if readers[name] != 1:
                raise ValueError(
                    f"NaivePattern.nodes[{name!r}]: an optional node has "
                    f"exactly one reader in the pattern, or none (a sink); "
                    f"it has {readers[name]}.")
            if name in named:
                raise ValueError(
                    f"NaivePattern.nodes[{name!r}]: an optional node inside "
                    f"the chain names no `io_mapping` port — the port would "
                    f"not exist when the node is absent.")

    def port_node(self, name: str, bound: Mapping[str, Any]) -> Optional[str]:
        """The pattern node a port that names `name` resolves to in one
        match: `name` itself when bound, else — an absent optional sink —
        the node its one internal input names, followed the same way."""
        nodes = self.nodes or {}
        while name not in bound:
            spec = nodes.get(name)
            if spec is None or not spec.optional:
                return None
            (through,) = spec.internal_inputs()
            name = through.rsplit(".", 1)[0]
        return name

    def _symmetries(self) -> List[Tuple[Dict[str, str], Dict[str, str]]]:
        """Every relabelling `(node map, external map)` of the pattern onto
        itself: same signature per node, internal edges carried to internal
        edges, external operands renamed consistently. Patterns are a few
        nodes, and a node can only go to one of equal signature, so the
        search is tiny."""
        nodes = self.nodes or {}
        names = sorted(nodes)
        choices = [[m for m in names
                    if nodes[m].signature() == nodes[n].signature()]
                   for n in names]
        out: List[Tuple[Dict[str, str], Dict[str, str]]] = []

        def edges_hold(node_map: Dict[str, str]) -> bool:
            """Every internal edge between two nodes already placed is
            carried onto an edge of their images: a partial relabelling
            that breaks one cannot be completed, so the search drops it
            here rather than after placing every node (a 28-node plan has
            factorially many label orders among its equal signatures)."""
            for name, target in node_map.items():
                for ref, mapped in zip(
                        [*nodes[name].inputs, *nodes[name].optional_inputs],
                        [*nodes[target].inputs,
                         *nodes[target].optional_inputs], strict=True):
                    head, _, tail = ref.partition(".")
                    if head == "external" or head not in node_map:
                        continue
                    if mapped != f"{node_map[head]}.{tail}":
                        return False
            return True

        def assign(i: int, node_map: Dict[str, str]) -> None:
            if i == len(names):
                ext = self._carried_externals(node_map)
                if ext is not None:
                    out.append((dict(node_map), ext))
                return
            for target in choices[i]:
                if target in node_map.values():
                    continue
                node_map[names[i]] = target
                if edges_hold(node_map):
                    assign(i + 1, node_map)
                del node_map[names[i]]

        assign(0, {})
        return out

    def _carried_externals(self, node_map: Dict[str, str]
                           ) -> Optional[Dict[str, str]]:
        """The external renaming `node_map` induces, or None when it breaks
        an edge (an internal ref not carried onto its image, or one external
        renamed two ways)."""
        nodes = self.nodes or {}
        ext: Dict[str, str] = {}
        for name, node in nodes.items():
            image = nodes[node_map[name]]
            for ref, mapped in zip(
                    [*node.inputs, *node.optional_inputs],
                    [*image.inputs, *image.optional_inputs], strict=True):
                head, _, tail = ref.partition(".")
                m_head, _, m_tail = mapped.partition(".")
                if head == "external":
                    if m_head != "external":
                        return None
                    if ext.setdefault(tail, m_tail) != m_tail:
                        return None
                elif (m_head, m_tail) != (node_map[head], tail):
                    return None
        if len(set(ext.values())) != len(ext):
            return None
        return ext

    def _reject_interchangeable_ports(self) -> None:
        port_nodes: Dict[str, str] = {}
        port_externals: Dict[str, str] = {}
        for port, target in self.io_mapping.items():
            head, _, tail = target.partition(".")
            if head == "external":
                port_externals[port] = tail
            else:
                port_nodes[port] = head
        for node_map, ext in self._symmetries():
            moved = sorted(
                f"{port}: {name} <-> {node_map[name]}"
                for port, name in port_nodes.items()
                if node_map.get(name, name) != name)
            moved += sorted(
                f"{port}: external.{name} <-> external.{ext[name]}"
                for port, name in port_externals.items()
                if ext.get(name, name) != name)
            if moved:
                raise ValueError(
                    "NaivePattern: ports bind to interchangeable pattern "
                    f"nodes ({'; '.join(moved)}). The edges and op "
                    "constraints do not say which candidate is which; "
                    "declare `feeds:` (the input role each one's output "
                    "becomes) on the nodes the ports name."
                )

    def is_parallel(self) -> bool:
        """True when this pattern uses the variable-arity sibling form."""
        return self.parallel is not None

    def ops(self) -> FrozenSet[str]:
        """Every op a node of this pattern may match: each node's `op`, or
        each of its `op_choices`."""
        forms = ((self.parallel,) if self.parallel is not None
                 else tuple((self.nodes or {}).values()))
        return frozenset(op for form in forms
                         for op in ([form.op] if form.op else
                                    form.op_choices or ()))

    def expanded_nodes(self) -> Dict[str, "PatternNode"]:
        """Return the explicit-nodes dict, or empty for parallel form.

        Parallel-form siblings are interchangeable and have no per-node
        names — callers handling parallel form should branch on
        `is_parallel()` first.
        """
        if self.parallel is not None:
            return {}
        return dict(self.nodes or {})

    def external_input_names(self) -> List[str]:
        """Return the ordered set of distinct `external.<name>` refs."""
        seen: List[str] = []
        if self.parallel is not None:
            for ref in self.parallel.shared_inputs:
                if ref.startswith("external."):
                    name = ref[len("external."):]
                    if name not in seen:
                        seen.append(name)
            return seen
        for pn in self.expanded_nodes().values():
            for inp in pn.inputs:
                if inp.startswith("external."):
                    name = inp[len("external."):]
                    if name not in seen:
                        seen.append(name)
        return seen



class FusionOpSpec(BaseModel):
    """Spec-scoped virtual op spec for declarative operator fusion.

    Loaded from `backend/inference/fusion_plans/*.yaml`. Each scheme
    carries ONE piece of load-bearing data: a `naive_pattern` declaring
    the subgraph the matcher should look for plus the `io_mapping` from
    fused-op operand names to matched-node ports. The graph rewrite derives
    input/output topology; each analysis layer derives its fused value by
    rebasing with that layer's combine law.

    The class is `extra="forbid"` — any field outside `naive_pattern`
    in the YAML is rejected at load time. This is deliberate: the old
    schema accumulated `flops` / `params` / `formula_symbols` /
    `dtype_env` / `category` / `signature` / `shape_out` fields that
    were author-declared but redundant with siblings' actual costs (or
    in some cases never consumed at all), and authors had a 100% miss
    rate getting them to agree with the engine. The slim schema makes
    it structurally impossible to author divergent data.

    Per CLAUDE.md §2.6 (extended): `ops.py` stays fusion-agnostic; this
    class lives at the schema layer because it shares Pydantic with the
    rest of the schema, but the directory it loads from sits under
    `backend/inference/` to preserve the analysis-vs-spec separation.
    """
    naive_pattern: NaivePattern
    #: Where a framework's source shows the launch writes its output into a
    #: buffer of its own rather than rewriting its input in place — the
    #: source evidence, as a string; empty when the match's topology says
    #: all there is to say. Only a plan that absorbs a re-partition needs it
    #: (a `slice` / `concat` pair around the lanes it computes): without it
    #: the byte price reads that pair as lanes nobody moved
    #: (`fusion._passed_through`) and bills the computed lanes alone, while
    #: a kernel that allocates its output reads and writes every lane
    #: (`fused_inv_rope`'s whole of o). A launch that quantises its output
    #: says so by naming the cast as a member (`quantise_activation`), not
    #: here — that is topology, and the match sees it.
    allocates_output: str = ""
    #: The dtype the launch writes each of its outputs at where that is not
    #: the activation dtype the deployment chose, keyed by the output's
    #: `io_mapping` port (`output` the primary): MiniMax-M3's block-max
    #: indexer scores land as float32 (`{output: float32}`), sglang's GDN
    #: gate launch writes g and beta as float32 (`{g_out: float32,
    #: beta_out: float32}`). A port not listed is written as its edge
    #: carries it. Only for a fused op with no primitive registration — a
    #: registered op (`indexer_logits`) states its output width once, on the
    #: op, and `Fusion.named` refuses a second statement.
    output_dtype: Dict[str, str] = {}
    #: The scale sidecar's STORAGE tile on the quantised output record, when
    #: the launch pads it: `{rows: 128, groups: 4}` is FlashInfer's F8_128x4
    #: (vLLM `mxfp8_scale_bytes`, query_quant.py:31-33 — token rows rounded
    #: to 128, 32-value groups to 4, the padded positions stored as zeros),
    #: which at a 64-token cell writes twice the scale bytes the generic
    #: per-group law prices. Unset: the record is exactly tokens × groups.
    #: A fact about the record the launch writes, not a cost — the bytes
    #: stay derived in `traffic`.
    output_scale_tile: Optional[Dict[str, int]] = None

    model_config = {"extra": "forbid"}

    @model_validator(mode="after")
    def _output_scale_tile_is_a_tile(self) -> "FusionOpSpec":
        if self.output_scale_tile is None:
            return self
        if "quantise_activation" not in self.naive_pattern.ops():
            raise ValueError(
                "output_scale_tile describes the quantised output record; "
                "the plan's pattern has no `quantise_activation` member")
        if set(self.output_scale_tile) != {"rows", "groups"}:
            raise ValueError(
                f"output_scale_tile is {{rows, groups}}, got "
                f"{sorted(self.output_scale_tile)}")
        for key, value in self.output_scale_tile.items():
            if not isinstance(value, int) or value <= 0:
                raise ValueError(
                    f"output_scale_tile[{key!r}]: {value!r} is not a "
                    f"positive int")
        return self

    @model_validator(mode="after")
    def _output_dtype_names_output_ports(self) -> "FusionOpSpec":
        """Each key is an output port of `io_mapping` (a target that is a
        pattern node, not `external.*`), each value a dtype the width table
        knows: a width on a port the launch does not write, or a guessed
        width, is refused here rather than mispriced later."""
        from backend.arch.dtypes import UnknownDtype, canonical
        outputs = {port for port, target in
                   self.naive_pattern.io_mapping.items()
                   if not str(target).startswith("external.")}
        for port, dtype in self.output_dtype.items():
            if port not in outputs:
                raise ValueError(
                    f"output_dtype[{port!r}]: not an output port of the "
                    f"pattern's io_mapping {sorted(outputs)}")
            if not isinstance(dtype, str):
                raise ValueError(
                    f"output_dtype[{port!r}]: {dtype!r} is not a dtype name")
            try:
                canonical(dtype)
            except UnknownDtype as e:
                raise ValueError(f"output_dtype[{port!r}]: {e}") from e
        return self
